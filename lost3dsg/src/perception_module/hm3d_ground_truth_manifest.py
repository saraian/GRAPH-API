#!/usr/bin/env python3
"""Esporta la geometria ground truth HM3D senza envelope artificiali.

Quando disponibili, usa polyloop XZ e AABB XYZ nativi di Habitat-Sim. Le build
con dati nativi incompleti vengono integrate dalla mesh semantica: le stanze
derivano dai triangoli di pavimento/soffitto (muri come ultima risorsa) e gli
oggetti mancanti sono associati tramite il loro ID semantico. Non costruisce
stanze unendo bounding box di oggetti.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import struct
from pathlib import Path

import numpy as np


WALL_CATEGORY_NAMES = frozenset({
    "wall", "wall panel", "fireplace wall", "shower wall", "partition", "column",
})


def _xyz(value):
    return np.asarray([float(value[0]), float(value[1]), float(value[2])])


def _category(entity):
    category = getattr(entity, "category", None)
    try:
        name = str(category.name()).strip().lower()
    except (AttributeError, TypeError):
        name = str(category or "").strip().lower()
    try:
        index = int(category.index())
    except (AttributeError, TypeError, ValueError):
        index = None
    return name, index


def _region_key(value):
    """Canonicalize Habitat-Sim IDs (``_0``) and semantic.txt IDs (``0``)."""
    text = str(value).strip()
    return text[1:] if text.startswith("_") and text[1:].isdigit() else text


def _scene_number(scene: Path):
    match = re.match(r"(\d+)", scene.parent.name)
    return match.group(1) if match else scene.stem.split(".")[0]


def _cluster_heights(values, tolerance):
    groups = []
    for height in sorted(float(v) for v in values if math.isfinite(float(v))):
        if not groups or height - float(np.median(groups[-1])) > tolerance:
            groups.append([height])
        else:
            groups[-1].append(height)
    return [float(np.median(group)) for group in groups]


def _aabb(entity):
    # In alcune Habitat-Sim builds SemanticObject.aabb is an envelope of the
    # semantic mesh.  The per-object OBB is the reliable geometry; convert it
    # to an axis-aligned box in Habitat coordinates first.
    obb = getattr(entity, "obb", None)
    if obb is not None:
        # Do not use ``obb.aabb`` or ``obb.to_aabb``: in the affected binding
        # both return the same scene-level envelope as SemanticObject.aabb.
        try:
            center = _xyz(obb.center)
            sizes = _xyz(obb.sizes)
            rotation = getattr(obb, "rotation", None)
            if rotation is not None:
                matrix = np.asarray(rotation, dtype=float)
                if matrix.shape == (3, 3):
                    half = np.abs(matrix) @ (sizes / 2.0)
                    low, high = center - half, center + half
                    if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                        return low, high
            low, high = center - sizes / 2.0, center + sizes / 2.0
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                return low, high
        except (AttributeError, TypeError, ValueError):
            pass

    box = getattr(entity, "aabb", None)
    return _range3d_aabb(box)


def _range3d_aabb(box):
    if box is None:
        return None
    try:
        # SemanticObject.aabb in Habitat-Sim exposes ``center`` and ``sizes``
        # (the API used by HOV-SG).  Some Magnum-backed bindings additionally
        # expose min/max, so retain those only as an API compatibility path;
        # both branches still read the same native Habitat AABB.
        center = _xyz(box.center)
        sizes = _xyz(box.sizes)
        low, high = center - sizes / 2.0, center + sizes / 2.0
    except (AttributeError, TypeError, ValueError):
        try:
            low, high = _xyz(box.min), _xyz(box.max)
        except (AttributeError, TypeError, ValueError):
            return None
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        return None
    if np.any(high < low) or float(np.max(high - low)) <= 1e-6:
        return None
    return low, high


def _semantic_paths(scene):
    """Trova semantic.glb/.txt anche nel layout HM3D a cartelle sorelle."""
    direct = scene.with_name(scene.name.replace(".basis.glb", ".semantic.glb"))
    candidates = [
        direct,
        Path(str(direct).replace("hm3d-val-habitat-v0.2", "hm3d-val-semantic-annots-v0.2")),
    ]
    for mesh in candidates:
        text = mesh.with_suffix(".txt")
        if mesh.is_file() and text.is_file():
            return mesh, text
    raise FileNotFoundError(
        "asset semantici non trovati; attesi vicino alla scena o nella cartella "
        "hm3d-val-semantic-annots-v0.2: " + ", ".join(str(p) for p in candidates)
    )


def _read_glb(path):
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError(f"file non GLB: {path}")
    offset, gltf, binary = 12, None, b""
    while offset + 8 <= len(raw):
        length, kind = struct.unpack_from("<II", raw, offset)
        chunk = raw[offset + 8:offset + 8 + length]
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + length
    if gltf is None:
        raise RuntimeError(f"chunk JSON assente nel GLB: {path}")
    return gltf, binary


def _accessor(gltf, binary, index):
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    formats = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
    sizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    counts = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    component, width = acc["componentType"], counts[acc["type"]]
    stride = view.get("byteStride", sizes[component] * width)
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    fmt = "<" + formats[component] * width
    values = [struct.unpack_from(fmt, binary, start + i * stride)
              for i in range(acc["count"])]
    if acc.get("normalized") and component != 5126:
        limits = {5120: 127.0, 5121: 255.0, 5122: 32767.0,
                  5123: 65535.0, 5125: 4294967295.0}
        values = [tuple(max(-1.0, x / limits[component]) for x in row) for row in values]
    return values


def _node_matrix(node):
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=float).reshape((4, 4), order="F")
    x, y, z, w = (float(v) for v in node.get("rotation", (0, 0, 0, 1)))
    matrix = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 0],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 0],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), 0],
        [0, 0, 0, 1],
    ], dtype=float)
    matrix[:3, :3] *= np.asarray(node.get("scale", (1, 1, 1)), dtype=float)[None, :]
    matrix[:3, 3] = np.asarray(node.get("translation", (0, 0, 0)), dtype=float)
    return matrix


def _mesh_transforms(gltf):
    nodes = gltf.get("nodes", [])
    scene_index = int(gltf.get("scene", 0))
    scenes = gltf.get("scenes", [])
    roots = scenes[scene_index].get("nodes", []) if scene_index < len(scenes) else []
    if not roots:
        children = {int(c) for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    result = {}
    def visit(index, parent):
        world = parent @ _node_matrix(nodes[index])
        if "mesh" in nodes[index]:
            result.setdefault(int(nodes[index]["mesh"]), []).append(world)
        for child in nodes[index].get("children", []):
            visit(int(child), world)
    for root in roots:
        visit(int(root), np.identity(4))
    return result


def _semantic_records(path):
    records = {}
    for row in csv.reader(path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if len(row) < 3 or not row[0].strip().isdigit():
            continue
        color = row[1].strip().lstrip("#")
        if not re.fullmatch(r"[0-9a-fA-F]{6}", color):
            continue
        records[tuple(int(color[i:i+2], 16) for i in (0, 2, 4))] = {
            "object_id": int(row[0]), "category_name": row[2].strip().strip('"').lower(),
            "region_id": row[3].strip() if len(row) > 3 else None,
        }
    return records


def _semantic_mesh_aabbs(mesh_path, text_path):
    """Calcola un AABB per colore/istanza direttamente dai triangoli HM3D."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow è necessario per leggere le texture semantic.glb") from exc
    records = _semantic_records(text_path)
    gltf, binary = _read_glb(mesh_path)
    transforms, images, boxes, triangles_by_color = _mesh_transforms(gltf), {}, {}, {}
    for mesh_index, mesh in enumerate(gltf.get("meshes", [])):
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            positions = _accessor(gltf, binary, attrs["POSITION"])
            indices = ([int(v[0]) for v in _accessor(gltf, binary, primitive["indices"])]
                       if "indices" in primitive else list(range(len(positions))))
            vertex_colors = _accessor(gltf, binary, attrs["COLOR_0"]) if "COLOR_0" in attrs else None
            texcoords, image, texture_transform = None, None, None
            if any(key.startswith("TEXCOORD_") for key in attrs) and "material" in primitive:
                material = gltf.get("materials", [])[primitive["material"]]
                texture = material.get("pbrMetallicRoughness", {}).get("baseColorTexture")
                if texture is not None:
                    transform_ext = texture.get("extensions", {}).get("KHR_texture_transform", {})
                    texcoord_set = int(transform_ext.get("texCoord", texture.get("texCoord", 0)))
                    texcoord_name = f"TEXCOORD_{texcoord_set}"
                    if texcoord_name not in attrs:
                        raise RuntimeError(
                            f"{mesh_path}: materiale richiede {texcoord_name}, accessor assente"
                        )
                    texcoords = _accessor(gltf, binary, attrs[texcoord_name])
                    texture_transform = transform_ext or None
                    source = gltf["textures"][texture["index"]]["source"]
                    if source not in images:
                        item = gltf["images"][source]
                        view = gltf["bufferViews"][item["bufferView"]]
                        start = view.get("byteOffset", 0)
                        images[source] = Image.open(io.BytesIO(
                            binary[start:start + view["byteLength"]])).convert("RGB")
                    image = images[source]
            for transform in transforms.get(mesh_index, [np.identity(4)]):
                for start in range(0, len(indices) - 2, 3):
                    tri = indices[start:start + 3]
                    if image is not None:
                        uv = np.mean([texcoords[i][:2] for i in tri], axis=0)
                        if texture_transform:
                            offset = np.asarray(texture_transform.get("offset", (0.0, 0.0)), dtype=float)
                            scale = np.asarray(texture_transform.get("scale", (1.0, 1.0)), dtype=float)
                            angle = float(texture_transform.get("rotation", 0.0))
                            scaled = uv * scale
                            uv = offset + np.asarray([
                                math.cos(angle) * scaled[0] - math.sin(angle) * scaled[1],
                                math.sin(angle) * scaled[0] + math.cos(angle) * scaled[1],
                            ])
                        u, v = np.clip(uv, 0, 0.999999)
                        # glTF 2.0 defines UV (0, 0) at the upper-left of the
                        # image. PIL uses the same origin, therefore flipping
                        # v here assigns triangles to unrelated atlas texels.
                        color = image.getpixel((int(u * image.width),
                                                int(v * image.height)))
                    elif vertex_colors is not None:
                        rgb = np.mean([vertex_colors[i][:3] for i in tri], axis=0)
                        color = tuple(int(round(float(c) * 255)) if float(c) <= 1 else int(round(c)) for c in rgb)
                    else:
                        continue
                    if color not in records:
                        continue
                    points = []
                    for i in tri:
                        p = transform @ np.array([*positions[i][:3], 1.0])
                        p = p[:3] / p[3] if abs(p[3]) > 1e-12 else p[:3]
                        points.append(np.array([p[0], p[2], -p[1]]))  # HM3D Z-up -> Habitat Y-up
                    low, high = np.min(points, axis=0), np.max(points, axis=0)
                    old = boxes.get(color)
                    boxes[color] = (low, high) if old is None else (np.minimum(old[0], low), np.maximum(old[1], high))
                    triangles_by_color.setdefault(color, []).append(
                        np.asarray(points, dtype=float).tolist())
    return [{**records[color], "color_rgb": list(color), "aabb": box,
             "triangles": triangles_by_color.get(color, [])}
            for color, box in boxes.items()]


def _reconstruct_region_from_walls(objects, region_id, resolution=0.05):
    """Recover a room polygon from its semantic structural triangles.

    The HM3D semantic scene loaded by some Habitat-Sim builds exposes region
    IDs but no region polyloops or object AABBs.  Prefer the actual floor (or
    ceiling) triangles, whose union is the room footprint.  If neither is
    available, rasterize the walls and recover their enclosed component.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV è necessario per ricostruire le regioni HM3D dai triangoli"
        ) from exc
    region_objects = [obj for obj in objects
                      if _region_key(obj.get("region_id") or "") == _region_key(region_id)]
    walls = [obj for obj in region_objects
             if str(obj.get("category_name", "")).strip().lower()
             in WALL_CATEGORY_NAMES]
    triangles = [np.asarray(triangle, dtype=float) for obj in walls
                 for triangle in obj.get("triangles", [])
                 if np.asarray(triangle).shape == (3, 3)]
    structural_triangles = [np.asarray(triangle, dtype=float)
                            for obj in region_objects
                            if str(obj.get("category_name", "")).strip().lower()
                            in WALL_CATEGORY_NAMES | {"floor", "ceiling"}
                            for triangle in obj.get("triangles", [])
                            if np.asarray(triangle).shape == (3, 3)]
    if not structural_triangles:
        return None
    vertical_low = float(min(triangle[:, 1].min()
                             for triangle in structural_triangles))
    vertical_high = float(max(triangle[:, 1].max()
                              for triangle in structural_triangles))

    def surface_footprint(category):
        surface = [np.asarray(triangle, dtype=float)
                   for obj in region_objects
                   if str(obj.get("category_name", "")).strip().lower() == category
                   for triangle in obj.get("triangles", [])
                   if np.asarray(triangle).shape == (3, 3)]
        if not surface:
            return None
        projected_surface = [triangle[:, [0, 2]] for triangle in surface]
        projected_walls = [triangle[:, [0, 2]] for triangle in triangles]
        footprint_triangles = projected_surface + projected_walls
        # Symmetric padding keeps morphology away from the image border.  A
        # mask starting at pixel zero used to enlarge max bounds by one voxel.
        padding = max(0.40, 2.0 * resolution)
        surface_low = (np.min([points.min(axis=0) for points in footprint_triangles],
                              axis=0) - padding)
        surface_high = (np.max([points.max(axis=0) for points in footprint_triangles],
                               axis=0) + padding)
        surface_shape = np.maximum(
            3, np.ceil((surface_high - surface_low) / resolution).astype(int) + 3)
        if int(np.prod(surface_shape)) > 4_000_000:
            return None
        surface_mask = np.zeros((int(surface_shape[1]), int(surface_shape[0])),
                                dtype=np.uint8)
        for triangle in projected_surface:
            points = np.rint((triangle - surface_low) / resolution).astype(np.int32)
            if abs(float(cv2.contourArea(points.reshape(-1, 1, 2)))) >= 1.0:
                cv2.fillPoly(surface_mask, [points], 1)
        # Include the complete wall geometry assigned to this region.  Most
        # wall faces are vertical and collapse to line segments in X-Z, so
        # rasterize those with a small physical thickness instead of dropping
        # them as zero-area triangles.
        wall_thickness = max(1, int(round(0.10 / resolution)))
        for triangle in projected_walls:
            points = np.rint((triangle - surface_low) / resolution).astype(np.int32)
            area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2))))
            if area >= 1.0:
                cv2.fillPoly(surface_mask, [points], 1)
            else:
                cv2.polylines(surface_mask, [points], False, 1,
                              thickness=wall_thickness)
        if not np.any(surface_mask):
            return None
        # Join wall/floor annotation seams.  Doors do not create a false
        # connection here because the filled room surface is already on one
        # side; this only prevents a separately meshed wall strip from being
        # discarded as a smaller external contour.
        close_size = max(3, int(round(0.40 / resolution)) | 1)
        surface_mask = cv2.morphologyEx(
            surface_mask, cv2.MORPH_CLOSE,
            np.ones((close_size, close_size), dtype=np.uint8))
        contours, _ = cv2.findContours(surface_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = cv2.approxPolyDP(max(contours, key=cv2.contourArea),
                                   max(1.0, 0.03 / resolution), True).reshape(-1, 2)
        if len(contour) < 3:
            return None
        polygon = [[float(surface_low[0] + point[0] * resolution),
                    float(surface_low[1] + point[1] * resolution)]
                   for point in contour]
        points = np.asarray(polygon, dtype=float)
        box_low = np.asarray([points[:, 0].min(), vertical_low,
                              points[:, 1].min()], dtype=float)
        box_high = np.asarray([points[:, 0].max(), vertical_high,
                               points[:, 1].max()], dtype=float)
        source = (f"semantic_{category}_and_wall_triangles" if projected_walls
                  else f"semantic_{category}_triangles")
        return polygon, (box_low, box_high), source

    # Floor annotations are normally the most direct footprint.  Ceiling is
    # a reliable substitute for regions where carpet replaces the floor label.
    surface_result = surface_footprint("floor") or surface_footprint("ceiling")
    if surface_result is not None:
        return surface_result
    if not triangles:
        return None

    # ``_semantic_mesh_aabbs`` has already converted vertices to Habitat's
    # Y-up frame [x, y, z].  A room therefore lives in the X-Z plane.  Using
    # columns [0, 1] here creates an elevation silhouette and used to produce
    # plausible-looking, but geometrically false, room polygons.
    projected = [triangle[:, [0, 2]] for triangle in triangles]
    lows = np.asarray([points.min(axis=0) for points in projected], dtype=float)
    highs = np.asarray([points.max(axis=0) for points in projected], dtype=float)
    low = lows.min(axis=0) - 0.5
    high = highs.max(axis=0) + 0.5
    shape = np.maximum(3, np.ceil((high - low) / resolution).astype(int) + 1)
    if int(np.prod(shape)) > 4_000_000:
        resolution *= np.sqrt(float(np.prod(shape)) / 4_000_000)
        shape = np.maximum(3, np.ceil((high - low) / resolution).astype(int) + 1)
    mask = np.zeros((int(shape[1]), int(shape[0])), dtype=np.uint8)
    for triangle in projected:
        points = np.rint((triangle - low) / resolution).astype(np.int32)
        area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2))))
        if area >= 1.0:
            cv2.fillPoly(mask, [points], 1)
        else:
            # Vertical wall surfaces collapse to line segments in X-Z.
            cv2.polylines(mask, [points], False, 1,
                          thickness=max(1, int(round(0.10 / resolution))))
    # Close door-sized gaps in the wall segments.  This operates on the
    # region-specific triangle raster only; it cannot merge two rooms.
    close_size = max(3, int(round(0.75 / resolution)) | 1)
    kernel = np.ones((close_size, close_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def result_from_contour(contour):
        contour = contour.reshape(-1, 2)
        if len(contour) < 3:
            return None
        polygon = [[float(low[0] + point[0] * resolution),
                    float(low[1] + point[1] * resolution)] for point in contour]
        points = np.asarray(polygon, dtype=float)
        box_low = np.asarray([points[:, 0].min(), vertical_low,
                              points[:, 1].min()], dtype=float)
        box_high = np.asarray([points[:, 0].max(), vertical_high,
                               points[:, 1].max()], dtype=float)
        return polygon, (box_low, box_high), "semantic_wall_triangles"

    def wall_mask_contour(source):
        # Wall instances can be split at corners and around doors.  A small
        # dilation joins those pieces while keeping the contour derived from
        # the projected wall triangles.
        join_size = max(3, int(round(0.35 / resolution)) | 1)
        joined = cv2.dilate(source, np.ones((join_size, join_size), dtype=np.uint8))
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        contour = cv2.approxPolyDP(contour, max(1.5, 0.04 / resolution), True)
        return result_from_contour(contour)

    def wall_points_hull():
        # Last geometric reconstruction step: use only the projected points
        # of this region's semantic wall triangles.  This is intentionally a
        # wall-derived contour, not an object-AABB fallback.
        points = np.concatenate(projected, axis=0)
        points = np.unique(np.rint((points - low) / resolution).astype(np.int32), axis=0)
        if len(points) < 3:
            return None
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        hull = cv2.approxPolyDP(hull, max(1.5, 0.04 / resolution), True)
        hull = hull.reshape(-1, 2)
        if len(hull) < 3 or cv2.contourArea(hull.reshape(-1, 1, 2)) < 4.0:
            return None
        return result_from_contour(hull)

    free = (mask == 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(free, connectivity=4)
    if count <= 1:
        return wall_mask_contour(mask) or wall_points_hull()
    boundary = np.unique(np.concatenate((labels[0, :], labels[-1, :],
                                         labels[:, 0], labels[:, -1])))
    candidates = [(int(np.count_nonzero(labels == idx)), idx)
                  for idx in range(1, count) if idx not in set(boundary)]
    if not candidates:
        # A region may have doors/openings, so its free-space component can
        # leak to the raster boundary.  Extract the outer contour of the
        # wall-triangle mask itself; this remains based on the semantic wall
        # triangles and is not an object-AABB/envelope fallback.
        return wall_mask_contour(mask) or wall_points_hull()
    _, chosen = max(candidates)
    component = (labels == chosen).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.5, 0.04 / resolution)
    contour = cv2.approxPolyDP(contour, epsilon, True)
    return result_from_contour(contour)


def _wall_triangle_coverage(polygon, objects, region_id, tolerance):
    """Count wall triangles represented by a region polygon within tolerance."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV è necessario per validare i muri HM3D") from exc
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    centroids = [np.asarray(triangle, dtype=float)[:, [0, 2]].mean(axis=0)
                 for obj in objects
                 if _region_key(obj.get("region_id") or "") == _region_key(region_id)
                 and str(obj.get("category_name", "")).strip().lower()
                 in WALL_CATEGORY_NAMES
                 for triangle in obj.get("triangles", [])
                 if np.asarray(triangle).shape == (3, 3)]
    covered = sum(
        cv2.pointPolygonTest(contour, tuple(map(float, point)), True) >= -tolerance
        for point in centroids
    )
    return covered, len(centroids)


def _grid_spec(low, high, resolution):
    shape = np.maximum(1, np.ceil((high - low) / resolution).astype(int))
    return {
        "origin_m": [float(v) for v in low],
        "resolution_m": float(resolution),
        "shape_xyz": [int(v) for v in shape],
        "flattening": "x + nx * (z + nz * y)",
    }


def _voxel_mask(low, high, spec):
    origin = np.asarray(spec["origin_m"])
    resolution = float(spec["resolution_m"])
    nx, ny, nz = spec["shape_xyz"]
    first = np.maximum(0, np.floor((low - origin) / resolution).astype(int))
    last = np.minimum([nx - 1, ny - 1, nz - 1],
                      np.floor((high - origin) / resolution).astype(int))
    if np.any(last < first):
        return []
    result = []
    for iy in range(first[1], last[1] + 1):
        for iz in range(first[2], last[2] + 1):
            base = nx * (iz + nz * iy)
            result.extend(range(base + first[0], base + last[0] + 1))
    return result


def _region_mask(low, high, origin_xz, shape_xz, resolution, floor_index):
    nx, nz = shape_xz
    first = np.maximum(0, np.floor((low[[0, 2]] - origin_xz) / resolution).astype(int))
    last = np.minimum([nx - 1, nz - 1],
                      np.floor((high[[0, 2]] - origin_xz) / resolution).astype(int))
    if np.any(last < first):
        return []
    plane = nx * nz
    return [int(ix + nx * iz + plane * floor_index)
            for iz in range(first[1], last[1] + 1)
            for ix in range(first[0], last[0] + 1)]


def _extract_texture_geometry(scene, semantic_mesh, semantic_text,
                              floor_tolerance, selected_floor,
                              region_resolution=0.10):
    """Fallback HM3D v0.2: geometria dalla texture semantica del GLB."""
    objects = _semantic_mesh_aabbs(semantic_mesh, semantic_text)
    if not objects:
        raise RuntimeError(f"nessuna istanza decodificata da {semantic_mesh}")

    region_ids = sorted({_region_key(obj.get("region_id")) for obj in objects
                         if obj.get("region_id") not in (None, "")},
                        key=lambda value: (not value.isdigit(),
                                           int(value) if value.isdigit() else value))
    region_geometry = {}
    missing_regions = []
    for rid in region_ids:
        reconstructed = _reconstruct_region_from_walls(
            objects, rid, resolution=region_resolution)
        if reconstructed is None:
            missing_regions.append(rid)
        else:
            region_geometry[rid] = reconstructed
    if missing_regions:
        raise RuntimeError(
            "GT rifiutata: nessuna impronta strutturale affidabile per le "
            f"regioni {missing_regions}"
        )
    floors = _cluster_heights([box[0][1] for _, box, _ in region_geometry.values()],
                              floor_tolerance)
    region_floor = {
        rid: int(np.argmin(np.abs(np.asarray(floors) - box[0][1])))
        for rid, (_, box, _) in region_geometry.items()
    }
    all_floors = list(floors)
    if selected_floor is not None:
        if selected_floor < 0 or selected_floor >= len(floors):
            raise ValueError(f"--floor-index={selected_floor} fuori intervallo 0..{len(floors)-1}")
        keep = {rid for rid, fi in region_floor.items() if fi == selected_floor}
        objects = [o for o in objects if _region_key(o.get("region_id") or "unknown") in keep]
        region_geometry = {rid: geometry for rid, geometry in region_geometry.items()
                           if rid in keep}
        floors = [floors[selected_floor]]
        region_floor = {rid: 0 for rid in keep}

    categories = {name: i for i, name in enumerate(sorted(
        {o["category_name"] for o in objects}))}
    gt_objects = []
    suspicious = []
    for obj in objects:
        low, high = obj["aabb"]
        size = high - low
        row = {"object_id": str(obj["object_id"]),
               "category_id": categories[obj["category_name"]],
               "category_name": obj["category_name"],
               "region_id": obj.get("region_id"),
               "color_rgb": obj["color_rgb"],
               "aabb_min_m": low.tolist(), "aabb_max_m": high.tolist(),
               "geometry_source": "semantic_glb_texture"}
        gt_objects.append(row)
        if float(np.max(size)) > 8.0 or float(np.prod(size)) <= 1e-9:
            suspicious.append({"object_id": row["object_id"],
                               "category_name": row["category_name"],
                               "size_m": size.tolist()})

    gt_regions = []
    for rid, (polygon, (low, high), geometry_source) in region_geometry.items():
        covered_walls, wall_count = _wall_triangle_coverage(
            polygon, objects, rid, max(0.10, 2.0 * region_resolution))
        coverage = covered_walls / wall_count if wall_count else 1.0
        if coverage < 0.999:
            raise RuntimeError(
                f"GT rifiutata: regione {rid} copre {covered_walls}/{wall_count} "
                "triangoli di muro"
            )
        gt_regions.append({"region_id": rid, "floor_index": region_floor[rid],
                           "category_id": None, "category_name": "",
                           "polygon_xz_m": polygon,
                           "aabb_min_m": low.tolist(), "aabb_max_m": high.tolist(),
                           "geometry_source": geometry_source,
                           "wall_triangle_count": wall_count,
                           "wall_triangle_outlier_count": wall_count - covered_walls,
                           "wall_triangle_coverage_pct": (
                               round(100.0 * coverage, 4) if wall_count else None),
                           "geometry_is_exact": False})
    return {
        "scene": _scene_number(scene), "construction_time_s": None,
        "predicted_floors_m": [], "ground_truth_floors_m": floors,
        "predicted_regions": [], "ground_truth_regions": gt_regions,
        "rooms": [{"region_id": r["region_id"], "predicted_label": None,
                   "ground_truth_label": "", "approximately_correct": None}
                  for r in gt_regions],
        "category_embeddings": [], "predicted_objects": [],
        "ground_truth_objects": gt_objects, "retrieval_trials": [],
        "representation_files": [],
        "geometry_space": {"coordinate_frame": "Habitat: x,z orizzontali; y verticale",
                           "region_geometry": "polygon_xz_m",
                           "object_geometry": "aabb_min_m/aabb_max_m"},
        "categories": [{"category_id": i, "category_name": n}
                       for n, i in categories.items()],
        "ground_truth_source": {
            "scene": str(scene.resolve()), "semantic_mesh": str(semantic_mesh.resolve()),
            "semantic_descriptor": str(semantic_text.resolve()),
            "method": "semantic_glb_structural_triangle_parser",
            "uv_origin": "upper-left (glTF 2.0; no vertical flip)",
            "selected_floor_index": selected_floor,
            "all_floor_heights_m": all_floors,
            "suspicious_object_count": len(suspicious),
            "suspicious_objects": suspicious[:100]},
    }


def extract(sim, scene: Path, region_resolution=0.10, object_voxel=0.10,
            floor_tolerance=0.50, selected_floor=None):
    semantic = sim.semantic_scene
    habitat_regions = [r for r in (getattr(semantic, "regions", None) or []) if r is not None]
    habitat_objects = [o for o in (getattr(semantic, "objects", None) or []) if o is not None]

    # Object GT geometry is authoritative only when exposed by Habitat-Sim.
    # The semantic texture remains useful for reconstructing missing *region*
    # boundaries, but must never supply or complete object bounding boxes.
    native_object_count = sum(_aabb(obj) is not None for obj in habitat_objects)
    if not habitat_regions:
        raise RuntimeError("SemanticScene senza regioni native")
    # Habitat-Sim adds the synthetic Unknown_0 object (semantic_id 0), which
    # has no corresponding geometry in HM3D annotations and therefore keeps a
    # zero AABB.  Require coverage only for actual annotated instances.
    annotated_objects = [obj for obj in habitat_objects
                         if int(getattr(obj, "semantic_id", 0)) != 0]
    annotated_native_count = sum(_aabb(obj) is not None for obj in annotated_objects)
    missing_objects = [
        f"{getattr(obj, 'id', '?')} (semantic_id={getattr(obj, 'semantic_id', '?')})"
        for obj in annotated_objects if _aabb(obj) is None
    ]
    if missing_objects:
        print(
            "ATTENZIONE: AABB nativo non trovato per "
            + ", ".join(missing_objects)
            + "; genero comunque gli altri oggetti.",
            flush=True,
        )

    semantic_mesh, semantic_text = _semantic_paths(scene)
    texture_objects = _semantic_mesh_aabbs(semantic_mesh, semantic_text)
    annotated_region_ids = {
        _region_key(obj.get("region_id")) for obj in texture_objects
        if obj.get("region_id") not in (None, "")
    }
    if annotated_region_ids:
        native_region_ids = {_region_key(region.id) for region in habitat_regions}
        missing_native_regions = annotated_region_ids - native_region_ids
        if missing_native_regions:
            raise RuntimeError(
                "SemanticScene nativa con regioni mancanti: "
                f"{sorted(missing_native_regions)}"
            )
        habitat_regions = [region for region in habitat_regions
                           if _region_key(region.id) in annotated_region_ids]

    regions = []
    for region in habitat_regions:
        # Habitat-Sim uses Y-up coordinates; room polygons live in the
        # horizontal X-Z plane, not X-Y.
        # Habitat-Sim exposes Vector2 points for region polyloops in some
        # versions and Vector3-like points in others.
        polygon = []
        for p in (getattr(region, "poly_loop_points", None) or []):
            try:
                polygon.append([float(p[0]), float(p[2]) if len(p) >= 3
                                else float(p[1])])
            except (TypeError, IndexError):
                polygon = []
                break
        box = _aabb(region)
        geometry_source = "semantic_region_poly_loop"
        if box is None or len(polygon) < 3:
            print(
                f"ATTENZIONE: geometria nativa non trovata per la regione "
                f"{_region_key(region.id)}; continuo con le altre regioni.",
                flush=True,
            )
            continue
        floor_height = float(getattr(region, "floor_height", box[0][1]))
        if geometry_source == "semantic_object_aabb_envelope":
            floor_height = float(box[0][1])
        if not math.isfinite(floor_height):
            floor_height = float(box[0][1])
        regions.append((_region_key(region.id), polygon, box, floor_height, region,
                        geometry_source))
    if len(regions) != len(habitat_regions):
        missing_geometry = sorted(
            {_region_key(region.id) for region in habitat_regions} -
            {item[0] for item in regions}
        )
        print(
            "ATTENZIONE: regioni senza geometria nativa ignorate: "
            + ", ".join(missing_geometry),
            flush=True,
        )

    floors = _cluster_heights([item[3] for item in regions], floor_tolerance)
    region_floor = {region_id: int(np.argmin(np.abs(np.asarray(floors) - floor_height)))
                    for region_id, _, _, floor_height, _, _ in regions}
    all_floors = list(floors)
    if selected_floor is not None:
        if selected_floor < 0 or selected_floor >= len(floors):
            raise ValueError(
                f"--floor-index={selected_floor} fuori intervallo; "
                f"la scena contiene {len(floors)} piani (0..{len(floors)-1})"
            )
        selected_region_ids = {
            region_id for region_id, index in region_floor.items() if index == selected_floor
        }
        regions = [item for item in regions if item[0] in selected_region_ids]
        habitat_objects = [obj for obj in habitat_objects
                           if getattr(obj, "region", None) is not None
                           and _region_key(obj.region.id) in selected_region_ids]
        floors = [floors[selected_floor]]
        region_floor = {region_id: 0 for region_id in selected_region_ids}
        if not regions:
            raise RuntimeError(f"nessuna regione trovata per il piano {selected_floor}")

    gt_regions, room_rows = [], []
    for region_id, polygon, (low, high), floor_height, region, geometry_source in regions:
        floor_index = region_floor[region_id]
        name, category_id = _category(region)
        covered_walls, wall_count = _wall_triangle_coverage(
            polygon, texture_objects, region_id,
            max(0.10, 2.0 * region_resolution))
        coverage = covered_walls / wall_count if wall_count else 1.0
        if geometry_source != "semantic_region_poly_loop" and coverage < 0.999:
            raise RuntimeError(
                f"GT rifiutata: regione {region_id} copre "
                f"{covered_walls}/{wall_count} triangoli di muro"
            )
        gt_regions.append({
            "region_id": region_id,
            "polygon_xz_m": polygon,
            "floor_index": floor_index,
            "floor_height_m": floor_height,
            "category_id": category_id,
            "category_name": name,
            "geometry_source": geometry_source,
            "geometry_is_exact": geometry_source == "semantic_region_poly_loop",
            "wall_triangle_count": wall_count,
            "wall_triangle_outlier_count": wall_count - covered_walls,
            "wall_triangle_coverage_pct": (
                round(100.0 * coverage, 4)
                if wall_count else None),
            "aabb_min_m": low.tolist(),
            "aabb_max_m": high.tolist(),
        })
        room_rows.append({
            "region_id": region_id,
            "predicted_label": None,
            "ground_truth_label": name,
            "approximately_correct": None,
        })

    native_objects = []
    for obj in habitat_objects:
        box = _aabb(obj)
        if box is None:
            continue
        name, native_category_id = _category(obj)
        native_objects.append((obj, box, name, native_category_id))
    if selected_floor is not None:
        allowed_regions = {_region_key(region_id) for region_id, *_ in regions}
        texture_objects = [obj for obj in texture_objects
                           if _region_key(obj.get("region_id") or "unknown")
                           in allowed_regions]

    # Object boxes come exclusively from Habitat-Sim SemanticObject.aabb.
    object_rows = []
    for obj, (low, high), name, native_category_id in native_objects:
        region = getattr(obj, "region", None)
        semantic_id = int(obj.semantic_id)
        object_rows.append({
            "object_id": str(obj.id),
            "semantic_id": semantic_id,
            "native_category_id": native_category_id,
            "category_name": name,
            "region_id": _region_key(region.id) if region is not None else None,
            "aabb_min_m": low.tolist(),
            "aabb_max_m": high.tolist(),
            "geometry_source": "semantic_object_obb_to_aabb",
        })
    if not object_rows:
        raise RuntimeError("nessun AABB nativo esposto da Habitat-Sim")
    category_ids = {name: index for index, name in enumerate(
        sorted({row["category_name"] for row in object_rows}))}
    gt_objects = [{**row, "category_id": category_ids[row["category_name"]]}
                  for row in sorted(object_rows, key=lambda item: item["semantic_id"])]

    return {
        "scene": _scene_number(scene),
        "construction_time_s": None,
        "predicted_floors_m": [],
        "ground_truth_floors_m": floors,
        "predicted_regions": [],
        "ground_truth_regions": gt_regions,
        "rooms": room_rows,
        "category_embeddings": [],
        "predicted_objects": [],
        "ground_truth_objects": gt_objects,
        "retrieval_trials": [],
        "representation_files": [],
        "geometry_space": {
            "coordinate_frame": "Habitat: x,z orizzontali; y verticale",
            "region_geometry": "polygon_xz_m",
            "object_geometry": "aabb_min_m/aabb_max_m",
        },
        "categories": [{"category_id": value, "category_name": name}
                       for name, value in category_ids.items()],
        "ground_truth_source": {
            "scene": str(scene.resolve()),
            "semantic_mesh": str(semantic_mesh.resolve()) if semantic_mesh else None,
            "semantic_descriptor": str(semantic_text.resolve()) if semantic_text else None,
            "api": "Habitat-Sim SemanticScene/SemanticRegion/SemanticObject",
            "note": "Box oggetti derivati dall'OBB nativo Habitat-Sim; SemanticObject.aabb usato solo come fallback.",
            "object_geometry_source": "semantic_object_obb_to_aabb",
            "native_object_count": len(gt_objects),
            "texture_fallback_object_count": 0,
            "region_geometry_exact": all(
                row["geometry_is_exact"] for row in gt_regions
            ),
            "region_aabb_fallback_count": sum(
                not row["geometry_is_exact"] for row in gt_regions
            ),
            "selected_floor_index": selected_floor,
            "all_floor_heights_m": all_floors,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="file *.basis.glb della scena")
    parser.add_argument("--dataset-config", required=True, type=Path,
                        help="hm3d_annotated_basis.scene_dataset_config.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--region-resolution", type=float, default=0.10)
    parser.add_argument("--object-voxel", type=float, default=0.10)
    parser.add_argument("--floor-tolerance", type=float, default=0.50)
    parser.add_argument("--floor-index", type=int, default=None,
                        help="considera un solo piano (indice 0-based dal basso)")
    args = parser.parse_args()
    if args.region_resolution <= 0 or args.object_voxel <= 0 or args.floor_tolerance <= 0:
        parser.error("risoluzioni e tolleranza devono essere positive")
    if not args.scene.is_file():
        parser.error(f"scena inesistente: {args.scene}")
    if not args.dataset_config.is_file():
        parser.error(f"config inesistente: {args.dataset_config}")

    try:
        import habitat_sim
    except ImportError as exc:
        parser.error(f"habitat_sim necessario per gli AABB nativi: {exc}")
    else:
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(args.scene.resolve())
        backend.scene_dataset_config_file = str(args.dataset_config.resolve())
        backend.load_semantic_mesh = True
        # HM3D-Semantics v0.2 can store instance IDs in semantic textures.  Older
        # bindings expose the explicit switch below; newer/refactored bindings
        # select the semantic asset through the annotated dataset configuration.
        backend.requires_textures = True
        # This build must use semantic vertex colors to populate the native
        # SemanticObject OBB/AABB data.  Enabling semantic textures leaves the
        # semantic descriptor populated but keeps every native box at zero.
        if hasattr(backend, "use_semantic_textures"):
            backend.use_semantic_textures = False
        semantic_sensor = habitat_sim.CameraSensorSpec()
        semantic_sensor.uuid = "semantic"
        semantic_sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = [semantic_sensor]
        with habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent])) as sim:
            result = extract(sim, args.scene, args.region_resolution,
                             args.object_voxel, args.floor_tolerance,
                             args.floor_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(result['ground_truth_regions'])} regioni, "
          f"{len(result['ground_truth_objects'])} oggetti, "
          f"{len(result['ground_truth_floors_m'])} piani")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
