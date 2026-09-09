#!/usr/bin/env python3
"""Esporta la geometria GT HM3D caricata nativamente da Habitat-Sim.

Le regioni sono i polyloop XZ di ``SemanticRegion`` e gli oggetti sono gli
AABB XYZ di ``SemanticObject``. Non vengono inferite istanze dai colori della
texture e non vengono costruite stanze unendo bounding box di oggetti.
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
    box = getattr(entity, "aabb", None)
    if box is None:
        return None
    try:
        low, high = _xyz(box.min), _xyz(box.max)
    except (AttributeError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
        return None
    if float(np.max(high - low)) <= 1e-6:
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
    transforms, images, boxes, triangles = _mesh_transforms(gltf), {}, {}, {}
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
                    triangles.setdefault(color, []).append(
                        [[float(p[0]), float(p[2])] for p in points]
                    )
    return [{**records[color], "color_rgb": list(color), "aabb": box,
             "_footprint_triangles": triangles.get(color, [])}
            for color, box in boxes.items()]


def _region_floor_wall_polygon(objects, resolution=0.05):
    """Ricava un footprint XZ da triangoli floor, o wall come fallback.

    Usa solo la libreria standard: l'inviluppo convesso dei vertici preserva
    l'orientamento e l'estensione della superficie senza richiedere OpenCV,
    Shapely o altre dipendenze nell'ambiente Habitat.
    """
    preferred = [o for o in objects if o.get("category_name") == "floor"]
    source = "semantic_floor_mesh_footprint"
    if not preferred:
        preferred = [o for o in objects if o.get("category_name") == "wall"]
        source = "semantic_wall_mesh_footprint"
    triangles = [tri for obj in preferred for tri in obj.get("_footprint_triangles", [])]
    if not triangles:
        return None, None
    points = sorted({(round(float(p[0]) / resolution) * resolution,
                      round(float(p[1]) / resolution) * resolution)
                     for triangle in triangles for p in triangle})
    if len(points) < 3:
        return None, None

    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    polygon_xz = [[float(x), float(z)] for x, z in lower[:-1] + upper[:-1]]
    return polygon_xz, source


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
                              floor_tolerance, selected_floor):
    """Fallback HM3D v0.2: geometria dalla texture semantica del GLB."""
    objects = _semantic_mesh_aabbs(semantic_mesh, semantic_text)
    if not objects:
        raise RuntimeError(f"nessuna istanza decodificata da {semantic_mesh}")

    region_boxes = {}
    region_objects = {}
    for obj in objects:
        rid = str(obj.get("region_id") or "unknown")
        low, high = obj["aabb"]
        region_objects.setdefault(rid, []).append(obj)
        old = region_boxes.get(rid)
        region_boxes[rid] = ((low, high) if old is None else
                             (np.minimum(old[0], low), np.maximum(old[1], high)))
    floors = _cluster_heights([box[0][1] for box in region_boxes.values()],
                              floor_tolerance)
    region_floor = {rid: int(np.argmin(np.abs(np.asarray(floors) - box[0][1])))
                    for rid, box in region_boxes.items()}
    all_floors = list(floors)
    if selected_floor is not None:
        if selected_floor < 0 or selected_floor >= len(floors):
            raise ValueError(f"--floor-index={selected_floor} fuori intervallo 0..{len(floors)-1}")
        keep = {rid for rid, fi in region_floor.items() if fi == selected_floor}
        objects = [o for o in objects if str(o.get("region_id") or "unknown") in keep]
        region_boxes = {rid: box for rid, box in region_boxes.items() if rid in keep}
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
    for rid, (low, high) in region_boxes.items():
        polygon_xz, geometry_source = _region_floor_wall_polygon(
            region_objects.get(rid, []))
        if polygon_xz is None:
            polygon_xz = [[float(low[0]), float(low[2])],
                          [float(high[0]), float(low[2])],
                          [float(high[0]), float(high[2])],
                          [float(low[0]), float(high[2])]]
            geometry_source = "semantic_object_aabb_envelope"
        gt_regions.append({"region_id": rid, "floor_index": region_floor[rid],
                           "category_id": None, "category_name": "",
                           "polygon_xz_m": polygon_xz,
                           "aabb_min_m": low.tolist(), "aabb_max_m": high.tolist(),
                           "geometry_source": geometry_source,
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
            "method": "semantic_glb_texture_parser",
            "uv_origin": "upper-left (glTF 2.0; no vertical flip)",
            "selected_floor_index": selected_floor,
            "all_floor_heights_m": all_floors,
            "suspicious_object_count": len(suspicious),
            "suspicious_objects": suspicious[:100]},
    }


def extract(sim, scene: Path, region_resolution=0.10, object_voxel=0.10,
            floor_tolerance=0.50, selected_floor=None):
    semantic = sim.semantic_scene
    semantic_mesh, semantic_text = _semantic_paths(scene)
    return _extract_texture_geometry(scene, semantic_mesh, semantic_text,
                                     floor_tolerance, selected_floor)
    habitat_regions = [r for r in (getattr(semantic, "regions", None) or []) if r is not None]
    habitat_objects = [o for o in (getattr(semantic, "objects", None) or []) if o is not None]

    regions = []
    for region in habitat_regions:
        polygon = [[float(p[0]), float(p[1])]
                   for p in (getattr(region, "poly_loop_points", None) or [])]
        box = _aabb(region)
        geometry_source = "semantic_region_poly_loop"
        if box is None:
            # Some Habitat-Sim builds load HM3D region membership but leave
            # SemanticRegion geometry empty.  Preserve that membership and
            # derive only a clearly-labelled envelope from native object AABBs.
            member_boxes = []
            for obj in habitat_objects:
                obj_region = getattr(obj, "region", None)
                obj_box = _aabb(obj)
                if (obj_region is not None and str(obj_region.id) == str(region.id)
                        and obj_box is not None):
                    member_boxes.append(obj_box)
            if member_boxes:
                box = (np.min([item[0] for item in member_boxes], axis=0),
                       np.max([item[1] for item in member_boxes], axis=0))
                geometry_source = "semantic_object_aabb_envelope"
        if box is None:
            continue
        if len(polygon) < 3:
            low, high = box
            polygon = [[float(low[0]), float(low[2])],
                       [float(high[0]), float(low[2])],
                       [float(high[0]), float(high[2])],
                       [float(low[0]), float(high[2])]]
            if geometry_source == "semantic_region_poly_loop":
                geometry_source = "semantic_region_aabb_fallback"
        floor_height = float(getattr(region, "floor_height", box[0][1]))
        if geometry_source == "semantic_object_aabb_envelope":
            floor_height = float(box[0][1])
        if not math.isfinite(floor_height):
            floor_height = float(box[0][1])
        regions.append((str(region.id), polygon, box, floor_height, region,
                        geometry_source))
    if not regions:
        valid_object_boxes = sum(_aabb(obj) is not None for obj in habitat_objects)
        raise RuntimeError(
            "impossibile costruire regioni HM3D: "
            f"regions={len(habitat_regions)}, objects={len(habitat_objects)}, "
            f"objects_with_valid_aabb={valid_object_boxes}. "
            "La build Habitat-Sim non espone geometria semantica utilizzabile."
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
                           and str(obj.region.id) in selected_region_ids]
        floors = [floors[selected_floor]]
        region_floor = {region_id: 0 for region_id in selected_region_ids}
        if not regions:
            raise RuntimeError(f"nessuna regione trovata per il piano {selected_floor}")

    gt_regions, room_rows = [], []
    for region_id, polygon, (low, high), floor_height, region, geometry_source in regions:
        floor_index = region_floor[region_id]
        name, category_id = _category(region)
        gt_regions.append({
            "region_id": region_id,
            "polygon_xz_m": polygon,
            "floor_index": floor_index,
            "floor_height_m": floor_height,
            "category_id": category_id,
            "category_name": name,
            "geometry_source": geometry_source,
            "geometry_is_exact": geometry_source == "semantic_region_poly_loop",
            "aabb_min_m": low.tolist(),
            "aabb_max_m": high.tolist(),
        })
        room_rows.append({
            "region_id": region_id,
            "predicted_label": None,
            "ground_truth_label": name,
            "approximately_correct": None,
        })

    gt_objects = []
    native_objects = []
    for obj in habitat_objects:
        box = _aabb(obj)
        if box is None:
            continue
        name, native_category_id = _category(obj)
        native_objects.append((obj, box, name, native_category_id))
    if not native_objects:
        raise RuntimeError("nessun SemanticObject HM3D possiede un AABB valido")
    category_ids = {name: index for index, name in enumerate(
        sorted({item[2] for item in native_objects}))}
    for obj, (low, high), name, native_category_id in native_objects:
        region = getattr(obj, "region", None)
        gt_objects.append({
            "object_id": str(obj.id),
            "semantic_id": int(obj.semantic_id),
            "category_id": category_ids[name],
            "native_category_id": native_category_id,
            "category_name": name,
            "region_id": str(region.id) if region is not None else None,
            "aabb_min_m": low.tolist(),
            "aabb_max_m": high.tolist(),
        })

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
            "semantic_mesh": str(semantic_mesh.resolve()),
            "semantic_descriptor": str(semantic_text.resolve()),
            "api": "Habitat-Sim SemanticScene/SemanticRegion/SemanticObject",
            "note": "Polyloop, altezze e AABB letti dall'API nativa; nessuna inferenza da texture.",
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
        parser.error(f"habitat_sim non disponibile nell'ambiente Python: {exc}")
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = str(args.scene.resolve())
    backend.scene_dataset_config_file = str(args.dataset_config.resolve())
    backend.load_semantic_mesh = True
    # HM3D-Semantics v0.2 can store instance IDs in semantic textures.  Older
    # bindings expose the explicit switch below; newer/refactored bindings
    # select the semantic asset through the annotated dataset configuration.
    backend.requires_textures = True
    if hasattr(backend, "use_semantic_textures_if_found"):
        backend.use_semantic_textures_if_found = True
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
