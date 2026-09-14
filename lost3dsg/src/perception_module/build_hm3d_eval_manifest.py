#!/usr/bin/env python3
"""Unisce ground truth HM3D e artefatti di un run nel manifest di metrics_eval.py."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from metrics_eval import assignment


def _load(path, default, required=False):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if required:
            raise RuntimeError(f"impossibile leggere il JSON richiesto {path}: {exc}") from exc
        return default


def _polygon_mask(polygon_ros, spec, floor_index):
    """Rasterizza ROS (x,y) in Habitat (x,z)=(-ros_y,-ros_x)."""
    if not isinstance(polygon_ros, list) or len(polygon_ros) < 3:
        return []
    polygon = np.asarray([[-float(p[1]), -float(p[0])] for p in polygon_ros], dtype=float)
    origin = np.asarray(spec["origin_xz_m"], dtype=float)
    resolution = float(spec["resolution_m"])
    nx, nz = map(int, spec["shape_xz"])
    low = np.maximum(0, np.floor((polygon.min(axis=0) - origin) / resolution).astype(int))
    high = np.minimum([nx - 1, nz - 1],
                      np.floor((polygon.max(axis=0) - origin) / resolution).astype(int))
    if np.any(high < low):
        return []
    xs = np.arange(low[0], high[0] + 1)
    zs = np.arange(low[1], high[1] + 1)
    xx, zz = np.meshgrid(origin[0] + (xs + .5) * resolution,
                         origin[1] + (zs + .5) * resolution)
    inside = np.zeros(xx.shape, dtype=bool)
    x, z = polygon[:, 0], polygon[:, 1]
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        crosses = ((z[current] > zz) != (z[previous] > zz))
        boundary_x = ((x[previous] - x[current]) * (zz - z[current]) /
                      (z[previous] - z[current] + 1e-300) + x[current])
        inside ^= crosses & (xx < boundary_x)
        previous = current
    iz, ix = np.nonzero(inside)
    plane = nx * nz
    return (xs[ix] + nx * zs[iz] + plane * int(floor_index)).astype(int).tolist()


def _predicted_aabb(bbox):
    """Read the run's ROS/z-up AABB without changing its frame."""
    required = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    if not isinstance(bbox, dict) or not all(k in bbox for k in required):
        return None
    try:
        values = {key: float(bbox[key]) for key in required}
    except (TypeError, ValueError):
        return None
    if (not all(math.isfinite(value) for value in values.values())
            or any(values[f"{axis}_min"] > values[f"{axis}_max"]
                   for axis in "xyz")):
        return None
    low = np.asarray([values["x_min"], values["y_min"], values["z_min"]])
    high = np.asarray([values["x_max"], values["y_max"], values["z_max"]])
    return low, high


def _habitat_aabb_to_ros(row):
    """Keep the canonical GT AABB in the manifest coordinate frame.

    The canonical GT object boxes are Habitat Y-up AABBs and need one
    conversion to the run's ROS Z-up frame.  Region polygons are handled
    separately because the canonical manifests already store their projected
    world coordinates.
    """
    try:
        low = np.asarray(row["aabb_min_m"], dtype=float)
        high = np.asarray(row["aabb_max_m"], dtype=float)
        if low.shape != (3,) or high.shape != (3,):
            return row
        ros_low = np.asarray([-high[2], -high[0], low[1]])
        ros_high = np.asarray([-low[2], -low[0], high[1]])
        return {**row, "aabb_min_m": ros_low.tolist(),
                "aabb_max_m": ros_high.tolist()}
    except (KeyError, TypeError, ValueError):
        return row


def _habitat_polygon_to_ros(row):
    """Convert canonical Habitat (x,z) region points to ROS (x,y)."""
    polygon = row.get("polygon_xz_m") if isinstance(row, dict) else None
    if not isinstance(polygon, list):
        return row
    try:
        converted = [[-float(point[1]), -float(point[0])] for point in polygon]
    except (IndexError, TypeError, ValueError):
        return row
    return {**row, "polygon_xz_m": converted}


def _base_label(label):
    return re.sub(r"#\d+$", "", str(label)).strip().lower()


def _rotate_ros_aabb(box, yaw_deg):
    """Rotate a ROS AABB around the origin in the horizontal X-Y plane."""
    if box is None or not yaw_deg:
        return box
    low, high = box
    corners = np.asarray([
        [x, y, z] for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ], dtype=float)
    angle = math.radians(float(yaw_deg))
    rotation = np.asarray([[math.cos(angle), -math.sin(angle), 0.0],
                           [math.sin(angle), math.cos(angle), 0.0],
                           [0.0, 0.0, 1.0]])
    rotated = corners @ rotation.T
    return rotated.min(axis=0), rotated.max(axis=0)


def build(gt, run_dir, persistent_path=None, prediction_yaw_deg=0.0):
    room_doc = _load(run_dir / "room.json", {}, required=True)
    bev = _load(run_dir / "bev_data.json", {}, required=True)
    objects = _load(persistent_path or run_dir / "persistent_perception.json", [],
                    required=True)
    embeddings = _load(run_dir / "clip_embeddings.json", {})
    # Older capture runs used ``[]`` for an empty room snapshot.  Treat that
    # representation as an empty snapshot while still rejecting malformed
    # non-container JSON.
    if isinstance(room_doc, list) and not room_doc:
        room_doc = {"rooms": []}
    if not isinstance(room_doc, dict):
        raise RuntimeError("room.json deve contenere un oggetto JSON")
    if not isinstance(bev, dict):
        raise RuntimeError("bev_data.json deve contenere un oggetto JSON")
    if not isinstance(objects, list):
        raise RuntimeError("persistent_perception deve contenere una lista JSON")
    result = dict(gt)
    # Keep the complete evaluation manifest in ROS coordinates: run bboxes are
    # serialized by the perception stack in ROS (x,y,z), while HM3D GT is
    # generated in Habitat (x,y,z), Y-up.
    result["ground_truth_objects"] = [
        _habitat_aabb_to_ros(row) for row in gt.get("ground_truth_objects", [])
    ]
    result["ground_truth_regions"] = [
        _habitat_polygon_to_ros(row) for row in gt.get("ground_truth_regions", [])
    ]

    gt_floors = [float(v) for v in gt.get("ground_truth_floors_m", [])]
    run_floors = [float(v) for v in bev.get("floors", [])]
    if len(gt_floors) == 1 and run_floors:
        active = bev.get("agent", {}).get("z")
        candidates = run_floors
        selected = min(candidates, key=lambda h: abs(h - float(active))) if active is not None else candidates[0]
        result["predicted_floors_m"] = [selected]
    else:
        result["predicted_floors_m"] = run_floors

    floor_for_rooms = 0
    if len(gt_floors) > 1 and bev.get("agent", {}).get("z") is not None:
        floor_for_rooms = int(np.argmin(np.abs(np.asarray(gt_floors) - float(bev["agent"]["z"]))))
    predicted_regions = []
    predicted_room_by_index = []
    # This is the persisted source corresponding to the RViz room-area
    # MarkerArray.  ``rooms`` at the document root is only a compatibility
    # duplicate and must not be used as an independent source.
    room_source = room_doc.get("building", {}).get("rooms")
    if room_source is None:
        raise RuntimeError("room.json non contiene building.rooms, cioè room areas")
    for room in room_source:
        if not isinstance(room, dict) or room.get("active", True) is False:
            continue
        polygon = room.get("polygon", [])
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        # room.json is the JSON serialization of /room_areas_array in ROS
        # (x,y). The runtime prediction is copied verbatim.
        predicted_regions.append({"polygon_xz_m": [[float(p[0]), float(p[1])] for p in polygon],
                                  "room_id": room.get("room_id"),
                                  # room.json is an active-run snapshot; all
                                  # retained rooms belong to the observed
                                  # floor.  Recording it prevents accidental
                                  # cross-floor matching in multi-floor GT.
                                  "floor_index": floor_for_rooms,
                                  "predicted_label": room.get("semantic_label", "")})
        predicted_room_by_index.append(room)
    result["predicted_regions"] = predicted_regions

    # Le etichette stanza vengono confrontate dopo l'associazione geometrica.
    room_trials = []
    # Use the already converted copy.  ``gt`` is still in Habitat X-Z here;
    # comparing it with the ROS room polygons silently associates unrelated
    # rooms and makes the room table disagree with the metric calculation.
    gt_regions = result.get("ground_truth_regions", [])
    # A top-down polygon alone cannot distinguish vertically stacked rooms.
    # Restrict candidates to the active floor before the Hungarian matching.
    gt_region_candidates = [
        (index, region) for index, region in enumerate(gt_regions)
        if region.get("floor_index") is None
        or int(region["floor_index"]) == floor_for_rooms
    ]
    candidate_rows = [region for _, region in gt_region_candidates]
    for pi, candidate_index, score in assignment(
            predicted_regions, candidate_rows, 0.0):
        gi, _ = gt_region_candidates[candidate_index]
        predicted_label = str(predicted_room_by_index[pi].get("semantic_label", ""))
        ground_truth_label = str(gt_regions[gi].get("category_name", ""))
        room_trials.append({
            "predicted_region_id": predicted_regions[pi].get("room_id"),
            "ground_truth_region_id": gt_regions[gi].get("region_id"),
            "predicted_label": predicted_label,
            "ground_truth_label": ground_truth_label,
            "approximately_correct": predicted_label.strip().casefold() == ground_truth_label.strip().casefold(),
            "region_iou": score,
        })
    result["rooms"] = room_trials

    predicted_objects = []
    invalid_predicted_aabb_count = 0
    for obj in objects if isinstance(objects, list) else []:
        if not isinstance(obj, dict):
            invalid_predicted_aabb_count += 1
            continue
        fused = obj.get("fused_bbox")
        geometry = fused if isinstance(fused, dict) else obj.get("bbox")
        box = _rotate_ros_aabb(_predicted_aabb(geometry), prediction_yaw_deg)
        if box is None:
            invalid_predicted_aabb_count += 1
            continue
        row = {"object_id": obj.get("object_id"), "label": obj.get("label"),
               "room_id": obj.get("room_id"),
               "bbox_source": "fused_bbox" if isinstance(fused, dict) else "bbox",
               "aabb_min_m": box[0].tolist(), "aabb_max_m": box[1].tolist()}
        # HOV-SG evaluates the appearance embedding belonging to this object.
        # It is serialized in the persistent object's bbox, not in the
        # bbox-free top-level object record.
        bbox_data = obj.get("bbox") if isinstance(obj.get("bbox"), dict) else {}
        embedding = bbox_data.get("clip_embedding", obj.get("clip_embedding"))
        if isinstance(embedding, list):
            row["embedding"] = embedding
        predicted_objects.append(row)
    result["predicted_objects"] = predicted_objects

    # Una matrice categorie è valida solo se ogni categoria GT ha un embedding
    # con la stessa dimensionalità. Non vengono fabbricati vettori mancanti.
    category_vectors = []
    category_dimension = None
    for category in sorted(gt.get("categories", []), key=lambda x: int(x["category_id"])):
        name = category["category_name"]
        vector = embeddings.get(name) if isinstance(embeddings, dict) else None
        try:
            vector_array = np.asarray(vector, dtype=float)
        except (TypeError, ValueError):
            vector_array = np.asarray([])
        if (not isinstance(vector, list) or vector_array.ndim != 1
                or vector_array.size == 0 or not np.all(np.isfinite(vector_array))
                or (category_dimension is not None and vector_array.size != category_dimension)):
            category_vectors = []
            break
        category_vectors.append(vector)
        category_dimension = int(vector_array.size)
    result["category_embeddings"] = category_vectors
    result["construction_time_s"] = bev.get("stats", {}).get("elapsed_sec")
    representation_candidates = [
        Path(persistent_path or run_dir / "persistent_perception.json"),
        run_dir / "room.json", run_dir / "bev_data.json", run_dir / "clip_embeddings.json",
    ]
    result["representation_files"] = list(dict.fromkeys(
        str(path.resolve()) for path in representation_candidates if path.is_file()
    ))
    result["adapter_notes"] = {
        "coordinates": "ROS (x,y,z), Z-up; GT boxes and region polygons converted once",
        "prediction_yaw_deg": float(prediction_yaw_deg),
        "active_floor_index": floor_for_rooms,
        "category_embeddings_complete": bool(category_vectors),
        "source_object_count": len(objects),
        "invalid_predicted_aabb_count": invalid_predicted_aabb_count,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--persistent-perception", type=Path,
                        help="JSON persistente alternativo, eventualmente arricchito con CLIP")
    parser.add_argument("--prediction-yaw-deg", type=float, default=0.0,
                        choices=(0.0, 90.0, 180.0, 270.0),
                        help="rotazione ROS delle predizioni attorno all'origine")
    args = parser.parse_args()
    gt = _load(args.ground_truth, None)
    if not isinstance(gt, dict):
        parser.error(f"manifest GT non valido: {args.ground_truth}")
    for required in ("room.json", "bev_data.json"):
        if not (args.run_dir / required).is_file():
            parser.error(f"artefatto mancante: {args.run_dir / required}")
    persistent_path = args.persistent_perception or args.run_dir / "persistent_perception.json"
    if not persistent_path.is_file():
        parser.error(f"artefatto mancante: {persistent_path}")
    try:
        result = build(gt, args.run_dir, persistent_path, args.prediction_yaw_deg)
    except RuntimeError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(result['predicted_regions'])} regioni e "
          f"{len(result['predicted_objects'])} oggetti predetti")


if __name__ == "__main__":
    main()
