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


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
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
    """Inverte hab->ROS=(-hab_z,-hab_x,hab_y) usato dal feed Habitat."""
    required = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    if not isinstance(bbox, dict) or not all(k in bbox for k in required):
        return None
    low = np.asarray([-bbox["y_max"], bbox["z_min"], -bbox["x_max"]], dtype=float)
    high = np.asarray([-bbox["y_min"], bbox["z_max"], -bbox["x_min"]], dtype=float)
    return (low, high) if np.all(np.isfinite(low)) and np.all(np.isfinite(high)) else None


def _base_label(label):
    return re.sub(r"#\d+$", "", str(label)).strip().lower()


def build(gt, run_dir):
    room_doc = _load(run_dir / "room.json", {})
    bev = _load(run_dir / "bev_data.json", {})
    objects = _load(run_dir / "persistent_perception.json", [])
    embeddings = _load(run_dir / "clip_embeddings.json", {})
    result = dict(gt)

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
    for room in room_doc.get("rooms", []) if isinstance(room_doc, dict) else []:
        if isinstance(room, dict) and room.get("active", True) is False:
            continue
        polygon = room.get("polygon", [])
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        predicted_regions.append({"polygon_xz_m": [[-float(p[1]), -float(p[0])] for p in polygon],
                                  "room_id": room.get("room_id"),
                                  "predicted_label": room.get("semantic_label", "")})
        predicted_room_by_index.append(room)
    result["predicted_regions"] = predicted_regions

    # Le etichette stanza vengono confrontate dopo l'associazione geometrica.
    room_trials = []
    gt_regions = gt.get("ground_truth_regions", [])
    for pi, gi, score in assignment(predicted_regions, gt_regions, 0.0):
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
    for obj in objects if isinstance(objects, list) else []:
        box = _predicted_aabb(obj.get("bbox"))
        if box is None:
            continue
        row = {"object_id": obj.get("object_id"), "label": obj.get("label"),
               "room_id": obj.get("room_id"),
               "aabb_min_m": box[0].tolist(), "aabb_max_m": box[1].tolist()}
        embedding = embeddings.get(obj.get("label")) if isinstance(embeddings, dict) else None
        if isinstance(embedding, list):
            row["embedding"] = embedding
        predicted_objects.append(row)
    result["predicted_objects"] = predicted_objects

    # Una matrice categorie è valida solo se ogni categoria GT ha un embedding
    # con la stessa dimensionalità. Non vengono fabbricati vettori mancanti.
    category_vectors = []
    for category in sorted(gt.get("categories", []), key=lambda x: int(x["category_id"])):
        name = category["category_name"]
        vector = embeddings.get(name) if isinstance(embeddings, dict) else None
        if not isinstance(vector, list):
            category_vectors = []
            break
        category_vectors.append(vector)
    result["category_embeddings"] = category_vectors
    result["construction_time_s"] = bev.get("stats", {}).get("elapsed_sec")
    result["representation_files"] = [
        str((run_dir / name).resolve()) for name in
        ("persistent_perception.json", "room.json", "bev_data.json", "clip_embeddings.json")
        if (run_dir / name).is_file()
    ]
    result["adapter_notes"] = {
        "coordinates": "habitat->ROS=(-hab_z,-hab_x,hab_y); ROS->habitat=(-ros_y,ros_z,-ros_x)",
        "active_floor_index": floor_for_rooms,
        "category_embeddings_complete": bool(category_vectors),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gt = _load(args.ground_truth, None)
    if not isinstance(gt, dict):
        parser.error(f"manifest GT non valido: {args.ground_truth}")
    for required in ("room.json", "persistent_perception.json", "bev_data.json"):
        if not (args.run_dir / required).is_file():
            parser.error(f"artefatto mancante: {args.run_dir / required}")
    result = build(gt, args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.output}: {len(result['predicted_regions'])} regioni e "
          f"{len(result['predicted_objects'])} oggetti predetti")


if __name__ == "__main__":
    main()
