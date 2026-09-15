#!/usr/bin/env python3
"""Calcola le metriche HOV-SG da manifest JSON prodotti da run Habitat.

I valori non misurabili non vengono inventati: restano null e sono elencati in
``missing_inputs``. Eseguire con ``--example`` per vedere lo schema completo.
"""
from __future__ import annotations
import argparse, csv, json, math, subprocess, sys, time
from functools import lru_cache
from pathlib import Path
import numpy as np

TOP_K = (5, 10, 25, 100, 250, 500)
HM3D_OBJECT_TYPES = Path(__file__).with_name("HM3D_CountsOfObjectTypes.csv")
HM3D_TEXT_FEATURES = Path(__file__).with_name("text_feats_HM3DSEM_LABELS.npy")

# HM3D contains architectural surfaces in the same object vocabulary as
# movable objects.  They are useful for geometry/room extraction, but should
# not affect object detection/classification metrics.
STRUCTURAL_OBJECT_LABELS = {
    "wall", "floor", "ceiling", "panel", "wall panel", "fireplace wall",
    "shower wall", "shower floor", "shower ceiling", "partition", "column",
    "door", "doorway", "door frame", "door jamb", "window", "window frame",
}


def is_structural_object(row):
    """Return whether *row* describes a structural architectural surface."""
    for key in ("category_name", "label", "predicted_label"):
        value = row.get(key)
        if value is None:
            continue
        # Predicted labels commonly have an instance suffix (e.g. wall#2).
        label = str(value).strip().casefold().split("#", 1)[0].strip()
        if (label in STRUCTURAL_OBJECT_LABELS or label.startswith("flooring")
                or label.endswith((" wall", " floor", " ceiling"))):
            return True
    # Include compositional HM3D labels (door frame, shower door frame,
    # window frame, open doorway, ...), while _structural_kind keeps hardware
    # such as door knobs available as ordinary object classes.
    return _structural_kind(row) is not None


STRUCTURAL_ELEMENT_LABELS = {
    "door": {"door", "doorway", "open doorway", "opening", "entrance",
             "passage", "portal"},
    "window": {"window"},
}
WALL_LABELS = {"wall", "wall panel", "fireplace wall", "shower wall",
               "partition", "column"}
SENSOR_HEIGHT = 1.5


def _row_label(row):
    for key in ("category_name", "label", "predicted_label", "type"):
        value = row.get(key)
        if value is not None:
            return str(value).strip().casefold().split("#", 1)[0].strip()
    return ""


def _structural_kind(row):
    label = _row_label(row)
    for kind, labels in STRUCTURAL_ELEMENT_LABELS.items():
        if label in labels:
            return kind
    # HM3D has compositional names such as ``door frame`` and ``shower door
    # frame``.  Treat those as door geometry too, while keeping small door
    # hardware (knobs, handles, hinges, locks) out of the door count.
    if ("door" in label or "doorway" in label) and not any(
            token in label for token in ("knob", "handle", "hinge", "lock")):
        return "door"
    if "window" in label:
        return "window"
    return "wall" if label in WALL_LABELS else None


def _active_floor_index(scene):
    """Resolve the current floor index, if the manifest records one."""
    candidates = (
        scene.get("active_floor_index"),
        scene.get("selected_floor_index"),
        (scene.get("adapter_notes") or {}).get("active_floor_index"),
        (scene.get("ground_truth_source") or {}).get("selected_floor_index"),
    )
    for value in candidates:
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _floor_index_maps(scene):
    regions = scene.get("predicted_regions", [])
    gt_regions = scene.get("ground_truth_regions", [])
    pred_map = {str(r.get("room_id")): r.get("floor_index") for r in regions
                if r.get("room_id") is not None and r.get("floor_index") is not None}
    gt_map = {str(r.get("region_id")): r.get("floor_index") for r in gt_regions
              if r.get("region_id") is not None and r.get("floor_index") is not None}
    return pred_map, gt_map


def _on_active_floor(row, scene, predicted, region_map):
    active = _active_floor_index(scene)
    if active is None:
        return True
    floor = row.get("floor_index")
    if floor is None:
        key = "room_id" if predicted else "region_id"
        floor = region_map.get(str(row.get(key)))
    # In manifests produced from a run, predicted rooms without an explicit
    # floor index are already the active-room set. Keep those rows. A GT
    # object without floor membership metadata must also be kept: dropping it
    # would erase the entire GT set when native HM3D regions have no geometry.
    if floor is None:
        return True
    # A GT manifest generated with --floor-index remaps the retained floor to
    # local index 0, while preserving the original selected index in metadata.
    source = scene.get("ground_truth_source") or {}
    gt_floor_indexes = {r.get("floor_index") for r in scene.get("ground_truth_regions", [])
                        if r.get("floor_index") is not None}
    if (not predicted and source.get("selected_floor_index") is not None
            and gt_floor_indexes == {0}):
        return floor == 0
    return floor == active


def filtered_scene(scene, include_regions=False):
    """Copy a scene with structural and (when known) off-floor items removed."""
    pred_map, gt_map = _floor_index_maps(scene)
    result = dict(scene)
    result["predicted_objects"] = [
        row for row in scene.get("predicted_objects", [])
        if not is_structural_object(row) and _on_active_floor(row, scene, True, pred_map)
    ]
    result["ground_truth_objects"] = [
        row for row in scene.get("ground_truth_objects", [])
        if not is_structural_object(row) and _on_active_floor(row, scene, False, gt_map)
    ]
    if include_regions and _active_floor_index(scene) is not None:
        active = _active_floor_index(scene)
        source = scene.get("ground_truth_source") or {}
        gt_floor_indexes = {row.get("floor_index") for row in scene.get("ground_truth_regions", [])
                            if row.get("floor_index") is not None}
        local_gt_floor = (source.get("selected_floor_index") is not None
                          and gt_floor_indexes == {0})
        for key in ("predicted_regions", "ground_truth_regions"):
            result[key] = [row for row in scene.get(key, [])
                           if row.get("floor_index") is None
                           or row.get("floor_index") == (0 if key == "ground_truth_regions" and local_gt_floor else active)]
    return result


def _structural_box_metrics(predicted, ground_truth, kind, threshold):
    pred = [_visible_structural_box(row) for row in predicted
            if _structural_kind(row) == kind]
    gt = [_visible_structural_box(row) for row in ground_truth
          if _structural_kind(row) == kind]
    pred = [row for row in pred if row is not None]
    gt = [row for row in gt if row is not None]
    all_scores = [[geometry_iou(p, g) for g in gt] for p in pred]
    matches = assignment(pred, gt, threshold) if pred and gt else []
    tp = len(matches)
    return {
        "predicted": len(pred), "ground_truth": len(gt), "matched": tp,
        "mean_iou_3d": round(float(np.mean([max(scores) for scores in all_scores
                                             if scores])), 4)
        if all_scores and gt else None,
        "iou_threshold": threshold,
        "sensor_height_m": SENSOR_HEIGHT,
    }


def _visible_structural_box(row):
    """Clip a structural AABB to the part visible below the camera height."""
    try:
        low = [float(v) for v in row["aabb_min_m"]]
        high = [float(v) for v in row["aabb_max_m"]]
        if len(low) != 3 or len(high) != 3:
            return None
        high[2] = min(high[2], SENSOR_HEIGHT)
        if high[2] <= low[2]:
            return None
        clipped = dict(row)
        clipped["aabb_min_m"] = low
        clipped["aabb_max_m"] = high
        return clipped
    except (KeyError, TypeError, ValueError):
        return None


def _gt_wall_segment(row):
    low = np.asarray(row.get("aabb_min_m", []), dtype=float)
    high = np.asarray(row.get("aabb_max_m", []), dtype=float)
    if low.shape != (3,) or high.shape != (3,) or np.any(high <= low):
        return None
    # The manifest is already in ROS (x,y horizontal; z vertical).  Represent
    # each thin GT wall AABB by its centre line and retain its vertical extent.
    if high[0] - low[0] >= high[1] - low[1]:
        return {"start": {"x": low[0], "y": (low[1] + high[1]) / 2},
                "end": {"x": high[0], "y": (low[1] + high[1]) / 2},
                "z_min": low[2], "z_max": high[2]}
    return {"start": {"x": (low[0] + high[0]) / 2, "y": low[1]},
            "end": {"x": (low[0] + high[0]) / 2, "y": high[1]},
            "z_min": low[2], "z_max": high[2]}


def _wall_prediction_aabb(row, default_thickness=.06):
    try:
        x0, y0 = float(row["start"]["x"]), float(row["start"]["y"])
        x1, y1 = float(row["end"]["x"]), float(row["end"]["y"])
        z0, z1 = float(row["z_min"]), float(row["z_max"])
    except (KeyError, TypeError, ValueError):
        return None
    try:
        thickness = float(row.get("thickness_m", default_thickness))
    except (TypeError, ValueError):
        thickness = default_thickness
    if not np.isfinite(thickness) or thickness <= 0.0:
        thickness = default_thickness
    half = thickness / 2.0
    return {"aabb_min_m": [min(x0, x1) - half, min(y0, y1) - half, min(z0, z1)],
            "aabb_max_m": [max(x0, x1) + half, max(y0, y1) + half, max(z0, z1)]}


def _wall_footprint_iou(predicted, ground_truth, resolution=.02):
    """Geometric IoU of wall footprints in the ROS XY floor plane."""
    try:
        import cv2
        p0 = np.array([float(predicted["start"]["x"]), float(predicted["start"]["y"])])
        p1 = np.array([float(predicted["end"]["x"]), float(predicted["end"]["y"])])
        thickness = float(predicted.get("thickness_m", .06))
        low = np.asarray(ground_truth["aabb_min_m"], dtype=float)
        high = np.asarray(ground_truth["aabb_max_m"], dtype=float)
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length <= 1e-9 or low.shape != (3,) or high.shape != (3,):
            return 0.0
        normal = np.array([-direction[1], direction[0]]) / length
        half = max(thickness, 1e-6) / 2.0
        pred_poly = np.array([p0 + normal * half, p1 + normal * half,
                              p1 - normal * half, p0 - normal * half])
        gt_poly = np.array([[low[0], low[1]], [high[0], low[1]],
                            [high[0], high[1]], [low[0], high[1]]])
        points = np.vstack((pred_poly, gt_poly))
        origin = points.min(axis=0) - resolution
        shape = np.ceil((points.max(axis=0) - origin) / resolution).astype(int) + 2
        if np.any(shape <= 0) or np.any(shape > 10000):
            return 0.0
        pred_mask = np.zeros((int(shape[1]), int(shape[0])), np.uint8)
        gt_mask = np.zeros_like(pred_mask)
        def raster(poly, mask):
            px = np.rint((poly - origin) / resolution).astype(np.int32)
            cv2.fillPoly(mask, [px], 1)
        raster(pred_poly, pred_mask); raster(gt_poly, gt_mask)
        union = np.count_nonzero(pred_mask | gt_mask)
        return float(np.count_nonzero(pred_mask & gt_mask) / union) if union else 0.0
    except (ImportError, KeyError, TypeError, ValueError):
        return 0.0


def _wall_metrics(predicted, ground_truth, iou_threshold=.5):
    pred = [row for row in predicted if isinstance(row, dict)]
    gt = [row for row in ground_truth if _structural_kind(row) == "wall"
          and "aabb_min_m" in row and "aabb_max_m" in row]
    scores_matrix = np.asarray([[_wall_footprint_iou(p, g) for g in gt]
                                for p in pred], dtype=float) if pred and gt else np.zeros((0, 0))
    matches = []
    if scores_matrix.size:
        from scipy.optimize import linear_sum_assignment
        ii, jj = linear_sum_assignment(scores_matrix, maximize=True)
        matches = [(int(i), int(j), float(scores_matrix[i, j]))
                   for i, j in zip(ii, jj) if scores_matrix[i, j] >= iou_threshold]
    scores = [score for _, _, score in matches]
    tp = len(scores); total_p, total_g = len(pred), len(gt)
    return {
        "predicted": total_p, "ground_truth": total_g, "matched": tp,
        "mean_iou_xy": round(float(np.mean(np.max(scores_matrix, axis=1))), 4)
        if scores_matrix.size else None,
        "iou_threshold": iou_threshold,
        "wall_iou_geometry": "XY footprint raster IoU",
        "wall_thickness_source": "walls.json thickness_m",
    }


def structural_metrics(scenes, object_iou=.5):
    """Evaluate run-produced doors, windows and depth-derived wall segments."""
    doors, windows, walls = [], [], []
    structural_ground_truth = []
    for scene in scenes:
        doors += [row for row in scene.get("predicted_structural_elements", [])
                  if _structural_kind(row) == "door"]
        windows += [row for row in scene.get("predicted_structural_elements", [])
                    if _structural_kind(row) == "window"]
        walls += list(scene.get("predicted_walls", []))
        _, gt_map = _floor_index_maps(scene)
        structural_ground_truth += [
            row for row in scene.get("ground_truth_objects", [])
            if _on_active_floor(row, scene, False, gt_map)
        ]
    return {"doors": _structural_box_metrics(doors, structural_ground_truth,
                                               "door", object_iou),
            "windows": _structural_box_metrics(windows, structural_ground_truth,
                                                 "window", object_iou)}

@lru_cache(maxsize=1)
def hm3d_object_types(path=HM3D_OBJECT_TYPES):
    """Return the HM3D class names in the order used for text embeddings."""
    try:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            return [row["Object Type Name"].strip()
                    for row in csv.DictReader(stream, delimiter=";")
                    if row.get("Object Type Name", "").strip()]
    except OSError:
        return []

@lru_cache(maxsize=1)
def hm3d_text_features(path=HM3D_TEXT_FEATURES):
    """Load the precomputed ViT-H/14 text features used by HOV-SG.

    HOV-SG's ``get_label_feats(..., "HM3DSEM_LABELS", ...)`` pairs this file
    with ``HM3D_CountsOfObjectTypes.csv``.  The validation prevents an
    accidental feature file from being evaluated with the HM3D label order.
    """
    try:
        features = np.load(path)
    except (OSError, ValueError):
        return None
    return features if features.ndim == 2 and len(features) == len(hm3d_object_types()) else None

def load(path):
    files = [path] if path.is_file() else sorted(path.glob("*.json"))
    out = []
    for filename in files:
        value = json.loads(filename.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else [value]
        if path.is_file():
            out.extend(rows)
            continue
        # A run directory also contains room.json, persistent objects and old
        # metric reports.  Only ingest documents that actually use the scene
        # manifest schema; otherwise those files become bogus extra scenes.
        manifest_keys = {
            "predicted_floors_m", "ground_truth_floors_m", "predicted_regions",
            "ground_truth_regions", "predicted_objects", "ground_truth_objects",
        }
        out.extend(row for row in rows
                   if isinstance(row, dict) and manifest_keys.intersection(row))
    return out

def mask(value):
    a = np.asarray(value)
    return set(np.flatnonzero(a.reshape(-1))) if a.dtype == bool else set(map(int, a.reshape(-1)))

def iou(a, b):
    a, b = mask(a), mask(b)
    return len(a & b) / len(a | b) if a or b else 0.0

def _aabb_iou(a, b):
    try:
        alo, ahi = np.asarray(a["aabb_min_m"],float), np.asarray(a["aabb_max_m"],float)
        blo, bhi = np.asarray(b["aabb_min_m"],float), np.asarray(b["aabb_max_m"],float)
    except (KeyError, TypeError, ValueError):
        return 0.0
    if (any(value.shape != (3,) for value in (alo, ahi, blo, bhi))
            or not all(np.all(np.isfinite(value)) for value in (alo, ahi, blo, bhi))
            or np.any(ahi < alo) or np.any(bhi < blo)):
        return 0.0
    intersection = np.maximum(0.0, np.minimum(ahi,bhi)-np.maximum(alo,blo))
    iv=float(np.prod(intersection)); av=float(np.prod(np.maximum(0,ahi-alo))); bv=float(np.prod(np.maximum(0,bhi-blo)))
    return iv/(av+bv-iv) if av+bv-iv>0 else 0.0

def _polygon_iou(a, b, resolution=.05):
    pa,pb=np.asarray(a["polygon_xz_m"],float),np.asarray(b["polygon_xz_m"],float)
    if pa.shape[0]<3 or pb.shape[0]<3: return 0.0
    low=np.minimum(pa.min(0),pb.min(0)); high=np.maximum(pa.max(0),pb.max(0))
    shape=np.maximum(1,np.ceil((high-low)/resolution).astype(int)+3)
    if int(np.prod(shape))>2_000_000:
        resolution*=math.sqrt(float(np.prod(shape))/2_000_000); shape=np.maximum(1,np.ceil((high-low)/resolution).astype(int)+3)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python è necessario per l'IoU dei poligoni") from exc
    def raster(poly):
        image=np.zeros((int(shape[1]),int(shape[0])),np.uint8)
        points=np.rint((poly-low)/resolution).astype(np.int32)
        cv2.fillPoly(image,[points],1)
        return image.astype(bool)
    ma,mb=raster(pa),raster(pb); union=np.count_nonzero(ma|mb)
    return float(np.count_nonzero(ma&mb)/union) if union else 0.0


def _region_overlap_shares(predicted, ground_truth, resolution=.05):
    """Return HOV-SG/HyDRA directional overlaps (pred-covered, GT-covered).

    HOV-SG calls ``find_intersection_share`` twice.  Its denominator is the
    second point cloud, so the first value below is intersection/prediction
    and the second is intersection/GT.  Region polygons are rasterized at the
    same 5 cm resolution used by HOV-SG's BEV point clouds.
    """
    pa = np.asarray(predicted["polygon_xz_m"], dtype=float)
    pb = np.asarray(ground_truth["polygon_xz_m"], dtype=float)
    low = np.minimum(pa.min(axis=0), pb.min(axis=0))
    high = np.maximum(pa.max(axis=0), pb.max(axis=0))
    shape = np.maximum(1, np.ceil((high - low) / resolution).astype(int) + 3)
    if int(np.prod(shape)) > 2_000_000:
        resolution *= math.sqrt(float(np.prod(shape)) / 2_000_000)
        shape = np.maximum(1, np.ceil((high - low) / resolution).astype(int) + 3)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python è necessario per l'overlap delle regioni") from exc

    def raster(poly):
        image = np.zeros((int(shape[1]), int(shape[0])), np.uint8)
        points = np.rint((poly - low) / resolution).astype(np.int32)
        cv2.fillPoly(image, [points], 1)
        return image.astype(bool)

    pred_mask, gt_mask = raster(pa), raster(pb)
    intersection = np.count_nonzero(pred_mask & gt_mask)
    pred_count, gt_count = np.count_nonzero(pred_mask), np.count_nonzero(gt_mask)
    return (float(intersection / pred_count) if pred_count else 0.0,
            float(intersection / gt_count) if gt_count else 0.0)


def hovsg_region_overlap_matrices(predicted, ground_truth, resolution=.05):
    """Return the two directional room-overlap matrices used by HOV-SG.

    Rows are predicted rooms and columns are GT rooms.  The first matrix is
    normalized by predicted-room area (HyDRA precision); the second by GT-room
    area (HyDRA recall).  HOV-SG's room association score is the maximum of
    those two values, rather than polygon IoU.
    """
    over_pred = np.zeros((len(predicted), len(ground_truth)), dtype=float)
    over_gt = np.zeros_like(over_pred)
    for pi, pred_row in enumerate(predicted):
        for gi, gt_row in enumerate(ground_truth):
            if ("polygon_xz_m" in pred_row and
                    "polygon_xz_m" in gt_row):
                over_pred[pi, gi], over_gt[pi, gi] = _region_overlap_shares(
                    pred_row, gt_row, resolution=resolution)
            else:
                # Retain mask-only fixture compatibility.  Identical masks
                # have identical denominators, so IoU is the least surprising
                # legacy fallback; real HM3D runs always use polygons.
                score = geometry_iou(pred_row, gt_row)
                over_pred[pi, gi] = score
                over_gt[pi, gi] = score
    return over_pred, over_gt


def hovsg_region_assignment(predicted, ground_truth, threshold=None,
                            resolution=.05):
    """One-to-one HOV-SG room association for serialized region polygons."""
    if not predicted or not ground_truth:
        return []
    over_pred, over_gt = hovsg_region_overlap_matrices(
        predicted, ground_truth, resolution)
    scores = np.maximum(over_pred, over_gt)
    try:
        from scipy.optimize import linear_sum_assignment
        ii, jj = linear_sum_assignment(scores, maximize=True)
    except ImportError:
        ii, jj = [], []
        for i, j in sorted(np.ndindex(scores.shape),
                           key=lambda pair: scores[pair], reverse=True):
            if i not in ii and j not in jj:
                ii.append(i)
                jj.append(j)
    return [(int(i), int(j), float(scores[i, j])) for i, j in zip(ii, jj)
            if threshold is None or scores[i, j] > threshold]

def geometry_iou(a,b):
    if "mask" in a and "mask" in b: return iou(a["mask"],b["mask"])
    if all(k in a and k in b for k in ("aabb_min_m","aabb_max_m")): return _aabb_iou(a,b)
    if "polygon_xz_m" in a and "polygon_xz_m" in b: return _polygon_iou(a,b)
    return 0.0


def _aabb_overlap(a, b):
    """HOV-SG ``association_metric: overlap`` for serialized AABBs."""
    try:
        al, ah = np.asarray(a["aabb_min_m"], float), np.asarray(a["aabb_max_m"], float)
        bl, bh = np.asarray(b["aabb_min_m"], float), np.asarray(b["aabb_max_m"], float)
        if any(x.shape != (3,) for x in (al, ah, bl, bh)):
            return 0.0
        intersection = float(np.prod(np.maximum(0.0, np.minimum(ah, bh) - np.maximum(al, bl))))
        av = float(np.prod(np.maximum(0.0, ah - al)))
        bv = float(np.prod(np.maximum(0.0, bh - bl)))
        return max(intersection / av if av else 0.0,
                   intersection / bv if bv else 0.0)
    except (KeyError, TypeError, ValueError):
        return 0.0


def hovsg_object_assignment(pred, gt):
    """Hungarian associations used by HOV-SG's HM3D top-k evaluator."""
    if not pred or not gt:
        return []
    scores = np.asarray([[_aabb_overlap(p, g) for g in gt] for p in pred])
    try:
        from scipy.optimize import linear_sum_assignment
        ii, jj = linear_sum_assignment(scores, maximize=True)
    except ImportError:
        ii, jj = [], []
    return [(int(i), int(j), float(scores[i, j])) for i, j in zip(ii, jj)]

def assignment(pred, gt, threshold):
    if not pred or not gt: return []
    scores = np.asarray([[geometry_iou(p, g) for g in gt] for p in pred])
    # When a threshold is supplied, maximize the number of *valid* pairs
    # first, then their total score.  A plain Hungarian assignment followed by
    # filtering can choose one excellent pair and discard several valid ones.
    weights = -scores
    if threshold is not None:
        valid = scores > threshold
        bonus = float(max(scores.size, 1) + 1)
        weights = np.where(valid, -(scores + bonus), bonus)
    try:
        from scipy.optimize import linear_sum_assignment
        ii, jj = linear_sum_assignment(weights)
    except ImportError:
        ii, jj = [], []
        pairs = sorted(np.ndindex(scores.shape), key=lambda z: (-int(threshold is not None and scores[z] > threshold), -scores[z]))
        for i, j in pairs:
            if i not in ii and j not in jj: ii.append(i); jj.append(j)
    return [(int(i), int(j), float(scores[i, j])) for i, j in zip(ii, jj)
            if threshold is None or scores[i, j] > threshold]


def _centre(row):
    try:
        explicit = np.asarray(row["center_m"], dtype=float)
        if explicit.shape == (3,) and np.all(np.isfinite(explicit)):
            return explicit
    except (KeyError, TypeError, ValueError):
        pass
    try:
        low = np.asarray(row["aabb_min_m"], dtype=float)
        high = np.asarray(row["aabb_max_m"], dtype=float)
        if low.shape == high.shape == (3,) and np.all(np.isfinite(low + high)):
            return (low + high) / 2.0
    except (KeyError, TypeError, ValueError):
        pass
    return None


def object_assignment(pred, gt, threshold):
    """Associate objects spatially by 3-D centre distance.

    The HOV-SG object metric uses the centre tolerance for detection.  AABB
    IoU is reported as a quality diagnostic, but must not turn two nearby
    detections into a false negative merely because their boxes do not overlap.
    """
    if not pred or not gt:
        return []
    centres_p, centres_g = [_centre(row) for row in pred], [_centre(row) for row in gt]
    if any(value is None for value in centres_p + centres_g):
        # Legacy manifests may contain masks only.  Preserve their geometric
        # association rather than silently reporting zero object matches.
        return assignment(pred, gt, threshold)
    distances = np.asarray([[float(np.linalg.norm(p - g)) if p is not None and g is not None else np.inf
                             for g in centres_g] for p in centres_p])
    valid = np.isfinite(distances) if threshold is None else distances <= threshold
    weights = distances.copy()
    finite = distances[np.isfinite(distances)]
    limit = float(finite.max() + 1.0) if finite.size else 1.0
    weights[~np.isfinite(weights)] = limit
    if threshold is not None:
        weights = np.where(valid, distances, limit + distances.clip(max=limit))
    try:
        from scipy.optimize import linear_sum_assignment
        ii, jj = linear_sum_assignment(weights)
    except ImportError:
        ii, jj = [], []
        for i, j in sorted(np.ndindex(distances.shape), key=lambda z: distances[z]):
            if valid[i, j] and i not in ii and j not in jj:
                ii.append(i); jj.append(j)
    return [(int(i), int(j), float(geometry_iou(pred[i], gt[j])))
            for i, j in zip(ii, jj) if valid[i, j]]

def _entity_id(row, index, kind):
    keys = ("room_id", "region_id") if kind == "region" else ("object_id", "id", "label")
    for key in keys:
        if row.get(key) is not None:
            return str(row[key])
    return str(index)

def match_details(scenes, region_threshold, object_threshold, include_all_pairs=False):
    """Restituisce tutte le associazioni, comprese quelle sotto soglia."""
    output = []
    for scene in scenes:
        scene_row = {"scene": str(scene.get("scene", "unknown"))}
        for name, pred_key, gt_key, threshold in (
            ("regions", "predicted_regions", "ground_truth_regions", region_threshold),
            ("objects", "predicted_objects", "ground_truth_objects", object_threshold),
        ):
            if name == "objects":
                object_scene = filtered_scene(scene)
                pred, gt = object_scene.get(pred_key, []), object_scene.get(gt_key, [])
            else:
                region_scene = filtered_scene(scene, include_regions=True)
                pred, gt = region_scene.get(pred_key, []), region_scene.get(gt_key, [])
            rows = []
            associations = (object_assignment(pred, gt, None)
                            if name == "objects" else hovsg_region_assignment(pred, gt, None))
            for pi, gi, score in associations:
                rows.append({
                    "predicted_index": pi,
                    "predicted_id": _entity_id(pred[pi], pi, name[:-1]),
                    "predicted_label": pred[pi].get("predicted_label", pred[pi].get("label")),
                    "ground_truth_index": gi,
                    "ground_truth_id": _entity_id(gt[gi], gi, name[:-1]),
                    "ground_truth_label": gt[gi].get("category_name"),
                    "iou": round(score, 6),
                    "passes_metric_threshold": True if threshold is None else score > threshold,
                })
            block = {"threshold": threshold, "assigned_matches": rows,
                     "predicted_count": len(pred), "ground_truth_count": len(gt)}
            if name == "objects":
                block["threshold_unit"] = "metres_3d"
            else:
                block["association_metric"] = "HOV-SG max directional overlap"
            if include_all_pairs:
                region_scores = None
                if name == "regions" and pred and gt:
                    over_pred, over_gt = hovsg_region_overlap_matrices(pred, gt)
                    region_scores = np.maximum(over_pred, over_gt)
                block["all_candidate_pairs"] = [
                    {"predicted_index": pi,
                     "predicted_id": _entity_id(p, pi, name[:-1]),
                     "ground_truth_index": gi,
                     "ground_truth_id": _entity_id(g, gi, name[:-1]),
                     "iou": round(float(region_scores[pi, gi]) if region_scores is not None
                                  else geometry_iou(p, g), 6)}
                    for pi, p in enumerate(pred) for gi, g in enumerate(gt)
                ]
            scene_row[name] = block
        output.append(scene_row)
    return output

def pct(values): return round(100 * sum(values) / len(values), 4) if values else None

def floor_regions(scenes, threshold):
    fh = ft = rh = pt = gt = 0
    region_precision_total = region_recall_total = 0.0
    for s in scenes:
        ps, gs = s.get("predicted_floors_m", []), s.get("ground_truth_floors_m", [])
        used_p, used_g = set(), set()
        candidates = [(abs(float(p)-float(g)), pi, gi)
                      for pi, p in enumerate(ps) for gi, g in enumerate(gs)]
        if candidates:
            distances = np.asarray([[d for d, p, g in candidates
                                     if p == pi and g == gi][0]
                                    for pi in range(len(ps)) for gi in range(len(gs))]
                                   ).reshape(len(ps), len(gs))
            valid = distances <= .5
            bonus = float(distances.size + distances.max() + 1.0)
            weights = np.where(valid, distances, bonus)
            try:
                from scipy.optimize import linear_sum_assignment
                ii, jj = linear_sum_assignment(weights)
            except ImportError:
                ii, jj = [], []
                for pi, gi in sorted(np.ndindex(distances.shape), key=lambda z: distances[z]):
                    if valid[pi, gi] and pi not in ii and gi not in jj:
                        ii.append(pi); jj.append(gi)
            for pi, gi in zip(ii, jj):
                if valid[pi, gi]: used_p.add(int(pi)); used_g.add(int(gi))
        fh += len(used_g); ft += len(gs)
        region_scene = filtered_scene(s, include_regions=True)
        pr = region_scene.get("predicted_regions", [])
        gr = region_scene.get("ground_truth_regions", [])
        pt += len(pr); gt += len(gr)
        if pr and gr:
            if not all("polygon_xz_m" in row for row in pr + gr):
                # Compatibility for legacy mask-only fixtures.  HM3D/HOV-SG
                # room evaluation always reaches the polygon branch above.
                rh += len(assignment(pr, gr, threshold))
                region_precision_total += len(assignment(pr, gr, threshold))
                region_recall_total += len(assignment(pr, gr, threshold))
                continue
            # This is the region metric reported by HOV-SG (HyDRA), not
            # polygon IoU: directional coverage is maximized independently
            # for every prediction and every GT region.
            overlap_pred, overlap_gt = hovsg_region_overlap_matrices(pr, gr, .05)
            region_precision_total += float(np.max(overlap_pred, axis=1).sum())
            region_recall_total += float(np.max(overlap_gt, axis=0).sum())
            # HOV-SG's acc@IoU=0.5 uses its symmetric overlap matrix and a
            # one-to-one Hungarian assignment.
            rh += len(hovsg_region_assignment(pr, gr, threshold, .05))
        else:
            rh += 0
    region_fp = pt - rh
    region_fn = gt - rh
    region_accuracy_denominator = rh + region_fp + region_fn
    return {"acc_f_pct": round(100*fh/ft,4) if ft else None,
            "region_precision_pct": round(100*region_precision_total/pt,4) if pt else None,
            "region_recall_pct": round(100*region_recall_total/gt,4) if gt else None,
            "region_accuracy_at_threshold_pct": (
                round(100*rh/region_accuracy_denominator, 4)
                if region_accuracy_denominator else None),
            "region_detection_precision_at_threshold_pct": (
                round(100*rh/pt, 4) if pt else None),
            "region_detection_recall_at_threshold_pct": (
                round(100*rh/gt, 4) if gt else None),
            "region_overlap_threshold": threshold,
            "region_association_metric": "HOV-SG max directional overlap",
            "floor_matches": fh, "floor_gt": ft, "region_matches": rh,
            "predicted_regions": pt, "ground_truth_regions": gt}

def rooms(scenes):
    exact, approx = [], []
    for s in scenes:
        for r in s.get("rooms", []):
            gt = str(r.get("ground_truth_label", "")).strip().casefold()
            if gt:
                exact.append(gt == str(r.get("predicted_label", "")).strip().casefold())
                if "approximately_correct" in r: approx.append(bool(r["approximately_correct"]))
    return {"acc_exact_pct": pct(exact), "acc_approx_pct": pct(approx), "rooms_evaluated": len(exact)}

def _semantic_classes(scene, embedding_count):
    """Resolve the labels associated with a scene's category-embedding rows.

    Full 1,624-class embeddings use HM3D_CountsOfObjectTypes.csv.  Smaller
    scene-local embedding matrices use the manifest's ordered ``categories``.
    Returning no labels prevents silently evaluating against a wrong ordering.
    """
    hm3d_classes = hm3d_object_types()
    if len(hm3d_classes) == embedding_count:
        return hm3d_classes
    categories = scene.get("categories", [])
    if not categories:
        # Older manifests only carry category_id.  Their embeddings are by
        # definition positional, so retain that documented representation.
        return [f"__category_{index}" for index in range(embedding_count)]
    try:
        categories = sorted(categories, key=lambda item: int(item["category_id"]))
    except (KeyError, TypeError, ValueError):
        return []
    labels = [str(item.get("category_name", "")).strip() for item in categories]
    return labels if len(labels) == embedding_count and all(labels) else []

def _scene_ground_truth_vocabulary(gt, category_embeddings, category_names):
    """Restrict classification candidates to labels present in this GT scene."""
    categories = np.asarray(category_embeddings, dtype=float)
    if categories.ndim != 2 or len(categories) != len(category_names):
        return [], []
    ground_truth_names = {
        str(row.get("category_name", "")).strip()
        for row in gt
        if str(row.get("category_name", "")).strip()
    }
    selected = {index for index, name in enumerate(category_names)
                if name in ground_truth_names}
    # Compatibilita' con i manifest posizionali meno recenti, che conservano
    # soltanto category_id (come lo schema --example originale).
    for row in gt:
        if str(row.get("category_name", "")).strip():
            continue
        category_id = row.get("category_id")
        if isinstance(category_id, (int, np.integer)) and 0 <= category_id < len(category_names):
            selected.add(int(category_id))
    indices = sorted(selected)
    if not indices:
        return [], []
    return categories[indices], [category_names[index] for index in indices]

def _top_k_semantics(matches, pred, gt, category_embeddings, category_names,
                     representative=TOP_K):
    """Match HOV-SG's ``object_semantics_eval_tp_auc`` for JSON manifests.

    In particular, the denominator is every Hungarian association (not only
    rows with a usable embedding), and the ranking is over the complete class
    vocabulary supplied by the evaluator.
    """
    categories = np.asarray(category_embeddings, dtype=float)
    if categories.ndim != 2 or not categories.size or len(category_names) != len(categories):
        return {}, None, 0
    category_norms = np.linalg.norm(categories, axis=1, keepdims=True)
    if np.any(~np.isfinite(categories)) or np.any(category_norms == 0):
        return {}, None, 0
    categories = categories / category_norms
    ranks = []
    for pi, gi, _ in matches:
        embedding = np.asarray(pred[pi].get("embedding", []), dtype=float)
        if (embedding.shape != (categories.shape[1],)
                or not np.all(np.isfinite(embedding))):
            continue
        gt_label = str(gt[gi].get("category_name", "")).strip()
        if not gt_label:
            category_id = gt[gi].get("category_id")
            if isinstance(category_id, (int, np.integer)) and 0 <= category_id < len(category_names):
                gt_label = category_names[category_id]
        if gt_label not in category_names:
            continue
        embedding_norm = float(np.linalg.norm(embedding))
        if embedding_norm == 0:
            continue
        dot_similarity = categories @ (embedding / embedding_norm)
        # HOV-SG usa np.argsort(dot_sim)[::-1]: manteniamo anche il suo
        # comportamento deterministico quando due classi hanno lo stesso score.
        ranked_classes = np.asarray(category_names)[np.argsort(dot_similarity)[::-1]]
        ranks.append(int(np.flatnonzero(ranked_classes == gt_label)[0]) + 1)

    def accuracy(k):
        # HOV-SG divides by len(col_ind), including associations whose
        # prediction cannot be classified. Such rows therefore count as 0.
        return sum(rank <= k for rank in ranks) / len(matches) if matches else None

    representative_accuracy = {k: accuracy(k) for k in representative}
    # Same sampling and normalization as top_k.py.  k=0 is deliberate: it is
    # the origin of the top-k accuracy curve used for the AUC.
    auc_k = list(range(0, len(category_names), 10))
    auc = (float(np.trapz([accuracy(k) for k in auc_k],
                          [k / len(category_names) for k in auc_k]))
           if ranks and auc_k else None)
    return representative_accuracy, auc, len(ranks)

def objects(scenes, threshold=.5):
    top_k_totals = {k: 0.0 for k in TOP_K}
    auc_total = 0.0
    classified_matches = 0
    semantic_match_total = 0
    geometric_matches = []
    predicted_total = ground_truth_total = 0
    for s in scenes:
        s = filtered_scene(s)
        pred, gt = s.get("predicted_objects", []), s.get("ground_truth_objects", [])
        predicted_total += len(pred); ground_truth_total += len(gt)
        matches = object_assignment(pred, gt, threshold)
        geometric_matches.extend(score for _, _, score in matches)
        # HOV-SG evaluates against the complete HM3DSEM label vocabulary, not
        # a scene-local closed set. Prefer the same precomputed 1,624 text
        # features used by HOV-SG; the manifest matrix is only a fallback for
        # environments where that asset is unavailable.
        # Prefer an explicitly serialized scene vocabulary: its ordering is
        # authoritative for small test/legacy manifests.  Fall back to the
        # canonical HM3D matrix only when the scene does not carry one.
        categories = s.get("category_embeddings") or None
        if categories is None and all(str(row.get("category_name", "")).strip() for row in gt):
            categories = hm3d_text_features()
        if categories is None:
            categories = []
        class_names = _semantic_classes(s, len(categories))
        # top_k.py evaluates the Hungarian object associations themselves;
        # its semantic curve is not additionally filtered by the IoU threshold.
        incompatible = (s.get("category_embedding_model") and
                        s.get("object_embedding_model") and
                        s["category_embedding_model"] != s["object_embedding_model"])
        # HOV-SG classification uses Hungarian AABB-IoU associations and
        # applies IoU > 0.5 after assignment; it is independent of the
        # centre-distance operating point used by the spatial table.
        semantic_matches = [] if incompatible else hovsg_object_assignment(pred, gt)
        semantic_match_total += len(semantic_matches)
        accuracy, auc, classified = _top_k_semantics(
            semantic_matches, pred, gt, categories, class_names)
        if accuracy and semantic_matches:
            for k, value in accuracy.items(): top_k_totals[k] += value * len(semantic_matches)
        if auc is not None: auc_total += auc * len(semantic_matches)
        classified_matches += classified
    top_k_accuracy = {k: (value / semantic_match_total if semantic_match_total else None)
                      for k, value in top_k_totals.items()}
    # No usable prediction embedding means classification was not measured.
    # Reporting 0% in that case confuses missing input with failed recognition.
    if classified_matches == 0:
        top_k_accuracy = {k: None for k in TOP_K}
    out = {f"top{k}_pct": round(100 * value, 4) if value is not None else None
           for k, value in top_k_accuracy.items()}
    # Keep the existing percent fields, and expose the unscaled values/names
    # returned by the evaluator in top_k.py for direct comparison.
    out["tp_top_k_acc"] = {str(k): round(value, 6) if value is not None else None
                           for k, value in top_k_accuracy.items()}
    out["top_k_auc"] = (round(auc_total / semantic_match_total, 6)
                        if semantic_match_total and classified_matches else None)
    out["auc_top_k"] = out["top_k_auc"]
    out["matched_objects"] = len(geometric_matches)
    out["predicted_objects"] = predicted_total
    out["ground_truth_objects"] = ground_truth_total
    out["precision_pct"] = round(100*len(geometric_matches)/predicted_total,4) if predicted_total else None
    out["recall_pct"] = round(100*len(geometric_matches)/ground_truth_total,4) if ground_truth_total else None
    out["object_precision_pct"] = out["precision_pct"]
    out["object_recall_pct"] = out["recall_pct"]
    out["matched_object_iou_mean"] = round(sum(geometric_matches)/len(geometric_matches),4) if geometric_matches else None
    out["box_quality_mean"] = {"iou_3d": out["matched_object_iou_mean"]}
    out["matched_objects_iou_gt_0.5"] = len(geometric_matches)
    out["classified_matched_objects"] = classified_matches
    out["top_k_eligible_pairs"] = sum(
        len(hovsg_object_assignment(filtered_scene(s).get("predicted_objects", []),
                                    filtered_scene(s).get("ground_truth_objects", [])))
        for s in scenes)
    out["top_k_unclassified_pairs"] = semantic_match_total - classified_matches
    out["top_k_embedding_coverage_pct"] = (round(100 * classified_matches / semantic_match_total, 4)
                                            if semantic_match_total else None)
    out["top_k_incompatible_model_pairs"] = sum(
        1 for s in scenes
        if s.get("category_embedding_model") and s.get("object_embedding_model")
        and s["category_embedding_model"] != s["object_embedding_model"])
    out["top_k_association_details"] = [{"scene": str(s.get("scene", "unknown")),
                                          "associations": [{"predicted_index": pi,
                                                             "ground_truth_index": gi,
                                                             "iou": round(score, 6)}
                                                            for pi, gi, score in hovsg_object_assignment(
                                                                filtered_scene(s).get("predicted_objects", []),
                                                                filtered_scene(s).get("ground_truth_objects", []))]}
                                         for s in scenes]
    semantic_labels = []
    for s in scenes:
        fs = filtered_scene(s)
        for pi, gi, _ in hovsg_object_assignment(fs.get("predicted_objects", []), fs.get("ground_truth_objects", [])):
            semantic_labels.append(str(fs["predicted_objects"][pi].get("label", "")).casefold() ==
                                   str(fs["ground_truth_objects"][gi].get("category_name", "")).casefold())
    out["semantic_accuracy_pct"] = round(100 * sum(semantic_labels) / len(semantic_labels), 4) if semantic_labels else 0
    out["distance_sweep"] = {str(distance): {"matched_objects": sum(
        len(object_assignment(filtered_scene(s).get("predicted_objects", []),
                              filtered_scene(s).get("ground_truth_objects", []), distance))
        for s in scenes)} for distance in (.1, .25, .5, 1.0)}
    return out

def room_objects(scenes, threshold=.5, region_threshold=.5):
    """Compare object occupancy per geometrically matched room.

    A global object match is counted for a room only when both the predicted object
    and the GT object belong to the same matched room.  This prevents an object
    detected in the wrong room from inflating that room's recall.
    """
    rows = []
    total_expected = total_predicted = total_matched = 0
    for scene in scenes:
        scene = filtered_scene(scene)
        pred = scene.get("predicted_objects", [])
        gt = scene.get("ground_truth_objects", [])
        # build_hm3d_eval_manifest emits predicted room -> GT region matches here.
        room_map = {}
        room_iou = {}
        for trial in scene.get("rooms", []):
            predicted_id = trial.get("predicted_region_id")
            ground_truth_id = trial.get("ground_truth_region_id")
            if predicted_id is None or ground_truth_id is None:
                continue
            if (region_threshold is not None
                    and float(trial.get("region_iou", 0.0)) <= region_threshold):
                continue
            room_map[str(predicted_id)] = str(ground_truth_id)
            room_iou[str(ground_truth_id)] = float(trial.get("region_iou", 0.0))

        matches = object_assignment(pred, gt, threshold)
        matched_by_room = {}
        matched_pred_ids = set()
        matched_gt_ids = set()
        for pi, gi, score in matches:
            p, g = pred[pi], gt[gi]
            pred_room = p.get("room_id")
            gt_room = g.get("region_id")
            mapped_room = room_map.get(str(pred_room)) if pred_room is not None else None
            if gt_room is None or mapped_room != str(gt_room):
                continue
            matched_by_room.setdefault(str(gt_room), []).append(score)
            matched_pred_ids.add(pi)
            matched_gt_ids.add(gi)

        gt_room_ids = {str(r.get("region_id")) for r in gt
                       if r.get("region_id") is not None}
        pred_room_ids = {str(r.get("room_id")) for r in pred
                         if r.get("room_id") is not None}
        matched_gt_room_ids = set(room_map.values())
        for room_id in sorted(gt_room_ids | matched_gt_room_ids):
            predicted_room_ids = {pid for pid, gid in room_map.items() if gid == room_id}
            expected = sum(1 for r in gt if str(r.get("region_id")) == room_id)
            predicted = sum(1 for r in pred if str(r.get("room_id")) in predicted_room_ids)
            matched = len(matched_by_room.get(room_id, []))
            missing = max(0, expected - matched)
            extra = max(0, predicted - matched)
            rows.append({
                "scene": str(scene.get("scene", "unknown")),
                "ground_truth_region_id": room_id,
                "predicted_region_ids": sorted(predicted_room_ids),
                "region_iou": round(room_iou.get(room_id, 0.0), 6),
                "expected_objects": expected,
                "predicted_objects": predicted,
                "matched_objects": matched,
                "missing_objects": missing,
                "extra_objects": extra,
                "precision_pct": round(100 * matched / predicted, 4) if predicted else None,
                "recall_pct": round(100 * matched / expected, 4) if expected else None,
                "f1_pct": round(200 * matched / (expected + predicted), 4)
                if expected + predicted else None,
            })
            total_expected += expected
            total_predicted += predicted
            total_matched += matched

        # A predicted object with no room assignment is intentionally not forced
        # into a room; report it in the aggregate as an unassigned extra.
        assigned_predicted = sum(1 for r in pred
                                 if r.get("room_id") is not None and
                                 str(r.get("room_id")) in room_map)
        unassigned_predicted = len(pred) - assigned_predicted
        total_predicted += unassigned_predicted

    return {
        "per_room": rows,
        "rooms_evaluated": len(rows),
        "expected_objects": total_expected,
        "predicted_objects": total_predicted,
        "matched_objects": total_matched,
        "missing_objects": max(0, total_expected - total_matched),
        "extra_objects": max(0, total_predicted - total_matched),
        "precision_pct": round(100 * total_matched / total_predicted, 4)
        if total_predicted else None,
        "recall_pct": round(100 * total_matched / total_expected, 4)
        if total_expected else None,
        "f1_pct": round(200 * total_matched / (total_expected + total_predicted), 4)
        if total_expected + total_predicted else None,
    }

def retrieval(scenes):
    trials = [t for s in scenes for t in s.get("retrieval_trials", [])]
    ret = [int(t["gt_rank"]) <= 10 and float(t["retrieved_iou"]) > .1 for t in trials
           if "gt_rank" in t and "retrieved_iou" in t]
    nav = [float(t["final_distance_m"]) <= 1 for t in trials if "final_distance_m" in t]
    return {"trials": len(trials), "retrieval_sr_at_10_pct": pct(ret), "navigation_sr_pct": pct(nav)}

def sizes(scenes, base):
    per = {}
    missing = []
    for s in scenes:
        paths = s.get("representation_files", [])
        if not paths and s.get("representation_file"): paths = [s["representation_file"]]
        total = 0
        found = False
        for value in paths:
            p = Path(value); p = p if p.is_absolute() else base/p
            try:
                if p.is_file(): total += p.stat().st_size; found = True
                elif p.is_dir(): total += sum(x.stat().st_size for x in p.rglob("*") if x.is_file()); found = True
                else: missing.append(str(p))
            except (OSError, PermissionError):
                missing.append(str(p))
        if paths and found: per[str(s.get("scene", "unknown"))] = round(total/1e6, 6)
    return {"size_mb_total": round(sum(per.values()),6) if per else None,
            "size_mb_per_scene": per, "missing_files": missing}

def evaluate(scenes, base=Path.cwd(), region_iou=.5, object_iou=.5,
             include_match_details=True, include_all_pairs=False):
    report = {"scenes":[str(s.get("scene","unknown")) for s in scenes],
              "table_ii_floor_regions":floor_regions(scenes,region_iou),
              "table_iii_rooms":rooms(scenes), "table_iv_objects":objects(scenes, object_iou),
              "table_structural_elements": structural_metrics(scenes, object_iou),
              "table_vi_room_objects":room_objects(scenes, object_iou, region_iou),
              "table_v_retrieval":retrieval(scenes), "table_vii_representation":sizes(scenes,base)}
    if include_match_details:
        report["match_details"] = match_details(
            scenes, region_iou, object_iou, include_all_pairs)
    times=[float(s["construction_time_s"]) for s in scenes if s.get("construction_time_s") is not None]
    report["construction_time_s"] = round(sum(times),4) if times else None
    missing = []
    if report["table_ii_floor_regions"]["acc_f_pct"] is None:
        missing.append("Table II: ground-truth floors")
    if report["table_iii_rooms"]["rooms_evaluated"] == 0:
        missing.append("Table III: ground-truth room labels")
    if report["table_iv_objects"]["classified_matched_objects"] == 0:
        missing.append("Table IV classification: category/prediction embeddings")
    if report["table_v_retrieval"]["trials"] == 0:
        missing.append("Table V: retrieval trials")
    if report["table_vii_representation"]["size_mb_total"] is None:
        missing.append("Table VII: representation files")
    report["missing_inputs"] = missing
    obj = report["table_iv_objects"]
    regions = report["table_ii_floor_regions"]
    report["summary"] = {
        **{f"top{k}": obj.get(f"top{k}_pct") / 100 if obj.get(f"top{k}_pct") is not None else None for k in TOP_K},
        "AUC_top_k_pct": obj.get("top_k_auc"),
        "Time_s": report.get("construction_time_s"),
        # AP is the association precision, independent of the optional
        # centre-distance operating point used for the detection table.
        "AP": (obj.get("top_k_eligible_pairs") / obj.get("predicted_objects")
               if obj.get("predicted_objects") else None),
        "Acc_F_pct": regions.get("acc_f_pct"),
        "Precision_regions_pct": regions.get("region_precision_pct"),
        "Recall_regions_pct": regions.get("region_recall_pct"),
    }
    return report

EXAMPLE={"scene":"00824","construction_time_s":118,"predicted_floors_m":[0],"ground_truth_floors_m":[.1],"predicted_regions":[{"mask":[0,1,2]}],"ground_truth_regions":[{"mask":[0,1,2,3]}],"rooms":[{"predicted_label":"bedroom","ground_truth_label":"bedroom","approximately_correct":True}],"category_embeddings":[[1,0],[0,1]],"predicted_objects":[{"mask":[0,1],"embedding":[1,0]}],"ground_truth_objects":[{"mask":[0,1],"category_id":0}],"retrieval_trials":[{"gt_rank":3,"retrieved_iou":.4,"final_distance_m":.8}],"representation_files":["persistent_perception.json","room.json"]}

def main():
    argv=sys.argv[1:]; launch=None
    if "--launch" in argv:
        index=argv.index("--launch"); launch=argv[index+1:]; argv=argv[:index]
        if launch[:1]==["--"]: launch=launch[1:]
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("path",nargs="?",type=Path); ap.add_argument("--output",type=Path); ap.add_argument("--region-iou",type=float,default=.5); ap.add_argument("--object-iou",type=float,default=.5); ap.add_argument("--no-thresholds",action="store_true",help="assegna e riporta i match senza filtro IoU"); ap.add_argument("--all-pairs",action="store_true",help="salva anche ogni coppia predetto-GT candidata"); ap.add_argument("--summary-only",action="store_true",help="non includere i dettagli dei match"); ap.add_argument("--example",action="store_true")
    a=ap.parse_args(argv)
    if a.example: print(json.dumps(EXAMPLE,indent=2)); return 0
    if not a.path: ap.error("PATH obbligatorio (oppure --example)")
    elapsed=None; code=0
    if launch is not None:
        if not launch: ap.error("--launch richiede un comando dopo --")
        started=time.monotonic()
        try:
            code=subprocess.run(launch,check=False).returncode
        except KeyboardInterrupt:
            # Ctrl-C is the normal way to finish a ros2 launch.  It must stop ROS,
            # not discard all measurements collected up to that point.
            code=130
        finally:
            elapsed=time.monotonic()-started
    region_iou = None if a.no_thresholds else a.region_iou
    object_iou = None if a.no_thresholds else a.object_iou
    report=evaluate(load(a.path), a.path.parent if a.path.is_file() else a.path,
                    region_iou, object_iou, not a.summary_only, a.all_pairs)
    if elapsed is not None: report.update(construction_time_s=round(elapsed,4),launch_exit_code=code)
    text=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False); print(text)
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(text+"\n",encoding="utf-8")
    return code
if __name__=="__main__": raise SystemExit(main())
