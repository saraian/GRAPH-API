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
    "shower wall", "shower floor", "shower ceiling",
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
    return False


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

def geometry_iou(a,b):
    if "mask" in a and "mask" in b: return iou(a["mask"],b["mask"])
    if all(k in a and k in b for k in ("aabb_min_m","aabb_max_m")): return _aabb_iou(a,b)
    if "polygon_xz_m" in a and "polygon_xz_m" in b: return _polygon_iou(a,b)
    return 0.0

def assignment(pred, gt, threshold):
    if not pred or not gt: return []
    scores = np.asarray([[geometry_iou(p, g) for g in gt] for p in pred])
    try:
        from scipy.optimize import linear_sum_assignment
        ii, jj = linear_sum_assignment(-scores)
    except ImportError:
        ii, jj = [], []
        for i, j in sorted(np.ndindex(scores.shape), key=lambda z: -scores[z]):
            if i not in ii and j not in jj: ii.append(i); jj.append(j)
    return [(int(i), int(j), float(scores[i, j])) for i, j in zip(ii, jj)
            if threshold is None or scores[i, j] > threshold]

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
                pred, gt = scene.get(pred_key, []), scene.get(gt_key, [])
            rows = []
            for pi, gi, score in assignment(pred, gt, None):
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
            if include_all_pairs:
                block["all_candidate_pairs"] = [
                    {"predicted_index": pi,
                     "predicted_id": _entity_id(p, pi, name[:-1]),
                     "ground_truth_index": gi,
                     "ground_truth_id": _entity_id(g, gi, name[:-1]),
                     "iou": round(geometry_iou(p, g), 6)}
                    for pi, p in enumerate(pred) for gi, g in enumerate(gt)
                ]
            scene_row[name] = block
        output.append(scene_row)
    return output

def pct(values): return round(100 * sum(values) / len(values), 4) if values else None

def floor_regions(scenes, threshold):
    fh = ft = rh = pt = gt = 0
    for s in scenes:
        ps, gs = s.get("predicted_floors_m", []), s.get("ground_truth_floors_m", [])
        used_p, used_g = set(), set()
        for d, pi, gi in sorted((abs(float(p)-float(g)), pi, gi) for pi,p in enumerate(ps) for gi,g in enumerate(gs)):
            if d <= .5 and pi not in used_p and gi not in used_g: used_p.add(pi); used_g.add(gi)
        fh += len(used_g); ft += len(gs)
        region_scene = filtered_scene(s, include_regions=True)
        pr = region_scene.get("predicted_regions", [])
        gr = region_scene.get("ground_truth_regions", [])
        rh += len(assignment(pr, gr, threshold)); pt += len(pr); gt += len(gr)
    return {"acc_f_pct": round(100*fh/ft,4) if ft else None,
            "region_precision_pct": round(100*rh/pt,4) if pt else None,
            "region_recall_pct": round(100*rh/gt,4) if gt else None,
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
        matches = assignment(pred, gt, threshold)
        geometric_matches.extend(score for _, _, score in matches)
        # HOV-SG evaluates against the complete HM3DSEM label vocabulary, not
        # a scene-local closed set. Prefer the same precomputed 1,624 text
        # features used by HOV-SG; the manifest matrix is only a fallback for
        # environments where that asset is unavailable.
        has_named_gt = all(str(row.get("category_name", "")).strip() for row in gt)
        categories = hm3d_text_features() if has_named_gt else None
        if categories is None:
            categories = s.get("category_embeddings", [])
        if categories is None:
            categories = []
        class_names = _semantic_classes(s, len(categories))
        # top_k.py evaluates the Hungarian object associations themselves;
        # its semantic curve is not additionally filtered by the IoU threshold.
        semantic_matches = assignment(pred, gt, None)
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
    out["object_precision_pct"] = round(100*len(geometric_matches)/predicted_total,4) if predicted_total else None
    out["object_recall_pct"] = round(100*len(geometric_matches)/ground_truth_total,4) if ground_truth_total else None
    out["matched_object_iou_mean"] = round(sum(geometric_matches)/len(geometric_matches),4) if geometric_matches else None
    out["matched_objects_iou_gt_0.5"] = len(geometric_matches)
    out["classified_matched_objects"] = classified_matches
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

        matches = assignment(pred, gt, threshold)
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
            if p.is_file(): total += p.stat().st_size; found = True
            elif p.is_dir(): total += sum(x.stat().st_size for x in p.rglob("*") if x.is_file()); found = True
            else: missing.append(str(p))
        if paths and found: per[str(s.get("scene", "unknown"))] = round(total/1e6, 6)
    return {"size_mb_total": round(sum(per.values()),6) if per else None,
            "size_mb_per_scene": per, "missing_files": missing}

def evaluate(scenes, base=Path.cwd(), region_iou=.5, object_iou=.5,
             include_match_details=True, include_all_pairs=False):
    report = {"scenes":[str(s.get("scene","unknown")) for s in scenes],
              "table_ii_floor_regions":floor_regions(scenes,region_iou),
              "table_iii_rooms":rooms(scenes), "table_iv_objects":objects(scenes, object_iou),
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
