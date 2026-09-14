"""Object evaluation v2: associate positions, then measure shape and semantics.

No GT labels, box IoU or embeddings enter the spatial assignment. Distances
are Euclidean in 3D, in metres. This is an indoor adaptation of the separation
between translation and box quality used in nuScenes, not its benchmark score.
"""
from collections import Counter, defaultdict
import math
import re

import numpy as np

import script_ledger
from scipy.optimize import linear_sum_assignment

DEFAULT_DISTANCE_M = 0.5
DISTANCE_THRESHOLDS_M = (0.1, 0.25, 0.5, 1.0)
# Equivalence only: do not collapse subclasses (table lamp -> lamp) or use
# substring matching (table != table lamp; towel != towel bar).
LABEL_ALIASES = {
    'couch': 'sofa', 'washbasin': 'sink', 'tub': 'bathtub',
    'telephone': 'phone', 'tv': 'television', 'trash can': 'bin',
    'trashcan': 'bin', 'garbage bin': 'bin', 'wastebasket': 'bin',
    'refrigerator': 'fridge',
}
UNKNOWN_LABELS = {'', 'unknown', 'unlabeled', 'unlabelled', 'void'}


def normalized_label(value):
    return re.sub(r'#\d+$', '', str(value or '').strip().casefold()).strip()


def canonical_label(value):
    label = normalized_label(value)
    return LABEL_ALIASES.get(label, label)


def label(row, predicted=False):
    return row.get('label', row.get('predicted_label')) if predicted else row.get('category_name')


def bounds(row):
    try:
        low = np.asarray(row['aabb_min_m'], dtype=float)
        high = np.asarray(row['aabb_max_m'], dtype=float)
        if (low.shape == high.shape == (3,) and np.all(np.isfinite([low, high]))
                and np.all(high >= low)):
            return low, high
    except (KeyError, TypeError, ValueError):
        pass
    return None


def center(row):
    # The adapter records the centre displayed by RViz when an OBB is used.
    if 'center_m' in row:
        try:
            value = np.asarray(row['center_m'], dtype=float)
            if value.shape == (3,) and np.all(np.isfinite(value)):
                return value
        except (TypeError, ValueError):
            pass
        return None
    box = bounds(row)
    return (box[0] + box[1]) / 2 if box is not None else None


def observed_at(row):
    """When a predicted object was last seen, or None when the manifest does not say."""
    for key in ("observed_at", "last_perception_timestamp"):
        value = (row or {}).get(key)
        try:
            out = float(value)
        except (TypeError, ValueError):
            continue
        if out == out:
            return out
    return None


def distances(pred, gt):
    """Centre-to-centre distance, INFINITE where the pair could not have coexisted.

    A scripted scene spawns, moves and removes objects while the run is under way, so a
    ground-truth row is not true for the whole run -- it is true for a window (script_ledger).
    Scoring a prediction against a row that was not true when the prediction was made is what made
    a spawned object a false positive and a removed one a false negative, neither of which is a
    perception error.

    The gate is HERE rather than in the assignment, because `match_costs` already treats an
    infinite cost as "not a candidate". Every caller therefore gets time-awareness without
    changing, and a run with no ledger is untouched: static rows carry no window and hold at every
    time, so the matrix is identical to what it was.
    """
    result = np.full((len(pred), len(gt)), np.inf)
    pc, gc = [center(p) for p in pred], [center(g) for g in gt]
    pt = [observed_at(p) for p in pred]
    for i, p in enumerate(pc):
        for j, g in enumerate(gc):
            if p is None or g is None:
                continue
            if not script_ledger.holds_at(gt[j], pt[i]):
                continue
            result[i, j] = np.linalg.norm(p - g)
    return result


def match_costs(costs, threshold):
    """Maximum cardinality within the gate, then minimum total distance."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('object distance threshold must be finite and positive')
    if not costs.size:
        return []
    valid = np.isfinite(costs) & (costs <= threshold)
    benefit = np.where(valid, min(costs.shape) + 1 - costs / threshold, 0.0)
    ii, jj = linear_sum_assignment(-benefit)
    return [(int(i), int(j), float(costs[i, j]))
            for i, j in zip(ii, jj) if valid[i, j]]


def associate(pred, gt, threshold=DEFAULT_DISTANCE_M):
    return match_costs(distances(pred, gt), threshold)


def box_quality(pred, gt):
    """Actual AABB IoU, translation-aligned size IoU and directional coverage.

    Thin/degenerate boxes have a usable centre but undefined volume metrics.
    """
    p, g = bounds(pred), bounds(gt)
    if p is None or g is None:
        return {}
    ps, gs = p[1] - p[0], g[1] - g[0]
    pv, gv = float(np.prod(ps)), float(np.prod(gs))
    if pv <= 0 or gv <= 0:
        return {}
    intersection = float(np.prod(np.maximum(0, np.minimum(p[1], g[1]) - np.maximum(p[0], g[0]))))
    aligned = float(np.prod(np.minimum(ps, gs)))
    return {
        'iou_3d': intersection / (pv + gv - intersection),
        'size_iou': aligned / (pv + gv - aligned),
        'predicted_volume_inside_gt': intersection / pv,
        'gt_volume_covered': intersection / gv,
        'predicted_to_gt_volume_ratio': pv / gv,
    }


def rate(n, d):
    return round(100 * n / d, 4) if d else None


def detection_counts(tp, predicted, ground_truth):
    return {'matched_objects': tp, 'predicted_objects': predicted,
            'ground_truth_objects': ground_truth,
            'false_positives': predicted - tp, 'false_negatives': ground_truth - tp,
            'precision_pct': rate(tp, predicted), 'recall_pct': rate(tp, ground_truth),
            'f1_pct': rate(2 * tp, predicted + ground_truth)}


def average_precision(scene_inputs, threshold):
    """Integral of the precision envelope over confidence-threshold recall.

    Ties enter together, so JSON ordering never changes AP. If all confidences
    are absent there is one operating point and AP = precision * recall.
    Mixed available/missing scores cannot define a ranking and return null.
    scene_inputs is a sequence of (cost_matrix, prediction_rows).
    """
    rows = [p for _, pred in scene_inputs for p in pred]
    scores = [p.get('confidence', p.get('score')) for p in rows]
    gt_count = sum(cost.shape[1] for cost, _ in scene_inputs)
    mode = 'confidence_thresholds'
    if all(s is None for s in scores):
        scores = [1.0] * len(rows)
        mode = 'single_operating_point_no_confidence'
    try:
        scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(scores)):
            return {'ap': None, 'ap_pct': None, 'ap_mode': 'missing_or_invalid_confidence'}
    except (TypeError, ValueError):
        return {'ap': None, 'ap_pct': None, 'ap_mode': 'missing_or_invalid_confidence'}
    if not gt_count:
        return {'ap': None, 'ap_pct': None, 'ap_mode': mode}
    curve = []
    for score in sorted(set(scores), reverse=True):
        offset = tp = count = 0
        for cost, pred in scene_inputs:
            active = np.flatnonzero(scores[offset:offset + len(pred)] >= score)
            offset += len(pred)
            tp += len(match_costs(cost[active], threshold))
            count += len(active)
        curve.append((tp / gt_count, tp / count))
    envelope = np.maximum.accumulate([p for _, p in curve][::-1])[::-1]
    prev = ap = 0.0
    for (recall, _), precision in zip(curve, envelope):
        ap += (recall - prev) * precision
        prev = recall
    return {'ap': round(float(ap), 6), 'ap_pct': round(100 * float(ap), 4),
            'ap_mode': mode}



def _scripted_counts(scenes, matches_by_scene):
    """Score the scene script's own objects as their OWN column.

    Pooling them with the static scene hides both. A spawned object that the robot found is a
    different result from a chair that was always there and was found: one says the system tracks
    change, the other says it maps a room. And a missed scripted object is the only miss whose
    ground truth we placed ourselves, so it is the one we can be sure was really there.

    Returns `present: False` when no scene had a script -- NOT zeros, which would read as a script
    that ran and was entirely missed (working rule 61: asserted, present-but-empty, and absent are
    three states, and the middle one is the interesting one).
    """
    total = matched = defaulted = 0
    per_object = {}
    for scene, matches in zip(scenes, matches_by_scene):
        gt = scene.get('ground_truth_objects', [])
        scripted_idx = {i for i, g in enumerate(gt) if isinstance(g, dict) and g.get('scripted')}
        if not scripted_idx:
            continue
        total += len(scripted_idx)
        hit = {gi for _, gi, _ in matches if gi in scripted_idx}
        matched += len(hit)
        for i in scripted_idx:
            name = str(gt[i].get('script_object_id') or gt[i].get('object_id'))
            slot = per_object.setdefault(name, {'poses': 0, 'found': 0})
            slot['poses'] += 1
            slot['found'] += int(i in hit)
            if gt[i].get('extents_source') == 'default':
                defaulted += 1
    if not total:
        return {'present': False,
                'note': 'no scene had a script; the static evaluation is unchanged'}
    # HOW MANY PREDICTIONS THE GATE COULD NOT JUDGE. holds_at() lets an untimed prediction match any
    # window on purpose -- refusing it would turn every pre-change bundle into 0% scripted recall,
    # and an absent timestamp cannot be told from a system that genuinely has none. But that
    # permissiveness EXEMPTS an untimed system from the very gate a timed one is held to, and until
    # this counter existed nothing in the output said so. MEASURED on identical geometry: with
    # observed_at set on all three predictions, precision 66.7% / scripted recall 50.0%; with the key
    # simply absent, 100% / 100%. A scripted column is comparable across systems ONLY when this is 0
    # for every system in the table; otherwise the untimed one is exempt, and must be marked so
    # rather than quoted.
    untimed = sum(observed_at(p) is None
                  for scene in scenes for p in scene.get('predicted_objects', []))
    predicted = sum(len(scene.get('predicted_objects', [])) for scene in scenes)
    return {
        'present': True,
        'untimed_predictions': untimed,
        'gate_applied_to_pct': round(100.0 * (predicted - untimed) / predicted, 4) if predicted else None,
        'comparable': untimed == 0,
        'poses': total,
        'poses_matched': matched,
        'poses_missed': total - matched,
        'recall_pct': round(100.0 * matched / total, 4),
        # OBJECT recall, beside pose recall, because pose recall has a CEILING BELOW 100%.
        # A predicted object carries ONE observation time and the assignment is one-to-one, so a
        # perfectly tracked object that was moved can satisfy only ONE of its two disjoint windows:
        # a flawless tracker scores 50% pose recall on a single move. Quoting pose recall alone
        # would report that ceiling as a failure. `poses_reachable` states the ceiling explicitly
        # rather than leaving a reader to discover it.
        'objects': len(per_object),
        'objects_found': sum(1 for v in per_object.values() if v['found']),
        'object_recall_pct': (round(100.0 * sum(1 for v in per_object.values() if v['found'])
                                    / len(per_object), 4) if per_object else None),
        'poses_reachable': len(per_object),
        'pose_recall_ceiling_pct': (round(100.0 * len(per_object) / total, 4) if total else None),
        # Volume metrics are meaningless on a pose whose size was fabricated (script_ledger's
        # default cube). This is the count, not a silent degradation.
        'box_quality_defaulted': defaulted,
        'box_quality_quotable': defaulted == 0,
        'per_object': per_object,
        'note': ('one POSE per spawn and per move: an object moved once contributes two poses, '
                 'and finding it at only one of them is a partial result, not a miss. A pose is '
                 'matched only by a prediction observed while that pose was true. With one '
                 'observation time per predicted object, pose recall cannot exceed '
                 'pose_recall_ceiling_pct -- read object_recall_pct beside it.'),
    }

def evaluate_geometry(scenes, threshold=DEFAULT_DISTANCE_M):
    inputs, matches_by_scene = [], []
    quality = defaultdict(list)
    center_errors = []
    invalid_pred = invalid_gt = ambiguous = 0
    for scene in scenes:
        pred, gt = scene.get('predicted_objects', []), scene.get('ground_truth_objects', [])
        cost = distances(pred, gt)
        matches = match_costs(cost, threshold)
        inputs.append((cost, pred))
        matches_by_scene.append(matches)
        invalid_pred += sum(center(p) is None for p in pred)
        invalid_gt += sum(center(g) is None for g in gt)
        for pi, gi, distance in matches:
            center_errors.append(distance)
            for key, value in box_quality(pred[pi], gt[gi]).items():
                quality[key].append(value)
            # Audit nearby alternatives without excluding difficult examples.
            alternatives = np.delete(cost[pi], gi)
            ambiguous += bool(np.any((alternatives <= threshold) &
                                     (np.abs(alternatives - distance) <= .05)))
    predicted = sum(len(p) for _, p in inputs)
    ground_truth = sum(c.shape[1] for c, _ in inputs)
    out = detection_counts(len(center_errors), predicted, ground_truth)
    out['scripted'] = _scripted_counts(scenes, matches_by_scene)
    out.update(average_precision(inputs, threshold))
    out.update({
        'protocol': 'objects_v2_center3d', 'distance_threshold_m': threshold,
        'distance_sweep': {
            f'{t:g}': detection_counts(sum(len(match_costs(c, t)) for c, _ in inputs),
                                      predicted, ground_truth)
            for t in sorted(set(DISTANCE_THRESHOLDS_M + (threshold,)))},
        'center_error_mean_m': round(float(np.mean(center_errors)), 6) if center_errors else None,
        'center_error_median_m': round(float(np.median(center_errors)), 6) if center_errors else None,
        'center_error_p90_m': round(float(np.percentile(center_errors, 90)), 6) if center_errors else None,
        'box_quality_mean': {key: round(float(np.mean(quality[key])), 6) if quality[key] else None
                             for key in ('iou_3d', 'size_iou', 'predicted_volume_inside_gt',
                                         'gt_volume_covered', 'predicted_to_gt_volume_ratio')},
        'box_quality_evaluated_pairs': len(quality['iou_3d']),
        'invalid_predicted_centers': invalid_pred, 'invalid_gt_centers': invalid_gt,
        'ambiguous_matches_within_5cm': ambiguous,
        'recall_scope': 'all nonstructural GT objects on selected floor; visibility unknown',
        'assignment': 'one-to-one, distance <= threshold, maximum count then minimum distance; no label or IoU gate',
    })
    return out, matches_by_scene


def evaluate_labels(scenes, matches_by_scene):
    correct = exact = total = unknown = known_gt = predicted = 0
    confusion = Counter()
    per_class = defaultdict(lambda: [0, 0])
    for scene, matches in zip(scenes, matches_by_scene):
        pred, gt = scene.get('predicted_objects', []), scene.get('ground_truth_objects', [])
        predicted += len(pred)
        known_gt += sum(canonical_label(label(g)) not in UNKNOWN_LABELS for g in gt)
        for pi, gi, _ in matches:
            p, g = canonical_label(label(pred[pi], True)), canonical_label(label(gt[gi]))
            if g in UNKNOWN_LABELS:
                unknown += 1
                continue
            total += 1
            correct += p == g
            exact += normalized_label(label(pred[pi], True)) == normalized_label(label(gt[gi]))
            confusion[(g, p)] += 1
            per_class[g][0] += p == g
            per_class[g][1] += 1
    return {
        'semantic_accuracy_pct': rate(correct, total),
        'semantic_exact_accuracy_pct': rate(exact, total),
        'semantic_macro_accuracy_pct': (round(float(np.mean([100 * a / n for a, n in per_class.values()])), 4)
                                        if per_class else None),
        'semantic_evaluated_pairs': total, 'semantic_correct_pairs': correct,
        'semantic_unknown_gt_pairs': unknown,
        'semantic_precision_pct': rate(correct, predicted - unknown),
        'semantic_recall_pct': rate(correct, known_gt),
        'semantic_confusions': [{'ground_truth': g, 'predicted': p, 'count': count}
                               for (g, p), count in sorted(confusion.items())],
        'semantic_protocol': 'labels on spatial matches; explicit equivalent aliases only; unknown GT excluded; missing predicted labels are errors',
        'label_aliases': LABEL_ALIASES,
    }
