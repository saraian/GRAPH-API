"""External replay evaluation. No native inference, clustering or tracking changes."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .replay_model import file_stamp, read_json


def validate_metrics(rows):
    previous = -math.inf
    for row in rows:
        stamp = row['time_s']
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < previous:
            raise ValueError('Metrics need finite, ordered acquisition-clock time_s')
        previous = stamp
        for key, metric in row['metrics'].items():
            if not all(k in metric for k in ('value', 'unit', 'source', 'scope')) or not metric['source'] or not metric['scope']:
                raise ValueError(f'Metric {key} lacks value/unit/source/scope')
            value = metric['value']
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
                raise ValueError(f'Metric {key} must be finite and nonnegative, or null')
            if key in ('precision', 'recall', 'accuracy', 'f1') and value is not None:
                limit = {'%': 100, 'fraction': 1}.get(metric['unit'])
                if limit is None or value > limit:
                    raise ValueError(f'Metric {key} has invalid score units/range')


def bounds(low, high):
    box = np.array([low, high], dtype=float)
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] < box[0]):
        raise ValueError('Invalid ground-truth/predicted box')
    return box


def match_boxes(predicted, truth, threshold):
    """Maximum-cardinality matching above threshold, then maximum total 3D IoU."""
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    if not predicted or not truth:
        return []
    from scipy.optimize import linear_sum_assignment
    p, g = np.array(predicted), np.array(truth)
    overlap = np.maximum(0, np.minimum(p[:, None, 1], g[None, :, 1]) - np.maximum(p[:, None, 0], g[None, :, 0])).prod(axis=2)
    union = (p[:, 1]-p[:, 0]).prod(axis=1)[:, None] + (g[:, 1]-g[:, 0]).prod(axis=1)[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    # One additional valid match outweighs every possible IoU tie-break combined.
    eligible = iou >= threshold
    reward = eligible * (min(len(p), len(g)) + 1 + iou)
    pi, gi = linear_sum_assignment(-reward)
    return [(int(a), int(b), float(iou[a, b])) for a, b in zip(pi, gi) if eligible[a, b]]


def evaluate(model_path, gt_path, output, threshold=0.25):
    model, gt = read_json(model_path), read_json(gt_path)
    if model['graph']['scope'] != 'final_snapshot':
        raise ValueError('This evaluator requires a final native snapshot')
    if not gt.get('geometry_space', {}).get('coordinate_frame', '').startswith('Habitat:'):
        raise ValueError('Ground truth must explicitly use Habitat Y-up coordinates')
    source_scene = Path(gt['ground_truth_source']['scene']).parent.name
    if source_scene != model['scene']:
        raise ValueError('Ground truth belongs to a different scene')
    if gt['ground_truth_source'].get('selected_floor_index') is not None:
        raise ValueError('Whole-scene evaluation requires unfiltered scene GT')
    # Static GT is invalid while a scheduled object remains spawned/moved.
    active = set()
    for action in model['actions']:
        oid = str(action['result']['object_id'])
        if action['action'] == 'remove':
            active.discard(oid)
        else:
            active.add(oid)
    if active:
        raise ValueError('Final dynamic world differs from static GT; provide dynamic GT evaluation instead')
    objects = [n for n in model['graph']['nodes'] if n['type'] == 'object']
    predicted, predicted_ids, missing = [], [], []
    for obj in objects:
        if 'corners' not in obj:
            missing.append(obj['id'])
            continue
        corners = np.asarray(obj['corners'], dtype=float)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            raise ValueError('Invalid predicted corners')
        predicted.append(bounds(corners.min(0), corners.max(0)))
        predicted_ids.append(obj['id'])
    truth_rows = gt['ground_truth_objects']
    if not truth_rows:
        raise ValueError('No native GT objects available')
    if any(row.get('geometry_source') != 'semantic_object_obb_to_aabb' for row in truth_rows):
        raise ValueError('Only native Habitat object boxes are accepted')
    truth_ids = [str(row['object_id']) for row in truth_rows]
    if len(set(truth_ids)) != len(truth_ids) or len({n['id'] for n in objects}) != len(objects):
        raise ValueError('Duplicate object IDs')
    all_truth = [bounds(r['aabb_min_m'], r['aabb_max_m']) for r in truth_rows]
    excluded = [{'object_id': oid, 'reason': 'zero-volume native 3D box'}
                for oid, box in zip(truth_ids, all_truth) if np.any(box[1] == box[0])]
    eligible = [(oid, box) for oid, box in zip(truth_ids, all_truth) if np.all(box[1] > box[0])]
    if not eligible:
        raise ValueError('No positive-volume native GT boxes available')
    truth_ids, truth = map(list, zip(*eligible))
    matches = match_boxes(predicted, truth, threshold)
    tp, npred, ngt = len(matches), len(objects), len(truth)
    source = {'model': file_stamp(model_path), 'ground_truth': file_stamp(gt_path), 'evaluator': file_stamp(__file__)}
    scope = (f'Final whole-scene class-agnostic AABB overlap at 3D IoU >= {threshold}; '
             'one-to-one maximum-cardinality matching; positive-volume native GT instances including structure/unseen rooms; '
             'zero-volume GT boxes excluded and listed; '
             'not visible-object detection accuracy, semantic accuracy, or a paper benchmark')
    metrics = {}
    for key, value in [('precision', tp / npred if npred else None), ('recall', tp / ngt),
                       ('f1', 2 * tp / (npred + ngt))]:
        metrics[key] = {'value': round(100*value, 3) if value is not None else None, 'unit': '%',
                        'label': 'Final box ' + ('F1' if key == 'f1' else key), 'scope': scope, 'source': source}
    report = {'schema': 'graphapi.baseline_metrics.v1', 'model_sha256': source['model']['sha256'],
              'scope': scope, 'iou_threshold': threshold, 'metrics': metrics, 'tp': tp, 'fp': npred-tp, 'fn': ngt-tp,
              'predicted_objects': npred, 'gt_objects': ngt, 'prediction_boxes_missing': missing,
              'gt_objects_raw': len(truth_rows), 'gt_boxes_excluded': excluded,
              'matches': [{'prediction_id': predicted_ids[a], 'gt_id': truth_ids[b], 'iou_3d': score}
                          for a, b, score in matches],
              'limitations': ['Clio task clusters and HOV instances have different granularity.',
                              'Whole-scene recall includes unobserved rooms; it is coverage-sensitive.',
                              'Static scene GT only; historical dynamic objects are not evaluated.']}
    with Path(output).open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--ground-truth', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iou', type=float, default=0.25)
    args = parser.parse_args()
    report = evaluate(args.model, args.ground_truth, args.output, args.iou)
    print(json.dumps({k: report[k] for k in ('tp', 'fp', 'fn', 'metrics')}))


if __name__ == '__main__':
    main()
