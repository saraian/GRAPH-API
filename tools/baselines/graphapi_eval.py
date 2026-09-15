"""End-to-end GRAPH-API evaluation adapters for unchanged native baselines.

The adapters convert native Clio and HOV-SG artifacts into the manifest used by
``metrics_eval.py`` and ``object_metrics.py``.  Scheduled lifecycle response is
evaluated separately from the final static tables because neither baseline
emits GRAPH-API lifecycle events.  Missing native concepts remain unsupported;
semantic primitives are never presented as native object instances.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .replay_model import ZUP_TO_HABITAT, bounds_corners, ply_points


SCHEMA = 'graphapi.baseline_end_to_end_eval.v1'
ROOT = Path(__file__).resolve().parents[2]
METRICS_ROOT = ROOT / 'lost3dsg/src/perception_module'


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Not JSON serializable: {type(value).__name__}')


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stamp(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest(),
            'bytes': path.stat().st_size}


def evaluators():
    """Import the repository's authoritative evaluators, not a copied fork."""
    location = str(METRICS_ROOT)
    if location not in sys.path:
        sys.path.insert(0, location)
    metrics = importlib.import_module('metrics_eval')
    objects = importlib.import_module('object_metrics')
    for module in (metrics, objects):
        if Path(module.__file__).resolve().parent != METRICS_ROOT.resolve():
            raise RuntimeError(f'Wrong evaluator imported: {module.__file__}')
    return metrics, objects


def require_ground_truth(gt):
    required = ('ground_truth_floors_m', 'ground_truth_regions',
                'ground_truth_objects', 'rooms', 'categories')
    missing = [key for key in required if key not in gt]
    if missing:
        raise ValueError('Full GRAPH-API ground truth is missing: ' + ', '.join(missing))
    if not gt['ground_truth_objects'] or not gt['ground_truth_regions']:
        raise ValueError('Full GRAPH-API ground truth has no objects or regions')
    frame = str((gt.get('geometry_space') or {}).get('coordinate_frame', ''))
    if not frame.startswith('Habitat:'):
        raise ValueError('Ground truth must use explicit Habitat Y-up coordinates')


def action_timeline(recording):
    recording = Path(recording)
    frames = read_rows(recording / 'frames.jsonl')
    actions = read_rows(recording / 'object_actions.jsonl')
    if not frames or any(row['index'] != index for index, row in enumerate(frames)):
        raise ValueError('Invalid recording frame order')
    for row in actions:
        if not row.get('result', {}).get('success'):
            raise ValueError('A scheduled action failed')
        first = int(row['after_frame']) + 1
        if first >= len(frames):
            raise ValueError('Scheduled action has no following observation')
        row['first_post_action_frame'] = first
        row['time_s'] = frames[first]['time_s']
    return frames, actions


def _template_label(value):
    name = Path(str(value)).stem
    if name.endswith('.object_config'):
        name = name[:-len('.object_config')]
    name = name.split(':', 1)[0]
    parts = name.split('_')
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return ' '.join(parts).strip() or 'dynamic object'


def _evaluation_id(result):
    return str(result.get('evaluation_object_id', result['object_id']))


def final_ground_truth(gt, frames, actions):
    """Return GT for the actual final world, including intentionally live objects."""
    result = dict(gt)
    objects = [dict(row) for row in gt['ground_truth_objects']]
    final_dynamic = {str(row.get('evaluation_object_id', row['object_id'])): row
                     for row in frames[-1].get('dynamic_ground_truth', [])}
    spawn = {_evaluation_id(row['result']): row for row in actions
             if row['action'] == 'spawn'}
    categories = [dict(row) for row in gt.get('categories', [])]
    category_ids = {str(row['category_name']): int(row['category_id'])
                    for row in categories}
    next_category = max(category_ids.values(), default=-1) + 1
    for object_id, dynamic in sorted(final_dynamic.items()):
        label = _template_label((spawn.get(object_id, {}).get('request') or {}).get(
            'template', dynamic.get('handle', 'dynamic object')))
        if label not in category_ids:
            category_ids[label] = next_category
            categories.append({'category_id': next_category, 'category_name': label})
            next_category += 1
        objects.append({'object_id': f'dynamic:{object_id}',
            'semantic_id': dynamic.get('semantic_id'), 'category_name': label,
            'category_id': category_ids[label], 'region_id': None,
            'aabb_min_m': dynamic['aabb_min'], 'aabb_max_m': dynamic['aabb_max'],
            'geometry_source': 'scheduled_dynamic_object_world_aabb'})
    result['ground_truth_objects'] = objects
    result['categories'] = categories
    result['dynamic_final_objects'] = len(final_dynamic)
    return result


def _active_floor(gt, acquisition):
    floors = np.asarray(gt['ground_truth_floors_m'], dtype=float)
    if not len(floors):
        return None
    height = acquisition.get('selected_floor_height_m')
    if height is None:
        return 0 if len(floors) == 1 else None
    return int(np.argmin(np.abs(floors - float(height))))


def _polygon_area(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        return 0.0
    return float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) -
                 np.dot(points[:, 1], np.roll(points[:, 0], -1))) / 2


def occupancy_outline(vertices, resolution=.05):
    """Trace the largest exterior of HOV-SG's native 2-D occupancy points."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('HOV room has no usable native 2-D vertices')
    if not np.isfinite(points).all() or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('Invalid HOV room geometry')
    origin = points.min(axis=0)
    indexes = np.rint((points - origin) / resolution).astype(int)
    cells = {tuple(value) for value in indexes}
    # Counter-clockwise edges around occupied cell squares.
    edges = set()
    for x, y in cells:
        if (x, y - 1) not in cells:
            edges.add(((x, y), (x + 1, y)))
        if (x + 1, y) not in cells:
            edges.add(((x + 1, y), (x + 1, y + 1)))
        if (x, y + 1) not in cells:
            edges.add(((x + 1, y + 1), (x, y + 1)))
        if (x - 1, y) not in cells:
            edges.add(((x, y + 1), (x, y)))
    outgoing = defaultdict(list)
    for edge in edges:
        outgoing[edge[0]].append(edge[1])
    loops = []
    unused = set(edges)
    while unused:
        first = min(unused)
        start, current = first
        loop = [start]
        unused.remove(first)
        limit = len(edges) + 1
        while current != start and len(loop) < limit:
            loop.append(current)
            choices = [target for target in outgoing[current]
                       if (current, target) in unused]
            if not choices:
                break
            target = min(choices)
            unused.remove((current, target))
            current = target
        if current == start and len(loop) >= 3:
            polygon = origin + (np.asarray(loop, dtype=float) - .5) * resolution
            loops.append(polygon)
    if not loops:
        raise ValueError('Could not trace HOV room occupancy boundary')
    polygon = max(loops, key=lambda value: abs(_polygon_area(value)))
    return polygon.tolist()


def _bounds_from_points(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return None
    if not np.isfinite(points).all():
        raise ValueError('Non-finite native object points')
    return points.min(axis=0), points.max(axis=0)


def _timing(result):
    value = read_json(Path(result) / 'execution_timing.json')
    phases = value.get('phases_s', {})
    return float(phases.get('run_wall_time', sum(float(phases.get(key, 0))
        for key in ('model_initialization', 'feature_map', 'hierarchy'))))


def hov_manifest(gt, recording, result):
    """Adapt native HOV-SG hierarchy files to a GRAPH-API scene manifest."""
    recording, result = Path(recording), Path(result)
    acquisition = read_json(recording / 'acquisition.json')
    active_floor = _active_floor(gt, acquisition)
    manifest = dict(gt)
    manifest['active_floor_index'] = active_floor
    graph = result / 'graph'
    floor_rows = [read_json(path) for path in sorted((graph / 'floors').glob('*.json'))]
    room_rows = [read_json(path) for path in sorted((graph / 'rooms').glob('*.json'))]
    object_paths = sorted((graph / 'objects').glob('*.json'))
    if not floor_rows or not room_rows or not object_paths:
        raise ValueError('HOV-SG hierarchy is incomplete')
    manifest['predicted_floors_m'] = [float(row['floor_zero_level']) for row in floor_rows]
    floor_index = {str(row['floor_id']): index for index, row in enumerate(floor_rows)}
    predicted_regions = []
    for row in room_rows:
        predicted_regions.append({'room_id': str(row['room_id']),
            'floor_index': active_floor if active_floor is not None else
                floor_index.get(str(row.get('floor_id'))),
            'polygon_xz_m': occupancy_outline(row['vertices'])})
    manifest['predicted_regions'] = predicted_regions
    predicted_objects = []
    for path in object_paths:
        row = read_json(path)
        ply = path.with_suffix('.ply')
        if not ply.is_file():
            raise ValueError(f'HOV object lacks its native 3-D PLY: {ply}')
        box = _bounds_from_points(ply_points(ply))
        if box is None:
            continue
        embedding = np.asarray(row.get('embedding'), dtype=float)
        item = {'object_id': str(row['object_id']), 'label': str(row.get('name', '')),
            'room_id': str(row['room_id']),
            'floor_index': active_floor if active_floor is not None else
                floor_index.get(str(row.get('floor_id', ''))),
            'aabb_min_m': box[0].tolist(), 'aabb_max_m': box[1].tolist()}
        if embedding.ndim == 1 and len(embedding) and np.isfinite(embedding).all():
            item['embedding'] = embedding.tolist()
        predicted_objects.append(item)
    if not predicted_objects:
        raise ValueError('HOV-SG native graph contains no usable 3-D object predictions')
    manifest['predicted_objects'] = predicted_objects
    assets_path = result / 'evaluation_assets/manifest.json'
    asset_meta = read_json(assets_path) if assets_path.is_file() else None
    room_labels = {row['room_id']: row.get('predicted_label') for row in
                   (asset_meta or {}).get('room_predictions', [])}
    metrics, _ = evaluators()
    matches = metrics.assignment(predicted_regions, manifest['ground_truth_regions'], None)
    manifest['rooms'] = []
    gt_regions = manifest['ground_truth_regions']
    for pi, gi, score in matches:
        predicted_label = room_labels.get(predicted_regions[pi]['room_id'])
        ground_truth_label = str(gt_regions[gi].get('category_name', ''))
        manifest['rooms'].append({'predicted_region_id': predicted_regions[pi]['room_id'],
            'ground_truth_region_id': str(gt_regions[gi].get('region_id')),
            'predicted_label': predicted_label, 'ground_truth_label': ground_truth_label,
            'approximately_correct': (predicted_label or '').casefold() ==
                                     ground_truth_label.casefold(),
            'region_iou': score})
    object_features = result / 'evaluation_assets/object_category_features.npy'
    if asset_meta and object_features.is_file():
        features = np.load(object_features)
        if (features.ndim != 2 or len(features) != len(asset_meta['object_categories'])
                or features.shape[1] != asset_meta['embedding_dimension']
                or not np.isfinite(features).all()):
            raise ValueError('Invalid HOV object category evaluation assets')
        manifest['category_embeddings'] = features.tolist()
        manifest['category_names'] = asset_meta['object_categories']
        manifest['category_embedding_model'] = asset_meta['embedding_model']
        manifest['object_embedding_model'] = asset_meta['embedding_model']
    else:
        manifest['category_embeddings'] = []
    manifest['construction_time_s'] = _timing(result)
    manifest['representation_files'] = [str(graph.resolve())]
    manifest['adapter_notes'] = {
        'baseline': 'hovsg', 'coordinate_frame': 'Habitat Y-up',
        'active_floor_index': active_floor,
        'room_geometry': 'largest exterior traced from native 5 cm occupancy points',
        'room_labels_available': bool(room_labels),
        'category_embeddings_available': bool(manifest['category_embeddings'])}
    return manifest


def clio_primitives(result, active_floor=None):
    """Read task-free semantic primitives from the final native DSG."""
    dsg = read_json(Path(result) / 'graph/backend/dsg.json')
    rows = []
    for node in dsg.get('nodes', []):
        if not isinstance(node.get('id'), int) or chr(node['id'] >> 56) != 's':
            continue
        attrs = node['attributes']
        bbox = attrs.get('bounding_box') or {}
        if bbox.get('type') != 'AABB':
            continue
        dimensions = np.asarray(bbox.get('dimensions'), dtype=float)
        center = np.asarray(bbox.get('world_P_center'), dtype=float)
        if (dimensions.shape != (3,) or center.shape != (3,)
                or not np.isfinite([dimensions, center]).all() or np.any(dimensions <= 0)):
            continue
        corners = bounds_corners(center, dimensions) @ ZUP_TO_HABITAT.T
        feature = np.asarray((attrs.get('semantic_feature') or {}).get('data'), dtype=float)
        row = {'object_id': str(node['id']), 'label': '', 'floor_index': active_floor,
            'aabb_min_m': corners.min(axis=0).tolist(),
            'aabb_max_m': corners.max(axis=0).tolist(),
            'evaluation_unit': 'semantic_primitive'}
        if feature.ndim == 1 and len(feature) and np.isfinite(feature).all():
            row['embedding'] = feature.tolist()
        rows.append(row)
    return rows


def clio_table_report(gt, recording, result):
    acquisition = read_json(Path(recording) / 'acquisition.json')
    active_floor = _active_floor(gt, acquisition)
    primitive_manifest = dict(gt, active_floor_index=active_floor,
        predicted_objects=clio_primitives(result, active_floor))
    _, objects = evaluators()
    filtered = [primitive_manifest]
    geometry, matches = objects.evaluate_geometry(filtered)
    labels = objects.evaluate_labels(filtered, matches)
    representation = [Path(result) / 'graph/backend/dsg.json',
                      Path(result) / 'graph/backend/mesh.ply']
    size = sum(path.stat().st_size for path in representation if path.is_file()) / 1e6
    reason = ('Task-free Clio has semantic primitives and places, but no native '
              'object-instance, room, or floor layer.')
    return {
        'table_ii_floor_regions': {'status': 'unsupported', 'reason': reason},
        'table_iii_rooms': {'status': 'unsupported', 'reason': reason},
        'table_iv_objects': {'status': 'unsupported',
            'reason': 'No task-conditioned Clio O nodes exist in semantic-mapping mode.'},
        'table_vi_room_objects': {'status': 'unsupported', 'reason': reason},
        'table_vii_representation': {'status': 'supported',
            'size_mb_total': round(size, 6),
            'files': [stamp(path) for path in representation if path.is_file()]},
        'construction_time_s': _timing(result),
        'semantic_primitive_proxy': {
            'status': 'descriptive_proxy', 'evaluation_unit': 'semantic_primitive',
            'must_not_be_compared_as_object_instance_performance': True,
            'geometry': geometry, 'labels': labels},
    }


def _without_retrieval(report):
    report.pop('table_v_retrieval', None)
    report['missing_inputs'] = [value for value in report.get('missing_inputs', [])
                                if not value.startswith('Table V:')]
    report['protocol_exclusions'] = ['Table V retrieval/navigation: excluded by experiment protocol']
    return report


def hov_support(manifest):
    room_labels = [row.get('predicted_label') for row in manifest.get('rooms', [])]
    gt_room_labels = [row.get('ground_truth_label') for row in manifest.get('rooms', [])]
    semantic = bool(manifest.get('category_embeddings'))
    return {
        'table_ii_floor_regions': {'status': 'supported'},
        'table_iii_rooms': {
            'status': ('supported' if room_labels and gt_room_labels and
                       all(room_labels) and all(gt_room_labels) else 'missing_inputs'),
            'reason': (None if room_labels and gt_room_labels and all(room_labels)
                       and all(gt_room_labels) else
                       'Native room/category evaluation assets or GT room labels are missing.')},
        'table_iv_objects': {'status': 'supported',
            'semantic_classification_status': ('supported' if semantic else 'missing_inputs'),
            'semantic_classification_reason': (None if semantic else
                'Native HOV object-category text features were not saved by this run.')},
        'table_vi_room_objects': {'status': 'supported'},
        'table_vii_representation': {'status': 'supported'},
        'table_v_retrieval': {'status': 'excluded_by_protocol'},
    }


def _iou_2d(a, b):
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return intersection / union if union else 0.0


def _aabb_iou(low_a, high_a, low_b, high_b):
    low_a, high_a = np.asarray(low_a), np.asarray(high_a)
    low_b, high_b = np.asarray(low_b), np.asarray(high_b)
    intersection = float(np.prod(np.maximum(0, np.minimum(high_a, high_b) -
                                            np.maximum(low_a, low_b))))
    va = float(np.prod(np.maximum(0, high_a - low_a)))
    vb = float(np.prod(np.maximum(0, high_b - low_b)))
    return intersection / (va + vb - intersection) if va + vb - intersection > 0 else 0.0


def _cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else -1.0


def _box_distance(point, low, high):
    point, low, high = map(lambda value: np.asarray(value, dtype=float),
                           (point, low, high))
    return float(np.linalg.norm(np.maximum(low - point, np.maximum(0, point - high))))


def _dynamic_by_frame(frames):
    return [{str(item.get('evaluation_object_id', item['object_id'])): item
             for item in row.get('dynamic_ground_truth', [])}
            for row in frames]


def _action_windows(actions, frame_count):
    result = []
    for index, action in enumerate(actions):
        object_id = _evaluation_id(action['result'])
        later = [row['first_post_action_frame'] for row in actions[index + 1:]
                 if _evaluation_id(row['result']) == object_id]
        result.append((action, min(later) if later else frame_count))
    return result


def _unpack_hov_masks(path):
    with np.load(path) as value:
        shape = tuple(int(v) for v in value['shape'])
        masks = np.unpackbits(value['packed'], axis=-1, count=shape[-1]).astype(bool)
        masks = masks.reshape(shape)
        features = np.asarray(value['mask_features'], dtype=float) if 'mask_features' in value else None
    if features is not None and (features.ndim != 2 or len(features) != len(masks)):
        raise ValueError(f'HOV mask feature alignment is invalid: {path}')
    return masks, features


def hov_temporal(recording, result, mask_iou=.25, feature_similarity=.80):
    """Measure HOV frame-local reaction and final-map persistence."""
    recording, result = Path(recording), Path(result)
    frames, actions = action_timeline(recording)
    dynamic = _dynamic_by_frame(frames)
    observations = read_rows(result / 'native_observations/observations.jsonl')
    sam = {row['frame_index']: row for row in observations if row['stage'] == 'sam_clip'}
    projected = {row['frame_index']: row for row in observations if row['stage'] == 'masks_3d'}
    if set(sam) != set(projected):
        raise ValueError('HOV SAM and projected-mask frame sets differ')
    frame_evidence = defaultdict(dict)
    reference_features = defaultdict(list)
    frame_masks = {}
    for frame_index in sorted(sam):
        archive = result / f'native_observations/{frame_index:06d}.masks.npz'
        masks, features = _unpack_hov_masks(archive)
        semantic = np.load(recording / f'semantic/{frame_index:06d}.npz')['semantic']
        masks3d = projected[frame_index]['masks']
        if len(masks) != len(masks3d):
            raise ValueError('HOV 2-D/3-D mask alignment differs')
        frame_masks[frame_index] = (features, masks3d)
        for object_id, truth in dynamic[frame_index].items():
            semantic_id = int(truth['semantic_id'])
            gt_mask = semantic == semantic_id
            if not np.any(gt_mask):
                continue
            scores = np.asarray([_iou_2d(mask, gt_mask) for mask in masks])
            selected = int(np.argmax(scores)) if len(scores) else None
            score = float(scores[selected]) if selected is not None else 0.0
            detected = score >= mask_iou
            item = {'visible_gt_pixels': int(np.count_nonzero(gt_mask)),
                'best_mask_iou_2d': score, 'detected': detected,
                'mask_index': selected if detected else None}
            if detected:
                box = masks3d[selected]
                if box.get('aabb_min') is not None:
                    item['aabb_iou_3d'] = _aabb_iou(
                        box['aabb_min'], box['aabb_max'], truth['aabb_min'], truth['aabb_max'])
                if features is not None:
                    reference_features[object_id].append(features[selected])
            frame_evidence[object_id][frame_index] = item
    final_objects = []
    for path in sorted((result / 'graph/objects').glob('*.json')):
        native = read_json(path)
        box = _bounds_from_points(ply_points(path.with_suffix('.ply')))
        feature = np.asarray(native.get('embedding'), dtype=float)
        if box is not None and feature.ndim == 1 and len(feature):
            final_objects.append({'object_id': str(native['object_id']),
                'aabb_min': box[0], 'aabb_max': box[1], 'feature': feature})
    rows = []
    for action, end in _action_windows(actions, len(frames)):
        object_id = _evaluation_id(action['result'])
        start = action['first_post_action_frame']
        opportunities = [index for index in sorted(sam) if start <= index < end
                         and object_id in frame_evidence and index in frame_evidence[object_id]]
        detections = [index for index in opportunities
                      if frame_evidence[object_id][index]['detected']]
        old_truth = dynamic[int(action['after_frame'])].get(object_id)
        new_truth = dynamic[start].get(object_id)
        target = old_truth if action['action'] == 'remove' else new_truth
        references = reference_features.get(object_id, [])
        reference = np.mean(references, axis=0) if references else None
        residual_frames = []
        final_candidates = []
        if reference is not None and target is not None:
            for frame_index in sorted(frame_masks):
                if not start <= frame_index < end:
                    continue
                features, masks3d = frame_masks[frame_index]
                if features is None:
                    continue
                candidates = []
                for mask_index, (feature, box) in enumerate(zip(features, masks3d)):
                    if box.get('aabb_min') is None:
                        continue
                    center = (np.asarray(box['aabb_min']) + np.asarray(box['aabb_max'])) / 2
                    similarity = _cosine(reference, feature)
                    distance = _box_distance(center, target['aabb_min'], target['aabb_max'])
                    if similarity >= feature_similarity and distance <= .5:
                        candidates.append({'mask_index': mask_index,
                            'feature_similarity': similarity,
                            'distance_to_scheduled_box_m': distance})
                if candidates:
                    residual_frames.append({'frame_index': frame_index,
                                            'candidates': candidates})
            for native in final_objects:
                center = (native['aabb_min'] + native['aabb_max']) / 2
                similarity = _cosine(reference, native['feature'])
                distance = _box_distance(center, target['aabb_min'], target['aabb_max'])
                if similarity >= feature_similarity and distance <= .5:
                    final_candidates.append({'native_object_id': native['object_id'],
                        'feature_similarity': similarity,
                        'distance_to_scheduled_box_m': distance})
        item = {'action': action['action'], 'object_id': object_id,
            'action_frame': start, 'action_time_s': action['time_s'],
            'visibility_opportunities': len(opportunities),
            'detections': len(detections),
            'visible_frame_recall_pct': (round(100 * len(detections) / len(opportunities), 4)
                                         if opportunities else None),
            'first_detection_frame': detections[0] if detections else None,
            'response_latency_s': (frames[detections[0]]['time_s'] - action['time_s']
                                   if detections else None),
            'post_action_feature_spatial_candidate_frames': residual_frames,
            'final_map_candidates_at_action_location': final_candidates,
            'identity_reference_available': reference is not None}
        if action['action'] == 'remove':
            item['removal_confirmed_in_final_map'] = (
                not final_candidates if reference is not None else None)
            item['status'] = ('evaluated_final_map_persistence' if reference is not None
                              else 'no_pre_removal_identity_reference')
        else:
            item['status'] = ('evaluated' if opportunities
                              else 'no_sampled_visible_opportunity')
            if not any(_evaluation_id(row['result']) == object_id
                       for row in actions[actions.index(action) + 1:]):
                item['presence_confirmed_in_final_map'] = (
                    bool(final_candidates) if reference is not None else None)
        rows.append(item)
    return {'status': 'supported', 'observation_unit': 'native frame-local SAM mask',
        'mask_iou_threshold': mask_iou, 'feature_similarity_threshold': feature_similarity,
        'actions': rows, 'sampled_frames': len(sam),
        'persistent_response_method': ('Final native HOV objects and post-action frame masks '
            'must match the observed object feature and lie within 0.5 m of its scheduled box.'),
        'limitations': [
            'HOV-SG constructs its persistent hierarchy after the whole batch; frame-local masks have no persistent native IDs.',
            'Removal is assessed in the final-map persistence block, not invented as a native delete event.'
        ], 'reference_features': {key: np.mean(value, axis=0).tolist()
                                  for key, value in reference_features.items() if value}}


def _clio_snapshot_nodes(result):
    history_path = Path(result) / 'native_graph_history.jsonl.gz'
    snapshots = read_rows(history_path)
    if not snapshots:
        raise ValueError('Clio has no native graph snapshots')
    semantic = any(any(node.get('type') == 'semantic_primitive'
                       for node in row.get('nodes', [])) for row in snapshots)
    if not semantic:
        return snapshots, False
    return snapshots, True


def _inside_box(node, truth, margin=.10):
    if 'corners' in node:
        center = np.asarray(node['corners'], dtype=float).mean(axis=0)
    elif node.get('centroid') is not None:
        center = np.asarray(node['centroid'], dtype=float)
    else:
        return False
    if center.shape != (3,) or not np.isfinite(center).all():
        return False
    low = np.asarray(truth['aabb_min'], dtype=float) - margin
    high = np.asarray(truth['aabb_max'], dtype=float) + margin
    return bool(np.all(center >= low) and np.all(center <= high))


def clio_temporal(recording, result):
    """Measure task-free Clio map evidence around every scheduled state change."""
    frames, actions = action_timeline(recording)
    dynamic = _dynamic_by_frame(frames)
    snapshots, has_semantic = _clio_snapshot_nodes(result)
    if not has_semantic:
        return {'status': 'unsupported_for_this_run',
            'reason': ('This run used observer v1, which omitted Clio semantic primitives. '
                       'Future runs save the required native s-layer history.'),
            'actions': []}
    stamps = np.asarray([row['time_s'] for row in snapshots])
    rows = []
    tracked_ids = defaultdict(set)
    for action, end in _action_windows(actions, len(frames)):
        object_id = _evaluation_id(action['result'])
        start = action['first_post_action_frame']
        before_index = max(0, int(np.searchsorted(stamps, action['time_s'], side='left')) - 1)
        before_ids = {node['id'] for node in snapshots[before_index]['nodes']
                      if node.get('type') == 'semantic_primitive'}
        previous_dynamic_ids = set(tracked_ids[object_id])
        evidence = []
        upper_time = frames[min(end, len(frames) - 1)]['time_s']
        selected_snapshots = np.flatnonzero((stamps >= action['time_s']) &
                                            (stamps <= upper_time))
        for snapshot_index in selected_snapshots:
            snapshot = snapshots[int(snapshot_index)]
            frame_index = min(range(start, min(end, len(frames))),
                key=lambda value: abs(frames[value]['time_s'] - snapshot['time_s']))
            truth = dynamic[frame_index].get(object_id)
            if truth is None:
                continue
            ids = [node['id'] for node in snapshot['nodes']
                   if node.get('type') == 'semantic_primitive'
                   and node['id'] not in before_ids and _inside_box(node, truth)]
            if ids:
                evidence.append((snapshot['time_s'], frame_index, ids))
        first = evidence[0] if evidence else None
        discovered = {value for _, _, ids in evidence for value in ids}
        if action['action'] != 'remove':
            tracked_ids[object_id].update(discovered)
        final_nodes = {node['id']: node for node in snapshots[-1]['nodes']
                       if node.get('type') == 'semantic_primitive'}
        retained = sorted(value for value in tracked_ids[object_id] if value in final_nodes)
        active = sorted(value for value in retained
                        if final_nodes[value].get('native_is_active'))
        item = {'action': action['action'], 'object_id': object_id,
            'action_frame': start, 'action_time_s': action['time_s'],
            'new_spatial_primitive_ids': sorted(discovered),
            'first_evidence_frame': first[1] if first else None,
            'response_latency_s': first[0] - action['time_s'] if first else None,
            'tracked_dynamic_evidence_ids': sorted(tracked_ids[object_id]),
            'retained_historical_ids_in_final_map': retained,
            'active_ids_in_final_map': active}
        if action['action'] == 'remove':
            item['pre_removal_tracked_ids'] = sorted(previous_dynamic_ids)
            item['removal_confirmed_active_layer'] = (
                not active if previous_dynamic_ids else None)
            item['historical_evidence_retained'] = (
                bool(retained) if previous_dynamic_ids else None)
            item['status'] = ('evaluated_final_activity_and_persistence'
                              if previous_dynamic_ids else 'no_pre_removal_spatial_evidence')
        else:
            item['status'] = 'evaluated' if discovered else 'no_new_spatial_primitive_evidence'
        rows.append(item)
    return {'status': 'supported', 'observation_unit': 'native Clio semantic primitive',
        'actions': rows, 'snapshots': len(snapshots),
        'method': ('New native s-layer IDs spatially inside the scheduled object box, '
                   'relative to the last pre-action graph snapshot. Removal checks '
                   'both native activity and retained historical nodes.'),
        'limitations': ['Semantic primitives are map fragments, not object instances.']}


def _dynamicgsg_snapshots(result):
    """Native per-frame object state, re-indexed into source recording frames."""
    result = Path(result)
    rows = read_rows(result / 'graph_stream.jsonl')
    if not rows:
        raise ValueError('DynamicGSG wrote no native graph stream for this run')
    mapping = read_rows(result / 'frame_mapping.jsonl')
    by_native = {int(row['exported_index']): row for row in mapping}
    snapshots = []
    for row in rows:
        native_index = int(row['frame'])
        entry = by_native.get(native_index)
        if entry is None:
            raise ValueError(f'Native frame {native_index} is outside the exported input')
        snapshots.append({'native_frame_index': native_index,
                          'frame_index': int(entry['source_index']),
                          'time_s': float(entry['time_s']),
                          'objects': row.get('objects', []),
                          'removed': [int(value) for value in row.get('removed', [])],
                          'num_gaussians': row.get('num_gaussians')})
    snapshots.sort(key=lambda row: (row['frame_index'], row['native_frame_index']))
    return snapshots


def dynamicgsg_temporal(recording, result):
    """Measure the native DynamicGSG object layer around every scheduled change."""
    frames, actions = action_timeline(recording)
    dynamic = _dynamic_by_frame(frames)
    snapshots = _dynamicgsg_snapshots(result)
    stamps = np.asarray([row['frame_index'] for row in snapshots])
    final = {int(node['idx']): node for node in snapshots[-1]['objects']}
    rows = []
    tracked_ids = defaultdict(set)
    for action, end in _action_windows(actions, len(frames)):
        object_id = _evaluation_id(action['result'])
        start = action['first_post_action_frame']
        before_index = max(0, int(np.searchsorted(stamps, start, side='left')) - 1)
        before_ids = {int(node['idx']) for node in snapshots[before_index]['objects']}
        previous_dynamic_ids = set(tracked_ids[object_id])
        evidence = []
        following = []
        removal_events = []
        for index in np.flatnonzero((stamps >= start) & (stamps < end)):
            snapshot = snapshots[int(index)]
            # A removed object leaves the dynamic ground truth, so the native delete
            # events must be read whether or not the object still has a scheduled box.
            gone = sorted(set(snapshot['removed']) & previous_dynamic_ids)
            if gone:
                removal_events.append({'frame_index': snapshot['frame_index'],
                                       'native_object_idx': gone})
            truth = dynamic[snapshot['frame_index']].get(object_id)
            if truth is None:
                continue
            inside = [int(node['idx']) for node in snapshot['objects']
                      if _inside_box(node, truth)]
            ids = [value for value in inside if value not in before_ids]
            if ids:
                evidence.append((snapshot['frame_index'], snapshot['time_s'], ids))
            # A move relocates an object that already exists, so an implementation that
            # carries the same native index to the new position is behaving correctly and
            # would show no NEW index. Record that separately instead of scoring it a miss.
            followed = sorted(set(inside) & previous_dynamic_ids)
            if followed:
                following.append((snapshot['frame_index'], snapshot['time_s'], followed))
        first = evidence[0] if evidence else None
        first_follow = following[0] if following else None
        discovered = {value for _, _, ids in evidence for value in ids}
        if action['action'] != 'remove':
            tracked_ids[object_id].update(discovered)
        retained = sorted(value for value in tracked_ids[object_id] if value in final)
        item = {'action': action['action'], 'object_id': object_id,
            'action_frame': start, 'action_time_s': action['time_s'],
            'new_native_object_idx': sorted(discovered),
            'first_evidence_frame': first[0] if first else None,
            'response_latency_s': first[1] - action['time_s'] if first else None,
            'tracked_dynamic_native_idx': sorted(tracked_ids[object_id]),
            'retained_native_idx_in_final_map': retained,
            'native_removal_events': removal_events,
            'tracked_idx_inside_new_box': sorted({v for _, _, ids in following for v in ids}),
            'first_tracked_follow_frame': first_follow[0] if first_follow else None,
            'follow_latency_s': (first_follow[1] - action['time_s']) if first_follow else None,
            'observed_snapshots_in_window': int(np.count_nonzero(
                (stamps >= start) & (stamps < end)))}
        if action['action'] == 'remove':
            item['pre_removal_tracked_native_idx'] = sorted(previous_dynamic_ids)
            item['removal_confirmed_in_final_map'] = (
                not retained if previous_dynamic_ids else None)
            item['native_delete_event_observed'] = (
                bool(removal_events) if previous_dynamic_ids else None)
            item['status'] = ('evaluated_native_deletion_and_persistence'
                              if previous_dynamic_ids else 'no_pre_removal_object_evidence')
        else:
            item['presence_confirmed_in_final_map'] = bool(retained) if discovered else None
            if discovered:
                item['status'] = 'evaluated'
            elif following:
                # The object kept its native identity and moved with the schedule.
                item['status'] = 'evaluated_tracked_identity_followed'
            else:
                item['status'] = 'no_native_object_evidence'
        rows.append(item)
    return {'status': 'supported',
        'observation_unit': 'native DynamicGSG Gaussian-backed object instance',
        'actions': rows, 'snapshots': len(snapshots),
        'method': ('New native object indices whose Gaussian centroid lies inside the '
                   'scheduled object box, relative to the last pre-action native snapshot. '
                   'Removal checks the native delete list and the final object set.'),
        'limitations': [
            'The native stream reports a Gaussian centroid per object, not a per-frame box, '
            'so containment is judged at the centroid.',
            'A move is credited either by a new native index inside the scheduled box or by a '
            'already-tracked index following the object there; the two are reported separately.',
            'Snapshots are appended only on frames where the native pipeline updates the map, '
            'so a window can contain fewer snapshots than sampled frames.']}


def dynamicgsg_objects(result, active_floor=None):
    """Read the final native object layer from the adapter's exported graph."""
    graph = read_json(Path(result) / 'dynamicgsg_graph.json')
    rows = []
    for node in graph.get('nodes', []):
        corners = np.asarray(node.get('corners'), dtype=float)
        if corners.shape != (8, 3) or not np.isfinite(corners).all():
            continue
        rows.append({'object_id': str(node['id']), 'label': str(node.get('label', '')),
            'floor_index': active_floor,
            'aabb_min_m': corners.min(axis=0).tolist(),
            'aabb_max_m': corners.max(axis=0).tolist(),
            'evaluation_unit': 'native_object_instance',
            'native_object_idx': node.get('native_object_idx'),
            'semantic_label_available': bool(node.get('semantic_label_available'))})
    if not rows:
        raise ValueError('DynamicGSG produced no usable 3-D object predictions')
    return rows, graph


def dynamicgsg_table_report(gt, recording, result):
    acquisition = read_json(Path(recording) / 'acquisition.json')
    active_floor = _active_floor(gt, acquisition)
    predicted, graph = dynamicgsg_objects(result, active_floor)
    manifest = dict(gt, active_floor_index=active_floor, predicted_objects=predicted)
    _, objects = evaluators()
    geometry, matches = objects.evaluate_geometry([manifest])
    semantic = bool(graph.get('semantic_output', {}).get('semantic_accuracy_eligible'))
    labels = objects.evaluate_labels([manifest], matches) if semantic else None
    representation = [Path(result) / name for name in
                      ('params_with_idx.npz', 'objects.pkl.gz', 'dynamicgsg_graph.json')]
    size = sum(path.stat().st_size for path in representation if path.is_file()) / 1e6
    reason = ('DynamicGSG builds a flat object layer over a Gaussian map; it has no '
              'native room or floor layer.')
    report = {
        'table_ii_floor_regions': {'status': 'unsupported', 'reason': reason},
        'table_iii_rooms': {'status': 'unsupported', 'reason': reason},
        'table_iv_objects': {'status': 'supported', 'evaluation_unit': 'native_object_instance',
            'geometry': geometry,
            'labels': labels if semantic else None,
            'semantic_classification_status': 'supported' if semantic else 'missing_inputs',
            'semantic_classification_reason': (None if semantic else
                'This run used the class-agnostic scope: the official DAM + qwen category '
                'postprocessor did not label every node.'),
            'evaluation_scope': graph.get('semantic_output', {}).get('evaluation_scope')},
        'table_vi_room_objects': {'status': 'unsupported', 'reason': reason},
        'table_vii_representation': {'status': 'supported',
            'size_mb_total': round(size, 6),
            'files': [stamp(path) for path in representation if path.is_file()]},
        'construction_time_s': _timing(result),
    }
    return report


DYNAMICGSG_FRAME_LINE = re.compile(r'^([0-9.]+)\tframe (\d+) num of objects:')


def dynamicgsg_frame_times(result):
    """Per-frame native wall time, read from the adapter's stamped native log."""
    stamps = []
    for line in (Path(result) / 'native.log').read_text(errors='replace').splitlines():
        match = DYNAMICGSG_FRAME_LINE.match(line)
        if match:
            stamps.append((int(match.group(2)), float(match.group(1))))
    stamps.sort()
    return [second - first for (_, first), (_, second) in zip(stamps, stamps[1:])]


def latency_report(baseline, result):
    result = Path(result)
    if baseline == 'dynamicgsg':
        rows = dynamicgsg_frame_times(result)
        scope = ('Native DynamicGSG wall time between consecutive frame-complete prints of '
                 'the native process: detection, association, merge and Gaussian '
                 'optimisation for one sampled frame. The first frame has no predecessor '
                 'and is not counted.')
    elif baseline == 'hovsg':
        rows = [row['stage_wall_time_s'] for row in
                read_rows(result / 'native_observations/observations.jsonl')
                if row['stage'] == 'sam_clip']
        scope = 'Native HOV-SG SAM/CLIP function call-to-return per sampled frame'
    else:
        observer = read_json(result / 'native_observer.json')
        values = [row['paired_transport_latency_s'] for row in
                  read_rows(result / 'native_receipts.jsonl')
                  if row['topic'] == '/dominic/forward/semantic/image_raw'
                  and row.get('paired_transport_latency_s') is not None]
        rows = values if observer.get('input_timing_origin') == 'before_rgb_publish' else []
        scope = observer.get('latency_scope')
    values = np.asarray(rows, dtype=float)
    return {'samples': len(values),
        'mean_ms': round(float(values.mean() * 1000), 6) if len(values) else None,
        'p50_ms': round(float(np.percentile(values, 50) * 1000), 6) if len(values) else None,
        'p95_ms': round(float(np.percentile(values, 95) * 1000), 6) if len(values) else None,
        'scope': scope}


def evaluate_baseline(baseline, recording, result, ground_truth, output):
    recording, result, output = map(Path, (recording, result, output))
    gt = read_json(ground_truth)
    require_ground_truth(gt)
    frames, actions = action_timeline(recording)
    final_gt = final_ground_truth(gt, frames, actions)
    metrics, objects = evaluators()
    if baseline == 'hovsg':
        manifest = hov_manifest(final_gt, recording, result)
        table_report = _without_retrieval(metrics.evaluate([manifest], output))
        filtered = metrics.filtered_scene(manifest)
        geometry, matches = objects.evaluate_geometry([filtered])
        labels = objects.evaluate_labels([filtered], matches)
        table_report['table_iv_objects_v2'] = {**geometry, **labels}
        temporal = hov_temporal(recording, result)
        support = hov_support(manifest)
    elif baseline == 'dynamicgsg':
        manifest = None
        table_report = dynamicgsg_table_report(final_gt, recording, result)
        table_report['protocol_exclusions'] = [
            'Table V retrieval/navigation: excluded by experiment protocol']
        temporal = dynamicgsg_temporal(recording, result)
        support = {key: {'status': value.get('status', 'supported'),
                         'reason': value.get('reason')}
                   for key, value in table_report.items()
                   if key.startswith('table_')}
        support['table_v_retrieval'] = {'status': 'excluded_by_protocol'}
    elif baseline == 'clio':
        manifest = None
        table_report = clio_table_report(final_gt, recording, result)
        table_report['protocol_exclusions'] = [
            'Table V retrieval/navigation: excluded by experiment protocol']
        temporal = clio_temporal(recording, result)
        support = {key: {'status': value.get('status', 'supported'),
                         'reason': value.get('reason')}
                   for key, value in table_report.items()
                   if key.startswith('table_')}
        support['table_v_retrieval'] = {'status': 'excluded_by_protocol'}
    else:
        raise ValueError(f'Unsupported baseline: {baseline}')
    report = {'schema': SCHEMA, 'baseline': baseline,
        'scene': read_json(recording / 'acquisition.json')['scene'],
        'tables': table_report, 'table_support': support,
        'temporal_object_actions': temporal,
        'latency': latency_report(baseline, result),
        'coverage': read_json(recording / 'acquisition.json').get('tour', {}),
        'input_sampling': (read_json(result / 'input_sampling.json')
                           if (result / 'input_sampling.json').is_file() else None),
        'inputs': {'ground_truth': stamp(ground_truth),
                   'acquisition': stamp(recording / 'acquisition.json'),
                   'frames': stamp(recording / 'frames.jsonl'),
                   'object_actions': stamp(recording / 'object_actions.jsonl'),
                   'baseline_result': stamp(result / 'baseline_result.json')},
        'dynamic_final_objects': final_gt['dynamic_final_objects'],
        'native_algorithm_modified': False,
        'interpretation': [
            'Final tables use the authoritative GRAPH-API evaluators.',
            'Temporal response is reconstructed from timestamped native outputs and scheduled dynamic GT.',
            'No unsupported metric is replaced with zero.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False,
                                 default=json_default) + '\n')
    if manifest is not None:
        persisted = dict(manifest)
        if persisted.get('category_embeddings'):
            persisted['category_embeddings'] = []
            persisted['category_embeddings_external'] = stamp(
                result / 'evaluation_assets/object_category_features.npy')
        (output.parent / f'{baseline}.eval-manifest.json').write_text(
            json.dumps(persisted, indent=2, allow_nan=False,
                       default=json_default) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True,
                        choices=('clio', 'hovsg', 'dynamicgsg'))
    parser.add_argument('--recording', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--ground-truth', type=Path,
                        help='Defaults to RECORDING/static-gt.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    ground_truth = args.ground_truth or args.recording / 'static-gt.json'
    report = evaluate_baseline(args.baseline, args.recording, args.result,
                               ground_truth, args.output)
    print(json.dumps({'output': str(args.output), 'baseline': args.baseline,
                      'temporal_status': report['temporal_object_actions']['status']}))


if __name__ == '__main__':
    main()
