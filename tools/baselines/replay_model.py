"""Read-only adapters into a baseline-neutral replay format. No inference is run here."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

SCHEMA = 'graphapi.baseline_replay.v1'
# All normalized spatial values use Habitat world coordinates (Y up).
ZUP_TO_HABITAT = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_stamp(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def ply_points(path):
    """Read native Open3D vertex PLYs without loading a baseline environment."""
    types = {'double': 'f8', 'float': 'f4', 'uchar': 'u1', 'uint8': 'u1', 'int': 'i4'}
    with Path(path).open('rb') as f:
        if f.readline().strip() != b'ply':
            raise ValueError(f'Not a PLY: {path}')
        props, count, vertex, fmt = [], None, False, None
        while True:
            line = f.readline().decode('ascii').strip()
            if not line:
                raise ValueError('Truncated PLY header')
            parts = line.split()
            if parts[0] == 'format':
                fmt = parts[1]
            elif parts[0] == 'element':
                vertex = parts[1] == 'vertex'
                if vertex:
                    count = int(parts[2])
            elif parts[0] == 'property' and vertex:
                if parts[1] not in types:
                    raise ValueError(f'Unsupported PLY property: {line}')
                props.append((parts[2], types[parts[1]]))
            elif line == 'end_header':
                break
        if count is None:
            raise ValueError('PLY has no vertices')
        if fmt == 'binary_little_endian':
            dtype = np.dtype([(name, '<' + kind) for name, kind in props])
            data = np.fromfile(f, dtype=dtype, count=count)
            if len(data) != count:
                raise ValueError('Truncated PLY vertices')
            return np.column_stack([data[a] for a in ('x', 'y', 'z')])
        if fmt == 'ascii':
            data = np.array([[float(v) for v in f.readline().split()] for _ in range(count)])
            names = [p[0] for p in props]
            return data[:, [names.index(a) for a in ('x', 'y', 'z')]]
        raise ValueError(f'Unsupported PLY format: {fmt}')


def bounds_corners(center, dimensions):
    return np.array(center) + np.array(list(itertools.product((-0.5, 0.5), repeat=3))) * dimensions


def clio_graph(root):
    path = root / 'graph/backend/dsg.json'
    raw = read_json(path)
    nodes = []
    for n in raw['nodes']:
        prefix = chr(n['id'] >> 56)
        if prefix not in ('O', 's', 'p', 'l'):
            continue  # Robot poses are a separate native layer.
        a = n['attributes']
        kind = {'O': 'object', 's': 'semantic_primitive',
                'p': 'place', 'l': 'room'}[prefix]
        node = {'id': str(n['id']), 'label': a.get('name') or f'{prefix}{n["id"] & ((1 << 56) - 1)}',
                'type': kind, 'position': (ZUP_TO_HABITAT @ a['position']).tolist()}
        b = a.get('bounding_box')
        if b and b['type'] == 'AABB':
            corners = bounds_corners(b['world_P_center'], b['dimensions']) @ ZUP_TO_HABITAT.T
            node['corners'] = corners.tolist()
        # Observation timestamps are retained as provenance, never used to invent graph history.
        node['native_first_observed_ns'] = a.get('first_observed_ns')
        if prefix == 's':
            node['native_last_observed_ns'] = a.get('last_observed_ns')
            node['native_is_active'] = bool(a.get('is_active', False))
            node['semantic_feature_dimension'] = len(
                (a.get('semantic_feature') or {}).get('data') or [])
        nodes.append(node)
    ids = {n['id'] for n in nodes}
    edges = [{'source': str(e['source']), 'target': str(e['target']), 'label': 'native edge'}
             for e in raw['edges'] if str(e['source']) in ids and str(e['target']) in ids]
    return {'nodes': nodes, 'edges': edges, 'scope': 'final_snapshot',
            'source': file_stamp(path),
            'excluded_layers': ['robot poses', 'dense semantic meshes',
                                'semantic feature vector values']}


def hovsg_graph(root):
    nodes, edges = [], []
    for kind in ('floor', 'room', 'object'):
        for path in sorted((root / 'graph' / (kind + 's')).glob('*.json')):
            a = read_json(path)
            node = {'id': kind + ':' + a[kind + '_id'], 'label': a.get('name', a[kind + '_id']), 'type': kind}
            points = ply_points(path.with_suffix('.ply'))
            if len(points):
                lo, hi = points.min(0), points.max(0)
                node.update(position=((lo + hi) / 2).tolist(), corners=bounds_corners((lo + hi) / 2, hi - lo).tolist())
            if kind != 'floor':
                parent = 'floor' if kind == 'room' else 'room'
                edges.append({'source': parent + ':' + a[parent + '_id'], 'target': node['id'], 'label': 'contains'})
            nodes.append(node)
    ids = {n['id'] for n in nodes}
    if any(e['source'] not in ids or e['target'] not in ids for e in edges):
        raise ValueError('Native HOV hierarchy has dangling references')
    return {'nodes': nodes, 'edges': edges, 'scope': 'final_snapshot',
            'source': file_stamp(root / 'baseline_result.json'), 'excluded_layers': ['navigation graph']}


def dashboard_graph(root):
    """Consume a recorded /graph_data response, e.g. from GRAPH-API or FOUND."""
    path = root / 'graph_data.json'
    raw = read_json(path)
    nodes, edges = [], []
    for item in raw['elements']['nodes']:
        a = item['data']
        node = {'id': str(a['id']), 'label': a.get('label', str(a['id'])), 'type': a.get('type', 'object')}
        if a.get('position') is not None:
            node['position'] = (ZUP_TO_HABITAT @ a['position']).tolist()
        b = a.get('bbox')
        if b and all(k in b for k in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')):
            lo, hi = np.array([b[k + '_min'] for k in 'xyz']), np.array([b[k + '_max'] for k in 'xyz'])
            node['corners'] = (bounds_corners((lo + hi) / 2, hi - lo) @ ZUP_TO_HABITAT.T).tolist()
        nodes.append(node)
    for item in raw['elements'].get('edges', []):
        a = item['data']
        edges.append({'source': str(a['source']), 'target': str(a['target']), 'label': a.get('label', '')})
    return {'nodes': nodes, 'edges': edges, 'scope': 'final_snapshot', 'source': file_stamp(path)}


def dynamicgsg_graph(root):
    """Read the external projection of native Gaussian/object outputs."""
    path = root / 'dynamicgsg_graph.json'
    raw = read_json(path)
    if raw.get('schema') != 'graphapi.dynamicgsg_graph.v1' or raw.get('coordinate_frame') != 'Habitat Y-up':
        raise ValueError('Unknown DynamicGSG graph projection')
    nodes = raw.get('nodes', [])
    if any(node.get('type') != 'object' for node in nodes):
        raise ValueError('DynamicGSG projection contains an unknown node type')
    return {'nodes': nodes, 'edges': raw.get('edges', []), 'scope': 'final_snapshot',
            'source': file_stamp(path), 'native_sources': raw.get('sources', []),
            'excluded_layers': ['camera trajectory', 'background Gaussians']}


READERS = {'clio': clio_graph, 'hovsg': hovsg_graph, 'dynamicgsg': dynamicgsg_graph,
           'dashboard': dashboard_graph}


def clio_metrics(result):
    """Read the native active-window timer, explicitly excluding inference latency."""
    path = result / 'graph/active_window/all_timing_raw.csv'
    if not path.is_file():
        return []
    with path.open() as f:
        samples = list(csv.DictReader(f))
    rows, total, previous = [], 0.0, -1.0
    for i, sample in enumerate(samples):
        stamp = int(sample['timestamp(ns)']) / 1e9
        elapsed = float(sample['elapsed(s)'])
        if not np.isfinite(stamp) or stamp < previous or not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid native active-window timing')
        total += elapsed
        previous = stamp
        rows.append({'time_s': stamp, 'metrics': {'stage_latency_ms': {
            'value': round(total * 1000 / (i + 1), 3), 'unit': 'ms',
            'label': 'Active-window mean time', 'samples': i + 1,
            'source': file_stamp(path),
            'scope': 'Native active_window/all elapsed(s), cumulative by input timestamp; excludes neural inference and end-to-end latency'}}})
    return rows


def normalize(baseline, recording, result, schedule, output, metrics=None, snapshots=None, map_ply=None):
    recording, result, output = map(lambda p: Path(p).resolve(), (recording, result, output))
    if output.exists():
        raise FileExistsError(output)
    meta = read_json(recording / 'acquisition.json')
    if not meta.get('complete'):
        raise ValueError('Recording is incomplete')
    if baseline != 'dashboard' and not read_json(result / 'baseline_result.json').get('complete'):
        raise ValueError('Baseline did not complete')
    frames = read_rows(recording / 'frames.jsonl')
    if not frames or len(frames) != meta['frames']:
        raise ValueError('Frame count disagrees with acquisition')
    if any(b['time_s'] <= a['time_s'] for a, b in zip(frames, frames[1:])):
        raise ValueError('Frame timestamps must strictly increase')
    for i, f in enumerate(frames):
        if f['index'] != i or Path(f['stem']).name != f['stem']:
            raise ValueError('Invalid frame identity')
        f['image'] = str(recording / 'rgb' / (f['stem'] + '.png'))
        if not Path(f['image']).is_file():
            raise FileNotFoundError(f['image'])
        f['pose'] = np.loadtxt(recording / 'pose' / (f['stem'] + '.txt')).reshape(4, 4).tolist()
    actions = read_rows(recording / 'object_actions.jsonl')
    for a in actions:
        if not a['result']['success']:
            raise ValueError('Failed scheduled action in completed recording')
        # Action happened AFTER this frame; first later observation is the safe boundary.
        index = a['after_frame'] + 1
        if not 0 <= index < len(frames):
            raise ValueError('Action has no subsequent observation')
        a['frame_index'] = index
        a['time_s'] = frames[index]['time_s']
        a['timing_scope'] = 'first recorded frame after acknowledged action'
    raw_schedule = read_json(schedule)
    if hashlib.sha256(Path(schedule).read_bytes()).hexdigest() != meta['schedule_sha256']:
        raise ValueError('Schedule is not the one used by the recording')
    levels = raw_schedule.get('schedule', [raw_schedule])
    if len(levels) == 1:
        level = levels[0]
    else:
        event = next((e for f in frames if f['reason'] == 'tour' for e in (f.get('event') or [])), None)
        if event is None:
            raise ValueError('Multi-storey schedule needs a recorded scan event to identify the toured floor')
        matches = [level for level in levels if abs(level['height'] - event['xyz'][1]) < 0.15]
        if len(matches) != 1:
            raise ValueError('Recorded tour floor does not select exactly one schedule level')
        level = matches[0]
    graph = READERS[baseline](result)
    if not graph['nodes']:
        raise ValueError('Empty graph')
    ids = {n['id'] for n in graph['nodes']}
    if len(ids) != len(graph['nodes']) or any(e['source'] not in ids or e['target'] not in ids for e in graph['edges']):
        raise ValueError('Graph IDs or references are invalid')
    timed_metrics = read_rows(metrics) if metrics else (clio_metrics(result) if baseline == 'clio' else [])
    from .replay_metrics import validate_metrics
    validate_metrics(timed_metrics)
    history = read_rows(snapshots) if snapshots else []
    if any(b['time_s'] <= a['time_s'] for a, b in zip(history, history[1:])):
        raise ValueError('Graph snapshots must be time ordered')
    for snapshot in history:
        if snapshot.get('scope') != 'recorded_snapshot' or not snapshot.get('source'):
            raise ValueError('Graph history must identify actual recorded snapshots')
    background = []
    if map_ply:
        points = ply_points(map_ply)
        # This is a context layer. It never feeds inference or supplies GT matches.
        height = level.get('height', 0)
        points = points[(points[:, 1] > height + 0.15) & (points[:, 1] < height + 1.7)]
        background = points[::max(1, len(points) // 12000), :].tolist()
    model = {'schema': SCHEMA, 'baseline': baseline, 'scene': Path(meta['scene']).parent.name,
             'frames': frames, 'actions': actions, 'trajectory': level['trajectory'], 'graph': graph,
             'graph_history': history, 'metrics_history': timed_metrics, 'map_points': background,
             'map_source': file_stamp(map_ply) if map_ply else None,
             'provenance': {'acquisition': file_stamp(recording / 'acquisition.json'),
                            'frames': file_stamp(recording / 'frames.jsonl'),
                            'actions': file_stamp(recording / 'object_actions.jsonl'),
                            'schedule': file_stamp(schedule), 'adapter': file_stamp(__file__)},
             'limitations': ['Final graph geometry is not a history of detections or object creation.',
                            'Action timing uses the first recorded observation after execution.',
                            'Precision/recall/accuracy need ground-truth matching and a stated denominator.',
                            'Projected final boxes do not perform depth-occlusion testing.']}
    timing = result / 'execution_timing.json'
    if timing.is_file():
        measured = read_json(timing)
        if measured.get('schema') != 'graphapi.baseline_timing.v1' or measured.get('clock') != 'time.perf_counter':
            raise ValueError('Unknown execution timing provenance')
        model['final_metrics'] = {'run_wall_time_s': {
            'value': measured['phases_s']['run_wall_time'], 'unit': 's',
            'label': 'Run wall time (final)', 'source': file_stamp(timing), 'scope': measured['scope']}}
        validate_metrics([{'time_s': 0, 'metrics': model['final_metrics']}])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(model, f, separators=(',', ':'), allow_nan=False)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', choices=READERS, required=True)
    for name in ('recording', 'result', 'schedule', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--metrics', help='Optional measured metrics JSONL, acquisition time_s and explicit source/scope')
    p.add_argument('--snapshots', help='Optional recorded normalized graph snapshots JSONL; no interpolation')
    p.add_argument('--map-ply', help='Optional Habitat-world context point cloud, visualization only')
    args = p.parse_args()
    model = normalize(**vars(args))
    print(json.dumps({'baseline': model['baseline'], 'frames': len(model['frames']), 'nodes': len(model['graph']['nodes'])}))


if __name__ == '__main__':
    main()
