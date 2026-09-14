"""Export display geometry from native results; never run or modify a baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .replay_model import ZUP_TO_HABITAT, file_stamp, ply_points, read_json


def clio_mesh(attributes, version, face_budget=1500):
    """Match clio_eval.utils.dsg_object_to_o3d's versioned local-to-world offset."""
    mesh = attributes['mesh']
    points = np.asarray(mesh['points'], dtype=float).reshape(-1, 3)
    faces = np.asarray(mesh['faces'], dtype=int).reshape(-1, 3)
    offset = np.asarray(attributes['position'], dtype=float)
    if version == (1, 0, 0):
        offset = offset - np.asarray(attributes['bounding_box']['dimensions']) / 2
    points = (points + offset) @ ZUP_TO_HABITAT.T
    invalid_faces = int(np.count_nonzero((faces < 0).any(1) | (faces >= len(points)).any(1)))
    # Preserve broken native topology as evidence. Show its actual vertices rather
    # than invent replacement faces or silently discard the affected object.
    if invalid_faces or not len(faces):
        used = np.arange(0, len(points), max(1, int(np.ceil(len(points) / 4500))))
        colors = mesh.get('colors', [])
        return {'primitive': 'points', 'positions': points[used].round(5).ravel().tolist(),
                'colors': [colors[i][c] / 255 for i in used for c in ('r', 'g', 'b')] if colors else [],
                'native_vertices': len(points), 'native_faces': len(faces),
                'rendered_vertices': len(used), 'rendered_faces': 0,
                'invalid_native_faces': invalid_faces,
                'display_note': 'Saved vertices only: native faces reference missing vertices'
                                if invalid_faces else 'Saved vertices only: native mesh has no faces'}
    faces = faces[::max(1, int(np.ceil(len(faces) / face_budget)))]
    used, indices = np.unique(faces.ravel(), return_inverse=True)
    colors = mesh.get('colors', [])
    return {'primitive': 'triangles', 'positions': points[used].round(5).ravel().tolist(),
            'indices': indices.tolist(),
            'colors': [colors[i][c] / 255 for i in used for c in ('r', 'g', 'b')] if colors else [],
            'native_vertices': len(points), 'native_faces': len(mesh['faces']),
            'rendered_vertices': len(used), 'rendered_faces': len(faces)}


def export_scene(model_path, result, output):
    model_path, result, output = map(Path, (model_path, result, output))
    if output.exists():
        raise FileExistsError(output)
    model = read_json(model_path)
    baseline = model['baseline']
    graph = model['graph']
    objects, sources = [], []
    if baseline == 'clio':
        source = result / 'graph/backend/dsg.json'
        stamp = file_stamp(source)
        if stamp['sha256'] != graph['source']['sha256']:
            raise ValueError('Native graph differs from replay provenance')
        raw = read_json(source)
        v = raw['SPARK_DSG_header']['version']
        version = tuple(v[k] for k in ('major', 'minor', 'patch'))
        for node in raw['nodes']:
            if chr(node['id'] >> 56) == 'O':
                objects.append({'id': str(node['id']),
                                **clio_mesh(node['attributes'], version)})
        sources.append(stamp)
    elif baseline == 'hovsg':
        for path in sorted((result / 'graph/objects').glob('*.json')):
            attr = read_json(path)
            cloud_path = path.with_suffix('.ply')
            points = ply_points(cloud_path)
            sampled = points[::max(1, int(np.ceil(len(points) / 2200)))]
            objects.append({'id': 'object:' + attr['object_id'], 'primitive': 'points',
                            'positions': sampled.round(5).ravel().tolist(), 'colors': [],
                            'native_vertices': len(points), 'rendered_vertices': len(sampled)})
            sources.extend([file_stamp(path), file_stamp(cloud_path)])
    elif baseline == 'dynamicgsg':
        graph_path = result / 'dynamicgsg_graph.json'
        native_graph = read_json(graph_path)
        if file_stamp(graph_path)['sha256'] != graph['source']['sha256']:
            raise ValueError('DynamicGSG graph differs from replay provenance')
        params_path = result / 'params_with_idx.npz'
        with np.load(params_path, allow_pickle=False) as params:
            points = np.asarray(params['means3D'], dtype=float)
            object_idx = np.asarray(params['object_idx']).reshape(-1)
        transform = np.asarray(native_graph['pipeline_to_habitat'], dtype=float)
        for node in graph['nodes']:
            selected = points[object_idx == node['native_object_idx']]
            selected = np.column_stack((selected, np.ones(len(selected)))) @ transform.T
            selected = selected[:, :3]
            sampled = selected[::max(1, int(np.ceil(len(selected) / 2200)))]
            objects.append({'id': node['id'], 'primitive': 'points',
                            'positions': sampled.round(5).ravel().tolist(), 'colors': [],
                            'native_vertices': len(selected), 'rendered_vertices': len(sampled)})
        sources.extend([file_stamp(graph_path), file_stamp(params_path)])
    else:
        raise ValueError(f'No native geometry adapter for {baseline}')
    expected = {n['id'] for n in graph['nodes'] if n['type'] == 'object'}
    if {o['id'] for o in objects} != expected:
        raise ValueError('Native geometry IDs differ from replay graph')
    for obj in objects:
        if not np.isfinite(obj['positions']).all():
            raise ValueError('Nonfinite native geometry')
    scene = {'schema': 'graphapi.baseline_scene.v1', 'baseline': baseline,
             'scope': 'final_snapshot', 'coordinate_frame': 'Habitat Y-up',
             'model_sha256': file_stamp(model_path)['sha256'],
             'graph': graph, 'objects': objects, 'sources': sources,
             'display_policy': 'Native geometry; deterministic vertex/face subsampling for display only. '
                               'Object colors identify saved IDs, not semantic classes. '
                               'Layer separation is a visual offset, not physical height.',
             'native_vertices': sum(o['native_vertices'] for o in objects),
             'rendered_vertices': sum(o['rendered_vertices'] for o in objects),
             'invalid_topology_objects': sum(bool(o.get('invalid_native_faces')) for o in objects)}
    output.write_text(json.dumps(scene, separators=(',', ':'), allow_nan=False))
    return {k: scene[k] for k in ('baseline', 'native_vertices', 'rendered_vertices')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('result')
    p.add_argument('output', help='Use <name>.replay.scene.json beside <name>.replay.json')
    a = p.parse_args()
    print(json.dumps(export_scene(a.model, a.result, a.output)))


if __name__ == '__main__':
    main()
