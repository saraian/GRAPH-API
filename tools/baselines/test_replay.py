"""Cross-reader checks for truthful synchronized replay and geometric projection."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tools.baselines.replay_model import SCHEMA, bounds_corners, clio_metrics, dashboard_graph, normalize, read_json
from tools.baselines.replay_render import Replay, project_segments
from tools.baselines.replay_server import create_app

ARTIFACTS = Path(__file__).resolve().parents[2] / 'artifacts/baselines'


class ReplayTests(unittest.TestCase):
    def test_dashboard_reader_preserves_graph_and_converts_axes(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            payload = {'elements': {'nodes': [
                {'data': {'id': 'one', 'label': 'chair', 'type': 'object', 'position': [1, 2, 3],
                          'bbox': {'x_min': 0, 'x_max': 2, 'y_min': 1, 'y_max': 3, 'z_min': 2, 'z_max': 4}}},
                {'data': {'id': 'room', 'label': 'room', 'type': 'room', 'position': [4, 5, 6]}}
            ], 'edges': [{'data': {'source': 'room', 'target': 'one', 'label': 'contains'}}]}}
            (Path(tmp)/'graph_data.json').write_text(json.dumps(payload))
            graph = dashboard_graph(Path(tmp))
            self.assertEqual(graph['nodes'][0]['position'], [1, 3, -2])
            self.assertEqual(graph['nodes'][0]['label'], 'chair')
            self.assertEqual(len(graph['nodes'][0]['corners']), 8)
            self.assertEqual(graph['edges'], [{'source': 'room', 'target': 'one', 'label': 'contains'}])
            self.assertEqual(graph['scope'], 'final_snapshot')

    def test_multistorey_uses_recorded_floor_and_refuses_ambiguity(self):
        import hashlib
        import json

        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in ('rgb', 'pose'):
                (root/directory).mkdir()
            rows = []
            for i in range(2):
                Image.new('RGB', (2, 2)).save(root/'rgb'/f'{i}.png')
                np.savetxt(root/'pose'/f'{i}.txt', np.eye(4))
                rows.append({'index': i, 'stem': str(i), 'time_s': float(i+1), 'reason': 'tour',
                             'event': [{'xyz': [0, 3, 0]}] if i else None})
            schedule = root/'schedule.json'
            schedule.write_text(json.dumps({'schedule': [
                {'height': 0, 'trajectory': [{'xyz': [0, 0, 0]}]},
                {'height': 3, 'trajectory': [{'xyz': [0, 3, 0]}]}]}))
            (root/'acquisition.json').write_text(json.dumps({'complete': True, 'frames': 2,
                'scene': '/scene/house.glb', 'schedule_sha256': hashlib.sha256(schedule.read_bytes()).hexdigest()}))
            (root/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            (root/'object_actions.jsonl').write_text('')
            (root/'graph_data.json').write_text(json.dumps({'elements': {'nodes': [
                {'data': {'id': 'obj', 'position': [1, 2, 3]}}], 'edges': []}}))
            model = normalize('dashboard', root, root, schedule, root/'good.json')
            self.assertEqual(model['trajectory'][0]['xyz'][1], 3)
            rows[1]['event'] = None
            (root/'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, 'recorded scan event'):
                normalize('dashboard', root, root, schedule, root/'bad.json')
            self.assertFalse((root/'bad.json').exists())

    def test_clio_mesh_native_offsets_and_broken_topology(self):
        from tools.baselines.replay_scene import clio_mesh
        attributes = {'position': [10, 20, 30], 'bounding_box': {'dimensions': [2, 4, 6]},
                      'mesh': {'points': [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                               'faces': [[0, 1, 2]], 'colors': []}}
        current = clio_mesh(attributes, (1, 0, 6))
        self.assertEqual(current['positions'][:3], [10, 30, -20])
        self.assertEqual(current['indices'], [0, 1, 2])
        old = clio_mesh(attributes, (1, 0, 0))
        self.assertEqual(old['positions'][:3], [9, 27, -18])
        attributes['mesh']['faces'] = [[0, 1, 9]]
        broken = clio_mesh(attributes, (1, 0, 6))
        self.assertEqual(broken['primitive'], 'points')
        self.assertEqual(broken['invalid_native_faces'], 1)
        self.assertEqual(broken['rendered_vertices'], 3)
        self.assertNotIn('indices', broken)

    @unittest.skipUnless((ARTIFACTS/'clio.replay.scene.json').is_file(), 'real 3D exports unavailable')
    def test_mounted_native_scene_and_provenance_rejection(self):
        import json
        import shutil

        from fastapi import FastAPI
        parent = FastAPI()
        parent.mount('/baseline-replay', create_app([ARTIFACTS/'clio.replay.json']))
        with TestClient(parent) as client:
            scene = client.get('/baseline-replay/api/scene/clio').json()
            self.assertEqual(len(scene['objects']), 106)
            self.assertEqual(scene['invalid_topology_objects'], 35)
            self.assertEqual(scene['scope'], 'final_snapshot')
            self.assertEqual(client.get('/baseline-replay/scene').status_code, 200)
            self.assertEqual(client.get('/baseline-replay/vendor/three.module.js').status_code, 200)
            self.assertEqual(client.get('/baseline-replay/api/feed/clio.jpg').headers['content-type'], 'image/jpeg')
            self.assertEqual(client.get('/baseline-replay/api/scene/unknown').status_code, 404)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/'clio.replay.json'
            shutil.copyfile(ARTIFACTS/'clio.replay.json', model)
            model.with_suffix('.scene.json').write_text(json.dumps({'model_sha256': 'wrong'}))
            with TestClient(create_app([model])) as client:
                self.assertEqual(client.get('/api/scene/clio').status_code, 409)

    def test_projection_asymmetric_axes_and_behind_camera(self):
        pose = np.eye(4)
        corners = bounds_corners([1, 1, -5], [.2, .2, .2])
        lines = project_segments(corners, pose, 640, 480, 90)
        self.assertEqual(len(lines), 12)
        for line in lines:
            for x, y in line:
                self.assertGreater(x, 320)
                self.assertLess(y, 240)
        self.assertEqual(project_segments(bounds_corners([0, 0, 5], [1, 1, 1]), pose, 640, 480, 90), [])

    def test_native_timing_units_and_negative_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'graph/active_window/all_timing_raw.csv'
            path.parent.mkdir(parents=True)
            path.write_text('timestamp(ns),elapsed(s)\n1000000000,0.01\n2000000000,0.03\n')
            rows = clio_metrics(Path(tmp))
            self.assertEqual(rows[-1]['time_s'], 2)
            self.assertEqual(rows[-1]['metrics']['stage_latency_ms']['value'], 20)
            self.assertNotIn('avg_latency_ms', rows[-1]['metrics'])
            path.write_text('timestamp(ns),elapsed(s)\n1000000000,-0.01\n')
            with self.assertRaises(ValueError):
                clio_metrics(Path(tmp))

    @unittest.skipUnless((ARTIFACTS/'hovsg.replay.json').is_file(), 'real Gin artifacts unavailable')
    def test_actual_action_boundaries_both_directions(self):
        replay = Replay(ARTIFACTS/'hovsg.replay.json')
        self.assertEqual(replay.model['schema'], SCHEMA)
        for action in replay.model['actions']:
            after = replay.state(replay.times[action['frame_index']])
            before = replay.state(replay.times[action['frame_index']-1])
            kind = action['action']
            self.assertEqual(after['action_counts'].get(kind, 0), before['action_counts'].get(kind, 0)+1)
        self.assertEqual(len(replay.state(replay.end)['active']), 0)
        self.assertEqual(len(replay.state(replay.start)['actions']), 0)
        self.assertEqual(len(replay.state(replay.end)['scans']), 7)
        self.assertNotIn('precision', replay.state(replay.end)['metrics'])
        self.assertEqual(replay.state(replay.start)['graph'], replay.state(replay.end)['graph'])

    @unittest.skipUnless((ARTIFACTS/'hovsg.replay.json').is_file(), 'real Gin artifacts unavailable')
    def test_recorded_snapshot_never_looks_ahead(self):
        model = read_json(ARTIFACTS/'hovsg.replay.json')
        snapshot = copy.deepcopy(model['graph'])
        snapshot.update(time_s=10, scope='recorded_snapshot')
        model['graph_history'] = [snapshot]
        replay = Replay(model)
        self.assertEqual(replay.state(0)['graph']['nodes'], [])
        self.assertEqual(replay.state(11)['graph']['nodes'], snapshot['nodes'])

    @unittest.skipUnless((ARTIFACTS/'clio.replay.json').is_file(), 'real Gin artifacts unavailable')
    def test_real_dashboard_routes_and_isolation(self):
        app = create_app([ARTIFACTS/'clio.replay.json', ARTIFACTS/'hovsg.replay.json'])
        with TestClient(app) as client:
            self.assertEqual(len(client.get('/api/replays').json()), 2)
            clio = client.get('/api/graph/clio').json()
            hov = client.get('/api/graph/hovsg').json()
            self.assertEqual(len(clio['elements']['nodes']), 207)
            self.assertEqual(len(hov['elements']['nodes']), 187)
            self.assertEqual(client.get('/api/state/hovsg?t=64').json()['action_counts']['remove'], 1)
            self.assertEqual(client.get('/api/state/hovsg?t=0').json()['action_counts'], {})
            self.assertEqual(client.get('/api/state/unknown').status_code, 404)
            self.assertEqual(client.get('/api/video/clio').status_code, 404)
            response = client.get('/api/panel/hovsg.jpg?t=23')
            self.assertEqual(response.headers['content-type'], 'image/jpeg')
            self.assertTrue(response.content.startswith(b'\xff\xd8'))
            self.assertEqual(client.post('/api/state/hovsg').status_code, 405)


if __name__ == '__main__':
    unittest.main()
