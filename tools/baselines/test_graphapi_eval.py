import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools.baselines.graphapi_eval import (
    clio_temporal,
    final_ground_truth,
    hov_temporal,
    hov_manifest,
    occupancy_outline,
    require_ground_truth,
)
from tools.baselines.hovsg_run import prepare_sampled_recording, select_hov_frames
from tools.baselines.clio_graph_observer import snapshot
from tools.baselines.hovsg_run import verify_evaluation_outputs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


class EndToEndEvaluationTests(unittest.TestCase):
    def test_hov_sampling_is_uniform_and_does_not_read_gt_or_actions(self):
        frames = [{'index': index, 'visible_semantic_ids': []} for index in range(200)]
        for index in (11, 12, 13, 71, 75, 80):
            frames[index]['visible_semantic_ids'] = [1001]
        indices, selections = select_hov_frames(frames, 50)
        self.assertEqual(indices, [0, 50, 100, 150])
        reasons = {row['frame_index']: row['reasons'] for row in selections}
        self.assertEqual(set(reasons), {0, 50, 100, 150})
        self.assertTrue(all(value == ['uniform_stride'] for value in reasons.values()))

    def test_hov_sampled_recording_preserves_source_frame_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording, output = root / 'recording', root / 'output'
            for folder in ('rgb', 'depth', 'pose'):
                (recording / folder).mkdir(parents=True)
            output.mkdir()
            frames = []
            for index in range(6):
                stem = f'{index:06d}'
                frames.append({'index': index, 'stem': stem, 'time_s': float(index),
                    'visible_semantic_ids': [1001] if index in (1, 2) else []})
                for folder, suffix in (('rgb', '.png'), ('depth', '.png'),
                                       ('pose', '.txt')):
                    (recording / folder / (stem + suffix)).write_bytes(folder.encode())
            (recording / 'frames.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in frames))
            write_json(recording / 'acquisition.json', {'complete': True, 'frames': 6})
            view, indices, receipt = prepare_sampled_recording(recording, output, 5)
            self.assertEqual(indices, [0, 5])
            rows = [json.loads(row) for row in (view / 'frames.jsonl').read_text().splitlines()]
            self.assertEqual([row['index'] for row in rows], list(range(2)))
            self.assertEqual([row['source_frame_index'] for row in rows], indices)
            self.assertEqual((view / 'rgb/000001.png').resolve(),
                             (recording / 'rgb/000005.png').resolve())
            self.assertEqual(receipt['sampled_source_frame_indices'], indices)

    def test_hov_run_readiness_requires_aligned_features_and_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording, output = root / 'recording', root / 'output'
            recording.mkdir()
            (output / 'native_observations').mkdir(parents=True)
            (output / 'evaluation_assets').mkdir()
            (recording / 'frames.jsonl').write_text(''.join(
                json.dumps({'index': native, 'source_frame_index': source}) + '\n'
                for native, source in enumerate((0, 2))))
            native_rows = []
            for native_index, index in enumerate((0, 2)):
                masks = np.zeros((1, 2, 2), dtype=bool)
                np.savez_compressed(output / f'native_observations/{index:06d}.masks.npz',
                    packed=np.packbits(masks, axis=-1), shape=np.asarray(masks.shape),
                    mask_features=np.ones((1, 2)), global_feature=np.ones(2))
                native_rows += [
                    {'stage': 'sam_clip', 'frame_index': index,
                     'native_frame_index': native_index, 'masks': [{}]},
                    {'stage': 'masks_3d', 'frame_index': index,
                     'native_frame_index': native_index, 'masks': [{}]},
                ]
            write_json(output / 'native_observations/observer.json',
                       {'schema': 'graphapi.hovsg_native_observer.v2'})
            (output / 'native_observations/observations.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in native_rows))
            write_json(output / 'evaluation_assets/manifest.json', {
                'object_categories': ['a'], 'room_categories': ['room'],
                'embedding_dimension': 2})
            np.save(output / 'evaluation_assets/object_category_features.npy',
                    np.ones((1, 2)))
            np.save(output / 'evaluation_assets/room_category_features.npy',
                    np.ones((1, 2)))
            ready = verify_evaluation_outputs(recording, output, 1, True, {})
            self.assertTrue(ready['ready_for_end_to_end_evaluation'])
            native_rows[0]['frame_index'] = 1
            (output / 'native_observations/observations.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in native_rows))
            with self.assertRaisesRegex(RuntimeError, 'indices'):
                verify_evaluation_outputs(recording, output, 1, True, {})

    def test_clio_observer_keeps_compact_semantic_primitive_history(self):
        class Box:
            def is_valid(self):
                return True

            def corners(self):
                return np.asarray([[x, y, z] for x in (0, 1)
                                   for y in (0, 1) for z in (0, 1)])

        attrs = SimpleNamespace(name='', position=np.asarray([1, 2, 3]),
            first_observed_ns=np.asarray([10]), last_observed_ns=np.asarray([20]),
            is_active=True, bounding_box=Box())
        native = SimpleNamespace(id=SimpleNamespace(value=(ord('s') << 56) + 7),
                                 attributes=attrs)
        graph = SimpleNamespace(nodes=[native], edges=[], interlayer_edges=[])
        row = snapshot(graph, 30, '/backend/dsg')
        self.assertEqual(row['nodes'][0]['type'], 'semantic_primitive')
        self.assertEqual(row['nodes'][0]['native_last_observed_ns'], [20])
        self.assertTrue(row['nodes'][0]['native_is_active'])
        self.assertNotIn('semantic_feature', row['nodes'][0])

    def test_hov_manifest_uses_native_3d_ply_not_2d_json_vertices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording, result = root / 'recording', root / 'result'
            recording.mkdir()
            write_json(recording / 'acquisition.json', {'selected_floor_height_m': 0})
            for name in ('floors', 'rooms', 'objects'):
                (result / 'graph' / name).mkdir(parents=True)
            write_json(result / 'execution_timing.json',
                       {'phases_s': {'run_wall_time': 3}})
            write_json(result / 'graph/floors/0.json',
                       {'floor_id': '0', 'floor_zero_level': 0})
            write_json(result / 'graph/rooms/0_0.json',
                       {'room_id': '0_0', 'floor_id': '0',
                        'vertices': [[0, 0], [.05, 0], [0, .05]]})
            write_json(result / 'graph/objects/0_0_0.json',
                       {'object_id': '0_0_0', 'room_id': '0_0',
                        'name': 'chair', 'vertices': [[99, 99]],
                        'embedding': [1, 0]})
            (result / 'graph/objects/0_0_0.ply').write_text(
                'ply\nformat ascii 1.0\nelement vertex 2\n'
                'property float x\nproperty float y\nproperty float z\nend_header\n'
                '1 2 3\n4 5 6\n')
            gt = {'ground_truth_floors_m': [0],
                  'ground_truth_regions': [{'region_id': 'r',
                                            'polygon_xz_m': [[0, 0], [1, 0], [0, 1]]}],
                  'ground_truth_objects': [], 'rooms': [], 'categories': []}
            fake = SimpleNamespace(assignment=lambda *args: [])
            with patch('tools.baselines.graphapi_eval.evaluators',
                       return_value=(fake, None)):
                manifest = hov_manifest(gt, recording, result)
            self.assertEqual(len(manifest['predicted_objects']), 1)
            self.assertEqual(manifest['predicted_objects'][0]['aabb_min_m'], [1., 2., 3.])
            self.assertEqual(manifest['predicted_objects'][0]['aabb_max_m'], [4., 5., 6.])

    def test_full_gt_contract_and_final_dynamic_state(self):
        gt = {'ground_truth_floors_m': [0], 'ground_truth_regions': [{}],
              'ground_truth_objects': [{'object_id': 'static'}], 'rooms': [{}],
              'categories': [{'category_id': 0, 'category_name': 'chair'}],
              'geometry_space': {'coordinate_frame': 'Habitat: Y up'}}
        require_ground_truth(gt)
        frames = [{'dynamic_ground_truth': []}, {'dynamic_ground_truth': [{
            'object_id': 4, 'semantic_id': 1000004,
            'aabb_min': [0, 0, 0], 'aabb_max': [1, 1, 1]}]}]
        actions = [{'action': 'spawn', 'request': {'template': '035_power_drill'},
                    'result': {'object_id': 4}}]
        result = final_ground_truth(gt, frames, actions)
        self.assertEqual(result['dynamic_final_objects'], 1)
        self.assertEqual(result['ground_truth_objects'][-1]['category_name'], 'power drill')
        frames[-1]['dynamic_ground_truth'] = []
        self.assertEqual(final_ground_truth(gt, frames, actions)['dynamic_final_objects'], 0)
        with self.assertRaisesRegex(ValueError, 'missing'):
            require_ground_truth({'ground_truth_objects': []})

    def test_native_hov_occupancy_is_traced_without_convex_fill(self):
        vertices = [[x, y] for x, y in
                    [(0, 0), (.05, 0), (.10, 0), (0, .05), (0, .10)]]
        polygon = occupancy_outline(vertices)
        self.assertGreaterEqual(len(polygon), 6)
        self.assertGreater(abs(sum(
            polygon[i][0] * polygon[(i + 1) % len(polygon)][1] -
            polygon[i][1] * polygon[(i + 1) % len(polygon)][0]
            for i in range(len(polygon))) / 2), 0)

    def _recording(self, root):
        recording = root / 'recording'
        (recording / 'semantic').mkdir(parents=True)
        frames = []
        for index in range(5):
            alive = index in (1, 2)
            dynamic = ([{'object_id': 1, 'semantic_id': 1000001,
                         'aabb_min': [0, 0, 0], 'aabb_max': [1, 1, 1]}]
                       if alive else [])
            frames.append({'index': index, 'time_s': index + 1.,
                           'dynamic_ground_truth': dynamic})
            semantic = np.zeros((2, 2), dtype=np.uint32)
            if alive:
                semantic[0, 0] = 1000001
            np.savez_compressed(recording / f'semantic/{index:06d}.npz',
                                semantic=semantic)
        (recording / 'frames.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in frames))
        actions = [
            {'action': 'spawn', 'after_frame': 0,
             'result': {'success': True, 'object_id': 1}},
            {'action': 'remove', 'after_frame': 2,
             'result': {'success': True, 'object_id': 1}},
        ]
        (recording / 'object_actions.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in actions))
        return recording

    def test_hov_temporal_appearance_and_disappearance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = self._recording(root)
            result = root / 'hov'
            native = result / 'native_observations'
            (result / 'graph/objects').mkdir(parents=True)
            native.mkdir(parents=True)
            rows = []
            for index in range(5):
                alive = index in (1, 2)
                mask = np.zeros((1 if alive else 0, 2, 2), dtype=bool)
                if alive:
                    mask[0, 0, 0] = True
                features = np.asarray([[1., 0.]] if alive else [], dtype=np.float16).reshape(-1, 2)
                np.savez_compressed(native / f'{index:06d}.masks.npz',
                    packed=np.packbits(mask, axis=-1), shape=np.asarray(mask.shape),
                    mask_features=features, global_feature=np.asarray([0, 1], dtype=np.float16))
                rows.append({'frame_index': index, 'stage': 'sam_clip',
                             'stage_wall_time_s': .1, 'masks': [{}] if alive else []})
                rows.append({'frame_index': index, 'stage': 'masks_3d',
                             'stage_wall_time_s': .01,
                             'masks': ([{'mask_index': 0, 'aabb_min': [0, 0, 0],
                                         'aabb_max': [1, 1, 1]}] if alive else [])})
            (native / 'observations.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows))
            report = hov_temporal(recording, result)
            self.assertEqual(report['actions'][0]['visible_frame_recall_pct'], 100)
            self.assertTrue(report['actions'][1]['removal_confirmed_in_final_map'])
            self.assertEqual(report['actions'][1]['status'],
                             'evaluated_final_map_persistence')

    def test_clio_temporal_keeps_history_separate_from_activity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recording = self._recording(root)
            result = root / 'clio'
            result.mkdir()
            corners = [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]
            snapshots = [
                {'time_s': 1., 'nodes': []},
                {'time_s': 2., 'nodes': [{'id': 's1', 'type': 'semantic_primitive',
                                          'corners': corners, 'native_is_active': True}]},
                {'time_s': 3., 'nodes': [{'id': 's1', 'type': 'semantic_primitive',
                                          'corners': corners, 'native_is_active': True}]},
                {'time_s': 4., 'nodes': [{'id': 's1', 'type': 'semantic_primitive',
                                          'corners': corners, 'native_is_active': False}]},
            ]
            with gzip.open(result / 'native_graph_history.jsonl.gz', 'wt') as stream:
                for row in snapshots:
                    stream.write(json.dumps(row) + '\n')
            report = clio_temporal(recording, result)
            self.assertEqual(report['actions'][0]['new_spatial_primitive_ids'], ['s1'])
            removal = report['actions'][1]
            self.assertTrue(removal['removal_confirmed_active_layer'])
            self.assertTrue(removal['historical_evidence_retained'])


if __name__ == '__main__':
    unittest.main()
