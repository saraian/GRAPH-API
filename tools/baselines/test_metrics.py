import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.baselines.replay_metrics import bounds, evaluate, match_boxes, validate_metrics
from tools.baselines.timing import phase, save


class MetricsTests(unittest.TestCase):
    def test_duplicate_predictions_do_not_inflate_recall(self):
        box = bounds([0, 0, 0], [1, 1, 1])
        self.assertEqual(len(match_boxes([box, box], [box], .25)), 1)
        self.assertEqual(match_boxes([box], [bounds([3, 3, 3], [4, 4, 4])], .25), [])
        self.assertEqual(match_boxes([], [box], .25), [])
        with self.assertRaises(ValueError):
            match_boxes([], [], 0)

    def test_maximum_cardinality_before_iou(self):
        import numpy as np
        # IoUs: P0-G0=0.9, P0-G1=.4, P1-G0=.4, P1-G1=0.
        # A greedy strongest-pair choice loses a valid second match.
        pred = [bounds([0, 0, 0], [9, 1, 1]), bounds([6, 0, 0], [10, 1, 1])]
        gt = [bounds([0, 0, 0], [10, 1, 1]), bounds([0, 0, 0], [3.6, 1, 1])]
        matches = match_boxes(pred, gt, .25)
        self.assertEqual({(a, b) for a, b, _ in matches}, {(0, 1), (1, 0)})
        with self.assertRaises(ValueError):
            bounds([np.nan, 0, 0], [1, 1, 1])

    def test_final_evaluation_denominators_and_scene_guard(self):
        from tools.baselines.replay_model import bounds_corners
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = {'scene': 'scene-a', 'actions': [], 'graph': {'scope': 'final_snapshot', 'nodes': [
                {'id': 'a', 'type': 'object', 'corners': bounds_corners([.5]*3, [1]*3).tolist()},
                {'id': 'missing', 'type': 'object'}]}}
            gt = {'geometry_space': {'coordinate_frame': 'Habitat: Y up'},
                  'ground_truth_source': {'scene': '/data/scene-a/mesh.glb'},
                  'ground_truth_objects': [{'object_id': 'gt1', 'aabb_min_m': [0]*3, 'aabb_max_m': [1]*3,
                                            'geometry_source': 'semantic_object_obb_to_aabb'}]}
            model_path, gt_path = root/'model.json', root/'gt.json'
            model_path.write_text(json.dumps(model))
            gt_path.write_text(json.dumps(gt))
            report = evaluate(model_path, gt_path, root/'report.json')
            self.assertEqual((report['tp'], report['fp'], report['fn']), (1, 1, 0))
            self.assertEqual(report['metrics']['precision']['value'], 50)
            self.assertEqual(report['metrics']['recall']['value'], 100)
            gt['ground_truth_objects'].insert(0, {'object_id': 'empty', 'aabb_min_m': [0]*3,
                'aabb_max_m': [0]*3, 'geometry_source': 'semantic_object_obb_to_aabb'})
            gt_path.write_text(json.dumps(gt))
            filtered = evaluate(model_path, gt_path, root/'filtered.json')
            self.assertEqual((filtered['gt_objects_raw'], filtered['gt_objects'], filtered['fn']), (2, 1, 0))
            self.assertEqual(filtered['gt_boxes_excluded'][0]['object_id'], 'empty')
            self.assertEqual(filtered['matches'][0]['gt_id'], 'gt1')
            self.assertIn('zero-volume GT boxes excluded', filtered['scope'])
            valid = gt['ground_truth_objects'].pop()
            gt_path.write_text(json.dumps(gt))
            with self.assertRaisesRegex(ValueError, 'No positive-volume'):
                evaluate(model_path, gt_path, root/'all-empty.json')
            gt['ground_truth_objects'].append(valid)
            gt['ground_truth_source']['scene'] = '/data/wrong/mesh.glb'
            gt_path.write_text(json.dumps(gt))
            with self.assertRaisesRegex(ValueError, 'different scene'):
                evaluate(model_path, gt_path, root/'bad.json')
            self.assertFalse((root/'bad.json').exists())

    def test_nonfinite_scores_and_clock_are_rejected(self):
        base = {'time_s': 1, 'metrics': {'precision': {'value': 50, 'unit': '%', 'source': 'real', 'scope': 'test'}}}
        validate_metrics([base])
        for bad in (float('nan'), float('inf'), -1, 101, True):
            base['metrics']['precision']['value'] = bad
            with self.assertRaises(ValueError):
                validate_metrics([base])
        base['metrics']['precision']['value'] = None
        validate_metrics([base])
        base['time_s'] = float('nan')
        with self.assertRaises(ValueError):
            validate_metrics([base])

    def test_metrics_sidecar_identity_and_final_scope(self):
        import shutil

        from fastapi.testclient import TestClient

        from tools.baselines.replay_model import file_stamp
        from tools.baselines.replay_server import create_app
        source = Path(__file__).resolve().parents[2] / 'artifacts/baselines/hovsg.replay.json'
        if not source.exists():
            self.skipTest('Real replay artifact unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)/'hovsg.replay.json'
            shutil.copyfile(source, model)
            report = {'schema': 'graphapi.baseline_metrics.v1', 'model_sha256': 'wrong',
                      'metrics': {'precision': {'value': 25, 'unit': '%', 'source': 'fixture', 'scope': 'final whole scene'}}}
            path = model.with_suffix('.metrics.json')
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'different replay model'):
                create_app([model])
            report['model_sha256'] = file_stamp(model)['sha256']
            path.write_text(json.dumps(report))
            with TestClient(create_app([model])) as client:
                for t in (0, 60, 0):
                    state = client.get(f'/api/state/hovsg?t={t}').json()
                    self.assertNotIn('precision', state['metrics'])
                    self.assertEqual(state['final_metrics']['precision']['value'], 25)

    def test_timing_observes_clock_and_does_not_report_failed_phase(self):
        rows = {}
        with patch('tools.baselines.timing.time.perf_counter', side_effect=[10, 12.5]):
            with phase(rows, 'native_call'):
                pass
        self.assertEqual(rows, {'native_call': 2.5})
        with self.assertRaises(RuntimeError):
            with phase(rows, 'failed'):
                raise RuntimeError('native failed')
        self.assertNotIn('failed', rows)
        with tempfile.TemporaryDirectory() as tmp:
            save(tmp, rows, 'native call only')
            record = json.loads((Path(tmp)/'execution_timing.json').read_text())
            self.assertEqual(record['phases_s']['native_call'], 2.5)
            self.assertNotIn('avg_latency_ms', record)


if __name__ == '__main__':
    unittest.main()
