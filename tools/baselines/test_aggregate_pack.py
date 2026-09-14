import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.baselines.aggregate_pack import aggregate


class AggregatePackTests(unittest.TestCase):
    def test_official_tables_temporal_actions_and_causal_latency_are_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_root = root / 'runs/scene-a'
            recording = scene_root / 'recording'
            pair = scene_root / 'pair-attempt-1'
            evaluation = pair / 'evaluation'
            clio = pair / 'results/clio'
            hov = pair / 'results/hovsg'
            for path in (recording, evaluation, clio, hov / 'native_observations'):
                path.mkdir(parents=True, exist_ok=True)
            (root / 'pack-manifest.json').write_text(json.dumps(
                {'pack': 'A', 'scenes': [{'scene': 'scene-a'}]}))
            (recording / 'static-gt.json').write_text('{}')
            (recording / 'frames.jsonl').write_text(json.dumps(
                {'index': 0, 'time_s': 0.0}) + '\n')
            (recording / 'object_actions.jsonl').write_text('')
            temporal = {'status': 'supported', 'actions': [
                {'action': 'spawn', 'object_id': '1', 'first_evidence_frame': 0},
                {'action': 'remove', 'object_id': '1',
                 'removal_confirmed_active_layer': True}]}
            for baseline, size, seconds in (('clio', 2.5, 3.0), ('hovsg', 1.0, 4.0)):
                report = {'tables': {'table_vii_representation': {'size_mb_total': size},
                                     'construction_time_s': seconds},
                          'temporal_object_actions': temporal}
                (evaluation / f'{baseline}.json').write_text(json.dumps(report))
            (clio / 'native_observer.json').write_text(json.dumps(
                {'input_timing_origin': 'before_rgb_publish'}))
            (clio / 'native_receipts.jsonl').write_text(json.dumps({
                'topic': '/dominic/forward/semantic/image_raw',
                'paired_transport_latency_s': .01}) + '\n')
            (hov / 'native_observations/observations.jsonl').write_text(json.dumps(
                {'stage': 'sam_clip', 'stage_wall_time_s': .2}) + '\n')
            state = {'complete': True, 'phase': 'evaluation_complete', 'jobs': {
                'clio': {'evaluation': {'output': str(evaluation / 'clio.json')},
                         'final_output': str(clio)},
                'hovsg': {'evaluation': {'output': str(evaluation / 'hovsg.json')},
                          'final_output': str(hov)}}}
            (pair / 'pair-status.json').write_text(json.dumps(state))
            metrics = SimpleNamespace(
                evaluate=lambda manifests, output: {
                    'table_v_retrieval': {'trials': 0},
                    'table_vii_representation': {'size_mb_total': 1.0}},
                filtered_scene=lambda manifest: manifest)
            objects = SimpleNamespace(
                evaluate_geometry=lambda manifests: (
                    {'precision_pct': 50.0, 'recall_pct': 25.0}, []),
                evaluate_labels=lambda manifests, matches: {'f1_pct': 33.333})
            output = root / 'pack-metrics'
            with patch('tools.baselines.aggregate_pack.evaluators',
                       return_value=(metrics, objects)), patch(
                       'tools.baselines.aggregate_pack.hov_manifest',
                       return_value={'scene': 'scene-a'}), patch(
                       'tools.baselines.aggregate_pack.final_ground_truth',
                       return_value={}):
                result = aggregate(root, output)
            self.assertTrue(result['complete'])
            self.assertNotIn('table_v_retrieval', result['baselines']['hovsg']['tables'])
            self.assertEqual(result['baselines']['clio']['latency']['mean_ms'], 10.0)
            self.assertEqual(result['baselines']['hovsg']['latency']['mean_ms'], 200.0)
            self.assertEqual(result['baselines']['clio']['temporal_object_actions']
                             ['appearance_action_recall_pct'], 100.0)
            self.assertEqual(result['baselines']['clio']['temporal_object_actions']
                             ['removals_confirmed'], 1)
            self.assertTrue((output / 'report.json').is_file())
            self.assertTrue((output / 'metrics.csv').is_file())


if __name__ == '__main__':
    unittest.main()
