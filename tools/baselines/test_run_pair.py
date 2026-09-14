import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.baselines.run_pair import run


class PairSchedulingTests(unittest.TestCase):
    def test_parallel_mode_assigns_distinct_gpus_and_requires_both_readiness_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording, inputs = root / 'recording', root / 'inputs'
            recording.mkdir(); inputs.mkdir()
            row = {'index': 0, 'stem': '000000', 'time_s': 0.0, 'reason': 'tour',
                   'event': [{'lap': 0, 'stop': 0}]}
            (recording / 'frames.jsonl').write_text(json.dumps(row) + '\n')
            for folder, suffix in (('rgb', '.png'), ('depth', '.png'),
                                   ('depth_m', '.npz'), ('pose', '.txt'),
                                   ('semantic', '.npz')):
                (recording / folder).mkdir()
                (recording / folder / ('000000' + suffix)).write_bytes(b'x')
            schedule = {'schedule': [{'trajectory': [{'stop': 0, 'scan_deg': 360}]}]}
            script = {'steps': []}
            (inputs / 'schedule.json').write_text(json.dumps(schedule))
            (inputs / 'script.json').write_text(json.dumps(script))
            import hashlib
            (recording / 'acquisition.json').write_text(json.dumps({'complete': True,
                'requested_laps': 1, 'frames': 1,
                'schedule_sha256': hashlib.sha256((inputs / 'schedule.json').read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256((inputs / 'script.json').read_bytes()).hexdigest()}))
            (recording / 'object_actions.jsonl').write_text('')
            gt = root / 'gt.json'
            gt.write_text(json.dumps({key: [1] for key in
                ('ground_truth_floors_m', 'ground_truth_regions',
                 'ground_truth_objects', 'rooms', 'categories')}))
            launcher = root / 'native-integration/tools/baselines/gin.sh'
            launcher.parent.mkdir(parents=True)
            launcher.write_text('''#!/bin/bash
name=$1
output=$3
mkdir -p "$output"
if [[ "$name" == clio ]]; then
  payload='{"complete":true,"evaluation_readiness":{"ready_for_temporal_object_action_evaluation":true}}'
else
  payload='{"complete":true,"evaluation_readiness":{"ready_for_end_to_end_evaluation":true}}'
fi
sleep 0.1
printf '%s\n' "$payload" > "$output/baseline_result.json"
''')
            args = SimpleNamespace(run_root=root, acquisition_pid=os.getpid(),
                baseline_roots=root / 'native', gpu=1, clio_gpu=1, hov_gpu=0,
                parallel_distinct_gpus=True, laps=1, hov_skip_frames=50,
                clio_rate=1.0, clio_segmentation_confidence=.25,
                clio_mode='semantic-mapping', clio_tasks=[], container_prefix='test',
                ground_truth=gt)
            source = {'Clio-Baseline': {'head': 'a', 'tracked_changes': ''},
                      'HOV-Baseline': {'head': 'b', 'tracked_changes': ''}}
            def fake_run(command, check=False, **kwargs):
                if 'tools.baselines.graphapi_eval' in command:
                    output = Path(command[command.index('--output') + 1])
                    output.write_text('{}')
                return subprocess.CompletedProcess(command, 0)
            with patch('tools.baselines.run_pair.native_state', return_value=source), \
                    patch('tools.baselines.run_pair.subprocess.run', side_effect=fake_run), \
                    patch('tools.baselines.run_pair.subprocess.check_output',
                          return_value='0, 0, 0\n1, 0, 0'):
                status = run(args)
            self.assertTrue(status['complete'])
            self.assertTrue(status['parallel'])
            self.assertEqual(status['phase'], 'evaluation_complete')
            self.assertEqual(status['jobs']['clio']['gpu'], 1)
            self.assertEqual(status['jobs']['hovsg']['gpu'], 0)
            self.assertTrue(status['jobs']['clio']['verified_complete'])
            self.assertTrue(status['jobs']['hovsg']['verified_complete'])
            # Each baseline ingests the same recording at its own native rate: Clio the full
            # stream, HOV-SG the shared manifest. On the stride-50 manifest Clio produced 4
            # primitives where the contiguous stream produced 1,524 on the same scene.
            launched = {name: status['jobs'][name]['attempts'][0]['command'][3]
                        for name in ('clio', 'hovsg')}
            self.assertEqual(launched['clio'], str(root / 'recording'))
            self.assertEqual(launched['hovsg'], str(root / 'shared-input'))
            self.assertEqual(status['input_by_baseline']['clio']['path'], launched['clio'])
            self.assertEqual(status['input_by_baseline']['hovsg']['path'], launched['hovsg'])
            self.assertEqual(status['shared_observation_input']['used_by'], ['hovsg'])


if __name__ == '__main__':
    unittest.main()
