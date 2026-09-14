"""Negative controls for the real overnight publication/archive boundaries."""
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import sys
import time
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.baselines import night_remote, acquisition_watch
from tools.baselines.overnight import require_pair_pass, verify_inventory
from tools.baselines.audit_pair import clio_timing
from tools.baselines.pack_a_driver import (
    compact_completed_scene, completed_recording, stage_pair_inputs, verify_frozen,
    verify_native_revisions)
from tools.baselines.freeze_pack import inventory
from tools.baselines.pack_a_monitor import (
    blocked_on_current_snapshot, direct_child_active, direct_driver_command,
    frozen_manifest_digest, owned_containers, progress_age, stop_direct_child,
    stop_direct_driver)
from tools.baselines.run_pair import prune_regenerable


class Boundaries(unittest.TestCase):
    def test_pack_never_reuses_recording_without_current_observation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = Path(directory)
            (recording / 'static-gt.json').write_text('{}')
            marker = {'complete': True, 'ground_truth_recorded': True}
            (recording / 'acquisition.json').write_text(json.dumps(marker))
            self.assertFalse(completed_recording(recording))
            marker['action_observation_plan'] = {'complete': True}
            (recording / 'acquisition.json').write_text(json.dumps(marker))
            self.assertTrue(completed_recording(recording))

    def test_pair_inputs_use_canonical_names_for_declared_scene_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / 'inputs/scene'
            source.mkdir(parents=True)
            schedule = source / '00813-schedule.json'
            script = source / 'floor-0.script.json'
            schedule.write_text('{}')
            script.write_text('{}')
            pair = root / 'pair'
            pair.mkdir()
            staged = stage_pair_inputs(root, {
                'input_root': 'inputs/scene',
                'schedule': 'inputs/scene/00813-schedule.json',
                'script': 'inputs/scene/floor-0.script.json'}, pair)
            self.assertEqual((pair / 'inputs/schedule.json').resolve(), schedule)
            self.assertEqual((pair / 'inputs/script.json').resolve(), script)
            self.assertEqual(staged, {'schedule.json': str(schedule),
                                      'script.json': str(script)})

    def test_monitor_holds_terminal_failure_until_snapshot_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'frozen-files.json'
            manifest.write_bytes(b'first')
            status = {'phase': 'blocked',
                      'frozen_manifest_sha256': frozen_manifest_digest(root)}
            self.assertTrue(blocked_on_current_snapshot(root, status))
            manifest.write_bytes(b'repaired')
            self.assertFalse(blocked_on_current_snapshot(root, status))

    def test_launcher_releases_output_owner_and_preserves_native_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_bin = root / 'bin'
            fake_bin.mkdir()
            recording = root / 'recording'
            recording.mkdir()
            output = root / 'results/result'
            log = root / 'docker.jsonl'
            fake = fake_bin / 'docker'
            fake.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['FAKE_DOCKER_LOG']).open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
if args[:2] == ['image', 'inspect']:
    print('sha256:fake')
    raise SystemExit(0)
if args and args[0] == 'run' and '--entrypoint' not in args:
    Path(os.environ['FAKE_OUTPUT']).mkdir(parents=True)
    raise SystemExit(int(os.environ['FAKE_NATIVE_RC']))
raise SystemExit(0)
""")
            fake.chmod(0o755)
            env = dict(os.environ, PATH=str(fake_bin) + os.pathsep + os.environ['PATH'],
                       FAKE_DOCKER_LOG=str(log), FAKE_OUTPUT=str(output),
                       FAKE_NATIVE_RC='7', BASELINE_INTEGRATION_ROOT=str(root),
                       BASELINE_REPOS_ROOT=str(root))
            completed = subprocess.run([
                'bash', str(Path(__file__).with_name('gin.sh')), 'hovsg',
                str(recording), str(output)], env=env, check=False)
            self.assertEqual(completed.returncode, 7)
            calls = [json.loads(row) for row in log.read_text().splitlines()]
            self.assertEqual(len(calls), 3)
            self.assertNotIn('--entrypoint', calls[1])
            self.assertIn('--entrypoint', calls[2])
            self.assertEqual(calls[2][calls[2].index('--entrypoint') + 1], 'chown')
            self.assertIn('/results/result', calls[2])

    def test_direct_monitor_stops_exact_child_and_driver_process_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            driver = subprocess.Popen([
                sys.executable, '-c', 'import time; time.sleep(60)',
                'tools.baselines.pack_a_driver', str(root)], start_new_session=True)
            child = subprocess.Popen([
                sys.executable, '-c', 'import time; time.sleep(60)',
                'tools.baselines.runtime', str(root)], start_new_session=True)
            try:
                status = {'driver_pid': driver.pid, 'child_pid': child.pid}
                deadline = time.monotonic() + 2
                while not direct_child_active(root, status) and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(direct_child_active(root, status))
                self.assertTrue(stop_direct_driver(root, status, grace_seconds=5))
                self.assertLess(driver.wait(timeout=5), 0)
                self.assertLess(child.wait(timeout=5), 0)
            finally:
                for process in (child, driver):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)

    def test_direct_monitor_stops_orphaned_recorded_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            child = subprocess.Popen([
                sys.executable, '-c', 'import time; time.sleep(60)',
                'tools.baselines.run_pair', str(root)], start_new_session=True)
            try:
                status = {'driver_pid': 999999999, 'child_pid': child.pid}
                deadline = time.monotonic() + 2
                while not direct_child_active(root, status) and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(stop_direct_child(root, status, grace_seconds=5))
                self.assertLess(child.wait(timeout=5), 0)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)

    def test_monitor_progress_ignores_resource_heartbeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            pair = root / 'runs/scene/pair-attempt-1'
            results = pair / 'results'
            results.mkdir(parents=True)
            status_path = pair / 'pair-status.json'
            heartbeat = pair / 'pair-resources.jsonl'
            native_log = results / 'clio.log'
            for path in (status_path, heartbeat, native_log):
                path.write_text('{}\n')
            old = time.time() - 5 * 3600
            os.utime(status_path, (old, old))
            os.utime(native_log, (old, old))
            self.assertGreater(progress_age(root, {'pair_root': str(pair)}), 4 * 3600)
            self.assertLess(time.time() - heartbeat.stat().st_mtime, 5)

    def test_direct_monitor_command_pins_parallel_pack_policy(self):
        args = SimpleNamespace(clio_gpu=1, hov_gpu=0, hov_skip_frames=50,
                               maximum_attempts=10, minimum_free_gib=15)
        command = direct_driver_command(Path('/tmp/pack'), args)
        self.assertIn('tools.baselines.pack_a_driver', command)
        self.assertEqual(command[command.index('--clio-gpu') + 1], '1')
        self.assertEqual(command[command.index('--hov-gpu') + 1], '0')
        self.assertEqual(command[command.index('--hov-skip-frames') + 1], '50')

    def test_monitor_only_claims_receipted_pack_containers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = root / 'runs/scene/pair-attempt-1'
            pair.mkdir(parents=True)
            (pair / 'pair-status.json').write_text(json.dumps({'jobs': {'clio': {
                'attempts': [{'container': 'graphapi-pack-a-00813-clio'},
                             {'container': 'graphapi-e2e-preflight-clio'},
                             {'container': 'graphapi-pack-a-00813-clio;bad'}]}}}))
            self.assertEqual(owned_containers(root, {'pair_root': str(pair)}),
                             ['graphapi-pack-a-00813-clio'])
            self.assertEqual(owned_containers(root, {'pair_root': '/tmp/another/pair'}), [])

    def test_pack_inventory_follows_only_declared_source_link(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'pack'
            source = Path(directory) / 'source'
            (root / 'integration').mkdir(parents=True)
            (root / 'inputs').mkdir()
            (source / 'lost3dsg/test').mkdir(parents=True)
            (source / 'lost3dsg/src/perception_module').mkdir(parents=True)
            external = Path(directory) / 'external'
            scene = external / 'scene.basis.glb'
            scene.parent.mkdir()
            scene.write_bytes(b'scene')
            scene.with_suffix('.navmesh').write_bytes(b'navmesh')
            semantic = external / 'scene.semantic.glb'
            semantic.write_bytes(b'semantic')
            descriptor = external / 'scene.semantic.txt'
            descriptor.write_text('semantic descriptor')
            objects = root / 'assets/objects'
            objects.mkdir(parents=True)
            object_config = objects / 'medium.object_config.json'
            object_config.write_text(json.dumps({
                'collision_asset': '../collision.glb',
                'render_asset': '../render.glb'}))
            (objects.parent / 'collision.glb').write_bytes(b'collision')
            (objects.parent / 'render.glb').write_bytes(b'render')
            (root / 'source').symlink_to(source, target_is_directory=True)
            dataset = root / 'inputs/dataset.json'
            dataset.write_text(json.dumps({'stages': {'paths': {'.glb': [str(scene)]},
                'default_attributes': {'semantic_asset': str(semantic),
                'semantic_descriptor_filename': str(descriptor)}}}))
            script = root / 'inputs/script.json'
            script.write_text(json.dumps({'steps': [
                {'action': 'spawn', 'template': 'medium'}]}))
            (root / 'pack-manifest.json').write_text(json.dumps({'scenes': [{
                'mesh': str(scene), 'dataset': 'inputs/dataset.json',
                'script': 'inputs/script.json', 'objects': str(objects)}]}))
            revisions = {'schema': 'graphapi.native_baseline_revisions.v1',
                'repositories': {'Clio-Baseline': {'head': 'a', 'tracked_changes': ''},
                                 'HOV-Baseline': {'head': 'b', 'tracked_changes': ''}}}
            (root / 'native-revisions.json').write_text(json.dumps(revisions))
            (root / 'integration/adapter.py').write_text('adapter')
            (root / 'inputs/scene.glb').write_bytes(b'scene')
            (source / 'lost3dsg/test/feed.py').write_text('feed')
            (source / 'lost3dsg/src/perception_module/bridge.py').write_text('bridge')
            values = inventory(root)
            self.assertEqual(set(values), {'pack-manifest.json', 'native-revisions.json',
                'integration/adapter.py', 'inputs/scene.glb', 'inputs/dataset.json',
                'inputs/script.json', 'source/lost3dsg/test/feed.py',
                'source/lost3dsg/src/perception_module/bridge.py', str(scene),
                str(scene.with_suffix('.navmesh')), str(semantic), str(descriptor),
                'assets/objects/medium.object_config.json', 'assets/collision.glb',
                'assets/render.glb'})
            (root / 'frozen-files.json').write_text(json.dumps(values))
            self.assertEqual(verify_frozen(root), len(values))
            with patch('tools.baselines.pack_a_driver.native_state',
                       return_value=revisions['repositories']):
                self.assertEqual(verify_native_revisions(root), revisions['repositories'])

    def test_pack_compaction_requires_metrics_and_preserves_replay_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory) / 'scene'
            recording, pair = scene / 'recording', scene / 'pair-attempt-1'
            for folder in ('rgb', 'pose', 'depth', 'depth_m', 'semantic'):
                (recording / folder).mkdir(parents=True)
            (recording / 'acquisition.json').write_text(json.dumps({'frames': 1}))
            (recording / 'frames.jsonl').write_text(json.dumps({'index': 0}) + '\n')
            for name in ('static-gt.json', 'object_actions.jsonl', 'scan_events.jsonl'):
                (recording / name).write_text('{}\n')
            for folder, suffix in (('rgb', '.png'), ('pose', '.txt'), ('depth', '.png'),
                                   ('depth_m', '.npz'), ('semantic', '.npz')):
                (recording / folder / ('000000' + suffix)).write_bytes(folder.encode())
            for baseline in ('clio', 'hovsg'):
                (pair / 'results' / baseline / 'graph').mkdir(parents=True)
                (pair / 'results' / baseline / 'baseline_result.json').write_text('{}')
            (pair / 'results/clio/graph/backend').mkdir()
            (pair / 'results/clio/graph/backend/dsg.json').write_text('{}')
            (pair / 'results/clio/native_graph_history.jsonl.gz').write_bytes(b'history')
            (pair / 'results/clio/native_outputs.bag').write_bytes(b'native')
            (pair / 'pair-status.json').write_text(json.dumps({'complete': True,
                'phase': 'evaluation_complete', 'jobs': {
                    'clio': {'final_output': str(pair / 'results/clio')},
                    'hovsg': {'final_output': str(pair / 'results/hovsg')}}}))
            with self.assertRaises(RuntimeError):
                compact_completed_scene(scene, pair)
            (scene / 'metrics').mkdir()
            (scene / 'metrics/report.json').write_text('{}')
            marker = compact_completed_scene(scene, pair)
            receipt = json.loads(marker.read_text())
            self.assertTrue(receipt['complete'])
            self.assertEqual(receipt['retained_rgb_frames'], 1)
            self.assertGreater(receipt['reclaimed_bytes'], 0)
            self.assertTrue((recording / 'rgb/000000.png').is_file())
            self.assertTrue((recording / 'pose/000000.txt').is_file())
            self.assertTrue((pair / 'results/clio/native_graph_history.jsonl.gz').is_file())
            self.assertFalse((recording / 'depth_m').exists())
            self.assertFalse((pair / 'results/clio/native_outputs.bag').exists())
            self.assertEqual(compact_completed_scene(scene, pair), marker)

    def test_pack_freeze_and_regenerable_prune_are_hash_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / 'integration/adapter.py'
            frozen.parent.mkdir()
            frozen.write_bytes(b'frozen')
            (root / 'frozen-files.json').write_text(json.dumps({
                'integration/adapter.py': hashlib.sha256(b'frozen').hexdigest()}))
            self.assertEqual(verify_frozen(root), 1)
            frozen.write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                verify_frozen(root)
            output = root / 'result'
            output.mkdir()
            (output / 'input.bag').write_bytes(b'regenerable')
            receipt = prune_regenerable(output, ('input.bag',))
            self.assertEqual(receipt[0]['sha256'],
                             hashlib.sha256(b'regenerable').hexdigest())
            self.assertFalse((output / 'input.bag').exists())

    def test_independent_receipts_never_report_causal_latency(self):
        result=clio_timing([.010,-.002],{'latency_scope':'old observer'})
        self.assertIsNone(result['latency']['mean_ms'])
        self.assertEqual(result['independent_receipt_delta']['negative_samples'],1)
        self.assertAlmostEqual(result['independent_receipt_delta']['mean_ms'],4)
        self.assertIsNone(clio_timing([.010],{})['latency']['mean_ms'])

    def test_publisher_timing_requires_causal_order(self):
        observer={'input_timing_origin':'before_rgb_publish','latency_scope':'publish to output'}
        self.assertAlmostEqual(clio_timing([.01,.02],observer)['latency']['mean_ms'],15)
        with self.assertRaises(ValueError):clio_timing([-.001],observer)

    def test_acquisition_guard_stops_actual_own_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'acquisition-command.json').write_text(json.dumps([sys.executable,'-c','import time; time.sleep(60)']))
            with patch.object(sys,'argv',['acquisition_watch',str(root)]), patch.object(acquisition_watch.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
                self.assertEqual(acquisition_watch.main(),1)
            result=json.loads((root/'acquisition-watch.json').read_text())
            self.assertFalse(result['complete'])
            self.assertLess(result['returncode'],0)
            self.assertFalse(Path('/proc',str(result['pid'])).exists())

    def test_publication_rejects_each_baseline_failure(self):
        report={'gate':{'passed':True},'baselines':{x:{'required_checks_passed':True} for x in ('clio','hovsg')}}
        require_pair_pass(report)
        for name in ('clio','hovsg'):
            report['baselines'][name]['required_checks_passed']=False
            with self.assertRaises(ValueError):require_pair_pass(report)
            report['baselines'][name]['required_checks_passed']=True
        report['gate']['passed']=False
        with self.assertRaises(ValueError):require_pair_pass(report)

    def test_real_bytes_not_just_file_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'native.json';path.write_bytes(b'original')
            inventory={'files':{'native.json':{'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}}}
            verify_inventory(root,inventory)
            path.write_bytes(b'mutated!')
            with self.assertRaises(ValueError):verify_inventory(root,inventory)
            path.unlink()
            with self.assertRaises(ValueError):verify_inventory(root,inventory)

    def test_path_escape_rejected(self):
        for path in ('/dev/shm/graphapi_live','/dev/shm','/dev/shm/graphapi-baselines-overnight-20260913/runs/../../other'):
            with self.assertRaises(ValueError):night_remote.checked_root(path)
        self.assertEqual(night_remote.checked_root(str(night_remote.CURRENT)),night_remote.CURRENT)

    def test_frozen_source_change_refuses_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'adapter.py';path.write_text('version_a')
            (root/'frozen-files.json').write_text(json.dumps({'adapter.py':night_remote.digest(path)}))
            with patch.object(night_remote,'ROOT',root):
                self.assertEqual(night_remote.frozen()['checked_files'],1)
                path.write_text('version_b')
                with self.assertRaises(ValueError):night_remote.frozen()

    def test_reclaim_incomplete_bundle_never_launches_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'pair-status.json').write_text(json.dumps({'phase':'native_failure','complete':False}))
            with patch.object(night_remote,'checked_root',return_value=root), patch.object(night_remote.subprocess,'run') as delete:
                with self.assertRaises(ValueError):night_remote.handle({'action':'reclaim','root':str(root),'verified_files':{}})
                delete.assert_not_called()


if __name__=='__main__':unittest.main()
