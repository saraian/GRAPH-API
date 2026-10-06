"""Startup boundary tests: no dataset, GPU, ROS or private PAL image required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from graphapi_cli.configuration import ConfigurationError, environment, resolve_config, read_yaml
from graphapi_cli.docker import command
from graphapi_cli.launch import prepare_job, stop_operation
from graphapi_cli.registry import Registry, write_json
from graphapi_cli.batch import run_batch
from tools.startup.generate_catalogue import generate

ROOT = Path(__file__).resolve().parents[2]


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.patch = patch.dict(os.environ, {'XDG_CACHE_HOME': str(self.work / 'cache')}, clear=True)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.env, _ = environment(ROOT, inherited={'WORKSPACE_ROOT': str(self.work), 'HM3D_ROOT': str(self.work),
                                                   'XDG_CACHE_HOME': str(self.work / 'cache'), 'REGOLO_API_KEY': 'test-key'})

    def test_all_original_entrypoints_are_registered(self):
        recorded = json.loads((ROOT / 'config/entrypoints.json').read_text())
        self.assertEqual(recorded, generate(ROOT))
        self.assertFalse(any('test_' in e['path'] and e['kind'] == 'launch' for e in recorded.values()))

    def test_default_config_has_original_effective_values(self):
        import hashlib
        _, cfg = resolve_config(ROOT)
        original = read_yaml(ROOT / 'config/graphapi.yaml')
        # Normalized configuration from pre-refactor commit 0a34051. Stable after
        # committing compatibility symlinks; no dependence on the current HEAD.
        self.assertEqual(hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
                         '94bd21952f1909825a1248f94f5c17722ef2e4e5cea9f8eb586ad1de933b14f6')
        for section, values in original.items():
            for key, value in values.items():
                self.assertEqual(cfg[section][key], value, f'{section}.{key}')
        self.assertEqual(cfg['habitat']['localization_mode'], 'ground_truth')

    def test_legacy_config_alias_matches_canonical(self):
        self.assertEqual(resolve_config(ROOT, 'lost3dsg/test/tiago.yaml')[1], resolve_config(ROOT, 'config/tiago.yaml')[1])

    def test_duplicate_yaml_fails_before_launch(self):
        path = self.work / 'bad.yaml'
        path.write_text('habitat: {}\nhabitat: {}\n')
        with self.assertRaisesRegex(ConfigurationError, 'duplicate'):
            read_yaml(path)

    def test_old_managed_image_is_rebuilt_with_current_dependency_revision(self):
        from graphapi_cli.docker import ensure_image
        with patch('graphapi_cli.docker.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='1|<no value>\n')) as run:
            ensure_image(ROOT,self.env)
        build = run.call_args_list[-1].args[0]
        self.assertEqual(build[:2],['docker','build'])
        self.assertTrue(any(value.startswith('RUNTIME_BUILD_REVISION=') for value in build))

    def test_current_managed_image_is_reused(self):
        import hashlib
        from graphapi_cli.docker import ensure_image
        revision = hashlib.sha256((ROOT/'docker/sim/Dockerfile').read_bytes()).hexdigest()
        with patch('graphapi_cli.docker.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='2|'+revision+'\n')) as run:
            ensure_image(ROOT,self.env)
        self.assertEqual(run.call_count,1)

    def test_paths_and_environment_precedence(self):
        settings = self.work / 'settings.yaml'
        settings.write_text('paths:\n  workspace: here\nenvironment:\n  RVIZ: 1\n')
        env, _ = environment(ROOT, str(settings), {'RVIZ': '0'})
        self.assertEqual(env['WORKSPACE_ROOT'], str(self.work / 'here'))
        self.assertEqual(env['RVIZ'], '0')

    def test_gpu_leases_are_exclusive_and_releaseable(self):
        first = Registry(self.work)
        op, _, _ = first.reserve('sim', gpu='0')
        second = Registry(self.work / 'other')
        with self.assertRaisesRegex(ValueError, 'owned'):
            second.reserve('sim', gpu='0')
        first.release(op)
        second.reserve('sim', gpu='0')

    def test_ports_and_domains_are_distinct(self):
        reg = Registry(self.work)
        _, a, da = reg.reserve('sim', ports={'bridge': None, 'feed': None})
        _, b, db = reg.reserve('sim', ports={'bridge': None, 'feed': None})
        self.assertEqual(len(a),len(set(a.values())))
        self.assertEqual(len(b),len(set(b.values())))
        self.assertFalse(set(a.values()) & set(b.values()))
        self.assertNotEqual(da, db)

    def test_dashboard_port_can_restart_after_a_closed_http_connection(self):
        import socket
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        listener.bind(('127.0.0.1',0))
        port = listener.getsockname()[1]
        listener.listen()
        client = socket.create_connection(('127.0.0.1',port))
        server,_ = listener.accept()
        server.close()  # Server actively closes: its port can remain in TIME_WAIT.
        self.assertEqual(client.recv(1),b'')
        client.close()
        listener.close()
        with socket.socket() as old_probe:
            with self.assertRaises(OSError):
                old_probe.bind(('127.0.0.1',port))
        reg = Registry(self.work)
        operation,ports,_ = reg.reserve('dashboard',ports={'dashboard':port},need_domain=False)
        self.assertEqual(ports['dashboard'],port)
        reg.release(operation)

    def test_live_listener_still_blocks_dashboard_port_with_clear_error(self):
        import socket
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            listener.bind(('127.0.0.1',0))
            port = listener.getsockname()[1]
            listener.listen()
            with self.assertRaisesRegex(ValueError,f'dashboard port {port} is unavailable'):
                Registry(self.work).reserve('dashboard',ports={'dashboard':port},need_domain=False)

    def test_reserved_dashboard_error_identifies_the_owner(self):
        reg = Registry(self.work)
        operation,ports,_ = reg.reserve('dashboard',ports={'dashboard':None},need_domain=False)
        with self.assertRaisesRegex(ValueError,f'reserved by operation {operation}'):
            reg.reserve('dashboard',ports={'dashboard':ports['dashboard']},need_domain=False)
        reg.release(operation)

    def test_orphan_container_keeps_lease(self):
        reg = Registry(self.work)
        operation, _, _ = reg.reserve('sim', gpu='0')
        reg.db.execute('UPDATE leases SET pid=-999999,identity=NULL')
        reg.db.commit()
        with patch('graphapi_cli.registry.alive', return_value=False), patch('graphapi_cli.registry.owned_containers', return_value=['owned']):
            with self.assertRaises(ValueError):
                reg.reserve('sim', gpu='0')
        reg.release(operation)

    def test_latest_uses_completion_and_active_requires_one(self):
        reg = Registry(self.work)
        for name, state, created, finished in [('a','COMPLETED',1,8), ('b','FAILED',2,9), ('c','COMPLETED',3,7)]:
            write_json(reg.operations / name / 'status.json', dict(operation_id=name,state=state,created=created,finished=finished,bundles=[name]))
        self.assertEqual(reg.select('latest')['operation_id'], 'a')
        self.assertEqual(reg.select('latest-started')['operation_id'], 'c')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            reg.select('active')

    def test_recording_defaults_on_and_explicit_off(self):
        for disabled in (False, True):
            env, state = prepare_job(ROOT, {'mode':'sim','no_record':disabled}, self.env)
            self.assertEqual(env['GRAPHAPI_RECORD'], '0' if disabled else '1')
            self.assertEqual(state['recording'], not disabled)
            self.assertEqual(env['GRAPH_API_AUTOSTART'], '0')
            self.assertIn('config/.runtime/', env['GRAPH_API_CONFIG'])
            Registry(self.work).release(state['operation_id'])
            Path(env['GRAPH_API_CONFIG']).unlink()

    def test_private_modes_preserve_pose_and_replay_selection(self):
        bag = self.work / 'bag'
        bag.mkdir()
        (bag / 'metadata.yaml').write_text('rosbag2_bagfile_information: {}\n')
        for mode, source, expected in [('tiago',None,'config/tiago_robot.yaml'), ('bag',None,'config/tiago_bag.yaml'), ('bag','slam','config/tiago_bag_rtabmap.yaml')]:
            env, state = prepare_job(ROOT, dict(mode=mode,map_source=source,bag=str(bag)), self.env)
            self.assertEqual(state['config_source'], str(ROOT / expected))
            Registry(self.work).release(state['operation_id'])
            Path(env['GRAPH_API_CONFIG']).unlink()

    def test_habitat_command_has_no_host_interpreter(self):
        cmd = command(ROOT, self.env, ['script.py'], habitat=True, gpu=True)
        self.assertIn('/opt/habitat/bin/python', cmd)
        self.assertNotIn('/home/yuri/miniconda3/envs/habitat_env/bin/python', cmd)
        self.assertIn('device=0', cmd)

    def test_schedule_mount_never_masks_writable_workspace(self):
        schedule = self.work / 'schedules' / 'tour.json'
        schedule.parent.mkdir()
        schedule.write_text('{}')
        env = dict(self.env, FEED_SCHEDULE=str(schedule))
        cmd = command(ROOT, env, [], habitat=True)
        self.assertIn(f'{schedule.parent}:{schedule.parent}', cmd)
        self.assertNotIn(f'{schedule.parent}:{schedule.parent}:ro', cmd)

    def test_batch_keeps_exact_children_and_stops_on_failure(self):
        schedule = self.work / 'schedule.yaml'
        schedule.write_text('arms:\n- name: first\n  config: {habitat.hfov_deg: 80}\n- name: second\n  config: {habitat.hfov_deg: 90}\n')
        args = SimpleNamespace(schedule=str(schedule),config=None,local=None,gpus='0',force=False,dry_run=False,continue_on_failure=False)
        with patch('graphapi_cli.batch.start', return_value={'operation_id':'exact-id','bundles':['/exact/bundle'],'state':'FAILED','returncode':3}) as start:
            self.assertEqual(run_batch(ROOT,self.env,args),1)
            self.assertEqual(start.call_count,1)
        manifest = next((self.work / 'results').glob('batch_*.json'))
        row = json.loads(manifest.read_text())['arms'][0]
        self.assertEqual(row['operation_id'],'exact-id')
        self.assertEqual(row['bundles'],['/exact/bundle'])

    def test_same_batch_configuration_requires_declared_replica(self):
        schedule = self.work / 'dupes.yaml'
        schedule.write_text('arms:\n- name: first\n- name: second\n')
        args = SimpleNamespace(schedule=str(schedule),config=None,local=None,gpus='0',force=False,dry_run=True,continue_on_failure=False)
        with self.assertRaisesRegex(ValueError,'replicate_of'):
            run_batch(ROOT,self.env,args)

    def test_terminal_stop_never_signals_unrelated_private_session(self):
        reg = Registry(self.work)
        write_json(reg.operations / 'mine/status.json', dict(operation_id='mine',state='FAILED',created=1,mode='tiago',container='private',session='found_tiago',project=str(ROOT)))
        with patch('graphapi_cli.launch.owned_containers',return_value=[]), patch('graphapi_cli.launch.subprocess.run', return_value=SimpleNamespace(returncode=0,stdout='another-operation\n')) as run:
            stop_operation(self.work,'mine')
        self.assertEqual(run.call_count,1)
        self.assertIn('/ws/output/operation_id',run.call_args.args[0])

    def test_foreground_supervisor_rejects_successful_required_node_death(self):
        from graphapi_cli.launch import execute
        models = self.work / 'models'
        models.mkdir()
        for name in ('l2_encoder.onnx', 'l2_decoder.onnx'):
            (models / name).write_text('test placeholder')
        env, state = prepare_job(ROOT, {'mode':'sim','no_record':True}, dict(self.env, SAM_MODEL_DIR=str(models)))
        bundle = self.work / 'unexpected-child-exit'
        bundle.mkdir()
        (bundle / 'terminating_node.json').write_text('{"node":"PERCEPTION","exit_status":0}')
        Path(env['GRAPHAPI_RESULT_FILE']).write_text(str(bundle)+'\n')
        with patch('graphapi_cli.launch.ensure_image'), patch('graphapi_cli.docker.command',return_value=[sys.executable,'-c','pass']), patch('graphapi_cli.launch.owned_containers',return_value=[]), patch('graphapi_cli.launch.subprocess.run',return_value=SimpleNamespace(returncode=0)):
            result = execute(ROOT,{'mode':'sim','no_record':True},env,state)
        self.assertEqual(result['state'],'FAILED')
        self.assertEqual(result['returncode'],1)
        self.assertFalse((self.work / 'results/latest').exists())
        registry = Registry(self.work)
        self.assertEqual(registry.db.execute('SELECT count(*) FROM leases').fetchone()[0],0)
        Path(env['GRAPH_API_CONFIG']).unlink()

    def test_foreground_completion_requires_flushed_recording(self):
        from graphapi_cli.launch import execute
        models = self.work / 'models'
        models.mkdir()
        for name in ('l2_encoder.onnx', 'l2_decoder.onnx'):
            (models / name).write_text('test placeholder')
        for recorded in (False, True):
            env, state = prepare_job(ROOT, {'mode':'sim'}, dict(self.env, SAM_MODEL_DIR=str(models)))
            bundle = self.work / ('bundle-' + str(recorded))
            bundle.mkdir()
            (bundle / 'terminating_node.json').write_text('{"node":"FEED_ENDED","exit_status":0}')
            if recorded:
                (bundle / 'recording').mkdir()
                (bundle / 'recording/metadata.yaml').write_text('rosbag2_bagfile_information: {}')
                (bundle / 'recording_status.json').write_text('{"closed":true}')
            Path(env['GRAPHAPI_RESULT_FILE']).write_text(str(bundle)+'\n')
            with patch('graphapi_cli.launch.ensure_image'), patch('graphapi_cli.docker.command',return_value=[sys.executable,'-c','pass']), patch('graphapi_cli.launch.owned_containers',return_value=[]), patch('graphapi_cli.launch.subprocess.run',return_value=SimpleNamespace(returncode=0)):
                result = execute(ROOT,{'mode':'sim'},env,state)
            self.assertEqual(result['state'],'COMPLETED' if recorded else 'FAILED')
            Path(env['GRAPH_API_CONFIG']).unlink()
        self.assertEqual((self.work / 'results/latest').resolve(),bundle)

    def test_auxiliary_controllers_do_not_reserve_child_dds_domain(self):
        registry = Registry(self.work)
        op, _, domain = registry.reserve('dashboard',ports={'dashboard':None},need_domain=False)
        self.assertIsNone(domain)
        self.assertFalse(registry.db.execute("SELECT resource FROM leases WHERE resource LIKE 'dds:%'").fetchall())
        registry.release(op)

    def test_named_launch_help_never_executes_private_container(self):
        from graphapi_cli.cli import main
        with patch('graphapi_cli.catalogue.run_tool') as run:
            self.assertEqual(main(['launch','bag-slam','--help']),0)
        run.assert_not_called()

    def test_tool_dry_run_never_provisions_or_launches(self):
        from graphapi_cli.cli import main
        with patch('graphapi_cli.catalogue.run_tool',return_value=0) as run:
            self.assertEqual(main(['tools','run','launch-simulation','--dry-run','--','map_yaml:=/data/map.yaml']),0)
        self.assertTrue(run.call_args.args[-1])
        self.assertEqual(run.call_args.args[-2],['map_yaml:=/data/map.yaml'])

    def test_parallel_helper_uses_cli_gpu_and_config_and_returns_child_failure(self):
        hostbin = self.work / 'bin'
        hostbin.mkdir()
        children = self.work / 'children.jsonl'
        python = hostbin / 'python3'
        python.write_text('#!/usr/bin/python3\n'
                          'import json, os, sys\n'
                          'with open(os.environ["GRAPHAPI_TEST_CHILDREN"], "a") as f:\n'
                          '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n'
                          'raise SystemExit(1 if "hm3d_fail" in sys.argv else 0)\n')
        python.chmod(0o755)
        sleep = hostbin / 'sleep'
        sleep.write_text('#!/bin/bash\nexit 0\n')
        sleep.chmod(0o755)
        config = str(self.work / 'config with spaces.yaml')
        env = dict(self.env, PATH=str(hostbin) + ':' + os.defpath,
                   GRAPHAPI_ROOT=str(ROOT), GRAPH_API_CONFIG=config,
                   GRAPHAPI_TEST_CHILDREN=str(children), LOG_DIR=str(self.work))
        result = subprocess.run(['bash', str(ROOT / 'graphapi_cli/runtime/run_pipelines.sh'),
                                 '0,1', 'hm3d_first', 'hm3d_fail'], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        launched = [json.loads(line) for line in children.read_text().splitlines()]
        expected = [['-m', 'graphapi_cli.cli', 'run', 'sim', '--scene', scene,
                     '--gpu', gpu, '--config', config]
                    for scene, gpu in [('hm3d_first', '0'), ('hm3d_fail', '1')]]
        self.assertCountEqual(launched, expected)

    def test_baseline_options_separator_is_removed_before_calling_native_tool(self):
        from graphapi_cli.cli import main
        with patch('graphapi_cli.catalogue.run_tool', return_value=0) as run:
            result = main(['baseline', 'run', 'clio', '/data/recording', '/data/output',
                           '--', '--rate', '0.5'])
        self.assertEqual(result, 0)
        self.assertEqual(run.call_args.args[3],
                         ['clio', '/data/recording', '/data/output', '--rate', '0.5'])

    def test_help_never_imports_ros_habitat_or_cuda(self):
        result = subprocess.run([sys.executable,'-m','graphapi_cli.cli','--help'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('==SUPPRESS==',result.stdout)
        for command in ('status', 'logs', 'stop', 'attach'):
            self.assertRegex(result.stdout, rf'(?m)^    {command}\s+\S')


if __name__ == '__main__':
    unittest.main()
