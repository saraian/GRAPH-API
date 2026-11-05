"""Docker CLI boundary: paths, host effects, arguments and detached ownership."""
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from graphapi_cli.bootstrap import controller_command, host_action, mount_plan, write_arguments


ROOT = Path(__file__).resolve().parents[2]


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'config').mkdir()
        self.env = {'HOME': str(self.root), 'XDG_CACHE_HOME': str(self.root / 'cache')}
        cache = patch.dict(os.environ, {'XDG_CACHE_HOME': str(self.root / 'cache')})
        cache.start()
        self.addCleanup(cache.stop)

    def test_external_paths_and_missing_output_directory(self):
        outside = self.root / 'external'
        outside.mkdir()
        dataset = outside / 'dataset'
        dataset.mkdir()
        local = self.root / 'config/local.yaml'
        local.write_text(f'paths:\n  dataset: {dataset}\n  workspace: {outside}/new/work\n')
        _, _, args = mount_plan(self.root, ['status'], self.env)
        self.assertIn(f'{dataset}:{dataset}:ro', args)
        self.assertIn(f'{outside}:{outside}', args)

    def test_setup_flags_and_external_local_config(self):
        external = self.root / 'settings'
        external.mkdir()
        local = external / 'machine.yaml'
        local.write_text('paths:\n  workspace: output\nenvironment:\n  REGOLO_API_KEY: local\n')
        selected = self.root / 'selected'
        selected.mkdir()
        _, env, args = mount_plan(self.root, ['--local', str(local), 'setup', 'sim', '--workspace', str(selected)],
                                  dict(self.env, REGOLO_API_KEY='exported'))
        self.assertIn(f'{selected}:{selected}', args)
        self.assertIn(f'{external}:{external}', args)
        self.assertEqual(env['REGOLO_API_KEY'], 'exported')

    def test_argument_files_preserve_shell_literals(self):
        path = self.root / 'args'
        values = ['a b', '$(touch /tmp/never)', 'one\ntwo', '']
        write_arguments(path, values)
        self.assertEqual(path.read_bytes().split(b'\0')[:-1], [v.encode() for v in values])

    def test_editable_local_config_is_not_pinned_by_a_file_mount(self):
        local = self.root / 'config/local.yaml'
        local.write_text('paths: {}\n')
        _, _, args = mount_plan(self.root, ['--local',str(local),'init'], self.env)
        self.assertNotIn(f'{local}:{local}:ro', args)
        self.assertIn(f'{local.parent}:{local.parent}', args)

    def test_operation_file_mount_does_not_make_its_output_directory_read_only(self):
        workspace = self.root / 'workspace'
        directory = workspace / 'results/operations/test'
        directory.mkdir(parents=True)
        job = directory / 'job.json'
        job.write_text('{}')
        _, _, args = mount_plan(self.root, ['_job',str(job)], dict(self.env, WORKSPACE_ROOT=str(workspace)))
        self.assertIn(f'{workspace}:{workspace}', args)
        self.assertNotIn(f'{directory}:{directory}:ro', args)

    def test_host_dds_is_only_selected_for_physical_actions(self):
        env = dict(self.env, TIAGO_NET_IFACE='robot0')
        mode, values, args = host_action(self.root, ['run', 'tiago', 'physical'], env)
        self.assertEqual(mode, 'before')
        self.assertIn('TIAGO_NET_IFACE=robot0', values)
        self.assertEqual(args[-1], 'apply')
        self.assertEqual(env['SKIP_TIAGO_HOST_DDS'], '1')
        for command in (['run','bag','/bags/test'], ['run','tiago','bag','/bags/test'], ['run','tiago'], ['run','tiago','--help'], ['launch','bag-slam','--help']):
            self.assertEqual(host_action(self.root, command, self.env)[0], 'none')
        self.assertEqual(host_action(self.root, ['tiago','network','status'], self.env)[0], 'instead')
        self.assertEqual(host_action(self.root, ['tiago','start'], dict(self.env, TIAGO_BAG_PATH='/bags/test'))[0], 'none')

    def test_detached_supervisor_uses_cli_image_and_actor(self):
        endpoint = self.root / 'docker.sock'
        endpoint.touch()
        env = dict(self.env, GRAPHAPI_DOCKER_SOCKET=str(endpoint), GRAPHAPI_CLI_IMAGE='cli:test', CUDA_VISIBLE_DEVICES='1')
        args = controller_command(self.root, env, ['-m','graphapi_cli.cli','_job','/job.json'],
                                  name='supervisor', operation='exact-id')
        self.assertIn('cli:test', args)
        self.assertNotIn('graphapi-sim:latest', args)
        self.assertIn('graphapi.operation=exact-id', args)
        self.assertIn('GRAPHAPI_RESOURCE_ACTOR=supervisor', args)
        self.assertIn('CUDA_VISIBLE_DEVICES=1', args)
        self.assertIn('--sig-proxy=false', args)

    def test_detached_handoff_waits_for_docker_and_redirects_logs(self):
        from graphapi_cli.launch import start
        from graphapi_cli.registry import write_json
        directory = self.root / 'results/operations/exact-id'
        directory.mkdir(parents=True)
        state = {'operation_id':'exact-id', 'workspace':str(self.root), 'log':str(directory/'launch.log'),
                 'state':'PREPARING', 'actor':'parent', 'pid':7, 'process_identity':'parent-start'}
        write_json(directory/'status.json', state)
        env = dict(self.env, GRAPHAPI_RESOURCE_ACTOR='parent')
        with patch('graphapi_cli.launch.prepare_job',return_value=(env,state)), \
             patch('graphapi_cli.bootstrap.controller_command',return_value=['docker','run','cli:test']) as command, \
             patch('graphapi_cli.launch.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='container-id')) as run, \
             patch('graphapi_cli.launch.subprocess.Popen') as popen:
            result = start(self.root, {'mode':'sim'}, detach=True)
        self.assertIn('-d', run.call_args.args[0])
        self.assertTrue(run.call_args.kwargs['check'])
        self.assertEqual(command.call_args.kwargs['executable'], 'bash')
        self.assertIn('>>"$2"', command.call_args.args[2][1])
        self.assertEqual(result['actor'], 'graphapi-exact-id-supervisor')
        self.assertIsNone(result['pid'])
        popen.assert_not_called()

    def test_pending_detached_actor_retains_ownership(self):
        from graphapi_cli.registry import actor_alive
        with patch('graphapi_cli.registry.container_alive',return_value=True), \
             patch('graphapi_cli.registry.subprocess.run') as run:
            self.assertTrue(actor_alive('supervisor', None))
        run.assert_not_called()

    def test_wrapper_needs_no_python_and_preserves_exit_status(self):
        # The host PATH has shell utilities and a fake Docker CLI, but no Python.
        hostbin = self.root / 'bin'
        hostbin.mkdir()
        import shutil
        for name in ('bash','dirname','cksum','id','mkdir','mktemp','cat','rm','stat'):
            (hostbin / name).symlink_to(shutil.which(name))
        with (ROOT/'docker/cli/Dockerfile').open() as source:
            revision = subprocess.check_output(['cksum'], stdin=source, text=True).strip().replace(' ','-')
        old_venv = (ROOT/'.venv').stat().st_mtime_ns if (ROOT/'.venv').exists() else None
        fake = hostbin / 'docker'
        fake.write_text('''#!/usr/bin/env bash
if [[ $1 == image ]]; then printf '%s\\n' "$GRAPHAPI_TEST_REVISION"; exit 0; fi
for arg; do
  if [[ $arg == *:/plan ]]; then
    plan=${arg%:/plan}
    printf '%s\\0' --network=host > "$plan/docker.args"
    : > "$plan/host.env"; : > "$plan/host.args"
    printf none > "$plan/host.mode"
    exit 0
  fi
done
printf '%s\\n' "$@"
exit 42
''')
        fake.chmod(0o755)
        endpoint = self.root / 'docker.sock'
        connection = socket.socket(socket.AF_UNIX)
        connection.bind(str(endpoint))
        self.addCleanup(connection.close)
        env = dict(self.env, PATH=str(hostbin), DOCKER_HOST='unix://'+str(endpoint), GRAPHAPI_TEST_REVISION=revision)
        result = subprocess.run([str(ROOT/'graphapi'),'tools','list','literal $(false) with spaces'], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertIn('literal $(false) with spaces', result.stdout)
        self.assertIn('graphapi-cli:latest', result.stdout)
        new_venv = (ROOT/'.venv').stat().st_mtime_ns if (ROOT/'.venv').exists() else None
        self.assertEqual(new_venv, old_venv)


if __name__ == '__main__':
    unittest.main()
