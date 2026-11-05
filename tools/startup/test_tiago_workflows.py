"""TIAGO command boundaries: explicit inputs, profiles and operator output."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from graphapi_cli import cli
from graphapi_cli.bootstrap import host_action
from graphapi_cli.catalogue import catalogue, launch_name
from graphapi_cli.configuration import ConfigurationError, tiago_map_source
from graphapi_cli.launch import prepare_job
from graphapi_cli.registry import Registry

ROOT = Path(__file__).resolve().parents[2]


class TiagoWorkflowsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.local = self.work / 'local.yaml'
        self.local.write_text('{}\n')
        self.bag = self.work / 'recorded-tiago'
        self.bag.mkdir()
        (self.bag / 'metadata.yaml').write_text('rosbag2_bagfile_information: {}\n')
        clean = patch.dict(os.environ, {'XDG_CACHE_HOME': str(self.work / 'cache')}, clear=True)
        clean.start()
        self.addCleanup(clean.stop)
        self.env = {'WORKSPACE_ROOT': str(self.work), 'XDG_CACHE_HOME': str(self.work / 'cache'),
                    'GRAPHAPI_PAL_CONTAINER': 'private-test-container', 'REGOLO_API_KEY': 'test-key'}

    def prepare(self, mode, **options):
        inherited = dict(self.env, **options.pop('inherited', {}))
        request = dict(mode=mode, local=str(self.local), **options)
        if mode == 'bag':
            request['bag'] = str(self.bag)
        env, state = prepare_job(ROOT, request, inherited)
        self.addCleanup(Registry(self.work).release, state['operation_id'])
        self.addCleanup(Path(state['config']).unlink, missing_ok=True)
        return request, env, state

    def test_physical_command_has_no_bag_input(self):
        args = cli.parser().parse_args(['run', 'tiago', 'physical', '--map-source', 'slam'])
        self.assertEqual((args.mode, args.tiago_workflow), ('tiago', 'physical'))
        self.assertFalse(hasattr(args, 'bag'))

    def test_bag_command_keeps_playback_options(self):
        args = cli.parser().parse_args(['run', 'tiago', 'bag', str(self.bag), '--map-source', 'slam', '--rate', '0.5', '--loop'])
        self.assertEqual((args.mode, args.tiago_workflow, args.map_source, args.rate, args.loop), ('bag', 'bag', 'slam', 0.5, True))

    def test_bare_tiago_rejects_implicit_physical_launch(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            cli.parser().parse_args(['run', 'tiago'])
        self.assertEqual(raised.exception.code, 2)

    def test_old_bag_spelling_keeps_explicit_tiago_identity(self):
        args = cli.parser().parse_args(['run', 'bag', str(self.bag)])
        self.assertEqual((args.mode, args.tiago_workflow), ('bag', 'bag'))

    def test_main_dispatches_nested_bag_without_robot_actions(self):
        with patch('graphapi_cli.launch.start', return_value={'operation_id': 'test', 'mode': 'bag'}) as start, contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main(['--local', str(self.local), 'run', 'tiago', 'bag', str(self.bag), '--container', 'chosen-private', '--detach'])
        self.assertEqual(code, 0)
        request = start.call_args.args[1]
        self.assertEqual((request['mode'], request['bag']), ('bag', str(self.bag)))
        self.assertEqual(start.call_args.kwargs['inherited']['GRAPHAPI_PAL_CONTAINER'], 'chosen-private')
        self.assertEqual(json.loads(out.getvalue())['mode'], 'tiago-bag')

    def test_physical_defaults_and_saved_identity_are_explicit(self):
        request, env, state = self.prepare('tiago')
        self.assertEqual((request['map_source'], env['FOUND_START_RTABMAP']), ('slam', '1'))
        self.assertEqual((state['workflow'], state['runtime'], state['container']), ('tiago-physical', 'private-pal', 'private-test-container'))
        self.assertEqual(Path(state['config_source']).name, 'tiago_robot.yaml')
        self.assertEqual((state['map_frame'], state['map_topic']), ('found_map', '/rtabmap/map'))
        self.assertEqual(state['robot_address'], '10.68.0.1')
        self.assertTrue(state['perception'] and state['recording'])

    def test_recorded_bag_defaults_are_explicit(self):
        _, env, state = self.prepare('bag')
        self.assertEqual((state['workflow'], state['map_source'], env['FOUND_START_RTABMAP']), ('tiago-bag', 'recorded', '0'))
        self.assertEqual(Path(state['config_source']).name, 'tiago_bag.yaml')
        self.assertEqual((state['map_frame'], state['map_topic'], state['bag']), ('map', '/map', str(self.bag)))
        self.assertNotIn('robot_address', state)

    def test_inherited_slam_setting_selects_matching_bag_profile(self):
        _, env, state = self.prepare('bag', inherited={'FOUND_START_RTABMAP': '1'})
        self.assertEqual(Path(state['config_source']).name, 'tiago_bag_rtabmap.yaml')
        self.assertEqual((state['map_source'], env['TIAGO_BAG_FILTER_CONFLICTING']), ('slam', '1'))

    def test_explicit_recorded_source_overrides_stale_slam_and_filter(self):
        _, env, state = self.prepare('bag', map_source='recorded', inherited={'FOUND_START_RTABMAP': '1', 'TIAGO_BAG_FILTER_CONFLICTING': '1'})
        self.assertEqual((state['map_source'], env['FOUND_START_RTABMAP'], env['TIAGO_BAG_FILTER_CONFLICTING']), ('recorded', '0', '0'))

    def test_physical_selection_clears_inherited_bag_input(self):
        _, env, state = self.prepare('tiago', inherited={'TIAGO_BAG_PATH': str(self.bag), 'TIAGO_ROSBAG': str(self.bag)})
        self.assertNotIn('TIAGO_BAG_PATH', env)
        self.assertNotIn('TIAGO_ROSBAG', env)
        self.assertEqual(env['TIAGO_BAG_MODE'], '0')
        self.assertEqual(host_action(ROOT, ['run', 'tiago', 'physical'], {'TIAGO_BAG_PATH': str(self.bag)})[0], 'before')
        self.assertEqual(state['workflow'], 'tiago-physical')

    def test_perception_and_recording_opt_out_match_status(self):
        request, env, state = self.prepare('bag', no_record=True, inherited={'FOUND_START_PERCEPTION': '0'})
        self.assertTrue(request['no_perception'])
        self.assertFalse(state['perception'] or state['recording'])
        self.assertEqual(env['GRAPHAPI_RECORD'], '0')

    def test_wrong_input_map_source_is_rejected(self):
        for mode, source in [('tiago', 'recorded'), ('bag', 'robot')]:
            with self.subTest(mode=mode), self.assertRaises(ConfigurationError):
                tiago_map_source(mode, source, {})

    def test_status_and_json_expose_tiago_for_historical_rows(self):
        row = {'operation_id': 'old', 'mode': 'bag', 'state': 'RUNNING', 'bag': str(self.bag), 'map_source': 'recorded'}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cli._show_status(row)
        self.assertIn('tiago-bag', out.getvalue())
        self.assertIn('recorded-tiago', out.getvalue())
        self.assertIn('Map source: recorded', out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cli._emit(row, True)
        self.assertEqual(json.loads(out.getvalue())['mode'], 'tiago-bag')
        self.assertEqual(row['mode'], 'bag')  # Persisted selectors remain compatible.

    def test_private_doctor_uses_tiago_profile_and_reports_its_defaults(self):
        for mode, profile, source in [('tiago-physical', 'tiago_robot.yaml', 'slam'), ('tiago-bag', 'tiago_bag.yaml', 'recorded')]:
            with self.subTest(mode=mode), patch('graphapi_cli.cli.subprocess.run', return_value=SimpleNamespace(returncode=0)), contextlib.redirect_stdout(io.StringIO()) as out:
                code = cli.main(['--local', str(self.local), 'doctor', '--mode', mode])
            checks = json.loads(out.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual((checks['workflow'], Path(checks['config']).name, checks['map_source']), (mode, profile, source))
            self.assertNotIn('default_pose', checks)
            self.assertEqual(checks['runtime'], 'private-pal')

    def test_explicit_auxiliary_names_resolve_to_original_launch_files(self):
        entries = catalogue(ROOT)
        for name, old in [('tiago-gazebo', 'simulation'), ('tiago-navigation', 'simulation2'), ('tiago-bag-slam', 'bag-slam'), ('tiago-bag-slam-1', 'bag-slam-1'), ('tiago-bag-slam-2', 'bag-slam-2')]:
            self.assertEqual(entries[launch_name(name)], entries[launch_name(old)])
            with contextlib.redirect_stdout(io.StringIO()) as out:
                cli._emit({'operation_id': 'historical', 'mode': launch_name(old)}, True)
            self.assertEqual(json.loads(out.getvalue())['mode'], name)

    def test_auxiliary_help_does_not_launch_private_or_public_containers(self):
        with patch('graphapi_cli.catalogue.run_tool') as run, contextlib.redirect_stdout(io.StringIO()) as out:
            code = cli.main(['--local', str(self.local), 'launch', 'tiago-bag-slam', '--help'])
        self.assertEqual(code, 0)
        run.assert_not_called()
        self.assertIn('graphapi launch tiago-bag-slam', out.getvalue())
        self.assertIn('Environment: pal', out.getvalue())
