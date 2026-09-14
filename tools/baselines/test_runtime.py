"""Exercise the shared script executor through the baseline transport seam."""
import json
import tempfile
import types
import unittest
from pathlib import Path

from tools.baselines.runtime import Session, matching_event, source_module, validate_triggers

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = source_module(ROOT, 'script_runner')


class RuntimeTests(unittest.TestCase):
    def test_shared_executor_orders_real_callbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'script.json'
            p.write_text(json.dumps({'steps': [
                {'action': 'spawn', 'name': 'cup', 'template': 'cup', 'position': [0, 1, 0]},
                {'action': 'move', 'object': 'cup', 'position': [1, 1, 0], 'at_waypoint': {'stop': 1, 'lap': 0}},
                {'action': 'remove', 'object': 'cup', 'at_waypoint': {'stop': 1, 'lap': 1}},
            ]}))
            calls, replies = [], {}
            def publish(action, payload):
                calls.append(action)
                replies[payload['request_id']] = {'success': True, 'action': action, 'object_id': 42}
            def wait(trigger):
                calls.append(('waypoint', trigger))
                return {'success': True, 'event': trigger}
            runner = SCRIPTS.HabitatScriptRunner(Path(tmp), Path(tmp)/'state.json', publish,
                lambda action, timeout, request_id: replies.pop(request_id),
                'spawn', 'move', 'remove', capture_frame=lambda *a: 'real-callback-path.png', wait_waypoint=wait)
            result = runner.run(str(p))
            self.assertTrue(result['success'])
            self.assertEqual(result['active_object_count'], 0)
            self.assertEqual(calls, ['spawn', ('waypoint', {'stop': 1, 'lap': 0}), 'move',
                                     ('waypoint', {'stop': 1, 'lap': 1}), 'remove'])

    def test_missing_and_backwards_trigger_fail(self):
        schedule = {'trajectory': [{'stop': 0, 'scan_deg': 360}, {'stop': 1, 'scan_deg': 360}]}
        for triggers in ([{'stop': 9}], [{'stop': 1, 'lap': 1}, {'stop': 0, 'lap': 0}]):
            script = {'steps': [{'action': 'move', 'at_waypoint': t} for t in triggers]}
            with self.assertRaises(ValueError):
                validate_triggers(SCRIPTS.HabitatScriptRunner, script, schedule, 2)

    def test_same_waypoint_can_trigger_consecutive_actions(self):
        validate_triggers(SCRIPTS.HabitatScriptRunner,
            {'steps': [{'action': 'move', 'at_waypoint': 0}] * 2},
            {'trajectory': [{'stop': 0, 'scan_deg': 360}]}, 1)

    def test_finished_tour_cannot_satisfy_unseen_trigger(self):
        session = object.__new__(Session)
        session.setup_dispatch()
        session.last_event = {'stop': 0, 'lap': 0}
        session.tour = types.SimpleNamespace(house_done=True)
        with self.assertRaises(RuntimeError):
            session.wait_waypoint({'stop': 1})

    def test_action_frame_does_not_re_emit_the_trigger_scan_event(self):
        """A post-action frame is not a scan completion.

        It used to carry a copy of the trigger scan's scan_complete event, so
        scan_windows saw one scan twice and refused the whole recording.
        """
        from concurrent.futures import Future
        session = object.__new__(Session)
        session.setup_dispatch()
        session.last_event = {'event': 'scan_complete', 'lap': 0, 'stop': 0}
        session.sim = session.agent = None
        seen = []
        session.recording = types.SimpleNamespace(
            capture=lambda sim, agent, reason, event=None: seen.append((reason, event)))
        future = Future()
        session.requests.put(('capture', 'spawn', future))
        session.dispatch()
        self.assertEqual(seen, [('object:spawn', None)])

    def test_time_wait_does_not_stop_the_tour(self):
        import threading
        with tempfile.TemporaryDirectory() as tmp:
            tour = types.SimpleNamespace(house_done=False)
            session = object.__new__(Session)
            session.tour = tour
            session.last_event = None
            session.responses, session.actions = {}, []
            session.dynamic_instance_count, session.evaluation_ids = 0, {}
            session.recording = types.SimpleNamespace(fps=100, index=0, output=Path(tmp),
                                                       dynamic_semantic_id_offset=0,
                                                       evaluation_ids=session.evaluation_ids)
            session.recording.capture = lambda *a: 'frame.png'
            session.sim = types.SimpleNamespace(step_physics=lambda dt: None)
            session.agent = None
            owner = threading.get_ident()
            def tick():
                self.assertEqual(threading.get_ident(), owner)
                session.recording.index += 1
                if session.recording.index >= 8:
                    tour.house_done = True
            session.tick = tick
            session.controller = types.SimpleNamespace(execute=lambda payload: {
                'action': payload['action'], 'success': True, 'object_id': 1})
            script = Path(tmp) / 'timed.json'
            script.write_text(json.dumps({'steps': [
                {'action': 'wait', 'seconds': 0.04},
                {'action': 'spawn', 'name': 'box', 'template': 'box', 'position': [0, 0, 0]},
            ]}))
            runner = SCRIPTS.HabitatScriptRunner(Path(tmp), Path(tmp)/'state.json',
                session.publish, session.wait_result, 'spawn', 'move', 'remove',
                capture_frame=session.capture, wait_waypoint=session.wait_waypoint)
            outcome = session.drive(runner, str(script))
            self.assertTrue(outcome['success'])
            self.assertGreaterEqual(session.actions[0]['after_frame'], 2)
            self.assertTrue(tour.house_done)

    def test_lap_is_part_of_trigger(self):
        self.assertFalse(matching_event({'stop': 1, 'lap': 0}, {'stop': 1, 'lap': 1}))
        self.assertTrue(matching_event({'stop': 1, 'lap': 0}, {'stop': 1}))


if __name__ == '__main__':
    unittest.main()
