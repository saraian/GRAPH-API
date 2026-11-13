"""Late perception must invalidate a tour's already requested final merge."""
import ast
from collections import deque
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock


class LateFinalMerge(unittest.TestCase):
    def test_late_pairs_schedule_full_audit_only_after_final_scan(self):
        path = Path(__file__).resolve().parents[1] / "src/perception_module/object_manager_6.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObjectManagerService")
        cls.bases, cls.decorator_list = [], []
        cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
                    ("_scan_complete_callback", "_schedule_scan_merge", "_try_process")]
        env = dict(json=json, time=time, CFG={"perception": {"frame_queue_max": 8}},
                   ObjectTrackingService=NS(Request=NS, Response=NS), _stamp_key_str=str)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), env)
        node = env["ObjectManagerService"]()
        node.ga493_replay_capture = None
        node._scan_state_lock = threading.RLock()
        node._last_scan_id = node._scan_merge_pending = node._scan_merge_timer = None
        node._scan_merge_seen_ids = deque(maxlen=256)
        node._desc_lag_s = []
        node._scan_settle_s = lambda: 1.
        node._run_pending_scan_merge = lambda: None
        node.scan_merge_callback_group = object()
        node.create_timer = Mock(side_effect=lambda *args, **kw: Mock())
        node.object_services = NS(log_both=Mock())
        node._moving_since = None
        node._processed_pairs = 0
        node.object_tracking_callback = Mock()

        def receive_pair(cycle):
            msg = NS(header=NS(stamp=cycle))
            node._pending_descriptions = {("cycle", cycle): msg}
            node._pending_bboxes = {("cycle", cycle): msg}
            node._try_process()

        receive_pair("before_final")
        node.create_timer.assert_not_called()
        node._scan_complete_callback(NS(data=json.dumps(dict(
            scan_id="last", stops_total=33, laps_total=3, scans_done=99))))
        first_timer = node._scan_merge_timer
        receive_pair("after_final")
        self.assertEqual(node._scan_merge_pending, "last:late:2")
        self.assertTrue(node._scan_merge_pending_full_audit)
        first_timer.cancel.assert_called_once()
        self.assertEqual(node.object_tracking_callback.call_count, 2)
        self.assertEqual(node._pending_bboxes, {})
        self.assertEqual(node._pending_descriptions, {})
