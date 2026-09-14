import json
import tempfile
import time
import unittest
from pathlib import Path

from multi_floor_verify_actions import append_record, wait_active


class MultiFloorVerifyActionsTest(unittest.TestCase):
    def test_wait_active_requires_exact_visit(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state_path.write_text(json.dumps({
                "state": "ACTIVE", "session": {"visit_index": 2}
            }))
            with self.assertRaises(TimeoutError):
                wait_active(state_path, 1, time.monotonic() + 0.01)

    def test_action_record_carries_session_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "actions.jsonl"
            state = {"session": {"visit_index": 0, "session_id": "visit-000-floor_-2.60",
                                 "floor_id": "floor_-2.60", "transform_epoch": 0}}
            append_record(output, state, {"action": "spawn"},
                          {"success": True, "object_id": 7})
            record = json.loads(output.read_text())
            self.assertEqual(record["session_id"], "visit-000-floor_-2.60")
            self.assertEqual(record["result"]["object_id"], 7)


if __name__ == "__main__":
    unittest.main()
