import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from persistent_habitat_feed import (
    ObservationDrainBarrier,
    latest_perception_queue_depth,
    require_complete_drain,
    wait_for_record,
)


class PersistentFeedBarrierTest(unittest.TestCase):
    def test_queue_depth_reader_uses_latest_complete_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "perception_latencies.jsonl"
            path.write_bytes(
                b'{"cycle": 1, "queue_depth": 4}\n'
                b'{"cycle": 2, "queue_depth": 0}\n'
                b'{"cycle": 3, "queue_depth":'
            )
            self.assertEqual(latest_perception_queue_depth(path), 0)
            self.assertIsNone(latest_perception_queue_depth(path.with_name("missing")))

    def test_complete_drain_is_required(self):
        require_complete_drain(
            {"drain_complete": True, "drain_queue_start": 8, "drain_queue_end": 0},
            "visit-000",
        )
        for marker in (
            {"drain_complete": False, "drain_queue_start": 8, "drain_queue_end": 2},
            {"drain_complete": None, "drain_queue_start": None, "drain_queue_end": None},
        ):
            with self.assertRaisesRegex(RuntimeError, "without a measured complete drain"):
                require_complete_drain(marker, "visit-000")

    def test_drain_barrier_requires_stable_zero_and_times_out_unknown(self):
        readings = iter((3, 0, 0))
        barrier = ObservationDrainBarrier(lambda: next(readings), 10.0, 2.0)
        self.assertEqual(
            barrier.sample(100.0, 100.0),
            {"queue_depth": 3, "complete": False, "timed_out": False},
        )
        self.assertEqual(
            barrier.sample(101.0, 100.0),
            {"queue_depth": 0, "complete": False, "timed_out": False},
        )
        self.assertEqual(
            barrier.sample(103.0, 100.0),
            {"queue_depth": 0, "complete": True, "timed_out": False},
        )
        unknown = ObservationDrainBarrier(lambda: None, 5.0, 1.0)
        self.assertEqual(
            unknown.sample(206.0, 200.0),
            {"queue_depth": None, "complete": False, "timed_out": True},
        )

    def test_wait_ignores_previous_visit_until_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "activate.json"
            path.write_text(json.dumps({"visit_index": 0}))

            def replace():
                time.sleep(0.05)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps({"visit_index": 1, "session_id": "visit-1"}))
                temporary.replace(path)

            thread = threading.Thread(target=replace)
            thread.start()
            record = wait_for_record(path, 1, 1.0, "activation")
            thread.join()
            self.assertEqual(record["session_id"], "visit-1")

    def test_wait_times_out_when_identity_never_advances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "closed.json"
            path.write_text(json.dumps({"visit_index": 0}))
            with self.assertRaises(TimeoutError):
                wait_for_record(path, 1, 0.02, "closed database")


class DrainScopeTest(unittest.TestCase):
    """RULE 79. `drain_complete` must say which of its two halves actually asserted.

    `habitat_feed_host` cannot be imported on the host -- it pulls in ROS and Habitat -- so
    this reads the real source and evaluates the real expression, rather than restating the
    mapping in the test and proving only that the test agrees with itself.
    """

    SOURCE = Path(__file__).with_name("habitat_feed_host.py")

    def _expression(self):
        text = self.SOURCE.read_text(encoding="utf-8")
        marker = 'feed_stats["drain_scope"] = ('
        start = text.index(marker) + len(marker) - 1
        depth = 0
        for offset, character in enumerate(text[start:]):
            depth += (character == "(") - (character == ")")
            if depth == 0:
                return text[start:start + offset + 1]
        raise AssertionError("unbalanced drain_scope expression")

    def _evaluate(self, strict_drain, settle_pending_end):
        return eval(self._expression(), {}, {  # noqa: S307 - the repo's own source, not input
            "strict_drain": strict_drain, "settle_pending_end": settle_pending_end})

    def test_no_barrier_reports_no_scope(self):
        self.assertIsNone(self._evaluate(None, None))
        self.assertIsNone(self._evaluate(None, 0))

    def test_both_halves_measured_is_named_as_such(self):
        for pending in (0, 3):
            self.assertEqual(self._evaluate(object(), pending),
                             "queue_and_merges_both_measured")

    def test_unavailable_merge_signal_says_that_half_asserts_nothing(self):
        scope = self._evaluate(object(), None)
        self.assertIn("asserts nothing", scope)
        self.assertNotEqual(scope, "queue_and_merges_both_measured")

    def test_the_marker_actually_carries_the_key(self):
        text = self.SOURCE.read_text(encoding="utf-8")
        self.assertIn('"drain_scope": feed_stats["drain_scope"],', text)

    def test_the_verdict_is_not_gated_on_the_merge_signal(self):
        """Guard against a later edit turning a scope note into a refusal.

        Failing the drain on an unmeasured merge signal would refuse every legacy-engine
        certification -- a new failure, not a fix (rule 15).
        """
        text = self.SOURCE.read_text(encoding="utf-8")
        verdict = text[text.index('feed_stats["drain_complete"] = ('):]
        verdict = verdict[:verdict.index("\n    #")]
        self.assertNotIn("settle_pending_end", verdict)


if __name__ == "__main__":
    unittest.main()
