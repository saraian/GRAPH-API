import json
import tempfile
import unittest
from pathlib import Path

from multi_floor_session import (
    ACTIVE,
    COMPLETE,
    LOCALIZING,
    TRANSITION,
    SessionJournal,
    TransitionRefused,
    apply_transforms,
    build_session_specs,
    floor_id,
    parse_floor_sequence,
    validate_rigid_transform,
    write_plan,
)


IDENTITY = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        self.now += 1.0
        return self.now


class MultiFloorSessionTest(unittest.TestCase):
    def test_sequence_retains_revisit_and_selects_localization(self):
        specs = build_session_specs(parse_floor_sequence("-1.59,+1.35,-1.59"), "/tmp/house")
        self.assertEqual([s.floor_id for s in specs], [
            "floor_-1.59", "floor_+1.35", "floor_-1.59"])
        self.assertEqual([s.mode for s in specs], ["mapping", "mapping", "localization"])
        self.assertEqual(specs[2].source_database_path, specs[0].database_path)
        self.assertNotEqual(specs[2].database_path, specs[0].database_path)
        self.assertEqual([s.transform_epoch for s in specs], [0, 1, 2])

    def test_empty_and_non_finite_sequences_refuse(self):
        with self.assertRaises(ValueError):
            parse_floor_sequence("")
        with self.assertRaises(ValueError):
            parse_floor_sequence("nan")
        self.assertEqual(floor_id(-0.0), "floor_+0.00")

    def test_transform_requires_proper_rigid_rotation(self):
        self.assertEqual(validate_rigid_transform(IDENTITY), IDENTITY)
        reflected = [row[:] for row in IDENTITY]
        reflected[0][0] = -1
        with self.assertRaises(ValueError):
            validate_rigid_transform(reflected)

    def test_readiness_and_observation_barriers(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            specs = build_session_specs([-1.59, 1.35], directory)
            journal = SessionJournal(Path(directory) / "coordinator", clock=clock)
            journal.prepare(specs[0])
            self.assertEqual(journal.state["state"], LOCALIZING)
            ready_stamp = clock.now + 10
            evidence = {
                "session_id": specs[0].session_id,
                "floor_id": specs[0].floor_id,
                "map_id": specs[0].map_id,
                "transform_epoch": specs[0].transform_epoch,
                "pose_stamp": ready_stamp,
                "pose_source": "simulator",
                "tf_authorities": ["habitat_feed_node"],
                "quality": {"passed": True, "criterion": "simulator pose available"},
                "building_to_map": IDENTITY,
            }
            journal.mark_ready(evidence)
            journal.activate()
            self.assertEqual(journal.state["state"], ACTIVE)
            active_stamp = journal.state["active_since"] + 1
            self.assertTrue(journal.accept_observation({
                "session_id": specs[0].session_id,
                "floor_id": specs[0].floor_id,
                "transform_epoch": specs[0].transform_epoch,
                "stamp": active_stamp,
            }))
            stale = {
                "session_id": specs[0].session_id,
                "floor_id": specs[0].floor_id,
                "transform_epoch": specs[0].transform_epoch,
                "stamp": journal.state["active_since"] - 1,
            }
            self.assertFalse(journal.accept_observation(stale))
            journal.begin_drain()
            self.assertFalse(journal.accept_observation({**stale, "stamp": active_stamp + 1}))
            journal.finish_visit("a" * 64, more_visits=True)
            self.assertEqual(journal.state["state"], TRANSITION)
            journal.prepare(specs[1])

    def test_wrong_map_duplicate_tf_and_failed_quality_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            spec = build_session_specs([-1.59], directory)[0]
            base = {
                "session_id": spec.session_id,
                "floor_id": spec.floor_id,
                "map_id": spec.map_id,
                "transform_epoch": spec.transform_epoch,
                "pose_stamp": 1000.0,
                "pose_source": "rtabmap",
                "tf_authorities": ["rtabmap"],
                "quality": {"passed": True, "criterion": "covariance"},
            }
            for change in (
                {"map_id": "wrong-map"},
                {"tf_authorities": ["rtabmap", "feed"]},
                {"quality": {"passed": False, "criterion": "covariance"}},
            ):
                journal = SessionJournal(Path(directory) / str(len(change)) / str(change), clock=clock)
                journal.prepare(spec)
                with self.assertRaises(TransitionRefused):
                    journal.mark_ready({**base, **change})

    def test_plan_labels_teleport_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            specs = build_session_specs([-1.59, 1.35, -1.59], directory)
            write_plan(path, specs, "teleport")
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema"], 1)
            self.assertIn("not certified", payload["transport_claim"])
            self.assertEqual(len(payload["sessions"]), 3)

    def test_explicit_transform_file_must_cover_every_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            specs = build_session_specs([-1.59, 1.35], directory)
            path = Path(directory) / "transforms.json"
            path.write_text(json.dumps({"floor_-1.59": IDENTITY}))
            with self.assertRaises(ValueError):
                apply_transforms(specs, path)
            path.write_text(json.dumps({
                "floor_-1.59": IDENTITY,
                "floor_+1.35": IDENTITY,
            }))
            updated = apply_transforms(specs, path)
            self.assertEqual(updated[1].building_to_map[3], (0.0, 0.0, 0.0, 1.0))

    def test_finish_requires_real_digest_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            spec = build_session_specs([-1.59], directory)[0]
            journal = SessionJournal(Path(directory) / "journal", clock=clock)
            journal.prepare(spec)
            with self.assertRaises(TransitionRefused):
                journal.finish_visit("a" * 64, more_visits=False)
            journal.fail("readiness timeout")
            self.assertNotEqual(journal.state["state"], COMPLETE)
            with self.assertRaises(TransitionRefused):
                journal.fail("second failure")

    def test_journal_reopens_without_resetting_active_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            clock = Clock()
            location = Path(directory) / "journal"
            spec = build_session_specs([-1.59], directory)[0]
            journal = SessionJournal(location, clock=clock)
            journal.prepare(spec)
            reopened = SessionJournal(location, clock=clock)
            self.assertEqual(reopened.state["state"], LOCALIZING)
            self.assertEqual(reopened.state["session"]["session_id"], spec.session_id)


if __name__ == "__main__":
    unittest.main()
