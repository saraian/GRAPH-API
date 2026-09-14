#!/usr/bin/env python3
"""Offline checks for the Phase A observation transport contract."""

import importlib.util
import pathlib
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "perception_module"


def load_detection_types():
    spec = importlib.util.spec_from_file_location(
        "ga493_detection_types", MODULE / "detection_types.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_detection_archive():
    spec = importlib.util.spec_from_file_location(
        "ga493_detection_archive", MODULE / "detection_archive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stamp:
    def __init__(self, sec=0, nanosec=0):
        self.sec = sec
        self.nanosec = nanosec


class ObservationMessage:
    def __init__(self):
        self.schema_version = 0
        self.run_id = ""
        self.producer_id = ""
        self.capture_id = ""
        self.capture_identity_kind = ""
        self.capture_stamp = Stamp()
        self.camera_frame_id = ""
        self.cycle_id = ""
        self.detection_index = 0
        self.observation_id = ""


class ObservationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.types = load_detection_types()

    def test_identity_uses_pipeline_coordinates_only(self):
        first = self.types.make_observation_ref(
            "run", "producer", "cycle", 3, Stamp(12, 34), "camera")
        second = self.types.make_observation_ref(
            "run", "producer", "cycle", 3, Stamp(12, 34), "camera")
        self.assertEqual(first, second)
        self.assertEqual(first.observation_id, "producer:cycle:3")
        self.assertEqual(first.capture_id, "producer:12:34")
        self.assertEqual(first.capture_identity_kind, "stamp_within_producer")
        self.assertNotIn("label", first.as_dict())
        self.assertNotIn("bbox", first.as_dict())

    def test_dropped_first_output_does_not_renumber_second(self):
        refs = [
            self.types.make_observation_ref(
                "", "p", "c", index, Stamp(1, 2), "camera")
            for index in range(2)
        ]
        surviving = refs[1:]
        self.assertEqual(surviving[0].detection_index, 1)
        self.assertEqual(surviving[0].observation_id, "p:c:1")

    def test_message_round_trip_and_legacy_none(self):
        ref = self.types.make_observation_ref(
            "run", "p", "c", 7, Stamp(8, 9), "frame")
        message = ObservationMessage()
        self.types.write_observation_msg(message, ref)
        self.assertEqual(self.types.observation_dict_from_msg(message), ref.as_dict())
        self.assertEqual(
            self.types.observation_dict_from_msg(ref.as_dict()), ref.as_dict())
        legacy = ObservationMessage()
        self.assertIsNone(self.types.observation_dict_from_msg(legacy))

    def test_ros_contract_and_dependencies_are_registered(self):
        observation = (ROOT / "msg" / "ObservationRef.msg").read_text()
        self.assertIn("builtin_interfaces/Time capture_stamp", observation)
        for name in ("Bbox3d.msg", "ObjectDescription.msg"):
            self.assertIn(
                "ObservationRef observation", (ROOT / "msg" / name).read_text())
        for name in ("AddObject.srv", "UpdateObject.srv"):
            service = (ROOT / "srv" / name).read_text()
            for field in (
                "ObservationRef observation",
                "ObservationRef description_observation",
                "string observation_attempt_id",
                "string mutation_event_id",
                "string mutation_state",
                "uint64 object_revision",
                "uint64 geometry_epoch",
            ):
                self.assertIn(field, service)
        cmake = (ROOT / "CMakeLists.txt").read_text()
        self.assertIn('"msg/ObservationRef.msg"', cmake)
        self.assertIn("find_package(builtin_interfaces REQUIRED)", cmake)
        dependencies = [node.text for node in ET.parse(ROOT / "package.xml").findall("depend")]
        self.assertIn("builtin_interfaces", dependencies)

    def test_both_producers_assign_before_archive_and_publish(self):
        for name in ("perception_2.py", "perception_parallel.py"):
            source = (MODULE / name).read_text()
            assigned = source.index("det.observation = make_observation_ref")
            archived = source.index("self.save_visualizations", assigned)
            published = source.index("self._publish_bbox_array", assigned)
            self.assertLess(assigned, archived)
            self.assertLess(assigned, published)
            self.assertRegex(
                source, r"write_observation_msg\(\s*box_msg\.observation")
            self.assertRegex(
                source, r"write_observation_msg\(\s*obj_msg\.observation")

    def test_partial_receipts_cross_http_failure_boundary(self):
        bridge = (MODULE / "graph_api_bridge.py").read_text()
        manager = (MODULE / "object_manager_6.py").read_text()
        services = (MODULE / "object_services.py").read_text()
        self.assertGreaterEqual(
            bridge.count('"mutation_state": res.mutation_state'), 4)
        self.assertIn("error.payload = payload", manager)
        self.assertIn('failure.get("mutation_state", "incomplete")', manager)
        self.assertIn('"applied_in_memory"', services)
        self.assertIn('"applied_and_checkpointed"', services)
        self.assertIn("os.fsync(ledger.fileno())", services)
        self.assertIn("def record_direct_mutation", services)
        self.assertIn("self.object_services.record_direct_mutation", manager)

    def test_non_detection_events_are_append_only_and_counted(self):
        archive_module = load_detection_archive()
        with tempfile.TemporaryDirectory() as root:
            archive = archive_module.DetectionArchive(
                root, enabled=True, save_rgb=False)
            self.assertTrue(archive.record_event(
                "cycle_completed", frame_id="1.2", cycle_id="c",
                outcome="valid_empty", detection_count=0))
            self.assertTrue(archive.record_event(
                "capture_queue_evicted", frame_id="1.1", queue_depth=8))
            rows = [
                __import__("json").loads(line)
                for line in pathlib.Path(archive._path).read_text().splitlines()
            ]
            self.assertEqual([row["event"] for row in rows], [
                "cycle_completed", "capture_queue_evicted"])
            self.assertEqual(archive.stats()["events"], 2)
            archive._path = root
            self.assertFalse(archive.record_event("cannot_write"))
            self.assertFalse(archive.stats()["provenance_complete"])

    def test_fast_discard_sites_have_machine_readable_events(self):
        manager = (MODULE / "object_manager_6.py").read_text()
        for name in ("perception_2.py", "perception_parallel.py"):
            source = (MODULE / name).read_text()
            self.assertIn('"capture_queue_evicted"', source)
            self.assertIn('outcome="valid_empty"', source)
            self.assertIn('reason="motion_during_detection"', source)
        self.assertIn('"unpaired_description"', manager)
        self.assertIn('"sync_buffer_evicted"', manager)
        self.assertIn('reason="manager_motion_gate"', manager)
        self.assertIn('reason="observed_during_motion"', manager)


if __name__ == "__main__":
    unittest.main(verbosity=2)
