#!/usr/bin/env python3
"""confirmation.py: sightings are counted where every applied observation passes, survive a
move and a merge, and tentative objects reach no consumer (markers, saved map, relations).
Off (the default) changes nothing. Run: python3 test_confirmation.py
"""
import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rosstub  # noqa: E402

rosstub.install()
import confirmation  # noqa: E402
import cv_utils  # noqa: E402
import object_services  # noqa: E402
from object_info import Object  # noqa: E402
from world_model import wm  # noqa: E402

BOX = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}


def obs(cycle):
    return {"schema_version": 1, "run_id": "r", "producer_id": "p", "capture_id": "c",
            "capture_identity_kind": "k", "capture_stamp": {"sec": 1, "nanosec": 0},
            "camera_frame_id": "cam", "cycle_id": cycle, "detection_index": 0,
            "observation_id": f"p:{cycle}:0"}


def obj(oid, label="chair"):
    o = Object(label, (0.5, 0.5, 0.5), dict(BOX))
    o.object_id = oid
    return o


class Log:
    def __init__(self):
        self.rows = []

    def write(self, kind, subject, **fields):
        self.rows.append((kind, subject, fields))


def service(tmp):
    svc = object_services.ObjectServices.__new__(object_services.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    svc.log_both = lambda *a, **k: None
    svc.decision_log = Log()
    svc._mutation_sequence = 0
    svc._provenance_incomplete = False
    svc._mutation_ledger_path = os.path.join(tmp, "mutation_receipts.jsonl")
    return svc


def receipt(svc, o, cycle, state="applied_in_memory"):
    svc._write_mutation_receipt("e", SimpleNamespace(observation=obs(cycle)), state, "update", obj=o)


def off_changes_nothing():
    confirmation.CONFIRM_MIN_VIEWS = None
    objects = [obj("a")]
    assert confirmation.shown(objects) is objects
    assert confirmation.note_sighting(objects[0], "c1", 0.0) is False
    assert not hasattr(objects[0], "sighting_cycles")
    with tempfile.TemporaryDirectory() as tmp:
        svc = service(tmp)
        receipt(svc, objects[0], "c1")
        assert svc.decision_log.rows == []
        wm.persistent_perceptions[:] = objects
        old = os.environ.get("GRAPH_API_OUTPUT_DIR")
        os.environ["GRAPH_API_OUTPUT_DIR"] = tmp
        try:
            object_services.save_persistent_perceptions(svc)
        finally:
            if old is None:
                os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            else:
                os.environ["GRAPH_API_OUTPUT_DIR"] = old
        saved = json.loads(pathlib.Path(tmp, "persistent_perception.json").read_text())
        assert [e["object_id"] for e in saved] == ["a"]
        assert not {"sightings", "confirmed", "confirmed_at", "confirm_min_views"} & set(saved[0])
        assert not pathlib.Path(tmp, "tentative_perception.json").exists()
        wm.persistent_perceptions.clear()


def sightings_from_receipts():
    confirmation.CONFIRM_MIN_VIEWS = 2
    with tempfile.TemporaryDirectory() as tmp:
        svc = service(tmp)
        a = obj("a")
        receipt(svc, a, "c1")
        receipt(svc, a, "c1", state="applied_and_checkpointed")  # second stage, same cycle
        receipt(svc, a, "c2", state="refused")                    # not applied
        receipt(svc, a, "c1")                                     # same cycle again
        assert not confirmation.is_shown(a) and len(a.sighting_cycles) == 1
        receipt(svc, a, "c2")
        assert a.confirmed and a.confirmed_at["cycle_id"] == "c2"
        assert [r[0] for r in svc.decision_log.rows] == ["object_confirmed"]
        assert svc.decision_log.rows[0][2]["sightings"] == 2


def merge_and_move_keep_state():
    confirmation.CONFIRM_MIN_VIEWS = 2
    keeper, confirmed = obj("k"), obj("d")
    keeper.sighting_cycles = {"c1"}
    confirmed.sighting_cycles, confirmed.confirmed = {"c2", "c3"}, True
    confirmed.confirmed_at = {"by": "sighting"}
    assert confirmation.absorb(keeper, confirmed, 1.0) and keeper.confirmed
    assert keeper.sighting_cycles == {"c1", "c2", "c3"} and keeper.confirmed_at["inherited_from"] == "d"
    a, b = obj("a"), obj("b")
    a.sighting_cycles, b.sighting_cycles = {"c1"}, {"c2"}
    assert confirmation.absorb(a, b, 2.0) and a.confirmed_at["by"] == "merge"
    c, d = obj("c"), obj("d2")
    c.sighting_cycles, d.sighting_cycles = {"c1"}, {"c1"}
    assert not confirmation.absorb(c, d, 3.0) and not confirmation.is_shown(c)
    moved = obj("k")
    for attr in confirmation.CARRY_ATTRS:  # the move path copies exactly these
        setattr(moved, attr, getattr(keeper, attr))
    assert confirmation.is_shown(moved) and moved.sighting_cycles == keeper.sighting_cycles


def tentative_objects_reach_no_consumer():
    confirmation.CONFIRM_MIN_VIEWS = 2
    a, b, c = obj("a"), obj("b", "lamp"), obj("c", "table")
    for o, cycles in ((a, {"c1", "c2"}), (b, {"c1"}), (c, {"c3", "c4"})):
        o.sighting_cycles = cycles
        o.confirmed = len(cycles) >= 2
    a.relations = {"isNextTo": {"b", "c"}}
    wm.persistent_perceptions[:] = [a, b, c]
    old = os.environ.get("GRAPH_API_OUTPUT_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["GRAPH_API_OUTPUT_DIR"] = tmp
        try:
            object_services.save_persistent_perceptions(service(tmp))
            saved = json.loads(pathlib.Path(tmp, "persistent_perception.json").read_text())
            tentative = json.loads(pathlib.Path(tmp, "tentative_perception.json").read_text())
        finally:
            if old is None:
                os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            else:
                os.environ["GRAPH_API_OUTPUT_DIR"] = old
    assert sorted(e["object_id"] for e in saved) == ["a", "c"], saved
    assert [e["object_id"] for e in tentative] == ["b"] and tentative[0]["sightings"] == 1
    assert next(e for e in saved if e["object_id"] == "a")["relations"]["isNextTo"] == ["c"]
    assert all(e["confirm_min_views"] == 2 for e in saved + tentative)

    seen = []
    cv_utils._publish_bbox_markers = lambda node, objects, *a, **k: seen.append([o.object_id for o in objects])
    cv_utils._publish_centroid_markers = lambda node, objects, *a, **k: seen.append([o.object_id for o in objects])
    cv_utils.publish_persistent_bboxes(rosstub.Any(), wm, rosstub.Any())
    cv_utils.publish_persistent_centroids(rosstub.Any(), wm, rosstub.Any())
    assert seen == [["a", "c"], ["a", "c"]], seen

    # Under rosstub MarkerArray.markers keeps nothing, so the service-side publisher is observed
    # where it places each object's box: once per shown object.
    placed = []
    object_services.set_marker_from_bbox = lambda marker, bbox: placed.append(marker.id) or True
    object_services.publish_persistent_bboxes(rosstub.Any(), wm, rosstub.Any())
    expected = [object_services.stable_bbox_marker_id(o, i)
                for i, o in enumerate(wm.persistent_perceptions) if o.confirmed]
    assert placed == expected, (placed, expected)
    wm.persistent_perceptions.clear()


if __name__ == "__main__":
    failed = 0
    for test in (off_changes_nothing, sightings_from_receipts, merge_and_move_keep_state,
                 tentative_objects_reach_no_consumer):
        try:
            test()
            print(f"  ok    {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - reporting
            failed += 1
            print(f"  FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    confirmation.CONFIRM_MIN_VIEWS = None
    sys.exit(1 if failed else 0)
