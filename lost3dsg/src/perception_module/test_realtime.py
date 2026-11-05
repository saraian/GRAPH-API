#!/usr/bin/env python3
"""The real-time fixes change speed, not results: each fast path is checked against the old one.
Run: python3 test_realtime.py   (host, under rosstub)
"""
import array
import math
import pathlib
import random
import struct
import sys
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rosstub  # noqa: E402

rosstub.install()
import object_services  # noqa: E402


def boxes(n, seed):
    rnd = random.Random(seed)
    out = []
    for k in range(n):
        x, y, z = rnd.uniform(0, 8), rnd.uniform(0, 8), rnd.uniform(0, 2)
        sx, sy, sz = (rnd.choice([0.0, rnd.uniform(0.05, 1.5)]) for _ in range(3))
        if k % 7 == 0 and out:   # exactly touching / nested / stacked neighbours
            b = out[-1].bbox
            x, y = b["x_max"] + sx / 2, (b["y_min"] + b["y_max"]) / 2
            z = b["z_max"] + rnd.choice([0.0, 0.05, 0.3]) + sz / 2
        out.append(SimpleNamespace(object_id=f"o{k}", bbox={
            "x_min": x - sx / 2, "x_max": x + sx / 2, "y_min": y - sy / 2,
            "y_max": y + sy / 2, "z_min": z - sz / 2, "z_max": z + sz / 2}))
    out.append(SimpleNamespace(object_id="nobox", bbox=None))
    return out


def relations_full(objs):
    return {r for a in objs for b in objs if a is not b
            for r in object_services.infer_spatial_relations(a, b)}


def relations_fast(objs):
    return {r for i, j in object_services.spatial_relation_candidates(objs, chunk=17)
            for r in object_services.infer_spatial_relations(objs[i], objs[j])}


def cloud(points, big=False, f8=False, pad=4, height=1):
    """A hand-built PointCloud2: x, y, z at offsets 0/4/8 (or 0/8/16), `pad` bytes of padding."""
    fmt, size, kind = ("d", 8, 8) if f8 else ("f", 4, 7)
    step = 3 * size + pad
    width = len(points) // height
    order = ">" if big else "<"
    data = b"".join(struct.pack(order + fmt * 3, *p) + b"\x00" * pad for p in points)
    fields = [SimpleNamespace(name=n, offset=k * size, datatype=kind, count=1) for k, n in enumerate("xyz")]
    return SimpleNamespace(fields=fields, is_bigendian=big, point_step=step, row_step=step * width,
                           width=width, height=height, data=array.array("B", data))


def cloud_parse_matches_reference():
    import room_manager
    rnd = random.Random(3)
    pts = [(rnd.uniform(-5, 5), rnd.uniform(-5, 5), rnd.uniform(0, 3)) for _ in range(600)]
    pts[7] = (float("nan"), 1.0, 2.0)
    pts[100] = (1.0, 2.0, float("nan"))
    for big in (False, True):
        for f8 in (False, True):
            for height in (1, 3):
                got = room_manager.RoomManager.cloud_xyz(cloud(pts, big, f8, 4, height))
                want = [p for p in pts if not any(math.isnan(v) for v in p)]
                if not f8:   # float32 round trip, as read_points returns it
                    want = [struct.unpack("fff", struct.pack("fff", *p)) for p in want]
                assert got.shape == (598, 3) and got.dtype.name == "float64", got.shape
                assert (got == [list(p) for p in want]).all(), (big, f8, height)
    assert room_manager.RoomManager.cloud_xyz(cloud([])).shape == (0, 3)


def checkpoint_is_batched_and_receipts_follow_it():
    import json
    import os
    import tempfile

    from object_info import Object
    from world_model import wm
    rows = []
    node = SimpleNamespace(log_both=lambda *a, **k: None,
                           _write_mutation_receipt=lambda e, r, st, op, obj=None, fusion_applied=False:
                           rows.append((e, st, op, obj.object_id, obj.object_revision)))
    a = Object("chair", (0, 0, 0), {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1, "z_min": 0, "z_max": 1})
    a.object_id, a.object_revision = "a", 1
    wm.persistent_perceptions[:] = [a]
    old = os.environ.get("GRAPH_API_OUTPUT_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["GRAPH_API_OUTPUT_DIR"] = tmp
        path = os.path.join(tmp, "persistent_perception.json")
        try:
            object_services.defer_checkpoint_receipt(node, "e1", None, "add", a)
            a.object_revision = 2                      # a later change in the same pair
            object_services.defer_checkpoint_receipt(node, "e2", None, "update", a)
            assert object_services.checkpoint_pending(node) and not os.path.exists(path) and rows == []
            object_services.save_persistent_perceptions(node)
            assert [e["object_id"] for e in json.load(open(path))] == ["a"]
            assert rows == [("e1", "applied_and_checkpointed", "add", "a", 1),
                            ("e2", "applied_and_checkpointed", "update", "a", 2)], rows
            assert not object_services.checkpoint_pending(node)
            mtime = os.stat(path).st_mtime_ns
            object_services.save_persistent_perceptions(node)    # nothing changed: no rewrite
            assert os.stat(path).st_mtime_ns == mtime
            wm.persistent_perceptions.clear()                     # removal reaches the file
            object_services.save_persistent_perceptions(node)
            assert json.load(open(path)) == []
        finally:
            if old is None:
                os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            else:
                os.environ["GRAPH_API_OUTPUT_DIR"] = old


def intake_buffers_and_pairs_run_in_capture_order():
    import json
    import os
    import tempfile
    import threading

    import object_manager_6 as om6
    node = object.__new__(om6.ObjectManagerService)
    node._pending_descriptions, node._pending_bboxes = {}, {}
    node._sync_lock, node._received_cycles = threading.Lock(), set()
    node._n_bbox_msgs = node._n_desc_msgs = node._n_bbox_no_stamp = node._n_desc_no_stamp = 0
    node._processed_pairs = node._dropped_moving_pairs = 0
    node._moving_since, node._last_nopair_state, node._desc_lag_s = None, None, []
    node.ga493_replay_capture = None
    node.object_services = SimpleNamespace(log_both=lambda *a, **k: None)
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=0))
    rows, done = [], []
    node.decision_log = SimpleNamespace(write=lambda kind, subject, **f: rows.append((kind, subject)))
    node.object_tracking_callback = lambda req, resp: done.append(req.bboxes.cycle_id)
    node._pair_timer = SimpleNamespace(reset=lambda: None)

    def msg(cycle, sec, items):
        return SimpleNamespace(cycle_id=cycle, header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=0)),
                               boxes=items, descriptions=items)
    # ids sort against capture time: z < y < x by name, x < y < z by stamp
    for cycle, sec in (("z", 3), ("x", 1), ("y", 2)):
        node._bboxes_callback(msg(cycle, sec, []))
        node._descriptions_callback(msg(cycle, sec, []))
    assert done == []                             # intake only buffers
    node._pair_tick()
    assert done == ["x"]                          # one pair per tick, oldest capture first
    node._pair_tick()
    node._pair_tick()
    assert done == ["x", "y", "z"]
    for k in range(om6.SYNC_BUFFER_LIMIT + 2):    # overflow is a logged eviction, not a silent drop
        node._bboxes_callback(msg(f"b{k}", 10 + k, []))
    assert [r for r in rows if r[0] == "sync_buffer_evicted"] == [("sync_buffer_evicted", "<bbox_array>")] * 2
    with tempfile.TemporaryDirectory() as tmp:
        old = os.environ.get("GRAPH_API_OUTPUT_DIR")
        os.environ["GRAPH_API_OUTPUT_DIR"] = tmp
        try:
            with open(os.path.join(tmp, "detections.jsonl"), "w") as f:
                for cid in ("x", "lost1"):                # empty cycles: a cycle_completed row
                    f.write(json.dumps({"event": "cycle_completed", "cycle_id": cid, "frame_id": "f",
                                        "detection_count": 0}) + "\n")
                for cid in ("y", "lost2", "lost2"):       # cycles with detections: per-detection rows
                    f.write(json.dumps({"frame_id": "g", "observation": {"cycle_id": cid}}) + "\n")
                f.write(json.dumps({"event": "capture_queue_evicted", "cycle_id": "other"}) + "\n")
            node.get_logger = lambda: SimpleNamespace(info=lambda *a: None)
            node._reconcile_received()
            node._reconcile_received()               # once only
        finally:
            if old is None:
                os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            else:
                os.environ["GRAPH_API_OUTPUT_DIR"] = old
    assert [r for r in rows if r[0] == "om_never_received"] == [("om_never_received", "lost1"),
                                                                 ("om_never_received", "lost2")]


def refiner_drain_keeps_its_budget_and_requeues_the_rest():
    import time

    import hooks
    import object_manager_6 as om6
    node = object.__new__(om6.ObjectManagerService)
    drains = []

    class Counting(hooks.Reevaluation):   # an extension's drain() has side effects per item
        def drain(self):
            items = super().drain()
            drains.append([o for o, _ in items])
            return items
    node.reeval = Counting()
    for oid in ("a", "b", "c"):
        node.reeval.mark(oid, "updated")
    seen = []
    node._find_node = lambda oid: SimpleNamespace(object_id=oid)
    node._node_dict = lambda o: {"object_id": o.object_id}
    node._neighbours = lambda o, r: []
    node.refiner_hook = SimpleNamespace(name="t", refine=lambda n, nb: seen.append(n["object_id"]) or time.sleep(0.05))
    node.decision_log = SimpleNamespace(write=lambda *a, **k: None)
    node._drain_reevaluations(time.monotonic() + 0.01)
    assert seen == ["a"] and [o for o, _ in node._refine_backlog] == ["b", "c"], seen
    node.reeval.mark("d", "updated")
    node._drain_reevaluations(time.monotonic() + 0.01)          # the backlog first, no new drain
    assert seen == ["a", "b"] and drains == [["a", "b", "c"]], (seen, drains)
    node._drain_reevaluations(None)
    node._drain_reevaluations(None)
    assert seen == ["a", "b", "c", "d"] and drains == [["a", "b", "c"], ["d"]], (seen, drains)


def relation_candidates_are_exact():
    for seed in range(20):
        objs = boxes(120, seed)
        full = relations_full(objs)
        assert full and relations_fast(objs) == full, seed


if __name__ == "__main__":
    failed = 0
    for test in (relation_candidates_are_exact, cloud_parse_matches_reference,
                 checkpoint_is_batched_and_receipts_follow_it, intake_buffers_and_pairs_run_in_capture_order,
                 refiner_drain_keeps_its_budget_and_requeues_the_rest):
        try:
            test()
            print(f"  ok    {test.__name__}")
        except Exception as exc:  # noqa: BLE001 - reporting
            failed += 1
            print(f"  FAIL  {test.__name__}: {type(exc).__name__}: {exc!r}")
    sys.exit(1 if failed else 0)
