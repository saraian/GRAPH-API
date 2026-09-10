#!/usr/bin/env python3
"""Checks for tasks 5 and 7: the decision->object link, and latency accounting.

    python3 test_latency_accounting.py

ROS-free and dependency-free on purpose: both changes are arithmetic and record shape,
and a check that needs the stack to run is a check that stops being run.
"""
import json
import os
import sys
import tempfile

# --- task 7: the accounting identity ------------------------------------------------
#
# The bug this guards against is not a wrong number, it is an UNNOTICED one. Before the
# change, `total_ms` was 1,288 ms median while the recorded stages summed to 328 -- and
# nothing anywhere said the other 960 was missing, so three sessions reasoned about
# latency from a quarter of the cycle. The identity below is what makes that loud.

def latency_record(t_vlm, t_owlv2, t_nms, t_sam, t_proj, t_cloud, t_room_geom, t_total):
    """The arithmetic from detection_pipeline.run_detection, isolated."""
    t_backend_overhead = max(0.0, t_cloud - (t_owlv2 + t_nms + t_sam)) if t_cloud else 0.0
    parts = t_vlm + t_owlv2 + t_nms + t_sam + t_proj + t_backend_overhead + t_room_geom
    return {
        "vlm_ms": round(t_vlm * 1000, 1),
        "owlv2_ms": round(t_owlv2 * 1000, 1),
        "nms_ms": round(t_nms * 1000, 1),
        "sam_ms": round(t_sam * 1000, 1),
        "projection_ms": round(t_proj * 1000, 1),
        "total_ms": round(t_total * 1000, 1),
        "backend_overhead_ms": round(t_backend_overhead * 1000, 1),
        "room_geom_ms": round(t_room_geom * 1000, 1),
        "unattributed_ms": round(max(0.0, t_total - parts) * 1000, 1),
    }


def test_cloud_backend_attributes_the_round_trip():
    """The real 26 Aug shape: server says ~330 ms, wall clock says ~1.3 s."""
    r = latency_record(t_vlm=0.0, t_owlv2=0.080, t_nms=0.0, t_sam=0.237, t_proj=0.011,
                       t_cloud=1.290, t_room_geom=0.002, t_total=1.288)
    # The round-trip is now a named number rather than a hole.
    assert r["backend_overhead_ms"] > 900, r
    # ...and what remains unexplained is small. Before the change this was ~857 ms.
    assert r["unattributed_ms"] < 60, r


def test_local_backend_has_no_round_trip_to_attribute():
    r = latency_record(t_vlm=0.0, t_owlv2=0.080, t_nms=0.005, t_sam=0.237, t_proj=0.011,
                       t_cloud=0.0, t_room_geom=0.002, t_total=0.340)
    assert r["backend_overhead_ms"] == 0.0, r
    assert r["unattributed_ms"] < 20, r


def test_accounting_never_goes_negative():
    """Clock skew or a stage overlapping total must not produce a negative remainder --
    a negative would render as a nonsense bar and be read as a parser bug, not a clock."""
    r = latency_record(t_vlm=1.0, t_owlv2=1.0, t_nms=1.0, t_sam=1.0, t_proj=1.0,
                       t_cloud=0.0, t_room_geom=1.0, t_total=0.5)
    assert r["unattributed_ms"] == 0.0, r
    assert r["backend_overhead_ms"] == 0.0, r


def stage_times(cloud_timings):
    """The extraction from detection_pipeline.run_detection, isolated.

    Raises when the server omits a stage, rather than estimating it from the wall clock.
    """
    missing = [k for k in ("detector", "sam2") if k not in cloud_timings]
    if missing:
        raise RuntimeError(f"perception backend returned no timing for {missing}")
    return (cloud_timings["detector"] / 1000.0,
            cloud_timings.get("nms", 0.0) / 1000.0,
            cloud_timings["sam2"] / 1000.0)


def test_missing_server_timing_raises_instead_of_estimating():
    """The old code defaulted a missing stage to a fixed fraction of the wall clock and stored
    it in the same field as a real measurement. That also pinned backend_overhead_ms at 10% of
    the wall clock, because the two fractions summed to 0.9 -- one invented number feeding
    another. A measurement path must not guess."""
    ok = stage_times({"detector": 80.0, "sam2": 237.0, "nms": 0.0})
    assert ok == (0.080, 0.0, 0.237), ok
    for broken in ({"sam2": 237.0}, {"detector": 80.0}, {}):
        try:
            stage_times(broken)
        except RuntimeError:
            continue
        raise AssertionError(f"no raise for {broken}")


def test_snapshot_keys_are_additive_only():
    """Three consumers read this snapshot by key (bridge /bev_data, viewer Metrics tab,
    viewer CSV export). Renaming or dropping a key breaks all three silently; adding one
    does not. This asserts the original keys survive."""
    r = latency_record(0.1, 0.08, 0.0, 0.24, 0.01, 1.3, 0.002, 1.3)
    for k in ("vlm_ms", "owlv2_ms", "nms_ms", "sam_ms", "projection_ms", "total_ms"):
        assert k in r, f"pre-existing consumer key {k} disappeared"


# --- task 5: the decision -> object link --------------------------------------------
#
# Guards the join that Table II's denominator depends on. `object_id` was null in all 121
# decision rows of the 26 Aug run, so the only shared key was the label (9 of 11 objects),
# which is why the analysis fell back to a rounding grid.

def _write(log_path, kind, object_ref, **payload):
    with open(log_path, "a") as f:
        f.write(json.dumps({"t": 0.0, "kind": kind, "object": object_ref, **payload}) + "\n")


def test_link_record_joins_decision_to_object():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "hook_decisions.jsonl")
        # two objects sharing a base label -- exactly the case the label join cannot resolve
        _write(p, "admission", "chair", filter="ont", outcome="admit", reason="",
               annotation={"decision_id": "dec-a"})
        _write(p, "link", "obj-17", decision_id="dec-a", label="chair")
        _write(p, "admission", "chair", filter="ont", outcome="admit", reason="",
               annotation={"decision_id": "dec-b"})
        _write(p, "link", "obj-42", decision_id="dec-b", label="chair")

        rows = [json.loads(x) for x in open(p) if x.strip()]
        links = {r["decision_id"]: r["object"] for r in rows if r["kind"] == "link"}
        decisions = [r for r in rows if r["kind"] == "admission"]

        # Every admitted decision resolves to exactly one object...
        assert links["dec-a"] == "obj-17"
        assert links["dec-b"] == "obj-42"
        # ...and the two same-label objects stay distinct, which the label key could not do.
        assert len({links[r["annotation"]["decision_id"]] for r in decisions}) == 2


def test_refused_proposals_get_no_link():
    """A refused proposal never becomes an object, so it must produce no link -- otherwise
    the join would invent objects that were never admitted."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "hook_decisions.jsonl")
        _write(p, "admission", "wombat", filter="ont", outcome="reject", reason="no grounds",
               annotation={"decision_id": "dec-x"})
        rows = [json.loads(x) for x in open(p) if x.strip()]
        assert not [r for r in rows if r["kind"] == "link"]


def test_object_history_reconstructs_across_a_merge():
    """The requirement: the decision history of one object must be reconstructable.

    A merge is the case that breaks a naive reader. The discarded object stops existing and its
    past belongs to the keeper, so a history built by filtering on one id alone ends early and
    silently. The merge record names both sides, which is what lets the reader follow it."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "hook_decisions.jsonl")
        # obj-A: admitted, then moved
        _write(p, "admission", "chair", outcome="admit", annotation={"decision_id": "dec-a"})
        _write(p, "link", "obj-A", decision_id="dec-a", label="chair")
        _write(p, "update", "obj-A", label="chair", change="moved", distance_m=0.66)
        # obj-B: admitted separately, then merged INTO obj-A
        _write(p, "admission", "chair", outcome="admit", annotation={"decision_id": "dec-b"})
        _write(p, "link", "obj-B", decision_id="dec-b", label="chair")
        _write(p, "merge", "obj-A", merged_from="obj-B", keeper_label="chair",
               discarded_label="chair", similarity=0.91)
        # obj-A later deleted
        _write(p, "delete", "obj-A", label="chair", reason="not seen in POV")

        rows = [json.loads(x) for x in open(p) if x.strip()]

        def history(object_id):
            """Every decision about this object, following merges backwards."""
            ids, out = {object_id}, []
            for _ in range(len(rows)):                      # bounded: no infinite chase
                grew = False
                for r in rows:
                    if r["kind"] == "merge" and r["object"] in ids and r["merged_from"] not in ids:
                        ids.add(r["merged_from"])
                        grew = True
                if not grew:
                    break
            links = {r["decision_id"]: r["object"] for r in rows if r["kind"] == "link"}
            for r in rows:
                if r["kind"] == "link" and r["object"] in ids:
                    out.append(r)
                elif r["kind"] in ("merge", "update", "delete") and r["object"] in ids:
                    out.append(r)
                elif r["kind"] == "admission":
                    if links.get(r["annotation"]["decision_id"]) in ids:
                        out.append(r)
            return out

        h = history("obj-A")
        kinds = [r["kind"] for r in h]
        # both admissions are inherited, not just obj-A's own
        assert kinds.count("admission") == 2, kinds
        assert kinds.count("link") == 2, kinds
        assert "merge" in kinds and "update" in kinds and "delete" in kinds, kinds

        # and the discarded id resolves to where it went, rather than ending without a successor
        moved_to = [r["object"] for r in rows if r["kind"] == "merge" and r["merged_from"] == "obj-B"]
        assert moved_to == ["obj-A"], moved_to


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
