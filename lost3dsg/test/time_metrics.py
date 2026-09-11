#!/usr/bin/env python3
"""The time a run spent, and where it went.

    python3 time_metrics.py <bundle> [--output <bundle>/eval/time_metrics.json]

metrics_eval.py computes exactly one duration -- `construction_time_s`, the whole run -- and the
bundle carries per-cycle stage timings that nothing reads. This computes the six the owner asked
for on 2026-09-11, each from a named source, and returns **null rather than zero** wherever the
input is absent. A zero and a missing measurement are not the same claim, and this project has
lost days to the difference.

  latency_ms            every stage the detection cycle records, as median / p95 / max / n.
                        Source: perception_latencies.jsonl, one row per cycle.
  run_time_s            first to last timestamp seen anywhere in the bundle. Wall clock, so it
                        includes the gate, the build and the teardown -- it is not compute time.
  exploration_time_s    first to last FRAME. The feed's own span: how long the agent was being
                        driven around. Source: frame_poses.jsonl.
  movement_time_s       time in which the agent was MOVING. Perception fires only when the robot
                        is stopped, so this is time in which no detection could happen -- the
                        "time lost in movement". A frame counts as moving when the pose differs
                        from the previous one by more than 1 cm or 0.5 degrees.
  online_time_s         time inside remote calls: vlm_ms + wire_ms + backend_overhead_ms summed
                        over cycles. With a local VLM and a local detector this is near zero, and
                        that is the point of measuring it.
  first_frame_to_kg_s   from the first frame to the first admission decision reaching the store.
                        Sources: frame_poses.jsonl and hook_decisions.jsonl. This is the latency a
                        reader of the graph actually waits, as opposed to a cycle's own duration.

WHY MEDIAN AND p95 RATHER THAN A MEAN. The first cycle of every run in the archive used to include
a model download, and three archived maxima are nothing but their own first cycle. A mean hides
that; a median with a p95 beside it does not.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

STAGES = ("total_ms", "vlm_ms", "latency_ms", "sam_ms", "nms_ms", "projection_ms",
          "backend_overhead_ms", "wire_ms", "cycle_ms")
ONLINE_STAGES = ("vlm_ms", "wire_ms", "backend_overhead_ms")
MOVE_M = 0.01          # 1 cm
MOVE_RAD = 0.00873     # 0.5 degrees


def _rows(p: pathlib.Path):
    """Every parseable JSON line. A torn LAST line is tolerated and counted; a tear anywhere
    else is not silently dropped, because a swallowed row makes every denominator short."""
    out, torn = [], 0
    if not p.exists():
        return out, None
    lines = [x for x in p.read_text(errors="ignore").splitlines() if x.strip()]
    for i, x in enumerate(lines):
        try:
            out.append(json.loads(x))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            torn += 1
    return out, torn


def _pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return round(s[0], 1)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def stage_latencies(rows):
    """Per stage: median, p95, max, n. A stage no cycle recorded is absent, not zero."""
    out = {}
    for st in STAGES:
        vals = [float(r[st]) for r in rows
                if isinstance(r.get(st), (int, float))]
        if vals:
            out[st] = {"median": _pct(vals, .5), "p95": _pct(vals, .95),
                       "max": round(max(vals), 1), "n": len(vals)}
    return out


def movement_split(frames):
    """(exploration_time_s, movement_time_s, moving_frames, still_frames).

    Time is attributed to INTERVALS, never to points: each gap between consecutive frames is
    charged to moving or still according to whether the pose changed across it. Summing over
    points instead would invent intervals, which is its own recorded mistake in this project.
    """
    if len(frames) < 2:
        return (None, None, 0, 0)
    frames = sorted(frames, key=lambda f: f.get("stamp", 0))
    moving = still = 0.0
    n_move = n_still = 0
    for a, b in zip(frames, frames[1:]):
        dt = float(b.get("stamp", 0)) - float(a.get("stamp", 0))
        if dt <= 0 or dt > 30:      # a gap over 30 s is a stall, not motion; do not charge it
            continue
        dx = math.dist((a.get("x", 0), a.get("y", 0), a.get("z", 0)),
                       (b.get("x", 0), b.get("y", 0), b.get("z", 0)))
        dyaw = abs(float(b.get("yaw", 0)) - float(a.get("yaw", 0)))
        dyaw = min(dyaw, 2 * math.pi - dyaw)
        if dx > MOVE_M or dyaw > MOVE_RAD:
            moving += dt
            n_move += 1
        else:
            still += dt
            n_still += 1
    span = float(frames[-1].get("stamp", 0)) - float(frames[0].get("stamp", 0))
    return (round(span, 1), round(moving, 1), n_move, n_still)


def compute(bundle: pathlib.Path):
    lat_rows, lat_torn = _rows(bundle / "perception_latencies.jsonl")
    frames, _ = _rows(bundle / "frame_poses.jsonl")
    decisions, _ = _rows(bundle / "hook_decisions.jsonl")

    explore, moving, n_move, n_still = movement_split(frames)
    stages = stage_latencies(lat_rows)

    online = None
    if lat_rows:
        tot = 0.0
        seen = False
        for r in lat_rows:
            for st in ONLINE_STAGES:
                if isinstance(r.get(st), (int, float)):
                    tot += float(r[st])
                    seen = True
        online = round(tot / 1000.0, 1) if seen else None

    # The wall clock: the widest span any timestamped source in the bundle covers.
    stamps = ([float(f["stamp"]) for f in frames if isinstance(f.get("stamp"), (int, float))]
              + [float(d["t"]) for d in decisions if isinstance(d.get("t"), (int, float))])
    run_time = round(max(stamps) - min(stamps), 1) if len(stamps) >= 2 else None

    first_to_kg = None
    if frames and decisions:
        f0 = min(float(f["stamp"]) for f in frames if isinstance(f.get("stamp"), (int, float)))
        adm = [float(d["t"]) for d in decisions
               if isinstance(d.get("t"), (int, float))
               and d.get("kind") in (None, "admission")]
        if adm:
            first_to_kg = round(min(adm) - f0, 1)

    return {
        "latency_ms": stages or None,
        "run_time_s": run_time,
        "exploration_time_s": explore,
        "movement_time_s": moving,
        "movement_frames": n_move or None,
        "still_frames": n_still or None,
        "online_time_s": online,
        "first_frame_to_kg_s": first_to_kg,
        "cycles": len(lat_rows) or None,
        "decisions": len(decisions) or None,
        "sources": {
            "perception_latencies.jsonl": len(lat_rows),
            "frame_poses.jsonl": len(frames),
            "hook_decisions.jsonl": len(decisions),
        },
        "notes": [n for n in [
            "perception_latencies.jsonl is absent, so no stage latency and no online time"
            if not lat_rows else None,
            "frame_poses.jsonl is absent, so no exploration or movement time"
            if not frames else None,
            "no admission decision, so first_frame_to_kg_s cannot be computed"
            if frames and not decisions else None,
            f"the last latency row was truncated and excluded ({lat_torn} row)"
            if lat_torn else None,
        ] if n],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path, default=None)
    args = ap.parse_args()
    b = args.bundle.resolve()
    if not b.is_dir():
        print(f"!! no such bundle: {b}", file=sys.stderr)
        return 2
    m = compute(b)
    out = args.output or (b / "eval" / "time_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(f"wrote {out}")
    for k in ("run_time_s", "exploration_time_s", "movement_time_s", "online_time_s",
              "first_frame_to_kg_s"):
        print(f"  {k:<22} {m[k] if m[k] is not None else 'not measured'}")
    if m["latency_ms"]:
        for st, d in m["latency_ms"].items():
            print(f"  {st:<22} median {d['median']} p95 {d['p95']} max {d['max']} (n={d['n']})")
    else:
        print("  latency_ms             not measured")
    for n in m["notes"]:
        print(f"  note: {n}")
    return 0


def _selfcheck():
    import tempfile
    assert _pct([1, 2, 3, 4], .5) == 2.5, _pct([1, 2, 3, 4], .5)
    assert _pct([], .5) is None
    # movement is charged to INTERVALS, and a still interval is not movement
    frames = [{"stamp": 0, "x": 0, "y": 0, "z": 0, "yaw": 0},
              {"stamp": 1, "x": 1, "y": 0, "z": 0, "yaw": 0},     # moved 1 m in 1 s
              {"stamp": 2, "x": 1, "y": 0, "z": 0, "yaw": 0},     # still for 1 s
              {"stamp": 3, "x": 1, "y": 0, "z": 0, "yaw": 1.0}]   # turned in 1 s
    span, moving, nm, ns = movement_split(frames)
    assert (span, moving, nm, ns) == (3.0, 2.0, 2, 1), (span, moving, nm, ns)
    # A TURN IS MOVEMENT. Perception fires only when stopped, and 84% of one run's movement
    # budget went into turning, so counting a turn as "still" would hide the whole finding.
    assert movement_split(frames[2:])[1] == 1.0
    # a stall is not charged as motion
    assert movement_split([{"stamp": 0, "x": 0, "y": 0, "z": 0, "yaw": 0},
                           {"stamp": 100, "x": 9, "y": 0, "z": 0, "yaw": 0}])[1] == 0.0
    with tempfile.TemporaryDirectory() as d:
        b = pathlib.Path(d)
        (b / "frame_poses.jsonl").write_text(
            "".join(json.dumps(f) + "\n" for f in frames))
        (b / "perception_latencies.jsonl").write_text(
            json.dumps({"total_ms": 1000, "vlm_ms": 400, "wire_ms": 100,
                        "backend_overhead_ms": 50, "sam_ms": 200}) + "\n"
            + json.dumps({"total_ms": 3000, "vlm_ms": 900, "wire_ms": 200,
                          "backend_overhead_ms": 100, "sam_ms": 300}) + "\n")
        (b / "hook_decisions.jsonl").write_text(
            json.dumps({"t": 2.5, "kind": "admission", "outcome": "admit"}) + "\n")
        m = compute(b)
        assert m["exploration_time_s"] == 3.0 and m["movement_time_s"] == 2.0, m
        assert m["online_time_s"] == round((400 + 100 + 50 + 900 + 200 + 100) / 1000, 1), m
        assert m["first_frame_to_kg_s"] == 2.5, m
        assert m["latency_ms"]["total_ms"]["median"] == 2000.0, m["latency_ms"]["total_ms"]
        assert m["cycles"] == 2 and m["decisions"] == 1, m
        # AN ABSENT SOURCE IS null AND SAYS SO, never 0.
        b2 = pathlib.Path(d) / "empty"
        b2.mkdir()
        m2 = compute(b2)
        assert m2["latency_ms"] is None and m2["online_time_s"] is None, m2
        assert m2["exploration_time_s"] is None and m2["run_time_s"] is None, m2
        assert any("perception_latencies.jsonl is absent" in n for n in m2["notes"]), m2["notes"]
    print("time_metrics self-check: PASSED")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    raise SystemExit(main())
