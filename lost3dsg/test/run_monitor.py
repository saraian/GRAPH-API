#!/usr/bin/env python3
"""Watch a run while it happens, and say what it is doing.

    ./monitor.sh                 the newest run in results/
    ./monitor.sh <bundle>        that one
    ./monitor.sh --once          one snapshot, then exit
    ./monitor.sh --interval 30   seconds between lines (default 20)

WHY THIS EXISTS. A run takes hours and its only visible sign is a growing directory. Three times
today a run was declared healthy while the thing being watched was not the thing that mattered:
a tour that counted arrivals while scanning one waypoint forever, a bundle whose background writer
had produced nothing, and a gate-passing run that died on a viewer. So this reports the FEW fields
that distinguish those cases, and it reports them as they change rather than at the end.

WHAT IT WATCHES, and why each one is here rather than a prettier summary:
  container   a run with no container is over, whatever the directory looks like.
  gate        a run that failed preflight produced no measurement, so nothing after it counts.
  moved       total_distance_m. A run that rotates in place reads 5-9 m; a real tour reads tens.
  waypoints   reached / planned. `reached` climbing while `planned` stays 0 means no tour was
              planned at all, which is the shape that wasted six runs.
  decisions   rows in hook_decisions.jsonl. Zero during the mapping phase is normal; zero after it
              means perception admitted nothing.
  belief      entries in actual_perceptions.json, written by perception_2's background writer. A
              growing decisions count with an absent belief file is the split that hid a whole
              class of missing output.
  ended       terminating_node.ended once the run stops: tour_complete, operator_abort,
              mapping_time, node_death, unrecorded. `unrecorded` means the container never reached
              its own end -- it is not "unknown".

IT MAKES NO JUDGEMENT. It prints what the bundle says. A line saying "moved 5.4 m, waypoints 0/0"
is not flagged as bad, because whether that is bad depends on the arm.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent.parent


def _json(p: pathlib.Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _lines(p: pathlib.Path):
    """Count rows without loading the file. A decision log reaches tens of thousands of lines."""
    try:
        with p.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def container_up(name="graphapi_live"):
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=15).stdout
        return name in out.split()
    except Exception:
        return None  # docker unreachable is NOT the same as "no container"


def newest_bundle(results: pathlib.Path):
    cands = [p for p in results.glob("2026*") if p.is_dir()]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def snapshot(bundle: pathlib.Path):
    fs = _json(bundle / "feed_stats.json", {}) or {}
    meta = _json(bundle / "run_metadata.json", {}) or {}
    pf = _json(bundle / "preflight.json", {}) or {}
    belief = _json(bundle / "actual_perceptions.json", None)
    tn = meta.get("terminating_node") or {}
    verdict = pf.get("verdict")
    if verdict is None and pf.get("failed") is not None:
        verdict = "pass" if not pf["failed"] and not pf.get("skipped") else "fail"
    return {
        "container": container_up(),
        "gate": verdict,
        "moved_m": fs.get("total_distance_m"),
        "reached": fs.get("tour_waypoints_reached"),
        "planned": fs.get("tour_waypoints_planned"),
        "decisions": _lines(bundle / "hook_decisions.jsonl"),
        "belief": (len(belief) if isinstance(belief, list) else None),
        "ended": tn.get("ended") or (tn.get("node") if tn else None),
        "size_mb": round(sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
                         / 1e6, 1),
    }


def line(s: dict) -> str:
    def v(x, dash="-"):
        return dash if x is None else x
    c = {True: "up", False: "down", None: "docker?"}[s["container"]]
    return (f"{datetime.datetime.now():%H:%M:%S}  container {c:<7}  gate {str(v(s['gate'], '?')):<4}  "
            f"moved {v(s['moved_m']):>7} m  waypoints {v(s['reached'])}/{v(s['planned'])}  "
            f"decisions {v(s['decisions']):>6}  belief {v(s['belief']):>5}  "
            f"{s['size_mb']:>7} MB" + (f"  ended {s['ended']}" if s["ended"] else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?", type=pathlib.Path)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--results", type=pathlib.Path, default=REPO / "results")
    args = ap.parse_args()

    bundle = args.bundle or newest_bundle(args.results)
    if bundle is None:
        print(f"!! no run bundle under {args.results}", file=sys.stderr)
        return 2
    bundle = bundle.resolve()
    if not (bundle / "run_metadata.json").exists() and not (bundle / "feed_stats.json").exists():
        print(f"!! {bundle} does not look like a run bundle", file=sys.stderr)
        return 2
    print(f"watching {bundle}")
    if args.once:
        print(line(snapshot(bundle)))
        return 0

    # THE RUN IS OVER WHEN THE CONTAINER IS GONE *AND* THE DIRECTORY HAS STOPPED GROWING. Either
    # alone is wrong: the container exits before the archive finishes, and a stalled run keeps its
    # container. Two consecutive quiet checks, so one slow write does not end the watch.
    quiet = 0
    last_size = -1.0
    while True:
        s = snapshot(bundle)
        print(line(s), flush=True)
        if s["container"] is False and abs(s["size_mb"] - last_size) < 0.05:
            quiet += 1
            if quiet >= 2:
                print(f"run finished — {s['ended'] or 'no terminating_node recorded'}")
                print(f"bundle: {bundle}")
                return 0
        else:
            quiet = 0
        last_size = s["size_mb"]
        time.sleep(args.interval)


def _selfcheck():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        b = pathlib.Path(d) / "20260101_000000_scene"
        b.mkdir()
        (b / "feed_stats.json").write_text(json.dumps(
            {"total_distance_m": 62.76, "tour_waypoints_reached": 25, "tour_waypoints_planned": 0}))
        (b / "hook_decisions.jsonl").write_text("{}\n{}\n{}\n")
        s = snapshot(b)
        assert s["moved_m"] == 62.76 and s["reached"] == 25 and s["planned"] == 0, s
        assert s["decisions"] == 3, s
        # AN ABSENT BELIEF FILE IS None, NOT 0. That distinction is the whole reason this field is
        # here: "no belief file" and "a belief file with nothing in it" are different faults.
        assert s["belief"] is None, s
        (b / "actual_perceptions.json").write_text("[]")
        assert snapshot(b)["belief"] == 0, snapshot(b)
        # a torn json must not crash the watch
        (b / "feed_stats.json").write_text("{not json")
        assert snapshot(b)["moved_m"] is None
        txt = line(snapshot(b))
        assert "container" in txt and "belief" in txt, txt
        # newest_bundle picks by mtime and ignores files
        (pathlib.Path(d) / "notabundle.txt").write_text("x")
        assert newest_bundle(pathlib.Path(d)) == b
    print("run_monitor self-check: PASSED")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    raise SystemExit(main())
