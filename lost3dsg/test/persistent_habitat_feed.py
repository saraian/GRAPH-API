#!/usr/bin/env python3
"""Run ordered floor visits while retaining one Habitat simulator instance."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

from multi_floor_session import SessionJournal, SessionSpec, file_sha256


def _read_json(path: Path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def latest_perception_queue_depth(path: Path):
    """Return the newest complete queue-depth measurement, or None.

    The perception node appends one JSON object per completed cycle. The host can
    read while the last append is in progress, so scan backwards and ignore an
    incomplete trailing line instead of treating it as an empty queue.
    """
    try:
        lines = path.read_bytes().splitlines()
    except OSError:
        return None
    for raw in reversed(lines):
        try:
            value = json.loads(raw).get("queue_depth")
        except (ValueError, AttributeError):
            continue
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def require_complete_drain(marker, session_id: str):
    if marker.get("drain_complete") is not True:
        raise RuntimeError(
            f"floor session {session_id} ended without a measured complete drain: "
            f"queue_start={marker.get('drain_queue_start')!r} "
            f"queue_end={marker.get('drain_queue_end')!r}")


class ObservationDrainBarrier:
    """Track a consumer queue until it is stably empty or times out."""

    def __init__(self, read_depth, timeout_s: float, stable_s: float):
        self.read_depth = read_depth
        self.timeout_s = float(timeout_s)
        self.stable_s = float(stable_s)
        self.zero_since = None

    def sample(self, now: float, started_at: float):
        depth = self.read_depth()
        if depth == 0:
            if self.zero_since is None:
                self.zero_since = now
        else:
            self.zero_since = None
        return {
            "queue_depth": depth,
            "complete": (
                depth == 0
                and self.zero_since is not None
                and now - self.zero_since >= self.stable_s
            ),
            "timed_out": now - started_at >= self.timeout_s,
        }


def wait_for_record(path: Path, visit_index: int, timeout: float, kind: str):
    """Wait for an atomically replaced barrier record for exactly one visit."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            record = _read_json(path)
            if int(record.get("visit_index", -1)) == visit_index:
                return record
        time.sleep(0.2)
    raise TimeoutError(f"timed out waiting for {kind} visit {visit_index} at {path}")


def load_specs(plan_path: Path):
    plan = _read_json(plan_path)
    if plan.get("schema") != 1:
        raise ValueError(f"unsupported multi-floor plan schema {plan.get('schema')!r}")
    specs = [SessionSpec(
        **{**raw, "building_to_map": tuple(tuple(row) for row in raw["building_to_map"])}
    ) for raw in plan["sessions"]]
    return plan, specs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--coord-dir", required=True)
    parser.add_argument("--barrier-timeout", type=float, default=900.0)
    args = parser.parse_args()

    plan, specs = load_specs(Path(args.plan))
    coord_dir = Path(args.coord_dir).resolve()
    coord_dir.mkdir(parents=True, exist_ok=True)
    journal = SessionJournal(coord_dir)
    runtime = {
        "clock_start": time.time(),
        "frame_seq": 0,
        "recording_dir": str(coord_dir),
    }

    # Import after run_sim has exported the ordinary feed configuration. The driver reuses the
    # project's ScheduledTour/load_schedule implementation through habitat_feed_host.main().
    import habitat_feed_host as feed

    simulator = feed.make_sim()
    for index, planned in enumerate(specs):
        activation = wait_for_record(
            coord_dir / "activate.json", index, args.barrier_timeout, "activation")
        if activation.get("session_id") != planned.session_id:
            raise RuntimeError("activation session_id does not match the plan")
        if activation.get("floor_id") != planned.floor_id:
            raise RuntimeError("activation floor_id does not match the plan")
        if activation.get("mode") != planned.mode:
            raise RuntimeError("activation mode does not match the plan")
        source = activation.get("source_database_path")
        if planned.mode == "mapping" and source is not None:
            raise RuntimeError("mapping activation must not name a source database")
        if planned.mode == "localization" and not source:
            raise RuntimeError("localization activation requires a source database")
        output_dir = Path(activation["output_dir"]).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        active = dataclasses.replace(
            planned,
            output_dir=str(output_dir),
            database_path=str(output_dir / "rtabmap.db"),
            source_database_path=source,
        )
        feed.STATS_DIR = output_dir
        feed.SPAWN_FLOOR = active.floor
        context = {
            "floor_id": active.floor_id,
            "session_id": active.session_id,
            "visit_index": active.visit_index,
            "transform_epoch": active.transform_epoch,
            "map_id": active.map_id,
            "building_to_map": [list(row) for row in active.building_to_map],
            "transport": plan["transport"],
        }
        journal.prepare(active)
        pose_source = os.environ.get("FEED_POSE_SOURCE", "simulator").strip().lower()
        if pose_source != "simulator":
            raise RuntimeError(
                "persistent feed currently requires FEED_POSE_SOURCE=simulator; rtabmap pose "
                "readiness needs a gated RGB-D warm-up path before observations can be admitted")
        def pose_ready(measured):
            delta = abs(float(measured["habitat_floor"]) - active.floor)
            journal.mark_ready({
                **context,
                "pose_stamp": float(measured["pose_stamp"]),
                "pose_source": pose_source,
                "tf_authorities": ["habitat_feed_node"],
                "quality": {
                    "passed": delta <= 0.75,
                    "criterion": "fresh simulator pose within 0.75 m of requested floor",
                    "floor_error_m": delta,
                },
            })
            journal.activate()

        runtime["pose_ready"] = pose_ready
        runtime["observation_guard"] = journal.accept_observation
        runtime["drain_barrier"] = ObservationDrainBarrier(
            lambda path=output_dir / "perception_latencies.jsonl":
                latest_perception_queue_depth(path),
            feed.TOUR_END_SETTLE_S,
            feed.DRAIN_STABLE_S,
        )
        result = feed.main(sim=simulator, session_context=context, runtime=runtime)
        require_complete_drain(result["marker"], active.session_id)
        simulator = result["sim"]
        journal.begin_drain()

        closed = wait_for_record(
            coord_dir / "closed.json", index, args.barrier_timeout, "closed database")
        if closed.get("session_id") != active.session_id:
            raise RuntimeError("closed barrier session_id does not match the active session")
        database = Path(closed["database_path"])
        digest = file_sha256(database)
        if digest != closed.get("database_sha256"):
            raise RuntimeError("closed database hash does not match its barrier record")
        journal.finish_visit(digest, more_visits=index + 1 < len(specs))
    simulator.close()
    ended = coord_dir / "driver_ended.json"
    temporary = ended.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump({"reason": "all_floor_sessions_complete", "t": time.time(),
                   "visits": len(specs), "frames": runtime["frame_seq"]}, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, ended)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
