#!/usr/bin/env python3
"""Exercise one Habitat object across three persistent floor sessions."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def read_json(path: Path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def wait_active(state_path: Path, visit_index: int, deadline: float):
    while time.monotonic() < deadline:
        if state_path.is_file():
            state = read_json(state_path)
            session = state.get("session") or {}
            if state.get("state") == "ACTIVE" and session.get("visit_index") == visit_index:
                return state
        time.sleep(0.5)
    raise TimeoutError(f"visit {visit_index} did not become ACTIVE before the deadline")


def post_command(url: str, command: dict, deadline: float):
    body = json.dumps(command).encode("utf-8")
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=12.0) as response:
                result = json.loads(response.read().decode("utf-8"))
            if result.get("success") is not True:
                raise RuntimeError(f"Habitat rejected {command['action']}: {result}")
            return result
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(1.0)
    raise TimeoutError(f"control server did not accept {command['action']} before the deadline")


def append_record(path: Path, state: dict, command: dict, result: dict):
    session = state["session"]
    record = {
        "t": time.time(),
        "visit_index": session["visit_index"],
        "session_id": session["session_id"],
        "floor_id": session["floor_id"],
        "transform_epoch": session["transform_epoch"],
        "command": command,
        "result": result,
    }
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coord-dir", required=True)
    parser.add_argument("--control-url", default="http://127.0.0.1:7790/object_command")
    parser.add_argument("--timeout", type=float, default=21600.0)
    parser.add_argument("--visit-one-position", default="2.554,0.0,-3.004")
    args = parser.parse_args()

    coord = Path(args.coord_dir).resolve()
    deadline = time.monotonic() + args.timeout
    output = coord / "dynamic_actions.jsonl"

    state = wait_active(coord / "state.json", 0, deadline)
    spawn = {"action": "spawn", "template": "random", "request_id": "mf-visit-000-spawn"}
    result = post_command(args.control_url, spawn, deadline)
    append_record(output, state, spawn, result)
    object_id = int(result["object_id"])

    position = [float(value) for value in args.visit_one_position.split(",")]
    if len(position) != 3:
        raise ValueError("--visit-one-position must contain x,y,z")
    state = wait_active(coord / "state.json", 1, deadline)
    move = {"action": "move", "object_id": object_id, "position": position,
            "request_id": "mf-visit-001-move"}
    result = post_command(args.control_url, move, deadline)
    append_record(output, state, move, result)

    state = wait_active(coord / "state.json", 2, deadline)
    remove = {"action": "remove", "object_id": object_id,
              "request_id": "mf-visit-002-remove"}
    result = post_command(args.control_url, remove, deadline)
    append_record(output, state, remove, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
