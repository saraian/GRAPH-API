#!/usr/bin/env python3
"""Ground truth for a scripted scene: what was where, and WHEN.

A scene script (`FOUND-Dataset/scene_script.py`) spawns, moves and removes rigid objects while the
run is under way. The HM3D semantic annotation cannot describe any of that -- it is a static mesh --
so before this module the evaluator scored a scripted run against a world that was never there:
a spawned object had no ground-truth counterpart and scored as a false positive, a removed one
stayed in the ground truth and scored as a false negative, and a moved one scored as both. None of
those are perception errors, and nothing in the report could tell them from real ones.

WHAT THIS ADDS. The runner writes a LEDGER of executed steps, each with the wall-clock time it
took effect. This module turns that ledger into ground-truth rows that carry a validity window:

    valid_from  the moment the object began to exist AT THIS POSE (None = since the run began)
    valid_to    the moment it stopped existing at this pose (None = to the end of the run)

A `move` becomes TWO rows -- the old pose closing at the move, the new pose opening at it -- because
an object at two places at two times is two facts, not one fact with a changed field. Nothing else
in the evaluator needs to know about actions; it only needs to know when a row was true.

THE MATCHING RULE lives in `object_metrics.distances`: a prediction observed at time t can only
match a ground-truth row whose window contains t. Rows with no window (the static scene) match at
any time, so a run with no script behaves exactly as before -- the same code path, the same numbers.

Run: python3 script_ledger.py   (self-check)
"""
import json
import os

# A predicted object carries one observation time; the static scene carries none. Both must work.
ALWAYS = (None, None)


def _f(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def load_ledger(run_dir):
    """-> the ledger dict, or None when this run had no script.

    A missing ledger is the ordinary case and is NOT an error: it means a static scene. It is
    distinguished from a PRESENT BUT EMPTY ledger, which means the script ran and did nothing --
    the caller reports those differently, because one is 'no subject' and the other is 'a subject
    that produced no change'.
    """
    path = os.path.join(run_dir, "scene_script_ledger.json")
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _aabb_from_position(position, extents):
    """A ground-truth AABB around a placement. The ledger records a point and a size."""
    cx, cy, cz = (_f(v) for v in position)
    if None in (cx, cy, cz):
        return None
    ex, ey, ez = ((_f(v) or 0.0) / 2.0 for v in extents)
    return {"x_min": cx - ex, "x_max": cx + ex,
            "y_min": cy - ey, "y_max": cy + ey,
            "z_min": cz - ez, "z_max": cz + ez}


def ground_truth_rows(ledger, default_extents=(0.2, 0.2, 0.2)):
    """-> [row] for every pose a scripted object held, each with its validity window.

    The rows are shaped like the static ground truth (`object_id`, `category_name`, `aabb_*_m`) so
    the evaluator's existing filter and matcher accept them unchanged, plus three fields:
    `valid_from`, `valid_to` and `scripted: True`.
    """
    if not ledger:
        return []
    rows = []
    # object_id -> the index in `rows` of the pose that is currently open
    open_row = {}
    for entry in ledger.get("steps") or []:
        action = str(entry.get("action", "")).strip().lower()
        at = _f(entry.get("at"))
        oid = entry.get("object_id") or entry.get("name")
        if action == "wait" or oid is None:
            continue
        if action in ("move", "remove") and oid in open_row:
            rows[open_row.pop(oid)]["valid_to"] = at      # this pose stops being true here
        if action == "remove":
            continue
        position = entry.get("position")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            continue
        measured = entry.get("extents")
        box = _aabb_from_position(position, measured or default_extents)
        if box is None:
            continue
        row = {
            # WHERE THE SIZE CAME FROM. The ledger records a placement POINT; the object's real
            # extent is known only to the simulator. When the runner did not report it we fall back
            # to a fabricated cube, and that value must never be mistaken for a measurement
            # (working rule 5). Centre-distance matching is unaffected either way -- but every
            # volume metric is meaningless on a defaulted pose, so `box_quality_defaulted` in the
            # scripted report refuses to let one be quoted silently.
            "extents_source": "measured" if measured else "default",
            "extents_m": [float(v) for v in (measured or default_extents)],
            "object_id": f"script:{oid}:{len(rows)}",
            "script_object_id": str(oid),
            "category_name": str(entry.get("template") or entry.get("target_category") or "object"),
            "scripted": True,
            "valid_from": at,
            "valid_to": None,
            "step": entry.get("step"),
            "action": action,
            "geometry_source": "scene_script_placement",
        }
        row.update(box)
        row["aabb_min_m"] = [box["x_min"], box["y_min"], box["z_min"]]
        row["aabb_max_m"] = [box["x_max"], box["y_max"], box["z_max"]]
        open_row[oid] = len(rows)
        rows.append(row)
    return rows


def window(row):
    """-> (valid_from, valid_to) for any ground-truth row. Static rows are valid always."""
    if not isinstance(row, dict):
        return ALWAYS
    return (_f(row.get("valid_from")), _f(row.get("valid_to")))


def holds_at(row, when):
    """Was this row true at `when`?

    A row with no window is the static scene and holds at every time. A prediction with no time is
    treated the same way -- it cannot be placed on the timeline, so it is not excluded by one. That
    is deliberate: an instrument must not silently refuse a match because a field it wants is
    absent, which would turn a missing timestamp into a false positive.
    """
    low, high = window(row)
    if low is None and high is None:
        return True
    at = _f(when)
    if at is None:
        return True
    if low is not None and at < low:
        return False
    if high is not None and at >= high:
        return False
    return True


def _self_check():
    spawn_at, move_at, remove_at = 100.0, 200.0, 300.0
    ledger = {"steps": [
        {"step": 0, "action": "spawn", "object_id": "banana_1", "template": "banana",
         "at": spawn_at, "position": [1.0, 0.0, 1.0], "extents": [0.2, 0.2, 0.2]},
        {"step": 1, "action": "move", "object_id": "banana_1", "template": "banana",
         "at": move_at, "position": [5.0, 0.0, 5.0], "extents": [0.2, 0.2, 0.2]},
        {"step": 2, "action": "remove", "object_id": "banana_1", "at": remove_at},
        {"step": 3, "action": "wait", "seconds": 1.0, "at": remove_at + 1},
    ]}
    rows = ground_truth_rows(ledger)
    assert len(rows) == 2, f"a move is two poses, got {len(rows)}"
    # the size the runner reported is marked measured; a fabricated one is marked default
    assert all(r["extents_source"] == "measured" for r in rows), rows[0]
    bare = {"steps": [{"step": 0, "action": "spawn", "object_id": "x", "at": 1.0,
                       "position": [0.0, 0.0, 0.0]}]}          # no extents reported
    assert ground_truth_rows(bare)[0]["extents_source"] == "default"
    first, second = rows
    assert (first["valid_from"], first["valid_to"]) == (spawn_at, move_at), first
    assert (second["valid_from"], second["valid_to"]) == (move_at, remove_at), second
    assert first["aabb_min_m"] == [0.9, -0.1, 0.9], first["aabb_min_m"]
    assert all(r["scripted"] for r in rows)

    # the windows partition the timeline, and nothing is true before the spawn or after the remove
    assert not holds_at(first, 50.0), "nothing exists before its spawn"
    assert holds_at(first, 150.0) and not holds_at(second, 150.0), "only the first pose holds"
    assert holds_at(second, 250.0) and not holds_at(first, 250.0), "only the second pose holds"
    assert not holds_at(first, 350.0) and not holds_at(second, 350.0), "nothing survives the remove"
    # the boundary belongs to the NEW pose, so the two never both hold
    assert holds_at(second, move_at) and not holds_at(first, move_at)

    # a static row holds at every time, and an untimed prediction is never excluded
    static = {"object_id": "chair_1", "category_name": "chair"}
    assert holds_at(static, 0.0) and holds_at(static, 1e9)
    assert holds_at(first, None), "a prediction with no timestamp must not be refused"

    # a run with no script yields no rows at all
    assert ground_truth_rows(None) == [] and ground_truth_rows({}) == []
    # a present-but-empty ledger is a different state from an absent one
    assert ground_truth_rows({"steps": []}) == []
    print("ok  script_ledger self-check")


if __name__ == "__main__":
    _self_check()
