#!/usr/bin/env python3
"""A scripted scene is scored against the world AS IT WAS, not against one final snapshot.

THE DEFECT THIS CLOSES. A scene script spawns, moves and removes rigid objects while the run is
under way. The HM3D ground truth is a static semantic mesh and cannot contain any of them, so
before this the evaluator scored a scripted run against a world that never existed: a spawned
object had no counterpart and became a false positive, a removed one stayed in the ground truth and
became a false negative, and a moved one became both. None are perception errors and the report
could not tell them from real ones.

THE TESTS BELOW EXERCISE THE ANSWER WHERE IT IS PRESENT (working rules 60, 70): every one builds a
world where the right answer and the wrong answer DIFFER, and asserts the new behaviour produces
the right one and the old behaviour would not have. A test that only ran on a static scene would
pass identically before and after the change and would prove nothing.

Run: python3 test_scene_script_evaluation.py   (or under pytest)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "perception_module"))

import object_metrics as om          # noqa: E402
import script_ledger                 # noqa: E402

SPAWN_AT, MOVE_AT, REMOVE_AT = 100.0, 200.0, 300.0
OLD_POSE, NEW_POSE = [1.0, 0.0, 1.0], [5.0, 0.0, 5.0]


def _box(centre, half=0.1):
    return {"aabb_min_m": [c - half for c in centre], "aabb_max_m": [c + half for c in centre]}


def _ledger():
    return {"steps": [
        {"step": 0, "action": "spawn", "object_id": "banana_1", "template": "banana",
         "at": SPAWN_AT, "position": OLD_POSE},
        {"step": 1, "action": "move", "object_id": "banana_1", "template": "banana",
         "at": MOVE_AT, "position": NEW_POSE},
        {"step": 2, "action": "remove", "object_id": "banana_1", "at": REMOVE_AT},
    ]}


def _static_gt():
    return [dict(_box([0.0, 0.0, 0.0]), object_id="chair_1", category_name="chair")]


def _scene(predicted, scripted=True):
    gt = _static_gt() + (script_ledger.ground_truth_rows(_ledger()) if scripted else [])
    return [{"predicted_objects": predicted, "ground_truth_objects": gt}]


def test_a_static_run_is_completely_unchanged():
    """The whole point of gating rather than rewriting: no script, no difference."""
    pred = [dict(_box([0.0, 0.0, 0.0]), object_id="p1", observed_at=150.0)]
    out, _ = om.evaluate_geometry(_scene(pred, scripted=False))
    assert out["matched_objects"] == 1 and out["false_positives"] == 0
    assert out["scripted"]["present"] is False, "a static run must not claim a scripted result"


def test_the_object_is_found_where_it_was_when_it_was_seen():
    """Observed between the spawn and the move -> the FIRST pose is the match."""
    pred = [dict(_box(OLD_POSE), object_id="p1", observed_at=150.0)]
    out, _ = om.evaluate_geometry(_scene(pred))
    assert out["matched_objects"] == 1, "the object was exactly where the script had put it"
    assert out["false_positives"] == 0, "a correctly found scripted object is NOT a false positive"
    assert out["scripted"]["present"] is True
    assert out["scripted"]["poses"] == 2 and out["scripted"]["poses_matched"] == 1
    assert out["scripted"]["per_object"]["banana_1"] == {"poses": 2, "found": 1}


def test_the_same_box_at_the_wrong_time_does_not_match():
    """THE DISCRIMINATOR. Identical geometry, only the observation time differs.

    Seen BEFORE the spawn, the object was not there yet. A detection at that place and time is a
    real false positive and must score as one. If this passed, the gate would be measuring nothing.
    """
    pred = [dict(_box(OLD_POSE), object_id="p1", observed_at=50.0)]
    out, _ = om.evaluate_geometry(_scene(pred))
    assert out["matched_objects"] == 0, "nothing existed there before the spawn"
    assert out["false_positives"] == 1
    assert out["scripted"]["poses_matched"] == 0


def test_a_moved_object_matches_its_new_pose_and_not_its_old_one():
    pred = [dict(_box(NEW_POSE), object_id="p1", observed_at=250.0)]
    out, _ = om.evaluate_geometry(_scene(pred))
    assert out["matched_objects"] == 1, "after the move it is at the new pose"
    # and the OLD pose at the same instant is not a match
    stale = [dict(_box(OLD_POSE), object_id="p1", observed_at=250.0)]
    assert om.evaluate_geometry(_scene(stale))[0]["matched_objects"] == 0


def test_a_removed_object_is_not_expected_afterwards():
    """Nothing is predicted after the removal, and that must NOT count as a miss of a live pose."""
    out, _ = om.evaluate_geometry(_scene([]))
    assert out["matched_objects"] == 0
    assert out["scripted"]["poses_missed"] == 2, "both poses went unobserved in this run"
    # a prediction after the removal is a false positive, not a late match
    late = [dict(_box(NEW_POSE), object_id="p1", observed_at=350.0)]
    assert om.evaluate_geometry(_scene(late))[0]["false_positives"] == 1


def test_both_poses_found_is_a_full_result():
    pred = [dict(_box(OLD_POSE), object_id="p1", observed_at=150.0),
            dict(_box(NEW_POSE), object_id="p2", observed_at=250.0)]
    out, _ = om.evaluate_geometry(_scene(pred))
    assert out["matched_objects"] == 2
    assert out["scripted"]["poses_matched"] == 2 and out["scripted"]["recall_pct"] == 100.0
    assert out["scripted"]["per_object"]["banana_1"] == {"poses": 2, "found": 2}


def test_a_prediction_with_no_timestamp_is_not_silently_refused():
    """An absent time must not become an exclusion.

    Older bundles carry no observation time. Treating that as 'outside every window' would turn
    every scripted pose into a miss and every prediction into a false positive, and the report
    would look like a catastrophic regression caused entirely by a missing field.
    """
    pred = [dict(_box(OLD_POSE), object_id="p1")]      # no observed_at at all
    assert om.evaluate_geometry(_scene(pred))[0]["matched_objects"] == 1


def test_the_old_behaviour_would_have_got_these_wrong():
    """The control: with the windows stripped, the wrong-time cases stop being distinguishable.

    This is what the evaluator did before. It is asserted rather than described so that anyone who
    removes the gate sees this test fail rather than a number quietly change.
    """
    gt = _static_gt() + script_ledger.ground_truth_rows(_ledger())
    for row in gt:
        row.pop("valid_from", None)
        row.pop("valid_to", None)
    before_spawn = [dict(_box(OLD_POSE), object_id="p1", observed_at=50.0)]
    ungated, _ = om.evaluate_geometry([{"predicted_objects": before_spawn,
                                     "ground_truth_objects": gt}])
    assert ungated["matched_objects"] == 1, "without windows the wrong time still matches"
    gated, _ = om.evaluate_geometry(_scene(before_spawn))
    assert gated["matched_objects"] == 0, "with windows it correctly does not"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
