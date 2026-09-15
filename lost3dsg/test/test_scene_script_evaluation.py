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


# --- the frame join, which the tests above cannot see ---------------------------------------
# Every test above calls evaluate_geometry directly with hand-built rows. That is why all eight
# passed while the scripted ground truth was being appended to the manifest in the WRONG FRAME:
# the bug lives in build(), and nothing here called build(). These two close that gap.

def test_scripted_rows_are_converted_to_the_manifest_frame():
    """The ledger records HABITAT coordinates; the manifest is ROS.

    MEASURED on the real /DATA/GRAPH-API/lost3dsg/FOUND-Dataset/scripts/compiled_script.json: a
    scripted pose appended raw sits 6.811 m from where it belongs, against object_metrics'
    0.5 m gate. Not "less accurate" -- no scripted object could EVER match, and a perfect tracker
    would have scored 0% scripted recall with nothing in the report to say why.
    """
    import build_hm3d_eval_manifest as b
    src = open(b.__file__).read()
    i = src.index('scripted_gt = script_ledger.ground_truth_rows')
    tail = src[i:i + 900]
    assert "_habitat_aabb_to_ros(row) for row in scripted_gt" in tail, (
        "scripted ground-truth rows must go through _habitat_aabb_to_ros, like every other row")


def test_the_conversion_keeps_the_window_and_the_flag():
    """A converter that dropped valid_from would silently disable the gate it enables."""
    import build_hm3d_eval_manifest as b
    row = script_ledger.ground_truth_rows(_ledger())[0]
    out = b._habitat_aabb_to_ros(row)
    assert out["valid_from"] == row["valid_from"] and out["valid_to"] == row["valid_to"]
    assert out["scripted"] is True
    assert out["aabb_min_m"] != row["aabb_min_m"], "the frame must actually change"


def test_the_report_says_how_much_of_the_gate_it_applied():
    """An untimed prediction is exempt from the window by design; the report must admit it."""
    timed = [dict(_box(OLD_POSE), object_id="p1", observed_at=150.0)]
    untimed = [dict(_box(OLD_POSE), object_id="p1")]
    a, _ = om.evaluate_geometry(_scene(timed))
    b_, _ = om.evaluate_geometry(_scene(untimed))
    assert a["scripted"]["untimed_predictions"] == 0 and a["scripted"]["comparable"] is True
    assert b_["scripted"]["untimed_predictions"] == 1 and b_["scripted"]["comparable"] is False
    assert b_["scripted"]["gate_applied_to_pct"] == 0.0


# --- the three the baseline review turned up ------------------------------------------------

def test_pose_recall_has_a_ceiling_and_the_report_states_it():
    """A perfectly tracked moved object CANNOT reach 100% pose recall, and that is not a failure.

    A predicted object carries one observation time and the assignment is one-to-one, so it can
    satisfy only ONE of two disjoint windows. Quoting pose recall alone would report an arithmetic
    ceiling as a tracking failure.
    """
    pred = [dict(_box(OLD_POSE), object_id="p1", observed_at=150.0)]
    out, _ = om.evaluate_geometry(_scene(pred))
    sc = out["scripted"]
    assert sc["poses"] == 2 and sc["poses_matched"] == 1
    assert sc["recall_pct"] == 50.0, "the pose number a flawless tracker gets on one move"
    assert sc["objects"] == 1 and sc["objects_found"] == 1
    assert sc["object_recall_pct"] == 100.0, "the object WAS found; only one pose was reachable"
    assert sc["pose_recall_ceiling_pct"] == 50.0, "the ceiling must be stated, not discovered"


def test_a_fabricated_size_cannot_be_quoted_as_a_volume():
    """script_ledger invents a 0.2 m cube when the runner reported no extent.

    Centre matching is unaffected, but every volume metric on such a pose is meaningless. The
    report must say so rather than degrade silently (working rule 5).
    """
    pred = [dict(_box(OLD_POSE), object_id="p1", observed_at=150.0)]
    out, _ = om.evaluate_geometry(_scene(pred))          # the fixture reports no extents
    assert out["scripted"]["box_quality_defaulted"] == 2
    assert out["scripted"]["box_quality_quotable"] is False

    measured = {"steps": [dict(st, extents=[0.18, 0.09, 0.05]) for st in _ledger()["steps"]]}
    gt = _static_gt() + script_ledger.ground_truth_rows(measured)
    out2, _ = om.evaluate_geometry([{"predicted_objects": pred, "ground_truth_objects": gt}])
    assert out2["scripted"]["box_quality_defaulted"] == 0
    assert out2["scripted"]["box_quality_quotable"] is True


def test_the_report_names_which_protocol_produced_which_key():
    """metrics_eval's table is IoU-based and time-blind; object_metrics' is time-gated.

    They are merged into one dict, so without a provenance stamp a reader cannot tell a gated
    number from an ungated one -- which matters exactly when a script ran.
    """
    import run_hm3d_metrics as r
    src = open(r.__file__).read()
    assert "protocol_provenance" in src
    assert "v1_keys_not_meaningful_here" in src
    i = src.index('report["table_iv_objects"].update(geometry)')
    assert "v1_only = sorted(" in src[:i], "the v1 key set must be captured BEFORE the merge"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
