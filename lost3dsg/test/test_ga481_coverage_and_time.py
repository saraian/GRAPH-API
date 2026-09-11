#!/usr/bin/env python3
"""Coverage must not be measured through walls, and the four time levers must not break the route.

GA-481 (owner 2026-09-11: 100% of the rooms, and a run that is time-optimal).

THE OLD COVERAGE NUMBER WAS OPTIMISTIC BY CONSTRUCTION. It was a Euclidean distance transform over
free space, which is straight-line distance ignoring walls, so a stop in the hall counted the room
behind the wall as covered. Measured on the 33 scenes that were short: under the radius model 35 of
71 storeys read below 100%; under line of sight, 69 did. The number moved because the measure was
wrong, not because the schedules changed.

Run: python3 test_ga481_coverage_and_time.py   (or under pytest)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voronoi_roadmap as V  # noqa: E402


def _two_rooms():
    """A wall down the middle with a doorway at the top, and a stop in the left room."""
    free = np.zeros((41, 41), bool)
    free[1:40, 1:19] = True
    free[1:40, 22:40] = True
    free[2:5, 19:22] = True
    return free, (20, 9)


def test_line_of_sight_does_not_see_through_a_wall():
    free, (sy, sx) = _two_rooms()
    right = free.copy()
    right[:, :22] = False
    vis = V.visible_from(free, sy, sx)
    leaked = int((vis & right).sum())
    assert leaked < 0.15 * int(right.sum()), \
        f"line of sight put {leaked} cells of the far room in view through a wall"


def test_the_radius_model_does_see_through_a_wall():
    """The positive control. If this ever passes, the test above proves nothing."""
    free, (sy, sx) = _two_rooms()
    right = free.copy()
    right[:, :22] = False
    rad = V.covered_mask(free, [(sy, sx)], "radius", 1000, 1000)
    assert (rad & right).sum() > 0.9 * right.sum(), \
        "the radius model no longer leaks through walls, so it is not the control it was written as"


def test_line_of_sight_still_covers_its_own_room():
    free, (sy, sx) = _two_rooms()
    own = free.copy()
    own[:, 22:] = False
    vis = V.visible_from(free, sy, sx)
    got = int((vis & own).sum()) / int(own.sum())
    assert got > 0.9, f"line of sight saw only {got:.1%} of the room the stop stands in"


def test_range_capped_sight_stops_at_the_range():
    free = np.ones((81, 81), bool)
    near = V.visible_from(free, 40, 40, max_px=10)
    far = V.visible_from(free, 40, 40, max_px=30)
    assert int(near.sum()) < int(far.sum()), "the range cap did not shorten the view"
    ys, xs = np.nonzero(near)
    assert max(math.dist((40, 40), (y, x)) for y, x in zip(ys, xs)) <= 12, \
        "a capped ray reached well past its range"


def test_top_up_reaches_the_target():
    free, (sy, sx) = _two_rooms()
    added, before, after = V.top_up_stops(free, [(sy, sx)], "los", 60, 60,
                                          target=1.0, candidates=60, n_rays=120)
    assert after > before, "the top-up added stops and covered nothing more"
    assert after >= 0.999, f"the top-up stopped at {after:.3f}, short of the target"
    assert added, "the target was already met, so this proves nothing about the top-up"


def test_top_up_respects_its_ceiling():
    """One bad storey must not produce a schedule nobody can run."""
    free, (sy, sx) = _two_rooms()
    added, _b, _a = V.top_up_stops(free, [(sy, sx)], "los", 60, 60,
                                   target=1.0, candidates=20, max_add=1, n_rays=90)
    assert len(added) <= 1, f"max_add=1 added {len(added)} stops"


def _turning(pts):
    total, prev = 0.0, None
    for a, b in zip(pts, pts[1:]):
        h = math.atan2(b[1] - a[1], b[0] - a[0])
        if prev is not None:
            d = abs(h - prev) % (2 * math.pi)
            total += math.degrees(min(d, 2 * math.pi - d))
        prev = h
    return total


def test_shortcut_removes_turning_it_can_remove():
    zig = [(x * 0.5, 0.4 if x % 2 else -0.4) for x in range(21)]
    out = V.shortcut(zig, lambda x, z: True, 0.15)
    assert _turning(out) < 0.1 * _turning(zig), "the shortcut left the zig-zag in place"
    assert len(out) < len(zig)


def test_shortcut_never_cuts_through_a_wall():
    """The whole risk of straightening a path. Checked, not assumed."""
    def clear(x, z):
        return not (4.9 < x < 5.1 and z < 1.0)

    detour = [(0.0, 0.0), (4.5, 0.0), (4.5, 1.5), (5.5, 1.5), (5.5, 0.0), (9.0, 0.0)]
    kept = V.shortcut(detour, clear, 0.15)
    for a, b in zip(kept, kept[1:]):
        n = max(2, int(math.dist(a, b) / 0.05))
        for t in range(n + 1):
            f = t / n
            assert clear(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])), \
                f"the shortcut crossed the wall between {a} and {b}"


def test_two_opt_shortens_a_crossing_order_and_keeps_the_root():
    pts = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}

    def d(a, b):
        return math.dist(pts[a], pts[b])

    bad = [0, 2, 1, 3]
    good = V.two_opt(bad, d)
    cost = lambda o: sum(d(o[i], o[i + 1]) for i in range(len(o) - 1))  # noqa: E731
    assert cost(good) < cost(bad), "2-opt did not shorten a self-crossing tour"
    assert good[0] == bad[0], "2-opt moved the root; the run starts there"
    assert sorted(good) == sorted(bad), "2-opt lost or duplicated a stop"


def test_the_turn_step_is_one_number_everywhere():
    """The schedule's frame budget divides by it and the agent turns by it. Two copies is a bundle
    whose budget describes a run that did not happen."""
    host = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "habitat_feed_host.py"), encoding="utf-8").read()
    assert "TURN_STEP_DEG" in host, "habitat_feed_host.py has no single turn-step constant"
    for literal in ('ActuationSpec(amount=10.0)', 'deg / 10.0', 'act_once("turn_left", 10.0)'):
        assert literal not in host, f"a hardcoded turn angle survives: {literal}"


def test_a_full_scan_clears_the_merge_threshold():
    """THE REASON THE TURN STEP IS 10 AND NOT 20.

    A merge commits only after merge_min_consecutive consecutive sweeps see the same pair, and a
    detection cycle takes about 3.2 s. A full 360 scan is 360/step frames at 3 f/s:

        10 deg -> 36 frames = 12.0 s = 3.75 cycles
        20 deg -> 18 frames =  6.0 s = 1.87 cycles, below a threshold of 2 at EVERY stop

    20 was shipped on 2026-09-11 for the 36% of run time scans cost, and reverted the same day
    because the saving came out of the one thing the scan exists to produce.
    """
    fps, cycle_s, need = 3.0, 3.2, 2
    frames = 360.0 / V.SCAN_STEP_DEG
    cycles = frames / fps / cycle_s
    assert cycles >= need, (
        f"a full 360 scan at {V.SCAN_STEP_DEG:.0f} deg is {frames:.0f} frames = {cycles:.2f} "
        f"detection cycles, under the {need} a merge needs. No stop could ever commit one.")


def _args(**over):
    """The generator's defaults, as a namespace, so the floor can be exercised without a navmesh."""
    import types
    d = dict(min_scan_cycles=2.0, cycle_seconds=3.2, fps_budget=3.0, turn_step_deg=V.SCAN_STEP_DEG,
             stepped_scan=False, scan_hold_frames=0, scan_tilts="0")
    d.update(over)
    return types.SimpleNamespace(**d)


def test_no_stop_may_scan_for_fewer_frames_than_a_merge_needs():
    """649 of 1900 stops fell under this when the adaptive angle was free to choose."""
    import schedule_batch as SB
    a = _args()
    floor = SB.scan_floor_frames(a)
    assert floor >= 2 * 3.2 * 3.0 - 1, f"the floor is {floor} frames, under two detection cycles"
    for tiny in (0.0, 1.0, 10.0, 45.0):
        assert SB.scan_cost_frames(tiny, a) >= floor, \
            f"a {tiny} degree scan was priced at fewer frames than the floor"


def test_the_floor_moves_with_the_measured_cycle():
    """The floor is arithmetic on a MEASURED cycle time, not a number somebody liked."""
    import schedule_batch as SB
    assert SB.scan_floor_frames(_args(cycle_seconds=6.4)) > SB.scan_floor_frames(_args()), \
        "doubling the cycle time did not raise the floor, so the floor is not derived from it"


def test_the_stepped_plan_costs_what_it_says():
    import schedule_batch as SB
    plain = SB.scan_cost_frames(360.0, _args())
    stepped = SB.scan_cost_frames(360.0, _args(stepped_scan=True))
    two_tilt = SB.scan_cost_frames(360.0, _args(stepped_scan=True, scan_tilts="30,0"))
    assert stepped > plain, "stepped costs no more than continuous, so it holds no heading"
    assert two_tilt == 2 * stepped, "a second tilt did not double the scan"
    assert SB.scan_plan(_args())["mode"] == "continuous", "stepped is not off by default"
    assert SB.scan_plan(_args())["hold_frames"] == 1, \
        "a continuous scan holds a heading for more than one frame, which is not what it is"


def test_the_feed_host_reads_the_plan_rather_than_assuming_one():
    host = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "habitat_feed_host.py"), encoding="utf-8").read()
    for name in ("scan_plan", "hold_frames", "min_scan_frames", "tilts_deg"):
        assert name in host, f"ScheduledTour ignores {name}, so the schedule's plan is not executed"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if fails else 0)
