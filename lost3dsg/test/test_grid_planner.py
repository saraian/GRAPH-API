# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: Unit tests of the clearance-aware grid planner (grid_planner.py), the path follower (path_follower.py)
#         and the auto-mode candidate choice, on synthetic grids (no ROS, no Habitat needed).
# *****
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "perception_module"))

from frontier_extractor import GridInfo  # noqa: E402
from grid_planner import build_costmap, plan_path  # noqa: E402
from path_follower import follow_step  # noqa: E402

RES = 0.05
INFO = GridInfo(RES, 0.0, 0.0)
FREE, OCC, UNK = 0, 100, -1


def _corridor(width_m=1.2):
    """10 x 4 m map: an east-west corridor of the given width centred at y = 2 m, walls around it."""
    g = np.full((80, 200), OCC, np.int16)
    half = int(round(width_m / 2 / RES))
    g[40 - half:40 + half, 4:196] = FREE
    return g


def _plan(g, start, goal, params=None):
    cost, _ = build_costmap(g, INFO, params)
    return plan_path(cost, INFO, start, goal, params)


def _offsets_from(points, y_mid, x_lo, x_hi):
    pts = np.asarray(points)
    dense = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, 20):
            q = a + t * (b - a)
            if x_lo <= q[0] <= x_hi:
                dense.append(abs(q[1] - y_mid))
    return np.asarray(dense)


def test_path_stays_in_the_middle_of_a_corridor():
    # Start and goal both 0.3 m off the centre line, on the same side: a shortest path would hug that side.
    res, why = _plan(_corridor(1.2), (1.0, 1.75), (9.0, 1.75))
    assert res is not None, why
    off = _offsets_from(res["points"], 2.0, 2.5, 7.5)
    assert off.mean() < 0.1, off.mean()


def test_without_clearance_cost_the_path_hugs_the_side():
    res, _ = _plan(_corridor(1.2), (1.0, 1.75), (9.0, 1.75), {"clearance_weight": 0.0})
    off = _offsets_from(res["points"], 2.0, 2.5, 7.5)
    assert off.mean() > 0.15


def test_wall_blocks_and_unknown_is_not_traversed():
    g = _corridor(1.2)
    g[30:50, 100:103] = OCC                                        # wall across the corridor
    res, why = _plan(g, (1.0, 2.0), (9.0, 2.0))
    assert res is None and "no path" in why
    g = _corridor(1.2)
    g[30:50, 90:110] = UNK                                         # 1 m of unknown across it
    res, why = _plan(g, (1.0, 2.0), (9.0, 2.0))
    assert res is None


def test_goal_inside_inflation_is_snapped_and_reported():
    res, why = _plan(_corridor(1.2), (1.0, 2.0), (9.0, 2.55))     # 0.05 m from the wall
    assert res is not None, why
    assert 0.0 < res["snapped_goal_m"] <= 0.5
    assert res["points"][-1][1] < 2.45


def test_timing_400x400():
    rng = np.random.default_rng(0)
    g = np.full((400, 400), FREE, np.int16)
    g[rng.random(g.shape) < 0.002] = OCC
    g[:, :2] = OCC
    t0 = time.perf_counter()
    cost, _ = build_costmap(g, INFO)
    res, why = plan_path(cost, INFO, (0.5, 0.5), (19.0, 19.0))
    dt = time.perf_counter() - t0
    print(f"costmap + plan on 400x400: {dt * 1000:.0f} ms, {len(res['points'])} points")
    assert res is not None and dt < 1.0


def test_follower_turns_towards_the_path_and_arrives():
    path = [[0.0, 0.0], [2.0, 0.0]]
    a, _ = follow_step((0.0, 0.0), math.radians(90), path)         # facing +y, path along +x
    assert a == "turn_right"
    a, _ = follow_step((0.0, 0.0), math.radians(-90), path)
    assert a == "turn_left"
    a, _ = follow_step((0.0, 0.0), math.radians(3), path)          # within half a turn step
    assert a == "move_forward"
    a, rest = follow_step((1.9, 0.0), 0.0, path)
    assert a == "arrived" and rest[-1] == (2.0, 0.0)
    assert follow_step((0, 0), 0, [])[0] == "no_path"


def test_follower_drops_reached_points_and_uses_lookahead():
    path = [[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [1.0, 1.0]]
    a, rest = follow_step((0.05, 0.0), 0.0, path)
    assert rest[0] == (1.0, 0.0) and a == "move_forward"
    a, _ = follow_step((0.95, 0.0), 0.0, path)                     # corner: next target is up the y leg
    assert a == "turn_left"


def test_follower_keeps_the_path_while_turning_in_place():
    # Regression (real-agent test, 2026-11-22): with the robot behind the path start, every call dropped one
    # more point, so turning in place ate the path and the robot then drove straight at the goal.
    path = [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [-3.0, 2.0]]
    rest = path
    for _ in range(20):
        _, rest = follow_step((-0.3, 0.0), math.radians(180), rest)
    assert len(rest) >= 3 and rest[-1] == (-3.0, 2.0)


def test_choose_candidate_skips_blacklisted():
    sys.modules.setdefault("rclpy", None)
    from exploration_controller_node import choose_candidate
    cands = [{"id": 0, "viewpoint": [1.0, 1.0], "utility": 5.0},
             {"id": 1, "viewpoint": [3.0, 1.0], "utility": 2.0}]
    assert choose_candidate(cands, [], 0.5)["id"] == 0
    assert choose_candidate(cands, [[1.2, 1.1]], 0.5)["id"] == 1
    assert choose_candidate(cands, [[1.0, 1.0], [3.0, 1.0]], 0.5) is None
