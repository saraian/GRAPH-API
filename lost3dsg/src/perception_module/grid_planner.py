# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: Path planner on the rtabmap occupancy grid (the map the frontiers come from) for unknown-scene goto.
#         Traversable cells are those of frontier_extractor.frontier_mask (blind zone, seam filling, obstacle
#         inflation), and a clearance cost pushes the path towards the middle of corridors. Pure
#         numpy/cv2/skimage, no ROS import, so it can be unit-tested without a ROS environment.
# *****
"""Clearance-aware grid path planner.

cost(cell) = 1 + clearance_weight * max(0, (d_pref - d) / (d_pref - r))^2
  d      distance from the cell to the nearest occupied cell (metres)
  r      robot_radius_m (cells closer than r are not traversable)
  d_pref preferred_clearance_m: beyond it the cost is the plain path length

The minimum-cost path is found with skimage MCP_Geometric (Dijkstra over the 8-connected grid, in C), which
gives the same optimal path as A* with an admissible heuristic. The path is then simplified with
Douglas-Peucker, which keeps its centred shape (no line-of-sight shortcuts that would cut back to the walls).
"""

import math

import cv2
import numpy as np

from frontier_extractor import DEFAULT_PARAMS, cells_to_world, classify, frontier_mask, world_to_cell

DEFAULT_PLANNER_PARAMS = {
    "robot_radius_m": 0.20,
    "preferred_clearance_m": 0.60,
    "clearance_weight": 5.0,
    "allow_unknown": False,        # reserved: unknown cells are never traversed in this version
    "snap_radius_m": 1.5,          # start / goal snapping to the nearest traversable cell
    "simplify_tol_m": 0.10,        # Douglas-Peucker tolerance
}


def planner_params(params=None):
    out = dict(DEFAULT_PLANNER_PARAMS)
    out.update({k: v for k, v in (params or {}).items() if k in out and v is not None})
    return out


def build_costmap(grid, info, params=None, visited_xy=None, frontier_params=None):
    """Return (cost, traversable): float cost per cell (inf where not traversable) and the boolean mask."""
    p = planner_params(params)
    fp = dict(DEFAULT_PARAMS)
    fp.update(frontier_params or {})
    fp["robot_radius_m"] = p["robot_radius_m"]
    grid = np.asarray(grid, dtype=np.int16)
    _, traversable = frontier_mask(grid, info, fp, visited_xy)
    _, occupied, _ = classify(grid, fp)
    res = float(info.resolution)
    # Distance (metres) from every cell to the nearest occupied cell.
    dist = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float64) * res
    r = float(p["robot_radius_m"])
    d_pref = max(float(p["preferred_clearance_m"]), r + res)
    pen = np.clip((d_pref - dist) / (d_pref - r), 0.0, 1.0) ** 2
    cost = 1.0 + float(p["clearance_weight"]) * pen
    cost[~traversable] = np.inf
    return cost, traversable


def _snap(traversable, rc, radius_cells):
    """Nearest traversable cell to rc within radius_cells -> ((row, col), distance_cells) or (None, None)."""
    h, w = traversable.shape
    r0, c0 = rc
    if 0 <= r0 < h and 0 <= c0 < w and traversable[r0, c0]:
        return (r0, c0), 0.0
    r_lo, r_hi = max(0, r0 - radius_cells), min(h, r0 + radius_cells + 1)
    c_lo, c_hi = max(0, c0 - radius_cells), min(w, c0 + radius_cells + 1)
    if r_lo >= r_hi or c_lo >= c_hi:
        return None, None
    rr, cc = np.nonzero(traversable[r_lo:r_hi, c_lo:c_hi])
    if len(rr) == 0:
        return None, None
    rr, cc = rr + r_lo, cc + c_lo
    d2 = (rr - r0) ** 2 + (cc - c0) ** 2
    k = int(np.argmin(d2))
    if d2[k] > radius_cells * radius_cells:
        return None, None
    return (int(rr[k]), int(cc[k])), math.sqrt(float(d2[k]))


def _douglas_peucker(pts, tol):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = float(np.hypot(ab[0], ab[1]))
    rel = pts[1:-1] - a
    if n < 1e-9:
        d = np.hypot(rel[:, 0], rel[:, 1])
    else:
        d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / n
    k = int(np.argmax(d))
    if d[k] <= tol:
        return np.stack([a, b])
    left = _douglas_peucker(pts[:k + 2], tol)
    right = _douglas_peucker(pts[k + 1:], tol)
    return np.vstack([left[:-1], right])


def plan_path(cost, info, start_xy, goal_xy, params=None):
    """Plan on a costmap from build_costmap.

    Returns (result, reason): result is {points [[x, y], ...] map metres from start to goal, length_m, cost,
    snapped_start_m, snapped_goal_m}, or None with a short reason.
    """
    from skimage.graph import MCP_Geometric

    p = planner_params(params)
    res = float(info.resolution)
    traversable = np.isfinite(cost)
    rad = int(math.ceil(float(p["snap_radius_m"]) / res))
    s_rc, s_snap = _snap(traversable, world_to_cell(start_xy[0], start_xy[1], info), rad)
    if s_rc is None:
        return None, "start is not near traversable space"
    g_rc, g_snap = _snap(traversable, world_to_cell(goal_xy[0], goal_xy[1], info), rad)
    if g_rc is None:
        return None, "goal is not near traversable space"
    mcp = MCP_Geometric(cost)
    cum, _ = mcp.find_costs([s_rc], [g_rc], find_all_ends=False)
    if not np.isfinite(cum[g_rc]):
        return None, "no path on the known free space"
    cells = np.asarray(mcp.traceback(g_rc), dtype=np.int64)
    xs, ys = cells_to_world(cells[:, 0], cells[:, 1], info)
    pts = np.stack([xs, ys], axis=1)
    length = float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0
    simple = _douglas_peucker(pts, float(p["simplify_tol_m"]))
    # The goal keeps the requested point when it was traversable (the path ends on the cell centre otherwise).
    out = [[round(float(x), 3), round(float(y), 3)] for x, y in simple]
    if g_snap == 0.0:
        out[-1] = [round(float(goal_xy[0]), 3), round(float(goal_xy[1]), 3)]
    return {"points": out, "length_m": round(length, 3), "cost": round(float(cum[g_rc]) * res, 3),
            "snapped_start_m": round(s_snap * res, 3), "snapped_goal_m": round(g_snap * res, 3)}, ""
