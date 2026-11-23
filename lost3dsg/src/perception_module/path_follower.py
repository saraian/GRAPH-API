# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: Pure-pursuit follower that turns a planned path (ROS map metres) into Habitat primitive actions
#         (move_forward / turn_left / turn_right) for the task executor "primitive". Standard library only, so
#         the feed can import it and it can be unit-tested anywhere.
# *****
"""One decision per simulator frame.

ROS convention: x forward, y left, yaw counter-clockwise; habitat `turn_left` increases the ROS yaw.
"""

import math

DEFAULT_FOLLOW_PARAMS = {
    "lookahead_m": 0.4,          # pursue the first path point at least this far from the robot
    "waypoint_reached_m": 0.25,  # leading path points closer than this are dropped
    "goal_tol_m": 0.3,           # arrival: horizontal distance to the last path point
    "turn_step_deg": 10.0,       # habitat turn amount; turn while the heading error exceeds half of it
}


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def follow_step(pose_xy, yaw, path, params=None):
    """Return (action, remaining_path).

    action is "arrived", "move_forward", "turn_left", "turn_right", or "no_path" for an empty path.
    remaining_path drops the leading points already reached (the last point, the goal, is always kept).
    """
    p = dict(DEFAULT_FOLLOW_PARAMS)
    p.update({k: v for k, v in (params or {}).items() if v is not None})
    if not path:
        return "no_path", []
    x, y = float(pose_xy[0]), float(pose_xy[1])
    pts = [(float(q[0]), float(q[1])) for q in path]
    # The robot is on (or beside) the path segment nearest to it: points before that segment are behind
    # it even when it never came within waypoint_reached_m of them (a cut corner, an off-path start).
    # The carrot is measured along the path from the robot's projection onto that segment, so a robot that
    # drifted sideways steers back onto the path instead of heading for the next vertex on a parallel line.
    # A robot still before the start of the first segment (t = 0 there) has passed nothing: dropping the
    # first point then would, call after call, eat the whole path while the robot turns in place.
    proj = (x, y)
    if len(pts) > 1:
        best_k, best_d, best_t = 0, float("inf"), 0.0
        for k in range(len(pts) - 1):
            (ax, ay), (bx, by) = pts[k], pts[k + 1]
            vx, vy = bx - ax, by - ay
            n2 = vx * vx + vy * vy
            t = 0.0 if n2 < 1e-12 else max(0.0, min(1.0, ((x - ax) * vx + (y - ay) * vy) / n2))
            d = math.hypot(ax + t * vx - x, ay + t * vy - y)
            if d < best_d:
                best_k, best_d, best_t, proj = k, d, t, (ax + t * vx, ay + t * vy)
        if best_k > 0 or best_t > 0.0:
            pts = pts[best_k + 1:]
        else:
            proj = pts[0]
    while len(pts) > 1 and math.hypot(pts[0][0] - x, pts[0][1] - y) < p["waypoint_reached_m"]:
        pts.pop(0)
    gx, gy = pts[-1]
    if math.hypot(gx - x, gy - y) <= p["goal_tol_m"]:
        return "arrived", pts
    # Carrot: the point lookahead_m along the path from the robot (not the first vertex beyond lookahead_m,
    # which can lie metres past a corner and make the robot cut it into the wall).
    target = pts[-1]
    left, (cx, cy) = float(p["lookahead_m"]), proj
    for q in pts:
        seg = math.hypot(q[0] - cx, q[1] - cy)
        if seg >= left:
            target = (cx + (q[0] - cx) * left / seg, cy + (q[1] - cy) * left / seg)
            break
        left -= seg
        cx, cy = q
    err = _wrap(math.atan2(target[1] - y, target[0] - x) - float(yaw))
    half = math.radians(float(p["turn_step_deg"])) / 2.0
    if err > half:
        return "turn_left", pts
    if err < -half:
        return "turn_right", pts
    return "move_forward", pts
