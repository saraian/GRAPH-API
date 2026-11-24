#!/usr/bin/env python3
# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: Exploration controller for unknown-scene tasks. (1) Plans every travelling goto of executor
#         "primitive" on the rtabmap grid with the clearance-aware grid_planner and posts the path to the feed.
#         (2) In auto mode, sends a goto to the best information-gain candidate whenever the robot is idle.
#         (3) Publishes its state for the BEV status line. Manual commands from the GUI switch the feed's
#         auto_mode off, and the controller then only plans.
# *****
"""Exploration controller node.

Subscribes : /rtabmap/map (TRANSIENT_LOCAL + RELIABLE), /exploration/candidates (JSON), TF <world> <- <base>
HTTP       : GET  <feed>/revisit_status   auto_mode, the active goto (seq, phase, target), last outcome
             POST <feed>/goto_path        {seq, points [[x, y], ...]} or {seq, error}
             GET  <feed>/action?act=goto  auto-mode goto to the chosen candidate
Publishes  : /exploration/status  std_msgs/String JSON
                 {stamp, auto, phase, target, candidate_id, path_len_m, plan_ms, last_outcome, message}

Never exits on its own (a required launch component that exits shuts the run down); idle in Building mode.
"""

import json
import math
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request

import cv2
import numpy as np

from frontier_extractor import GridInfo, cells_to_world, world_to_cell
from grid_planner import build_costmap, plan_partial, plan_path, planner_params

STATUS_TOPIC = "/exploration/status"
CANDIDATES_TOPIC = "/exploration/candidates"


def _load_cfg():
    try:
        from config import CFG
        return CFG
    except Exception:
        return {}


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def feed_url(cfg):
    """Same resolution as graph_api_bridge.FEED_HOST."""
    svc = (cfg.get("services") or {}) if isinstance(cfg, dict) else {}
    return os.environ.get("FEED_HOST") or f"http://{svc.get('feed_host', '127.0.0.1')}:{svc.get('feed_port', 7790)}"


def choose_candidate(candidates, blacklist, radius_m):
    """Highest-utility candidate whose viewpoint is not within radius_m of a blacklisted point."""
    for c in sorted(candidates or [], key=lambda c: -float(c.get("utility", 0.0))):
        vp = c.get("viewpoint") or []
        if len(vp) < 2:
            continue
        if any(math.hypot(vp[0] - b[0], vp[1] - b[1]) <= radius_m for b in blacklist):
            continue
        return c
    return None


def object_clearance(x, y, box):
    dx = np.maximum(np.maximum(box["x_min"] - x, x - box["x_max"]), 0)
    dy = np.maximum(np.maximum(box["y_min"] - y, y - box["y_max"]), 0)
    return np.hypot(dx, dy)


def object_costmap(cost, info, box, radius):
    rows, cols = np.ogrid[:cost.shape[0], :cost.shape[1]]
    x, y = cells_to_world(rows, cols, info)
    distance = object_clearance(x, y, box)
    return np.where(distance <= radius, np.inf, cost), distance


def clear_to_object(grid, info, xy, box, occ_thresh):
    # Trace to the nearest surface. The cell containing the surface may be occupied by the object.
    edge = [np.clip(xy[i], box[a + "_min"], box[a + "_max"]) for i, a in enumerate("xy")]
    cells = max(1, math.ceil(math.dist(xy, edge) / (info.resolution * 0.5)))
    end = world_to_cell(*edge, info)
    for i in range(cells):
        t = i / cells
        r, c = world_to_cell(xy[0] + t * (edge[0] - xy[0]), xy[1] + t * (edge[1] - xy[1]), info)
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            return False
        if (r, c) != end and grid[r, c] >= occ_thresh:
            return False
    return True


def object_approach_goal(cost, grid, info, robot, box, radius, stand_off, excluded, tol, frontier):
    cost, distance = object_costmap(cost, info, box, radius)
    r, c = world_to_cell(*robot[:2], info)
    reachable = np.isfinite(cost).astype(np.uint8)
    if not (0 <= r < cost.shape[0] and 0 <= c < cost.shape[1] and reachable[r, c]):
        return None
    cv2.floodFill(reachable, None, (c, r), 2, flags=8)
    candidates = ((reachable == 2) & (grid >= 0) & (grid <= frontier.get("free_thresh", 25))
                  & (distance <= stand_off))
    rr, cc = np.nonzero(candidates)
    xs, ys = cells_to_world(rr, cc, info)
    for idx in np.argsort(np.hypot(xs - robot[0], ys - robot[1])):
        xy = (float(xs[idx]), float(ys[idx]))
        if any(math.dist(xy, old) <= tol for old in excluded):
            continue
        if clear_to_object(grid, info, xy, box, frontier.get("occ_thresh", 65)):
            return xy
    return None


def _node_class():
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    class ExplorationController(Node):
        def __init__(self):
            super().__init__("exploration_controller_node")
            cfg = _load_cfg()
            task = cfg.get("task") or {}
            self.task_mode = str(task.get("mode") or "Building")
            # Named task_executor: rclpy.Node already has an `executor` property (assigning it raises).
            self.task_executor = str(task.get("executor") or "primitive")
            ecfg = cfg.get("exploration") or {}
            acfg = ecfg.get("auto") or {}
            self.pparams = planner_params(ecfg.get("planner") or {})
            self.fparams = dict(ecfg.get("frontier") or {})
            self.replan_period = float((ecfg.get("planner") or {}).get("replan_period_s", 1.0))
            self.poll_period = 1.0 / max(0.5, float(acfg.get("poll_hz", 5.0)))
            self.max_failures = int(acfg.get("max_failures", 2))
            self.blacklist_radius = float(acfg.get("blacklist_radius_m", 0.5))
            self.goal_tol = float(acfg.get("goal_tol_m", 0.3))
            self.candidate_wait_s = float(acfg.get("candidate_wait_s", 8.0))
            self.world = (cfg.get("tf") or {}).get("world_frame", "map")
            bev = cfg.get("bev") or {}
            self.base = bev.get("base_frame", "base_link")
            self.map_topic = bev.get("map_topic", "/rtabmap/map")
            self.feed = feed_url(cfg)
            self.active = self.task_mode != "Building"
            if not self.active:
                self.get_logger().info(f"exploration controller idle (task={self.task_mode})")
                return
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            self.pub_status = self.create_publisher(String, STATUS_TOPIC, latched)
            self.buffer = Buffer()
            self.listener = TransformListener(self.buffer, self)
            self.map = None
            self.cost = None
            self.cost_key = None
            self.cost_info = None
            self.candidates = None            # (receive time, payload)
            self.target_found_sent = False
            self.object_approach = None
            self.object_attempts = {}
            self.checked_match = None
            self.object_standoff = float((task.get("objectnav") or {}).get("approach_distance_m", 0.6))
            self.visited = []
            # Goto bookkeeping
            self.planned_seq = None
            self.last_plan = 0.0
            self.prev_seq = None
            self.goto_end = time.monotonic()
            self.auto_target = None           # [x, y] of the goto this node sent, until it ends
            self.sent_at = None               # when that goto was sent, until the feed reports it
            self.bootstrapped = False         # the one scan-in-place used when nothing is ranked yet
            self.failures = {}                # rounded target -> count
            self.blacklist = []
            self.reached = []                 # targets of gotos that were reached (skipped like the blacklist)
            self.state = {"auto": None, "phase": "idle", "target": None, "candidate_id": None,
                          "path_len_m": None, "plan_ms": None, "last_outcome": None, "message": "starting"}
            self.last_status_json = None
            # *****
            # Date: 2026-11-23
            # Editor: WangZX
            # Remark: Plan on the drop-off-filtered map of negative_obstacle_node (a stairwell is a wall to the
            #         planner), raw map until the first filtered one arrives.
            # *****
            from negative_obstacle_node import FILTERED_MAP_TOPIC, PreferFilteredMap, filter_enabled
            if filter_enabled(cfg):
                self.map_source = PreferFilteredMap(self, self.map_topic, FILTERED_MAP_TOPIC, self.on_map, latched)
            else:
                self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, latched)
            self.create_subscription(String, CANDIDATES_TOPIC, self.on_candidates, latched)
            self.create_timer(self.poll_period, self.tick)
            self.get_logger().info(f"exploration controller active (task={self.task_mode}, executor="
                                   f"{self.task_executor}, feed={self.feed}) planner={self.pparams}")

        # ---- inputs -------------------------------------------------------------------------------
        def on_map(self, msg):
            self.map = msg

        def on_candidates(self, msg):
            try:
                self.candidates = (time.monotonic(), json.loads(msg.data))
            except ValueError:
                pass

        def robot(self):
            try:
                tf = self.buffer.lookup_transform(self.world, self.base, Time())
            except TransformException:
                return None
            t = tf.transform.translation
            return (t.x, t.y, t.z)

        def _http(self, path, body=None, timeout=0.5):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(self.feed + path, data=data,
                                         headers={"Content-Type": "application/json"} if data else {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())

        # ---- planning -----------------------------------------------------------------------------
        def costmap(self):
            msg = self.map
            if msg is None:
                return None, None
            info = msg.info
            key = (msg.header.stamp.sec, msg.header.stamp.nanosec, info.width, info.height, len(self.visited))
            if key != self.cost_key:
                if info.width <= 0 or info.height <= 0 or len(msg.data) != info.width * info.height:
                    return None, None
                grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
                gi = GridInfo(info.resolution, info.origin.position.x, info.origin.position.y,
                              _yaw(info.origin.orientation))
                self.cost, _ = build_costmap(grid, gi, self.pparams, self.visited, self.fparams)
                self.cost_info, self.cost_key = gi, key
            return self.cost, self.cost_info

        def plan_and_post(self, seq, target, robot, object_box=None):
            t0 = time.perf_counter()
            if math.hypot(target[0] - robot[0], target[1] - robot[1]) <= self.goal_tol:
                body = {"seq": seq, "points": [[robot[0], robot[1]], [target[0], target[1]]]}
                length = 0.0
            else:
                cost, info = self.costmap()
                if cost is None:
                    body, length = {"seq": seq, "error": f"no map on {self.map_topic} yet"}, None
                else:
                    if object_box is not None:
                        cost, _ = object_costmap(cost, info, object_box, self.pparams["robot_radius_m"])
                    res, why = plan_path(cost, info, robot[:2], target[:2], self.pparams)
                    # *****
                    # Date: 2026-11-23
                    # Editor: WangZX
                    # Remark: No full path: drive the partial path towards the goal and keep re-planning (an
                    #         obstacle seen from afar, like a low lintel, often vanishes on approach).
                    # *****
                    if res is None and self.pparams.get("allow_partial"):
                        part, pwhy = plan_partial(cost, info, robot[:2], target[:2], self.pparams)
                        if part is not None:
                            res = part
                            self.get_logger().info(
                                f"goto #{seq}: no full path ({why}); partial path {part['length_m']:.2f} m, "
                                f"ends {part['remaining_m']:.2f} m from the goal", throttle_duration_sec=3.0)
                        else:
                            why = f"{why}; {pwhy}"
                    if res is None:
                        body, length = {"seq": seq, "error": why}, None
                    else:
                        body, length = {"seq": seq, "points": res["points"]}, res["length_m"]
            ms = (time.perf_counter() - t0) * 1000.0
            try:
                self._http("/goto_path", body)
            except Exception as exc:
                self.get_logger().warn(f"POST /goto_path failed: {exc}", throttle_duration_sec=10.0)
                return
            self.planned_seq, self.last_plan = seq, time.monotonic()
            self.state.update(path_len_m=length, plan_ms=round(ms, 1))
            if body.get("error"):
                self.get_logger().warn(f"goto #{seq}: no grid path ({body['error']})")
            else:
                self.get_logger().info(f"goto #{seq}: path {length:.2f} m, {len(body['points'])} points, "
                                       f"{ms:.0f} ms", throttle_duration_sec=3.0)

        # ---- main loop ----------------------------------------------------------------------------
        def tick(self):
            try:
                self._tick()
            except Exception:
                self.get_logger().error("exploration controller tick failed:\n" + traceback.format_exc(),
                                        throttle_duration_sec=10.0)
            self.publish_status()

        def _tick(self):
            robot = self.robot()
            if robot is not None and (not self.visited or math.hypot(
                    robot[0] - self.visited[-1][0], robot[1] - self.visited[-1][1]) >= 0.25):
                self.visited.append(robot[:2])
                del self.visited[:-4000]
            try:
                st = self._http("/revisit_status")
            except Exception as exc:
                self.state.update(phase="offline", message=f"feed unreachable: {type(exc).__name__}")
                return
            auto = bool(st.get("auto_mode"))
            active = st.get("active")
            self.state["auto"] = auto
            if self.target_found_sent:
                return
            if self.task_mode == "ObjectNav":
                if not auto:
                    if self.object_approach is not None:
                        self.goto_end = time.monotonic()
                    self.object_approach = None
                elif self.object_step(robot, active, st):
                    return
            seq = active.get("seq") if active else None
            # A goto ended (or was replaced): remember when, and count failures of this node's own gotos.
            if self.prev_seq is not None and seq != self.prev_seq:
                self.goto_end = time.monotonic()
                last = st.get("last") or {}
                outcome = last.get("outcome")
                self.state["last_outcome"] = (outcome + (f": {last.get('detail')}" if last.get("detail") else "")
                                              if outcome else None)
                # A reached target is skipped from now on: a stale ranking (or a frontier that stays after the
                # scan) must not send the robot back to a place it has already scanned from.
                if outcome == "reached" and last.get("target_ros"):
                    self.reached.append(list(last["target_ros"][:2]))
                # Only a goto that ended counts; one replaced by a newer goto (a manual click) did not fail.
                if self.auto_target is not None and seq is None and outcome and outcome != "reached":
                    key = (round(self.auto_target[0], 1), round(self.auto_target[1], 1))
                    self.failures[key] = self.failures.get(key, 0) + 1
                    if self.failures[key] >= self.max_failures:
                        self.blacklist.append(list(self.auto_target))
                        self.get_logger().warn(f"candidate at ({self.auto_target[0]:.2f}, "
                                               f"{self.auto_target[1]:.2f}) blacklisted after "
                                               f"{self.failures[key]} failures")
                self.auto_target = None
            self.prev_seq = seq
            # A goto this node sent but the feed never started (refused: off navmesh, off storey): count it as
            # ended once the feed has had time to report it.
            if self.sent_at is not None:
                if seq is not None:
                    self.sent_at = None
                elif time.monotonic() - self.sent_at > 3.0:
                    self.sent_at = None
                    self.goto_end = time.monotonic()
                    last = st.get("last") or {}
                    self.state["last_outcome"] = f"{last.get('outcome', 'refused')}: {last.get('detail', '')}"
                    if self.auto_target is not None:
                        self.blacklist.append(list(self.auto_target))
                        self.get_logger().warn(f"auto goto to ({self.auto_target[0]:.2f}, "
                                               f"{self.auto_target[1]:.2f}) was not started by the feed; "
                                               "blacklisted")
                    self.auto_target = None

            if active:
                target = active.get("target_ros") or [0.0, 0.0, 0.0]
                self.state.update(phase=active.get("phase"), target=[round(target[0], 2), round(target[1], 2)])
                executor = st.get("executor") or self.task_executor
                if (active.get("phase") == "travel" and executor == "primitive" and robot is not None
                        and (seq != self.planned_seq or time.monotonic() - self.last_plan >= self.replan_period)):
                    self.plan_and_post(seq, target, robot)
                self.state["message"] = ("AUTO" if auto else "MANUAL") + f" goto #{seq} {active.get('phase')}"
                return

            self.state.update(phase="idle", target=None, path_len_m=None)
            if not auto:
                self.state.update(candidate_id=None, message="MANUAL: waiting for a BEV click or AUTO")
                return
            self.auto_step(robot)

        def object_step(self, robot, active, status):
            approach = self.object_approach
            if approach is None:
                if not self.candidates or robot is None or self.map is None:
                    return False
                received, payload = self.candidates
                found = payload.get("found_object") if payload.get("target_found") else None
                if not found or not found.get("bbox") or self.checked_match == received:
                    return False
                stamp = payload.get("frontier_stamp") or payload.get("stamp", 0)
                if time.time() - stamp > max(3.0, self.candidate_wait_s):
                    return False
                cost, info = self.costmap()
                if cost is None:
                    return False
                box = found["bbox"]
                grid = np.asarray(self.map.data).reshape(cost.shape)
                failed = self.object_attempts.setdefault(found["object_id"], [])
                goal = object_approach_goal(cost, grid, info, robot, box, self.pparams["robot_radius_m"],
                                            self.object_standoff, failed, self.goal_tol, self.fparams)
                self.checked_match = received
                if goal is None:
                    return False
                if not self.send_goto(robot, goal, None):
                    return False
                approach = {"goal": goal, "object": found, "sent": time.monotonic(), "seq": None}
                self.object_approach = approach
                # This goto supersedes the frontier goto and owns its own acknowledgement and retries.
                self.auto_target = self.sent_at = self.prev_seq = None
                self.state.update(phase="approach", message=f"ObjectNav: approaching {found['label']}")
                return True

            goal, found = approach["goal"], approach["object"]
            xy = (active or {}).get("target_ros") or []
            matches = len(xy) >= 2 and math.dist(xy[:2], goal) < 0.002
            if matches:
                approach["seq"] = active["seq"]
            if robot is not None and approach["seq"] is not None and self.map is not None:
                box = found["bbox"]
                cost, info = self.costmap()
                tolerance = self.goal_tol
                if (cost is not None and math.dist(robot[:2], goal) <= tolerance
                        and self.pparams["robot_radius_m"] <= object_clearance(*robot[:2], box)
                        <= self.object_standoff + tolerance
                        and clear_to_object(np.asarray(self.map.data).reshape(cost.shape), info,
                                            robot[:2], box, self.fparams.get("occ_thresh", 65))):
                    result = self._http("/action?act=task_done")
                    if result.get("success"):
                        self.target_found_sent = True
                        self.state.update(phase="target_found", target=None, candidate_id=None,
                                          message=f"ObjectNav: reached {found['label']}")
                    return True
            if matches:
                seq = active["seq"]
                if (active.get("phase") == "travel" and robot is not None
                        and (status.get("executor") or self.task_executor) == "primitive"
                        and (seq != self.planned_seq or time.monotonic() - self.last_plan >= self.replan_period)):
                    self.plan_and_post(seq, goal, robot, found["bbox"])
                return True
            if approach["seq"] is None and time.monotonic() - approach["sent"] < 3.0:
                return True
            self.object_attempts[found["object_id"]].append(goal)
            self.object_approach = None
            self.goto_end = time.monotonic()
            return False

        def auto_step(self, robot):
            if robot is None:
                self.state["message"] = "AUTO: waiting for the robot pose"
                return
            # *****
            # Date: 2026-11-23
            # Editor: WangZX
            # Remark: Always start with one 360-degree scan in place (a goto to the robot's own position: the feed
            #         scans on arrival), then choose a frontier from the ranking of the scanned map. Before, the scan
            #         ran only when nothing was ranked yet, so a start with a few frontiers drove off at once.
            # *****
            if not self.bootstrapped:
                self.send_goto(robot, robot, None)
                if self.sent_at is not None:          # sent: the feed scans on arrival; else retry next tick
                    self.bootstrapped = True
                    self.state["message"] = "AUTO: initial 360-degree scan in place"
                return
            since_end = time.monotonic() - self.goto_end
            got = self.candidates
            fresh = got is not None and got[0] > self.goto_end
            if not fresh and since_end < self.candidate_wait_s:
                self.state["message"] = "AUTO: waiting for the candidate ranking after the scan"
                return
            cands = (got[1].get("candidates") if got else None) or []
            pick = choose_candidate(cands, self.blacklist + self.reached, self.blacklist_radius)
            if pick is None:
                self.state.update(candidate_id=None,
                                  message="AUTO: exploration complete (no reachable candidate left)")
                return
            vp = pick["viewpoint"]
            self.send_goto(robot, vp, pick.get("id"))

        def send_goto(self, robot, xy, cid):
            q = urllib.parse.urlencode({"act": "goto", "x": f"{xy[0]:.3f}", "y": f"{xy[1]:.3f}",
                                        "z": f"{robot[2]:.3f}"})
            try:
                result = self._http("/action?" + q)
                if not result.get("success", True):
                    return False
            except Exception as exc:
                self.get_logger().warn(f"auto goto failed to send: {exc}", throttle_duration_sec=10.0)
                return
            self.auto_target = [float(xy[0]), float(xy[1])]
            # Wait for the feed to report the new goto (or to refuse it) before sending another.
            self.sent_at = time.monotonic()
            self.goto_end = time.monotonic() + 1e6
            self.state.update(candidate_id=cid, target=[round(xy[0], 2), round(xy[1], 2)],
                              message=f"AUTO: goto {'F%s' % cid if cid is not None else 'here'}")
            self.get_logger().info(f"auto goto -> {'F%s' % cid if cid is not None else 'scan in place'} "
                                   f"({xy[0]:.2f}, {xy[1]:.2f})")
            return True

        def publish_status(self):
            out = dict(self.state, stamp=time.time())
            # *****
            # Date: 2026-11-23
            # Editor: WangZX
            # Remark: Also publish the points auto mode skips (blacklisted + reached targets) and the skip radius,
            #         so the BEV can mark skipped candidates and star the candidate actually chosen (the gold star
            #         used to follow info_gain's best_id even when the controller had skipped it).
            # *****
            out["skipped"] = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in self.blacklist + self.reached]
            out["skip_radius_m"] = self.blacklist_radius
            text = json.dumps(out)
            self.pub_status.publish(String(data=text))

    return ExplorationController


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    try:
        node = _node_class()()
    except Exception:
        traceback.print_exc()
        from rclpy.node import Node
        node = Node("exploration_controller_node")
        node.get_logger().error("exploration controller disabled for this run: start-up failed (traceback above)")
    from rclpy.executors import ExternalShutdownException
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
