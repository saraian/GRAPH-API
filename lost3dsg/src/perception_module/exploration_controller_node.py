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

from frontier_extractor import GridInfo
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


def _node_class():
    import numpy as np
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

        def plan_and_post(self, seq, target, robot):
            t0 = time.perf_counter()
            if math.hypot(target[0] - robot[0], target[1] - robot[1]) <= self.goal_tol:
                body = {"seq": seq, "points": [[robot[0], robot[1]], [target[0], target[1]]]}
                length = 0.0
            else:
                cost, info = self.costmap()
                if cost is None:
                    body, length = {"seq": seq, "error": f"no map on {self.map_topic} yet"}, None
                else:
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

        def auto_step(self, robot):
            if robot is None:
                self.state["message"] = "AUTO: waiting for the robot pose"
                return
            since_end = time.monotonic() - self.goto_end
            got = self.candidates
            fresh = got is not None and got[0] > self.goto_end
            if not fresh and since_end < self.candidate_wait_s:
                self.state["message"] = "AUTO: waiting for the candidate ranking after the scan"
                return
            cands = (got[1].get("candidates") if got else None) or []
            pick = choose_candidate(cands, self.blacklist + self.reached, self.blacklist_radius)
            if pick is not None:
                self.bootstrapped = True
            if pick is None:
                if not self.bootstrapped:
                    # Nothing to rank yet (right after start the map is tiny): scan in place once.
                    self.bootstrapped = True
                    self.send_goto(robot, robot, None)
                    self.state["message"] = "AUTO: no candidates yet, scanning in place"
                else:
                    self.state.update(candidate_id=None,
                                      message="AUTO: exploration complete (no reachable candidate left)")
                return
            vp = pick["viewpoint"]
            self.send_goto(robot, vp, pick.get("id"))

        def send_goto(self, robot, xy, cid):
            q = urllib.parse.urlencode({"act": "goto", "x": f"{xy[0]:.3f}", "y": f"{xy[1]:.3f}",
                                        "z": f"{robot[2]:.3f}"})
            try:
                self._http("/action?" + q)
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
