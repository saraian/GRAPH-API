#!/usr/bin/env python3
# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: ROS 2 node that scores every frontier candidate with the CRMI geometric gain (info_gain.py),
#         discounted by distance and reachability, and publishes the ranking as JSON and RViz markers.
#         Compute-and-display only: auto_goto is a reserved switch. The semantic gain is a reserved
#         interface with weight 0.
# *****
"""Information gain node.

Subscribes : /exploration/frontiers  std_msgs/String JSON from frontier_node (latched)
             /rtabmap/map             nav_msgs/OccupancyGrid (TRANSIENT_LOCAL + RELIABLE)
Publishes  : /exploration/candidates         std_msgs/String JSON
                 {stamp, frame_id, frontier_stamp, compute_ms, weights, best_id,
                  candidates[{id, viewpoint, reachable, distance_m, geo_gain_bits, sem_gain, utility, rank}]}
             /exploration/candidate_markers  visualization_msgs/MarkerArray

Like frontier_node, it never exits on its own (a required component exiting shuts the run down) and
stays idle in Building mode.
"""

import json
import math
import sys
import time
import traceback

from frontier_extractor import GridInfo
from info_gain import DEFAULT_GAIN_PARAMS, SemanticGain, evaluate_candidates

FRONTIERS_TOPIC = "/exploration/frontiers"
CANDIDATES_TOPIC = "/exploration/candidates"
MARKERS_TOPIC = "/exploration/candidate_markers"


def _load_cfg():
    try:
        from config import CFG
        return CFG
    except Exception:  # config is optional for --selfcheck
        return {}


def gain_params(icfg):
    """Flatten exploration.info_gain (top level + sensor + belief) onto DEFAULT_GAIN_PARAMS."""
    flat = {}
    for section in (icfg, icfg.get("sensor") or {}, icfg.get("belief") or {}):
        flat.update({k: v for k, v in section.items() if not isinstance(v, dict)})
    return {k: flat.get(k, v) for k, v in DEFAULT_GAIN_PARAMS.items()}


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _node_class():
    import numpy as np
    from geometry_msgs.msg import Point
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import ColorRGBA, String
    from visualization_msgs.msg import Marker, MarkerArray

    class InfoGainNode(Node):
        def __init__(self):
            super().__init__("info_gain_node")
            cfg = _load_cfg()
            self.task = cfg.get("task") or {}
            self.task_mode = str(self.task.get("mode") or "Building")
            icfg = ((cfg.get("exploration") or {}).get("info_gain") or {})
            self.enabled = bool(icfg.get("enabled", True))
            self.min_period = float(icfg.get("min_period_s", 2.0))
            self.params = gain_params(icfg)
            self.world = (cfg.get("tf") or {}).get("world_frame", "map")
            self.map_topic = (cfg.get("bev") or {}).get("map_topic", "/rtabmap/map")
            self.active = self.enabled and self.task_mode != "Building"
            if not self.active:
                self.get_logger().info(
                    f"information gain idle (task={self.task_mode}, enabled={self.enabled})")
                return
            if icfg.get("auto_goto"):
                self.get_logger().warn("exploration.info_gain.auto_goto is a reserved switch, not "
                                       "implemented: candidates are computed and displayed only")
            # Semantic gain: reserved interface, only called when its weight is not 0.
            # *****
            # Date: 2026-11-22
            # Editor: WangZX
            # Remark: Named sem_context: rclpy.Node already has a read-only `context` property, and assigning
            #         it killed the node at start-up (live test, run 20261122T100236Z_dd7efad039).
            # *****
            self.semantic = SemanticGain(icfg.get("semantic") or {})
            self.sem_context = {"target": (self.task.get("objectnav") or {}).get("target", "")}

            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            self.pub_json = self.create_publisher(String, CANDIDATES_TOPIC, latched)
            self.pub_markers = self.create_publisher(MarkerArray, MARKERS_TOPIC, latched)
            self.map = None
            self.frontiers = None
            self.done_stamp = None
            self.last_run = 0.0
            # *****
            # Date: 2026-11-23
            # Editor: WangZX
            # Remark: Score on the drop-off-filtered map of negative_obstacle_node (beams stop at a stairwell
            #         instead of counting the floor below), raw map until the first filtered one arrives.
            # *****
            from negative_obstacle_node import FILTERED_MAP_TOPIC, PreferFilteredMap, filter_enabled
            if filter_enabled(cfg):
                self.map_source = PreferFilteredMap(self, self.map_topic, FILTERED_MAP_TOPIC, self.on_map, latched)
            else:
                self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, latched)
            self.create_subscription(String, FRONTIERS_TOPIC, self.on_frontiers, latched)
            self.create_timer(0.25, self.tick)
            self.get_logger().info(f"information gain active (task={self.task_mode}) params={self.params}")

        def on_map(self, msg):
            self.map = msg

        def on_frontiers(self, msg):
            try:
                self.frontiers = json.loads(msg.data)
            except ValueError:
                self.get_logger().warn("malformed frontier JSON ignored", throttle_duration_sec=10.0)

        def tick(self):
            fr, msg = self.frontiers, self.map
            if fr is None or msg is None:
                return
            if fr.get("stamp") == self.done_stamp or time.monotonic() - self.last_run < self.min_period:
                return
            self.last_run = time.monotonic()
            try:
                self.process(fr, msg)
                self.done_stamp = fr.get("stamp")
            except Exception:
                self.get_logger().error("information gain failed:\n" + traceback.format_exc())

        def process(self, fr, msg):
            info = msg.info
            if info.width <= 0 or info.height <= 0 or len(msg.data) != info.width * info.height:
                return
            grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
            gi = GridInfo(info.resolution, info.origin.position.x, info.origin.position.y,
                          _yaw(info.origin.orientation))
            out = evaluate_candidates(grid, gi, fr, self.params, self.semantic, self.sem_context)
            out.update({"stamp": time.time(), "frame_id": self.world})
            best = out["candidates"][0] if out["candidates"] else None
            self.get_logger().info(
                f"candidates={len(out['candidates'])} best="
                f"{'F%d G=%.2fb U=%.3f' % (best['id'], best['geo_gain_bits'], best['utility']) if best else '-'} "
                f"{out['compute_ms']:.0f} ms", throttle_duration_sec=5.0)
            self.pub_json.publish(String(data=json.dumps(out)))
            self.pub_markers.publish(self.markers(out, msg.header.stamp))

        def markers(self, out, stamp):
            arr = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            gold = ColorRGBA(r=1.0, g=0.8, b=0.1, a=1.0)
            white = ColorRGBA(r=0.95, g=0.95, b=0.95, a=0.95)
            for c in out["candidates"]:
                best = c["rank"] == 0
                t = Marker()
                t.header.frame_id, t.header.stamp = self.world, stamp
                t.ns, t.id, t.type, t.action = "candidate_gain", c["id"], Marker.TEXT_VIEW_FACING, Marker.ADD
                t.pose.position.x, t.pose.position.y, t.pose.position.z = c["viewpoint"][0], c["viewpoint"][1], 0.7
                t.pose.orientation.w = 1.0
                t.scale.z = 0.22 if best else 0.16
                t.color = gold if best else white
                t.text = f"F{c['id']} G={c['geo_gain_bits']:.1f}b U={c['utility']:.2f}"
                arr.markers.append(t)
                if best:
                    ring = Marker()
                    ring.header.frame_id, ring.header.stamp = self.world, stamp
                    ring.ns, ring.id, ring.type, ring.action = "best_candidate", 0, Marker.CYLINDER, Marker.ADD
                    ring.pose.position.x, ring.pose.position.y, ring.pose.position.z = \
                        c["viewpoint"][0], c["viewpoint"][1], 0.05
                    ring.pose.orientation.w = 1.0
                    ring.scale.x = ring.scale.y = 0.45
                    ring.scale.z = 0.05
                    ring.color = ColorRGBA(r=1.0, g=0.8, b=0.1, a=0.6)
                    arr.markers.append(ring)
                    arrow = Marker()
                    arrow.header.frame_id, arrow.header.stamp = self.world, stamp
                    arrow.ns, arrow.id, arrow.type, arrow.action = "best_candidate", 1, Marker.ARROW, Marker.ADD
                    arrow.points = [Point(x=float(c["viewpoint"][0]), y=float(c["viewpoint"][1]), z=1.2),
                                    Point(x=float(c["viewpoint"][0]), y=float(c["viewpoint"][1]), z=0.25)]
                    arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.06, 0.14, 0.18
                    arrow.color = gold
                    arr.markers.append(arrow)
            return arr

    return InfoGainNode


def _selfcheck():
    import numpy as np
    g = np.full((160, 160), -1, np.int16)
    g[40:120, 40:120] = 0
    payload = {"stamp": 1.0, "robot": [4.0, 4.0], "frontiers": [
        {"id": 0, "viewpoint": [5.9, 4.0], "reachable": True, "path_dist_m": 1.9},
        {"id": 1, "viewpoint": [4.0, 5.9], "reachable": False, "path_dist_m": -1.0}]}
    out = evaluate_candidates(g, GridInfo(0.05, 0.0, 0.0), payload, DEFAULT_GAIN_PARAMS)
    print(json.dumps(out)[:400])
    return 0 if out["candidates"] else 1


def main(args=None):
    if "--selfcheck" in (args or sys.argv):
        return _selfcheck()
    import rclpy
    rclpy.init(args=args)
    # *****
    # Date: 2026-11-22
    # Editor: WangZX
    # Remark: A required launch component that exits shuts the whole run down, so a start-up failure of this
    #         node is logged and replaced by an idle node instead of exiting.
    # *****
    try:
        node = _node_class()()
    except Exception:
        traceback.print_exc()
        from rclpy.node import Node
        node = Node("info_gain_node")
        node.get_logger().error("information gain disabled for this run: start-up failed (traceback above)")
    # *****
    # Date: 2026-11-22
    # Editor: WangZX
    # Remark: On SIGINT/SIGTERM rclpy shuts its context down itself (ExternalShutdownException), so shut down
    #         only if still running; otherwise the second shutdown printed a misleading traceback at run end.
    # *****
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
