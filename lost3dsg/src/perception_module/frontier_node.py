#!/usr/bin/env python3
# *****
# Date: 2026-11-19
# Editor: WangZX
# Remark: ROS 2 node that extracts frontiers from /rtabmap/map and publishes them as RViz markers
#         (/exploration/frontier_markers) and JSON (/exploration/frontiers) for the bridge and the gain module.
# *****
"""Frontier node.

Subscribes : /rtabmap/map (nav_msgs/OccupancyGrid, TRANSIENT_LOCAL + RELIABLE, same as ros_bev.py)
TF         : <world_frame> <- <bev.base_frame>
Publishes  : /exploration/frontiers         std_msgs/String  JSON {stamp, frame_id, robot, frontiers[]}
             /exploration/frontier_markers  visualization_msgs/MarkerArray

A launch-file component that exits shuts the whole run down (habitat_launch._record_process_exit),
so this node never exits on its own: errors are logged and the next map is tried, and in
Building mode (known scene, fixed tour) it stays idle.
"""

import json
# *****
# Date: 2026-11-20
# Editor: WangZX
# Remark: math is used for the visited-pose spacing check.
# *****
import math
import sys
import time
import traceback

from frontier_extractor import DEFAULT_PARAMS, GridInfo, extract_frontiers

FRONTIERS_TOPIC = "/exploration/frontiers"
MARKERS_TOPIC = "/exploration/frontier_markers"


def _load_cfg():
    try:
        from config import CFG
        return CFG
    except Exception:  # config is optional for --selfcheck
        return {}


def _yaw(q):
    import math
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _node_class():
    import numpy as np
    import rclpy
    from geometry_msgs.msg import Point
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from std_msgs.msg import ColorRGBA, String
    from tf2_ros import Buffer, TransformException, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray

    class FrontierNode(Node):
        def __init__(self):
            super().__init__("frontier_node")
            cfg = _load_cfg()
            self.task_mode = str((cfg.get("task") or {}).get("mode") or "Building")
            fcfg = ((cfg.get("exploration") or {}).get("frontier") or {})
            self.enabled = bool(fcfg.get("enabled", True))
            self.params = {k: fcfg.get(k, v) for k, v in DEFAULT_PARAMS.items()}
            self.period = 1.0 / max(0.05, float(fcfg.get("rate_hz", 1.0)))
            self.world = (cfg.get("tf") or {}).get("world_frame", "map")
            bev = cfg.get("bev") or {}
            self.base = bev.get("base_frame", "base_link")
            self.map_topic = bev.get("map_topic", "/rtabmap/map")
            self.active = self.enabled and self.task_mode != "Building"
            if not self.active:
                self.get_logger().info(
                    f"frontier extraction idle (task={self.task_mode}, enabled={self.enabled})")
                return

            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            self.pub_json = self.create_publisher(String, FRONTIERS_TOPIC, latched)
            self.pub_markers = self.create_publisher(MarkerArray, MARKERS_TOPIC, latched)
            self.buffer = Buffer()
            self.listener = TransformListener(self.buffer, self)
            self.latest = None
            self.processed_stamp = None
            self.last_run = 0.0
            # *****
            # Date: 2026-11-20
            # Editor: WangZX
            # Remark: Poses the robot has stood at (every visit_spacing_m), for clearing the camera blind zone.
            # *****
            self.visited = []
            self.visit_spacing_m = float(fcfg.get("visit_spacing_m", 0.25))
            self.max_visited = 4000
            self.prev_marker_ids = 0
            # *****
            # Date: 2026-11-23
            # Editor: WangZX
            # Remark: Read the drop-off-filtered map of negative_obstacle_node (stairwells / the floor below
            #         are occupied there), falling back to the raw map until the first filtered one arrives.
            # *****
            from negative_obstacle_node import FILTERED_MAP_TOPIC, PreferFilteredMap, filter_enabled
            if filter_enabled(cfg):
                self.map_source = PreferFilteredMap(self, self.map_topic, FILTERED_MAP_TOPIC, self.on_map, latched)
            else:
                self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, latched)
            self.create_timer(0.2, self.tick)
            self.get_logger().info(
                f"frontier extraction active (task={self.task_mode}) on {self.map_topic}, "
                f"{self.world} <- {self.base}, params={self.params}")

        def on_map(self, msg):
            self.latest = msg

        def _robot_xy(self):
            try:
                tf = self.buffer.lookup_transform(self.world, self.base, Time())
            except TransformException:
                return None
            t = tf.transform.translation
            return (t.x, t.y)

        def tick(self):
            robot = self._robot_xy()
            # *****
            # Date: 2026-11-20
            # Editor: WangZX
            # Remark: Record the visited pose at every 0.2 s tick (not only when extracting), so a fast move
            #         leaves no gap; the oldest poses are dropped past max_visited.
            # *****
            if robot is not None and (not self.visited or math.hypot(
                    robot[0] - self.visited[-1][0], robot[1] - self.visited[-1][1]) >= self.visit_spacing_m):
                self.visited.append(robot)
                del self.visited[:-self.max_visited]
            msg = self.latest
            if msg is None or time.monotonic() - self.last_run < self.period:
                return
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec, msg.info.width, msg.info.height)
            # Re-run when the map changes; the robot moving alone changes reachability only.
            if stamp == self.processed_stamp and robot is None:
                return
            self.last_run = time.monotonic()
            try:
                self.process(msg, robot)
                self.processed_stamp = stamp
            except Exception:
                self.get_logger().error("frontier extraction failed:\n" + traceback.format_exc())

        def process(self, msg, robot):
            if msg.header.frame_id and msg.header.frame_id != self.world:
                self.get_logger().warn(
                    f"grid frame {msg.header.frame_id!r} != {self.world!r}; skipped",
                    throttle_duration_sec=10.0)
                return
            info = msg.info
            if info.width <= 0 or info.height <= 0 or len(msg.data) != info.width * info.height:
                return
            grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
            gi = GridInfo(info.resolution, info.origin.position.x, info.origin.position.y,
                          _yaw(info.origin.orientation))
            t0 = time.perf_counter()
            frontiers = extract_frontiers(grid, gi, robot, self.params, self.visited)
            ms = (time.perf_counter() - t0) * 1000.0
            n_reach = sum(1 for f in frontiers if f.reachable)
            self.get_logger().info(
                f"frontiers={len(frontiers)} reachable={n_reach} grid={info.width}x{info.height} "
                f"{ms:.1f} ms", throttle_duration_sec=5.0)
            payload = {
                "stamp": time.time(),
                "frame_id": self.world,
                "robot": list(robot) if robot else None,
                "extract_ms": round(ms, 2),
                "frontiers": [f.to_dict() for f in frontiers],
            }
            self.pub_json.publish(String(data=json.dumps(payload)))
            self.pub_markers.publish(self.markers(frontiers, msg.header.stamp))

        def markers(self, frontiers, stamp):
            arr = MarkerArray()
            clear = Marker()
            clear.action = Marker.DELETEALL
            arr.markers.append(clear)
            orange = ColorRGBA(r=1.0, g=0.55, b=0.0, a=0.9)
            green = ColorRGBA(r=0.1, g=0.9, b=0.3, a=0.95)
            grey = ColorRGBA(r=0.6, g=0.6, b=0.6, a=0.8)
            white = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            for f in frontiers:
                m = Marker()
                m.header.frame_id, m.header.stamp = self.world, stamp
                m.ns, m.id, m.type, m.action = "frontier_cells", f.id, Marker.POINTS, Marker.ADD
                m.scale.x = m.scale.y = 0.06
                m.color = orange
                m.pose.orientation.w = 1.0
                m.points = [Point(x=float(x), y=float(y), z=0.05) for x, y in f.cells]
                arr.markers.append(m)

                v = Marker()
                v.header.frame_id, v.header.stamp = self.world, stamp
                v.ns, v.id, v.type, v.action = "frontier_viewpoints", f.id, Marker.SPHERE, Marker.ADD
                v.pose.position.x, v.pose.position.y, v.pose.position.z = f.viewpoint[0], f.viewpoint[1], 0.1
                v.pose.orientation.w = 1.0
                v.scale.x = v.scale.y = v.scale.z = 0.18
                v.color = green if f.reachable else grey
                arr.markers.append(v)

                t = Marker()
                t.header.frame_id, t.header.stamp = self.world, stamp
                t.ns, t.id, t.type, t.action = "frontier_ids", f.id, Marker.TEXT_VIEW_FACING, Marker.ADD
                t.pose.position.x, t.pose.position.y, t.pose.position.z = f.centroid[0], f.centroid[1], 0.4
                t.pose.orientation.w = 1.0
                t.scale.z = 0.2
                t.color = white
                t.text = f"F{f.id} {f.size_m:.1f}m"
                arr.markers.append(t)
            return arr

    return FrontierNode


def _selfcheck():
    import numpy as np
    g = np.full((60, 60), -1, np.int16)
    g[10:50, 10:30] = 0
    g[10:50, 10] = 100
    fr = extract_frontiers(g, GridInfo(0.05, 0.0, 0.0), (0.8, 1.5))
    print(json.dumps([f.to_dict() for f in fr])[:400])
    return 0 if fr else 1


def main(args=None):
    if "--selfcheck" in (args or sys.argv):
        return _selfcheck()
    import rclpy
    rclpy.init(args=args)
    node = _node_class()()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
