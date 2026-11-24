#!/usr/bin/env python3
# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: ROS 2 node that scores every frontier candidate with the CRMI geometric gain (info_gain.py),
#         discounted by distance and reachability, and publishes the ranking as JSON and RViz markers.
#         Compute-and-display only: auto_goto is a reserved switch. ObjectNav can add LOST semantic
#         similarity to the geometric gain when a target is configured.
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
import threading
import time
import traceback

from frontier_extractor import GridInfo
from info_gain import (DEFAULT_GAIN_PARAMS, SemanticGain, evaluate_candidates,
                       objects_in_camera_view)

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
    from lost3dsg.srv import QueryObjects

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
            objectnav_cfg = ((self.task.get("objectnav") or {})
                             if self.task_mode == "ObjectNav" else {})
            self.objectnav_target = str(objectnav_cfg.get("target") or "").strip()
            if self.task_mode == "ObjectNav" and self.objectnav_target:
                self.params["w_sem"] = float(objectnav_cfg.get("semantic_weight", 5.0))
            elif self.task_mode == "ObjectNav":
                self.params["w_sem"] = 0.0
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
            # *****
            # Date: 2026-11-24
            # Editor: WangZX
            # Remark: ObjectNav semantic score compares visible scene-graph objects with the parsed target
            #         through LOST similarity. Target parsing and embedding warm-up run in a worker thread.
            # *****
            self.semantic = SemanticGain(icfg.get("semantic") or {})
            habitat = cfg.get("habitat") or {}
            perception = cfg.get("perception") or {}
            hfov = float(habitat.get("hfov", 90.0))
            width = max(1, int(habitat.get("width", 640)))
            height = max(1, int(habitat.get("height", 480)))
            vfov = math.degrees(2.0 * math.atan(math.tan(math.radians(hfov) / 2.0) * height / width))
            self.sem_context = {
                "target": (self.task.get("objectnav") or {}).get("target", ""),
                "target_fields": None,
                "objects": None,
                "camera": {
                    "hfov_deg": hfov,
                    "vfov_deg": vfov,
                    "pitch_deg": float(habitat.get("camera_pitch_deg", 0.0)),
                    "height_m": float(habitat.get("camera_height_m",
                                                  perception.get("floor_fallback_camera_height_m", 1.5))),
                    "max_range_m": float(perception.get("fov_max_depth_m", 1.8)),
                },
            }
            self.object_query = self.create_client(QueryObjects, "/graph/query_objects")
            self.object_query_period = 2.0
            self.last_object_query = 0.0
            self.object_query_pending = False
            self.object_generation = 0
            self.target_parse_thread = None
            if self.task_mode == "ObjectNav":
                if self.objectnav_target:
                    fallback_target = {"label": self.objectnav_target,
                                       "color": None, "material": None, "description": None}
                    self.semantic.set_target(fallback_target)
                    self.sem_context["target_fields"] = fallback_target
                    self.target_parse_thread = threading.Thread(
                        target=self.prepare_objectnav_semantics, daemon=True,
                        name="objectnav-semantic-target")
                    self.target_parse_thread.start()
                else:
                    self.get_logger().warn("ObjectNav has no target; semantic gain is inactive")

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

        def prepare_objectnav_semantics(self):
            """Load LOST once and extract explicitly stated target attributes once per run."""
            try:
                from nlp_utils import (get_embedding, lost_similarity_detailed,
                                       semantic_similarity, world2vec)
                self.sem_context.update({"get_embedding": get_embedding,
                                         "lost_similarity_detailed": lost_similarity_detailed,
                                         "semantic_similarity": semantic_similarity,
                                         "embedding_model": world2vec,
                                         "object_embeddings": {}})
                self.object_generation += 1
                self.get_logger().info(
                    f"ObjectNav LOST similarity ready (SentenceTransformer="
                    f"{bool(getattr(world2vec, 'st_model', None))}, "
                    f"Word2Vec={bool(getattr(world2vec, 'w2v_model', None))})")
            except Exception as exc:
                self.get_logger().error(f"ObjectNav LOST similarity unavailable: {type(exc).__name__}: {exc}")
                return

            prompt = ("Extract only attributes explicitly stated in this ObjectNav target. Do not infer "
                      "typical colors, materials, or properties. Return JSON with required non-empty "
                      "'label' (the object category), and optional 'color', 'material', 'description'. "
                      "The description should contain explicit identifying details beyond category, "
                      "color, and material. Use null for each optional field that is not stated. "
                      "Target: " + json.dumps(self.objectnav_target, ensure_ascii=False))
            try:
                from cv_utils import vlm_call
                raw = vlm_call(prompt, timeout=30.0, response_format={"type": "json_object"},
                               request_kind="objectnav_target_fields")
                parsed = json.loads(raw)
                fields = {"label": str(parsed.get("label") or "").strip() or self.objectnav_target,
                          "color": self._known_target_field(parsed.get("color")),
                          "material": self._known_target_field(parsed.get("material")),
                          "description": self._known_target_field(parsed.get("description"))}
                self.semantic.set_target(fields)
                self.sem_context["target_fields"] = fields
                self.object_generation += 1
                self.get_logger().info(
                    "ObjectNav target attributes parsed: " + json.dumps(fields, ensure_ascii=False))
            except Exception as exc:
                self.get_logger().warn(
                    f"ObjectNav target attribute extraction failed; using label only: "
                    f"{type(exc).__name__}: {exc}", throttle_duration_sec=10.0)

        @staticmethod
        def _known_target_field(value):
            if value is None:
                return None
            value = str(value).strip()
            return value if value and value.lower() not in {"unknown", "none", "n/a", "null"} else None

        def on_map(self, msg):
            self.map = msg

        def on_frontiers(self, msg):
            try:
                self.frontiers = json.loads(msg.data)
            except ValueError:
                self.get_logger().warn("malformed frontier JSON ignored", throttle_duration_sec=10.0)

        def tick(self):
            fr, msg = self.frontiers, self.map
            self.refresh_scene_objects()
            if fr is None or msg is None:
                return
            work_stamp = (fr.get("stamp"), self.object_generation)
            if work_stamp == self.done_stamp or time.monotonic() - self.last_run < self.min_period:
                return
            self.last_run = time.monotonic()
            try:
                self.process(fr, msg)
                self.done_stamp = work_stamp
            except Exception:
                self.get_logger().error("information gain failed:\n" + traceback.format_exc())

        def refresh_scene_objects(self):
            """Refresh persistent scene-graph object boxes without blocking scoring."""
            now = time.monotonic()
            if self.object_query_pending or now - self.last_object_query < self.object_query_period:
                return
            if not self.object_query.service_is_ready():
                return
            req = QueryObjects.Request()
            req.uncertain_only = False
            req.room_id = ""
            req.label_filter = ""
            req.area_filter = []
            self.object_query_pending = True
            self.last_object_query = now
            future = self.object_query.call_async(req)

            def on_done(done):
                self.object_query_pending = False
                try:
                    response = done.result()
                    if response.success:
                        objects = json.loads(response.serialized_json or "[]")
                        objects = [o for o in objects if isinstance(o, dict)]
                        if objects != self.sem_context.get("objects"):
                            self.sem_context["objects"] = objects
                            self.object_generation += 1
                except Exception as exc:
                    self.get_logger().warn(f"scene-graph object query failed: {exc}",
                                           throttle_duration_sec=10.0)
            future.add_done_callback(on_done)

        def process(self, fr, msg):
            info = msg.info
            if info.width <= 0 or info.height <= 0 or len(msg.data) != info.width * info.height:
                return
            grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
            gi = GridInfo(info.resolution, info.origin.position.x, info.origin.position.y,
                          _yaw(info.origin.orientation))
            out = evaluate_candidates(grid, gi, fr, self.params, self.semantic, self.sem_context)
            target_fields = self.sem_context.get("target_fields") or {}
            out.update({"stamp": time.time(), "frame_id": self.world,
                        "task": self.task_mode,
                        "target": self.objectnav_target or None,
                        "target_fields": {k: target_fields.get(k) for k in
                                          ("label", "color", "material", "description")}
                        if self.task_mode == "ObjectNav" and target_fields else None})
            # Object detection at the robot's current pose is separate from the predicted visibility
            # at frontier viewpoints. Only this current-pose check may complete ObjectNav.
            if self.task_mode == "ObjectNav" and self.objectnav_target and fr.get("robot"):
                camera = dict(self.sem_context.get("camera") or {})
                camera["grid"], camera["grid_info"] = grid, gi
                current_visible = objects_in_camera_view(
                    fr["robot"], self.sem_context.get("objects"), camera, include_nodes=True)
                _, current_matches = self.semantic.score_visible_objects(current_visible, self.sem_context)
                target = self.sem_context.get("target_fields") or {}
                required_fields = ["label"] + [
                    key for key in ("color", "material", "description") if target.get(key)]
                threshold = float(((self.task.get("objectnav") or {}).get("found_threshold", 0.80)))
                label_threshold = float(((self.task.get("objectnav") or {}).get(
                    "found_label_threshold", 0.75)))
                found = next((row for row in sorted(
                    current_matches, key=lambda item: item.get("lost_similarity", 0.0), reverse=True)
                    if row.get("lost_similarity", 0.0) >= threshold
                    and row.get("label_similarity", 0.0) >= label_threshold
                    and all((row.get("evidence") or {}).get(key) for key in required_fields)), None)
                out["target_found"] = found is not None
                if found is not None:
                    node = next(row["_node"] for row in current_visible
                                if row["object_id"] == found["object_id"])
                    found["bbox"] = node["bbox"]
                out["found_object"] = found
            best = out["candidates"][0] if out["candidates"] else None
            if best and self.params.get("w_sem", 0.0):
                best_text = (f"F{best['id']} G={best['geo_gain_bits']:.2f}b "
                             f"S={best['sem_gain']:.2f} U={best['utility']:.3f}")
            elif best:
                best_text = f"F{best['id']} G={best['geo_gain_bits']:.2f}b U={best['utility']:.3f}"
            else:
                best_text = "-"
            self.get_logger().info(
                f"candidates={len(out['candidates'])} best={best_text} "
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
                sem_text = f" S={c.get('sem_gain', 0.0):.2f}" if self.params.get("w_sem", 0.0) else ""
                t.text = f"F{c['id']} G={c['geo_gain_bits']:.1f}b{sem_text} U={c['utility']:.2f}"
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
