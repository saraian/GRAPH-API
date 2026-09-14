#!/usr/bin/env python3
"""2D GVD room manager with conservative RTAB-Map 3D evidence fusion.

This version keeps the pipeline intentionally small and standard:

1. classify the occupancy grid into free / occupied;
2. compute a 2D GVD on the free space using obstacle labels;
3. prune short skeleton spurs;
4. detect doorway candidates as low-clearance skeleton branches;
5. cut the free space locally at those candidates;
6. watershed the cut free space into room regions;
7. extract polygons and track them over time.

The occupancy grid remains the source of navigability. Point clouds and
depth-derived wall segments classify which occupied structures are credible
room boundaries; they never create free space or close an observed doorway.
"""

import json
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from config import world_frame

file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = (current_dir.split('/install/')[0] if '/install/' in current_dir
                else os.path.abspath(os.path.join(current_dir, "../..")))


@dataclass
class Region:
    region_id: str
    polygon: List[List[float]]
    area_m2: float
    centroid: Tuple[float, float]
    room_id: Optional[str] = None
    misses: int = 0
    last_seen: float = field(default_factory=time.time)


class RoomManager:
    """Room manager compatible with object_manager_6.py (2D-GVD segmentation)."""

    _ROOM_PALETTE = [
        (0.90, 0.10, 0.10), (0.10, 0.60, 0.90), (0.95, 0.65, 0.05),
        (0.10, 0.80, 0.30), (0.75, 0.10, 0.85), (0.95, 0.90, 0.10),
        (0.10, 0.85, 0.85), (0.85, 0.30, 0.55), (0.45, 0.85, 0.10),
        (0.30, 0.30, 0.95), (0.95, 0.45, 0.15), (0.65, 0.65, 0.65),
    ]

    def __init__(self, w2v_model=None, node=None, map_topic='/rtabmap/map',
                 cloud_map_topic='/rtabmap/cloud_map', live_obstacle_topic=None,
                 live_empty_topic=None, live_ground_topic=None,
                 enable_live_local_grids=False,
                 cloud_ground_topic='/rtabmap/cloud_ground',
                 cloud_obstacles_topic='/rtabmap/cloud_obstacles'):
        self.w2v = w2v_model
        self.node = None
        self.tf_buffer = None
        self.tf_listener = None
        self.map_topic = map_topic
        self.cloud_map_topic = cloud_map_topic
        self.cloud_ground_topic = cloud_ground_topic
        self.cloud_obstacles_topic = cloud_obstacles_topic

        self.scene_graph: Dict[str, dict] = {}
        self.rooms = self.scene_graph
        self.regions: Dict[str, Region] = {}
        self.room_counter = 0
        self.region_counter = 0
        self.current_room_id = None
        self.previous_room_id = None
        self.current_room_changed = False
        # GA-137 FOLLOW-ON (rule 15: the room fix ARMS this). Walls are keyed BY ROOM.
        # `current_room_walls` was a single list initialised once and never cleared, so
        # every wall segment ever received accumulated into it and `finalize_current_room`
        # wrote the whole accumulation into EVERY room's record. That was invisible while
        # the segmentation produced exactly one room -- the union of all walls and the walls
        # of the only room are the same list. Switching gvd_method to `ridge` makes multiple
        # rooms real, and would have given every one of them every wall in the building.
        self._walls_by_room = {}
        # Physical walls are global. Room IDs are outputs of segmentation and may change;
        # keying observations by them made persistence depend circularly on its consumer.
        self._detected_wall_map = []
        self.current_room_wall_segments = []
        self.last_grid = None
        self.last_robot_xy = None
        self._last_save_time = 0.0
        self._last_grid_signature = None
        self._last_grid_data = None
        self._last_marker_ids = set()
        self._lock = threading.RLock()
        self._grid_sub = None
        self._marker_pub = None
        self._persistent_wall_marker_pub = None
        self._room_pub = None
        self._room_areas_pub = None
        self._cloud_sub = None
        self._latest_cloud_points = None
        self._latest_cloud_frame = None
        self._latest_cloud_stamp = None
        self._latest_cloud_received_at = None
        self._latest_cloud_ground_points = None
        self._latest_cloud_ground_received_at = None
        self._latest_cloud_ground_mask = None
        self._latest_cloud_obstacle_points = None
        self._latest_cloud_obstacle_received_at = None
        self._latest_cloud_observed_mask = None
        self._latest_cloud_nonwall_mask = None
        self._last_cloud_warn_time = 0.0
        self.last_segmentation_stats = {}
        self._pending_partition = None
        self._pending_partition_count = 0
        # Doorway hypotheses are deliberately kept separate from the GVD state.  The
        # GVD/wall detector proposes; a background VLM call is the only thing allowed
        # to promote a proposal to a room cut.
        self._doorway_vlm_executor = None
        self._doorway_vlm_pending = {}
        self._doorway_vlm_state = {}
        # Optional provider installed by object_manager_6. It must return frames
        # keyed by the requested room id; never use a global/latest camera frame
        # for room semantics.
        self._room_frame_provider = None
        self._room_vlm_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='room-vlm')
        self._room_vlm_pending = {}
        # Immutable doorway geometry. Once a doorway is validated, this registry
        # is the source of its cut; later GVD/occupancy updates must not move it.
        self._confirmed_door_cuts = {}
        self._window_vlm_pending = False
        self._window_vlm_last_call = 0.0
        self._window_vlm_state = []
        self._doorway_room_edges = {}
        self._latest_rgb = None
        self._latest_rgb_received_at = 0.0
        self._camera_info = None
        self._doorway_marker_pub = None

        from config import CFG  # local, as elsewhere in this file
        self._params = {
            # GA-137. Read from CFG so the switch is REACHABLE. `_params` is otherwise a
            # hardcoded dict and CFG["rooms"] is read nowhere in this file -- so adding the
            # key to config.py alone would have created a setting that exists and cannot be
            # reached, which is the defect class this review keeps finding. Checked before
            # shipping it, not after.
            'gvd_method': str(CFG.get('rooms', {}).get('gvd_method', 'ridge')),
            # 2D map classification
            'free_threshold': 20,
            'occupied_threshold': 50,
            'unknown_is_obstacle': True,
            'map_median_blur_ksize': 3,
            'min_room_area_m2': 1.5,
            'room_max_area_m2': 100.0,
            # Clutter filtering for topology extraction only
            'gvd_min_obstacle_length_m': 0.4,
            'gvd_wall_max_thickness_m': 0.18,
            'gvd_wall_min_aspect_ratio': 4.0,
            'gvd_wall_min_area_m2': 0.08,
            'gvd_wall_min_fill_ratio': 0.35,
            'gvd_wall_network_max_fill_ratio': 0.35,
            # Optional 3D support for structural obstacle filtering
            'enable_3d_structural_filter': bool(cloud_map_topic),
            'gvd_3d_min_points_per_cell': 4,
            'gvd_3d_min_vertical_span_m': 0.75,
            'gvd_3d_min_height_m': 0.25,
            'gvd_3d_mid_height_m': 0.80,
            'gvd_3d_high_height_m': 1.40,
            'gvd_3d_max_height_m': 2.20,
            'gvd_3d_required_height_bands': 2,
            'gvd_3d_min_points_per_band': 1,
            'gvd_3d_max_age_s': 15.0,
            'gvd_3d_floor_radius_m': 2.0,
            'gvd_3d_floor_min_points': 20,
            'gvd_3d_min_support_ratio': 0.05,
            'gvd_3d_support_dilation_px': 1,
            # Temporally fused wall_detector evidence for conservative doorway support.
            'enable_detected_wall_support': True,
            # Confirmed depth walls are stronger evidence than a compact 2D
            # occupancy blob (which may be furniture).  After an additional
            # confidence gate they can repair small free-space holes in the 2D
            # map, making the wall network usable as a room boundary.
            'detected_wall_reinforce_obstacles': True,
            # Walls must close the topology so a doorway cut can reach both wall
            # sides. Candidate generation below runs once on the unclosed topology,
            # so the VLM can inspect openings before this reinforcement is applied.
            'detected_wall_close_free_space': True,
            # A 5 cm wall raster can otherwise be crossed diagonally by the
            # 8-connected topology. This margin affects topology only.
            'detected_wall_topology_dilation_px': 1,
            # Two independent observations already satisfy the persistence gate
            # below.  Do not silently impose a second three-frame gate here.
            'detected_wall_topology_confidence': 0.33,
            'detected_wall_min_observations': 2,
            # Wall coordinates are expressed in ``map``.  Old observations may
            # be invalid after an RTAB-Map graph optimisation, so require them
            # to be seen again instead of reinforcing the topology forever.
            'detected_wall_max_age_s': 300.0,
            # Use the detector's own 0.5 m length floor. Requiring a longer
            # segment here discarded already validated, repeatedly observed wall
            # pieces (especially beside doors and partial occlusions).
            'detected_wall_min_length_m': 0.50,
            'detected_wall_min_vertical_extent_m': 0.90,
            'detected_wall_max_rms_m': 0.05,
            'detected_wall_min_confidence': 0.45,
            'detected_wall_map_max_segments': 256,
            # Viewpoint changes perturb a TLS line by several degrees/cell widths.
            # Fuse that jitter instead of restarting the observation counter.
            'detected_wall_merge_angle_deg': 15.0,
            'detected_wall_merge_distance_m': 0.22,
            'detected_wall_merge_gap_m': 0.50,
            'detected_wall_interpolation_gap_m': 0.20,
            # Keep the structural raster one cell wide. A thick band can seal a
            # narrow doorway and also makes two nearby wall surfaces look like a
            # single duplicate wall in RViz.
            'detected_wall_thickness_m': 0.06,
            # Close small acquisition gaps and imperfect corner junctions in the
            # confirmed wall network.  This is deliberately shorter than a door:
            # the network becomes topologically continuous without sealing a real
            # passage between two collinear wall pieces.
            'detected_wall_junction_gap_m': 0.30,
            'detected_wall_junction_angle_deg': 20.0,
            'detected_wall_door_endpoint_radius_m': 0.20,
            'detected_wall_door_endpoint_max_extension_m': 0.75,
            'detected_wall_door_min_support': 0.16,
            # Robust collinear wall gaps may create a topology cut without waiting
            # for a VLM result. The cut still must pass free-space and component
            # validation inside _detected_wall_doorway_cuts().
            'detected_wall_direct_door_cuts': True,
            'detected_wall_door_min_m': 0.55,
            # A measured wall pair can delimit a wide/open doorway; the generic
            # GVD door threshold remains conservative for clutter-induced gaps.
            'detected_wall_door_max_m': 2.00,
            'detected_wall_bottleneck_ratio': 0.90,
            # Beds and sofas commonly occupy 3--6 m².  Leaving those compact,
            # non-wall blobs in the topology makes the medial axis invent a room
            # around them when the cloud is sparse or temporarily stale.
            # Large furniture must be removed from the topology when obstacle
            # cloud evidence confirms it, even if the 2D blob is bigger than a
            # bed. Confirmed depth walls are protected separately.
            'gvd_topo_fill_max_area_m2': 20.0,
            'gvd_3d_nonwall_component_ratio': 0.01,
            'gvd_room_hole_fill_max_area_m2': 2.0,
            'room_nested_merge_max_area_m2': 8.0,
            'room_nested_merge_area_ratio': 0.30,
            'room_nested_merge_wall_support_max_ratio': 0.20,
            # GVD construction / pruning
            'gvd_site_min_separation_m': 0.30,
            'gvd_equidistance_tolerance_px': 1.5,
            'gvd_prune_min_branch_m': 0.35,
            'gvd_door_max_m': 1.20,
            'gvd_bottleneck_ratio': 0.80,
            # A narrow corridor produces several nearby clearance minima. Keep
            # one doorway hypothesis per local bottleneck instead of promoting
            # every raster fluctuation to a separate door.
            'gvd_door_nms_m': 1.60,
            'gvd_critical_endpoint_margin_m': 0.20,
            'gvd_cut_margin_px': 2,
            # Visual confirmation is lazy: no candidate, no image projection and no
            # request.  Calls run outside the ROS callback thread.
            'doorway_vlm_enabled': True,
            'doorway_vlm_min_confidence': 0.55,
            'doorway_vlm_min_confirmations': 1,
            'doorway_require_wall_support': True,
            'doorway_vlm_max_candidates_per_call': 8,
            'doorway_vlm_retry_s': 8.0,
            'doorway_vlm_max_image_age_s': 2.0,
            'doorway_vlm_state_max_age_s': 60.0,
            'doorway_vlm_key_resolution_m': 0.25,
            'doorway_vlm_show_rejected': False,
            'doorway_vlm_show_pending': False,
            'doorway_vlm_cluster_distance_m': 1.20,
            'doorway_vlm_camera_topic': '/camera/rgb',
            'doorway_vlm_camera_info_topic': '/camera/camera_info',
            'doorway_vlm_camera_frame': 'habitat_camera_optical',
            'window_vlm_enabled': True,
            'window_vlm_retry_s': 10.0,
            # Segmentation / region bookkeeping
            'min_region_pixels': 20,
            'region_match_distance_m': 3.0,
            'region_match_iou_min': 0.20,
            'max_region_misses': 8,
            'poly_approx_epsilon_m': 0.08,
            'room_assignment_tolerance_m': 0.12,
            # A split/merge must be stable across several map updates.  One
            # frame is especially unreliable while RTAB-Map closes a wall or
            # the depth wall detector is still accumulating evidence.
            'room_partition_change_confirmations': 3,
            # Losing a doorway cut merges identities and is much harder to undo
            # than temporarily keeping an old split.  Use asymmetric hysteresis.
            'room_split_change_confirmations': 3,
            'room_merge_change_confirmations': 8,
            # A partial/temporarily occluded map must never retire most of the
            # already segmented floor just because it repeats for a few frames.
            'room_partition_min_area_retention_ratio': 0.75,
            'room_hold_empty_partition': True,
            # A doorway cut already defines connected room components. Watershed
            # reassigns the remaining free pixels by distance and can move a room
            # boundary through an open area, so it is opt-in only.
            'room_use_watershed': False,
            # Compatible contour refinements below this IoU are also held;
            # otherwise room polygons visibly breathe at every map callback.
            'room_partition_stable_iou_min': 0.65,
            # Map callbacks are frequent. Skip a full GVD pass when occupancy is
            # unchanged or only a tiny local raster patch changed.
            'room_resegment_local_change_max_cells': 100,
            'room_resegment_local_change_max_bbox_fraction': 0.002,
            'gvd_min_robot_component_ratio': 0.15,
            # A confirmed wall may intentionally disconnect two rooms.  Keeping only the
            # robot's connected component (or the largest one) therefore made valid rooms
            # disappear after 3D wall reinforcement.  Preserve all meaningful floor
            # components and discard only small scan-noise islands.
            'gvd_keep_disconnected_rooms': True,
            'gvd_disconnected_component_min_area_m2': 1.5,
            # GA-30: the nearest-room radius used to be the literal 0.45 in
            # `max(tolerance, 0.45)`, so the tolerance knob above never reached it.
            'room_nearest_fallback_m': 0.45,
            # Persistence/output
            'save_period_s': 1.0,
            'room_stale_prune_s': 10.0,
            'discard_border_regions': False,
        }
        # GA-30. Every threshold above is reachable from config.yaml's `rooms:` block; a key
        # that is not a threshold (default_room_id) is left to its own reader.
        self._params.update({k: v for k, v in CFG.get('rooms', {}).items() if k in self._params})
        # The detector and its consumer must not silently qualify the same wall
        # against different geometry thresholds. ``walls`` is the single authority;
        # the legacy rooms keys remain accepted only when the shared key is absent.
        wall_quality = CFG.get('walls', {}) or {}
        shared = {
            'min_segment_length_m': 'detected_wall_min_length_m',
            'min_vertical_extent_m': 'detected_wall_min_vertical_extent_m',
            'max_inlier_rms_m': 'detected_wall_max_rms_m',
            'min_confidence': 'detected_wall_min_confidence',
        }
        for source, target in shared.items():
            if source in wall_quality:
                self._params[target] = wall_quality[source]

        if node is not None:
            self.bind_ros_node(node)

    # ------------------------------------------------------------------
    # ROS setup and basic helpers
    # ------------------------------------------------------------------

    def bind_ros_node(self, node):
        self.node = node
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, node)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        # cloud_ground/cloud_obstacles are often emitted by a live obstacle
        # detector with sensor-data QoS (BEST_EFFORT + VOLATILE).  Requesting
        # TRANSIENT_LOCAL here is incompatible with those publishers.  VOLATILE
        # subscribers remain compatible with RTAB-Map's RELIABLE publishers.
        cloud_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        # A REENTRANT GROUP FOR EVERYTHING THIS CLASS SUBSCRIBES TO.
        #
        # These used to land in the node's DEFAULT callback group, which is MUTUALLY EXCLUSIVE:
        # however many threads the executor has, one callback in that group runs at a time. The
        # group already holds the perception cycle (median 4.4 s, max 12.1 s on 20260911_181716)
        # and the GVD segmentation at about 1 Hz, so a cloud arriving every 0.9-5.1 s queued
        # behind them and was dropped by its depth-1 BEST_EFFORT queue before it was ever
        # scheduled.
        #
        # MEASURED end to end before changing anything: /rtabmap/cloud_map publishes every
        # 0.9-5.1 s; a subscriber with THIS class's exact QoS receives ~296,000 points per
        # message and parses all of them; the subscription is created ("3D structural filter
        # enabled" is in the log). And `_latest_cloud_points` stayed None for the whole run, so
        # room.json recorded `cloud_3d: {"source": "none", "reason": "no_fresh_cloud"}` and the
        # 3D structural filter never contributed to a single segmentation. Neither "Failed to
        # parse" nor "Ignoring stale 3D cloud" was ever logged -- nothing arrived to judge.
        #
        # The consequence was not subtle: with no 3D structure the medial-axis skeleton had 91
        # to 163 pixels for a 62 m2 storey, every doorway candidate was rejected on geometry,
        # no cut was proposed, and a storey that ground truth divides into 12 rooms stayed ONE.
        #
        # Reentrant, not a second mutually-exclusive group: the callbacks here write disjoint
        # state under `self._lock`, and a group that serialises them would reintroduce the same
        # head-of-line blocking between the grid and the cloud.
        self._cb_group = ReentrantCallbackGroup()
        self._grid_sub = node.create_subscription(
            OccupancyGrid, self.map_topic, self._slow_map_callback, qos, callback_group=self._cb_group)
        from config import CFG
        if self._params.get('doorway_vlm_enabled', True):
            image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self._rgb_sub = node.create_subscription(
                Image, self._params['doorway_vlm_camera_topic'],
                self._rgb_callback, image_qos, callback_group=self._cb_group)
            self._camera_info_sub = node.create_subscription(
                CameraInfo, self._params['doorway_vlm_camera_info_topic'],
                self._camera_info_callback, image_qos, callback_group=self._cb_group)
            self._doorway_vlm_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='doorway-vlm')
            self._log('info',
                      f"Doorway VLM enabled: image={self._params['doorway_vlm_camera_topic']}")
        self._marker_pub = node.create_publisher(MarkerArray, '/room_areas_array', qos)
        self._doorway_marker_pub = node.create_publisher(
            MarkerArray, '/room_doorway_vlm_markers', qos)
        self._window_vlm_pub = node.create_publisher(
            String, '/room_window_vlm_detections', qos)
        self._window_marker_pub = node.create_publisher(
            MarkerArray, '/room_window_vlm_markers', qos)
        self._persistent_wall_marker_pub = node.create_publisher(
            MarkerArray, '/room_detected_wall_markers', qos)
        self._room_pub = node.create_publisher(String, '/current_room', 10)
        self._room_areas_pub = node.create_publisher(String, '/room_areas', 10)
        if self.cloud_map_topic:
            self._cloud_sub = node.create_subscription(
                PointCloud2, self.cloud_map_topic, self._cloud_map_callback, cloud_qos, callback_group=self._cb_group)
            self._log(
                'info',
                f'RoomManager: 3D structural filter enabled from {self.cloud_map_topic}')
        self._cloud_ground_sub = None
        if self.cloud_ground_topic:
            self._cloud_ground_sub = node.create_subscription(
                PointCloud2, self.cloud_ground_topic, self._cloud_ground_callback, cloud_qos, callback_group=self._cb_group)
        self._cloud_obstacles_sub = None
        if self.cloud_obstacles_topic:
            self._cloud_obstacles_sub = node.create_subscription(
                PointCloud2, self.cloud_obstacles_topic,
                self._cloud_obstacles_callback, cloud_qos, callback_group=self._cb_group)
        if self.cloud_ground_topic or self.cloud_obstacles_topic:
            self._log(
                'info',
                f'RoomManager: floor/obstacle clouds enabled from '
                f'{self.cloud_ground_topic}, {self.cloud_obstacles_topic}')
        self._log(
            'info',
            f'RoomManager: subscribed to {self.map_topic} (simple 2D GVD segmentation)')

    def _rgb_callback(self, msg):
        """Keep only the newest camera frame; VLM work is never done in this callback."""
        try:
            height, width = int(msg.height), int(msg.width)
            channels = 4 if msg.encoding in ('rgba8', 'bgra8') else 3
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            raw = raw.reshape((height, int(msg.step // max(1, channels)), channels))[:, :width]
            if msg.encoding in ('bgr8', 'bgra8'):
                raw = raw[:, :, :3][:, :, ::-1]
            else:
                raw = raw[:, :, :3]
            with self._lock:
                self._latest_rgb = np.ascontiguousarray(raw)
                self._latest_rgb_received_at = time.monotonic()
        except Exception as exc:
            self._log('warn', f'Doorway VLM: cannot decode RGB frame: {exc}')

    def _camera_info_callback(self, msg):
        with self._lock:
            self._camera_info = msg

    @staticmethod
    def _rotate_point(q, point):
        qv = np.asarray([q.x, q.y, q.z], dtype=np.float64)
        v = np.asarray(point, dtype=np.float64)
        uv = np.cross(qv, v)
        uuv = np.cross(qv, uv)
        return v + 2.0 * (float(q.w) * uv + uuv)

    def _project_world_to_image(self, world_xy):
        if self.tf_buffer is None or self._camera_info is None:
            return None
        try:
            frame = self._camera_info.header.frame_id or self._params['doorway_vlm_camera_frame']
            tf = self.tf_buffer.lookup_transform(
                frame, world_frame(), rclpy.time.Time(), timeout=Duration(seconds=0.05))
            t = tf.transform.translation
            p = self._rotate_point(tf.transform.rotation,
                                   (float(world_xy[0]), float(world_xy[1]), 0.0))
            p += np.asarray([t.x, t.y, t.z], dtype=np.float64)
            if p[2] <= 0.05:
                return None
            k = self._camera_info.k
            u = float(k[0] * p[0] / p[2] + k[2])
            v = float(k[4] * p[1] / p[2] + k[5])
            if not (0 <= u < self._camera_info.width and 0 <= v < self._camera_info.height):
                return None
            return [u, v]
        except Exception:
            return None

    def _doorway_key(self, world_xy, theta=None):
        resolution = float(self._params.get('doorway_vlm_key_resolution_m', 0.25))
        resolution = max(0.05, resolution)
        key = f'{round(float(world_xy[0]) / resolution) * resolution:.2f},' \
              f'{round(float(world_xy[1]) / resolution) * resolution:.2f}'
        if theta is not None:
            sector_size = max(5.0, float(self._params.get(
                'doorway_vlm_cluster_angle_deg', 30.0)))
            sector = int(round((math.degrees(float(theta)) % 180.0) /
                               sector_size))
            key += f',a{sector}'
        return key

    @staticmethod
    def _doorway_angle_distance(theta_a, theta_b):
        """Smallest angle between two unoriented doorway lines, in radians."""
        if theta_a is None or theta_b is None:
            return 0.0
        delta = abs(float(theta_a) - float(theta_b)) % math.pi
        return min(delta, math.pi - delta)

    def _doorway_same_cluster(self, world_a, theta_a, world_b, theta_b, radius=None):
        if radius is None:
            radius = float(self._params.get('doorway_vlm_cluster_distance_m', 0.60))
        if float(np.linalg.norm(np.asarray(world_a) - np.asarray(world_b))) > radius:
            return False
        max_angle = math.radians(float(self._params.get(
            'doorway_vlm_cluster_angle_deg', 30.0)))
        return self._doorway_angle_distance(theta_a, theta_b) <= max_angle

    def _doorway_confirmed(self, world_xy, theta=None):
        key = self._nearby_doorway_key(world_xy, theta) or self._doorway_key(world_xy, theta)
        item = self._doorway_vlm_state.get(key)
        return bool(item and item.get('status') == 'confirmed')

    def _doorway_cut_confirmed(self, world_xy, theta=None):
        """Only doors/open doorways may split rooms; windows remain annotations."""
        key = self._nearby_doorway_key(world_xy, theta) or self._doorway_key(world_xy, theta)
        item = self._doorway_vlm_state.get(key)
        return bool(item and item.get('status') == 'confirmed' and item.get('cuttable'))

    def _nearby_doorway_key(self, world_xy, theta=None):
        radius = float(self._params.get('doorway_vlm_cluster_distance_m', 0.60))
        point = np.asarray(world_xy, dtype=float)
        nearest_key, nearest_distance = None, float('inf')
        for key, item in self._doorway_vlm_state.items():
            other = item.get('world')
            if other is None:
                continue
            distance = float(np.linalg.norm(point - np.asarray(other, dtype=float)))
            if (distance <= radius and distance < nearest_distance and
                    self._doorway_same_cluster(world_xy, theta, other,
                                               item.get('theta'))):
                nearest_key, nearest_distance = key, distance
        return nearest_key

    def _publish_doorway_markers(self):
        """Publish the VLM state in map coordinates for RViz."""
        if getattr(self, '_doorway_marker_pub', None) is None:
            return
        output = MarkerArray()
        clear = Marker()
        clear.header.frame_id = world_frame()
        clear.header.stamp = self.node.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        output.markers.append(clear)
        visible = [item for item in self._doorway_vlm_state.values()
                   if (item.get('status') == 'confirmed' or
                   (item.get('status') == 'pending_confirmation' and
                    self._params.get('doorway_vlm_show_pending', False))) or
                   (item.get('status') == 'rejected' and
                    self._params.get('doorway_vlm_show_rejected', False))]
        # Render one marker per physical doorway cluster, even if a previous run
        # already accumulated several nearby candidates in memory.
        compact = []
        cluster_radius = float(self._params.get('doorway_vlm_cluster_distance_m', 0.60))
        for item in sorted(visible, key=lambda value: value.get('confidence', 0.0), reverse=True):
            world = item.get('world')
            if world is None:
                continue
            if any(self._doorway_same_cluster(
                    world, item.get('theta'), other.get('world'), other.get('theta'))
                   for other in compact):
                continue
            compact.append(item)
        for marker_id, item in enumerate(compact):
            world = item.get('world')
            if not world:
                continue
            marker = Marker()
            marker.header.frame_id = world_frame()
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns = 'doorway_vlm'
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(world[0])
            marker.pose.position.y = float(world[1])
            marker.pose.position.z = 0.12
            marker.pose.orientation.w = 1.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.22
            status = item.get('status')
            if status == 'confirmed':
                marker.color.r, marker.color.g, marker.color.b = 0.1, 0.9, 0.15
            elif status == 'rejected':
                marker.color.r, marker.color.g, marker.color.b = 0.9, 0.1, 0.1
            else:
                marker.color.r, marker.color.g, marker.color.b = 0.95, 0.75, 0.05
            marker.color.a = 0.9
            output.markers.append(marker)
            label = Marker()
            label.header = marker.header
            label.ns = 'doorway_vlm_labels'
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose = marker.pose
            label.pose.position.z = 0.42
            label.scale.z = 0.18
            label.color = marker.color
            label.text = f"{status}: {item.get('type', 'unknown')}"
            output.markers.append(label)
        self._doorway_marker_pub.publish(output)

    def _encode_rgb_for_vlm(self, image):
        from PIL import Image as PILImage
        stream = io.BytesIO()
        # cv_utils.vlm_call supplies the shared PNG data-URI transport.
        PILImage.fromarray(image).save(stream, format='PNG')
        return base64.b64encode(stream.getvalue()).decode('ascii')

    def _submit_doorway_vlm(self, hypotheses, image):
        """One batched, lazy confirmation request. The worker never touches ROS state."""
        # Use the shared transport used by perception_2.  Besides keeping retries and
        # timeouts identical, it resolves REGOLO/OPENROUTER/api.txt credentials and
        # refuses to send the literal local-only ``ollama`` key to a remote endpoint.
        from cv_utils import vlm_call
        from PIL import Image as PILImage, ImageDraw
        annotated = PILImage.fromarray(image.copy())
        draw = ImageDraw.Draw(annotated)
        for hypothesis in hypotheses:
            px, py = [int(round(value)) for value in hypothesis['pixel']]
            draw.ellipse((px-10, py-10, px+10, py+10), outline=(255, 220, 0), width=3)
            draw.text((px+12, py-12), hypothesis['candidate_id'], fill=(255, 220, 0))
        encoded = self._encode_rgb_for_vlm(np.asarray(annotated))
        prompt = (
            'You are confirming geometric doorway hypotheses in the attached current robot image. '
            'For each candidate_id, inspect a small area around the supplied pixel. Confirm only '
            'a real architectural opening: a door, open doorway, or window. Reject walls, furniture '
            'gaps and occlusions. Return ONLY JSON: '
            '{"doorways":[{"candidate_id":"...","confirmed":true|false,'
            '"type":"door|window|none","bbox":[x1,y1,x2,y2],"confidence":0.0}]} .\n'
            'The bbox must be pixel coordinates in the original image. A candidate is confirmed '
            'only when its pixel is near the centre of a tight returned bbox. Do not use a '
            'frame-sized bbox. Windows may be reported, but they are not room passages.\n'
            'Candidates:\n' +
            json.dumps(hypotheses, ensure_ascii=False))
        raw = (vlm_call(prompt, encoded) or '').strip()
        raw = re.sub(r'^```json\s*|```$', '', raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return data.get('doorways', []) if isinstance(data, dict) else []

    def _submit_window_vlm(self, image):
        """Detect windows directly from RGB, without GVD hypotheses."""
        from cv_utils import vlm_call
        encoded = self._encode_rgb_for_vlm(image)
        prompt = (
            'Inspect the full current robot RGB image and detect visible architectural windows. '
            'Do not infer windows from walls, furniture, reflections, screens, paintings, or '
            'bright openings. Return ONLY JSON in this exact form: '
            '{"windows":[{"type":"window","bbox":[x1,y1,x2,y2],'
            '"confidence":0.0}]} . The bbox must be tight pixel coordinates in the original image. '
            'Return an empty list when no real window is visible.')
        raw = (vlm_call(prompt, encoded) or '').strip()
        raw = re.sub(r'^```json\s*|```$', '', raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return data.get('windows', []) if isinstance(data, dict) else []

    def _on_window_vlm_done(self, future):
        try:
            detections = future.result()
            valid = []
            for item in detections:
                if not isinstance(item, dict):
                    continue
                bbox = item.get('bbox', [])
                try:
                    x0, y0, x1, y1 = [float(value) for value in bbox]
                    confidence = float(item.get('confidence', 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if x1 <= x0 or y1 <= y0 or confidence < 0.55:
                    continue
                valid.append({'type': 'window',
                              'bbox': [x0, y0, x1, y1],
                              'confidence': confidence,
                              'updated_at': time.time()})
            self._window_vlm_state = valid
            if getattr(self, '_window_vlm_pub', None) is not None:
                self._window_vlm_pub.publish(String(
                    data=json.dumps({'windows': self._json_safe(valid)},
                                    ensure_ascii=False)))
            self._publish_window_markers(valid)
        except Exception as exc:
            self._log('warn', f'Window VLM failed: {exc}')
        finally:
            self._window_vlm_pending = False

    def _publish_window_markers(self, windows):
        """Draw RGB bounding boxes as camera-frame markers in RViz."""
        if (getattr(self, '_window_marker_pub', None) is None or
                self._camera_info is None or self.node is None):
            return
        output = MarkerArray()
        header = Marker().header
        header.frame_id = self._camera_info.header.frame_id or self._params.get(
            'doorway_vlm_camera_frame', 'camera_optical_frame')
        header.stamp = self.node.get_clock().now().to_msg()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        output.markers.append(clear)
        fx, fy = float(self._camera_info.k[0]), float(self._camera_info.k[4])
        cx, cy = float(self._camera_info.k[2]), float(self._camera_info.k[5])
        depth = 1.0
        for marker_id, item in enumerate(windows):
            x0, y0, x1, y1 = item['bbox']
            points = []
            for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
                point = Point()
                point.x = (float(u) - cx) * depth / max(fx, 1e-6)
                point.y = (float(v) - cy) * depth / max(fy, 1e-6)
                point.z = depth
                points.append(point)
            marker = Marker()
            marker.header = header
            marker.ns = 'window_vlm'
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.points = points
            marker.scale.x = 0.015
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.1, 0.9, 1.0, 0.95
            output.markers.append(marker)
            label = Marker()
            label.header = header
            label.ns = 'window_vlm_labels'
            label.id = marker_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = points[0]
            label.pose.position.z += 0.08
            label.pose.orientation.w = 1.0
            label.scale.z = 0.08
            label.color = marker.color
            label.text = f"window ({item['confidence']:.2f})"
            output.markers.append(label)
        self._window_marker_pub.publish(output)

    def _schedule_window_vlm(self):
        """Run an independent, throttled RGB-only window detector."""
        if (not self._params.get('window_vlm_enabled', True) or
                getattr(self, '_doorway_vlm_executor', None) is None or
                self._window_vlm_pending or self._doorway_vlm_pending):
            return
        now = time.time()
        if now - float(self._window_vlm_last_call) < float(
                self._params.get('window_vlm_retry_s', 10.0)):
            return
        with self._lock:
            image = None if self._latest_rgb is None else self._latest_rgb.copy()
            image_age = time.monotonic() - self._latest_rgb_received_at
        if image is None or image_age > float(self._params.get(
                'doorway_vlm_max_image_age_s', 2.0)):
            return
        self._window_vlm_pending = True
        self._window_vlm_last_call = now
        future = self._doorway_vlm_executor.submit(self._submit_window_vlm, image)
        future.add_done_callback(self._on_window_vlm_done)

    def _on_doorway_vlm_done(self, key, hypotheses, future):
        resegment_grid = None
        try:
            answers = future.result()
            by_id = {str(item.get('candidate_id')): item for item in answers
                     if isinstance(item, dict)}
            with self._lock:
                for hypothesis in hypotheses:
                    answer = by_id.get(hypothesis['candidate_id'], {})
                    bbox = answer.get('bbox', [])
                    confidence = float(answer.get('confidence', 0.0) or 0.0)
                    try:
                        x0, y0, x1, y1 = [float(value) for value in bbox]
                        bw, bh = max(0.0, x1-x0), max(0.0, y1-y0)
                        px, py = hypothesis['pixel']
                        # A huge bbox is not useful confirmation.  The candidate must
                        # be near the visual centre, not merely somewhere in the frame.
                        inside = (bw > 2.0 and bh > 2.0 and x0 <= px <= x1 and y0 <= py <= y1
                                  and abs(px-(x0+x1)*0.5) <= 0.75*bw
                                  and abs(py-(y0+y1)*0.5) <= 0.75*bh)
                    except (TypeError, ValueError):
                        inside = False
                        bbox = []
                    object_type = str(answer.get('type', 'none')).lower().strip()
                    normalized_type = object_type.replace('_', ' ').replace('-', ' ')
                    grid = self.last_grid
                    wall_score = 0.0
                    if grid is not None:
                        q = self._world_to_grid(
                            float(hypothesis['world'][0]),
                            float(hypothesis['world'][1]), grid)
                        if q is not None:
                            wall_score = self._door_wall_support_score(
                                int(q[1]), int(q[0]),
                                float(hypothesis.get('theta', 0.0)),
                                float(hypothesis.get('radius_px', 1.0)),
                                float(grid.info.resolution))
                    # A VLM-positive opening is a doorway only when both ends of
                    # the geometric opening have structural wall support.
                    wall_confirmed = wall_score >= float(self._params.get(
                        'detected_wall_door_min_support', 0.16))
                    previous = self._doorway_vlm_state.get(hypothesis['key'], {})
                    previously_validated = (
                        previous.get('status') == 'confirmed' and
                        previous.get('cuttable', False))
                    cuttable = (wall_confirmed or previously_validated) and normalized_type in (
                        'door', 'doorway', 'open doorway', 'opening', 'passage',
                        'open passage')
                    visual_confirmed = str(answer.get('confirmed', 'false')).lower() == 'true'
                    positive = (visual_confirmed and inside and
                                (wall_confirmed or previously_validated) and
                                confidence >= float(self._params['doorway_vlm_min_confidence']) and
                                normalized_type in (
                                    'door', 'doorway', 'window', 'open doorway',
                                'opening', 'passage', 'open passage'))
                    existing_lock = self._confirmed_door_cuts.get(hypothesis['key'])
                    if positive and existing_lock is None:
                        self._confirmed_door_cuts[hypothesis['key']] = {
                            'world': tuple(hypothesis['world']),
                            'left_world': (None if hypothesis.get('left_world') is None
                                           else tuple(hypothesis['left_world'])),
                            'right_world': (None if hypothesis.get('right_world') is None
                                            else tuple(hypothesis['right_world'])),
                            'theta': float(hypothesis.get('theta', 0.0)),
                            'created_at': time.time(),
                        }
                    lock = self._confirmed_door_cuts.get(hypothesis['key'])
                    geometry = lock or {
                        'world': tuple(hypothesis['world']),
                        'left_world': hypothesis.get('left_world'),
                        'right_world': hypothesis.get('right_world'),
                        'theta': float(hypothesis.get('theta', 0.0)),
                    }
                    positive_votes = (int(previous.get('positive_votes', 0)) + 1
                                      if positive else 0)
                    # One valid visual confirmation is immediately permanent until
                    # an explicit map reset. There is no time-based recheck.
                    confirmation_count = 1 if positive else int(
                        previous.get('confirmation_count', 0))
                    confirmed = positive or previous.get('status') == 'confirmed'
                    permanent = confirmed
                    recheck_at = 0.0
                    self._doorway_vlm_state[hypothesis['key']] = {
                        'status': ('confirmed' if confirmed
                                   else 'pending_confirmation' if positive else 'rejected'),
                        'cuttable': cuttable,
                        'positive_votes': positive_votes,
                        'confirmation_count': confirmation_count,
                        'permanent_confirmed': permanent,
                        'recheck_at': recheck_at,
                        'world': geometry['world'], 'pixel': hypothesis['pixel'],
                        'theta': float(geometry.get('theta', hypothesis.get('theta', 0.0))),
                        'radius_px': float(hypothesis.get('radius_px', 1.0)),
                        'left_world': geometry.get('left_world'),
                        'right_world': geometry.get('right_world'),
                        'type': object_type, 'confidence': confidence,
                        'wall_support_score': wall_score,
                        'bbox': bbox, 'updated_at': time.time(),
                    }
                self._doorway_vlm_pending.pop(key, None)
                self._publish_doorway_markers()
                resegment_grid = self.last_grid
        except Exception as exc:
            with self._lock:
                self._doorway_vlm_pending.pop(key, None)
            self._log('warn', f'Doorway VLM failed: {exc}')
        # Do not wait for a future map tick to make a confirmed doorway useful.  The
        # expensive VLM call is already complete; this short geometry pass runs off the
        # ROS subscription callback and is serialized by the same room lock.
        if resegment_grid is not None:
            self.process_grid(resegment_grid, True, force_resegment=True)

    def _schedule_doorway_confirmations(self, grid, points):
        if (not self._params.get('doorway_vlm_enabled', True) or
                getattr(self, '_doorway_vlm_executor', None) is None or not points):
            return
        # One in-flight batch is enough.  Without this guard every map callback
        # queued another batch while the previous request was still running.
        if self._doorway_vlm_pending:
            return
        with self._lock:
            image = None if self._latest_rgb is None else self._latest_rgb.copy()
            image_age = time.monotonic() - self._latest_rgb_received_at
        if image is None or image_age > float(self._params['doorway_vlm_max_image_age_s']):
            return
        hypotheses = []
        batch_worlds = []
        for index, point in enumerate(points[:int(self._params['doorway_vlm_max_candidates_per_call'])]):
            world = self._grid_to_world(point[1], point[0], grid)
            pixel = self._project_world_to_image(world)
            if pixel is None:
                continue
            theta = float(point[2])
            key = self._doorway_key(world, theta)
            nearby_key = self._nearby_doorway_key(world, theta)
            if nearby_key is not None:
                key = nearby_key
            if any(self._doorway_same_cluster(
                    world, theta, other[0], other[1]) for other in batch_worlds):
                continue
            previous = self._doorway_vlm_state.get(key, {})
            if previous.get('status') == 'confirmed':
                continue
            if key in self._doorway_vlm_pending:
                continue
            if (previous.get('status') in ('rejected', 'pending_confirmation') and
                    time.time() - float(previous.get('updated_at', 0.0)) <
                    float(self._params['doorway_vlm_retry_s'])):
                continue
            hypotheses.append({
                'candidate_id': f'c{index}', 'key': key,
                'world': world, 'pixel': pixel,
                'theta': theta, 'radius_px': float(point[3]),
                'left_world': self._grid_to_world(
                    point[1] - point[3] * math.sin(float(point[2])),
                    point[0] + point[3] * math.cos(float(point[2])), grid),
                'right_world': self._grid_to_world(
                    point[1] + point[3] * math.sin(float(point[2])),
                    point[0] - point[3] * math.cos(float(point[2])), grid),
                'gvd_width_m': round(2.0 * float(point[3]) * float(grid.info.resolution), 3),
            })
            batch_worlds.append((world, theta))
        if not hypotheses:
            return
        batch_key = '|'.join(item['key'] for item in hypotheses)
        self._doorway_vlm_pending[batch_key] = True
        future = self._doorway_vlm_executor.submit(
            self._submit_doorway_vlm, hypotheses, image)
        future.add_done_callback(
            lambda completed: self._on_doorway_vlm_done(batch_key, hypotheses, completed))
        self._publish_doorway_markers()

    def _log(self, level, text):
        if self.node is None:
            return
        logger = self.node.get_logger()
        if level in ('warn', 'warning'):
            logger.warning(text)
        elif level == 'error':
            logger.error(text)
        elif level == 'debug':
            logger.debug(text)
        else:
            logger.info(text)

    @staticmethod
    def _yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _robot_pose(self):
        if self.tf_buffer is None:
            return None
        for frame in ('base_link', 'base_footprint', 'robot_base'):
            try:
                tf = self.tf_buffer.lookup_transform(
                    world_frame(), frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.3))
                return float(tf.transform.translation.x), float(tf.transform.translation.y)
            except Exception:
                continue
        return None

    def _robot_height(self):
        """Return the current base height in map coordinates.

        RTAB-Map's cloud is global.  Restricting it to a height range relative
        to the current floor prevents another storey from being projected onto
        the same 2D occupancy grid.
        """
        if self.tf_buffer is None:
            return 0.0
        for frame in ('base_link', 'base_footprint', 'robot_base'):
            try:
                tf = self.tf_buffer.lookup_transform(
                    world_frame(), frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.3))
                return float(tf.transform.translation.z)
            except Exception:
                continue
        return 0.0

    @staticmethod
    def _transform_points(points, transform):
        """Vectorised geometry_msgs Transform application for an N x 3 cloud."""
        q = transform.rotation
        qv = np.asarray([q.x, q.y, q.z], dtype=np.float64)
        qw = float(q.w)
        # Quaternion rotation: v' = v + 2(qw(qv x v) + qv x (qv x v)).
        uv = np.cross(np.broadcast_to(qv, points.shape), points)
        uuv = np.cross(np.broadcast_to(qv, points.shape), uv)
        rotated = points + 2.0 * (qw * uv + uuv)
        t = transform.translation
        return rotated + np.asarray([t.x, t.y, t.z], dtype=np.float64)

    def _grid_to_world(self, px, py, grid):
        origin = grid.info.origin
        yaw = self._yaw(origin.orientation)
        x, y = px * grid.info.resolution, py * grid.info.resolution
        return [origin.position.x + math.cos(yaw)*x - math.sin(yaw)*y,
                origin.position.y + math.sin(yaw)*x + math.cos(yaw)*y]

    def _world_to_grid(self, wx, wy, grid):
        origin = grid.info.origin
        yaw = self._yaw(origin.orientation)
        dx, dy = wx-origin.position.x, wy-origin.position.y
        x = math.cos(yaw)*dx + math.sin(yaw)*dy
        y = -math.sin(yaw)*dx + math.cos(yaw)*dy
        if grid.info.resolution <= 0:
            return None
        return int(round(x/grid.info.resolution)), int(round(y/grid.info.resolution))

    def _cloud_map_callback(self, msg: PointCloud2):
        self._cloud_callback(msg, 'map')

    def _cloud_ground_callback(self, msg: PointCloud2):
        self._cloud_callback(msg, 'ground')

    def _cloud_obstacles_callback(self, msg: PointCloud2):
        self._cloud_callback(msg, 'obstacles')

    def _cloud_callback(self, msg: PointCloud2, kind='map'):
        try:
            raw_pts = list(point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
            if raw_pts:
                points = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in raw_pts], dtype=np.float64)
            else:
                points = np.empty((0, 3), dtype=np.float64)
        except Exception as exc:
            self._log('warn', f'Failed to parse 3D cloud: {exc}')
            return

        if points.size == 0:
            with self._lock:
                if kind == 'ground':
                    self._latest_cloud_ground_points = None
                    self._latest_cloud_ground_received_at = time.monotonic()
                elif kind == 'obstacles':
                    self._latest_cloud_obstacle_points = None
                    self._latest_cloud_obstacle_received_at = time.monotonic()
                else:
                    self._latest_cloud_points = None
                    self._latest_cloud_frame = msg.header.frame_id or world_frame()
                    self._latest_cloud_stamp = msg.header.stamp
                    self._latest_cloud_received_at = time.monotonic()
            return

        if points.ndim == 1:
            points = points.reshape(1, -1)
        points = points[:, :3].astype(np.float64, copy=False)

        target_frame = world_frame().lstrip('/')
        source_frame = (msg.header.frame_id or target_frame).lstrip('/')
        if source_frame != target_frame:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target_frame, source_frame, msg.header.stamp,
                    timeout=Duration(seconds=0.3))
                points = self._transform_points(points, tf.transform)
            except Exception as exc:
                now = time.time()
                if now - self._last_cloud_warn_time > 5.0:
                    self._log('warn', f'Cannot transform 3D cloud {target_frame}<-{source_frame}: {exc}')
                    self._last_cloud_warn_time = now
                return

        with self._lock:
            received_at = time.monotonic()
            if kind == 'ground':
                self._latest_cloud_ground_points = points
                self._latest_cloud_ground_received_at = received_at
            elif kind == 'obstacles':
                self._latest_cloud_obstacle_points = points
                self._latest_cloud_obstacle_received_at = received_at
            else:
                self._latest_cloud_points = points
                self._latest_cloud_frame = target_frame
                self._latest_cloud_stamp = msg.header.stamp
                self._latest_cloud_received_at = received_at

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _polygon_area(poly):
        if len(poly) < 3:
            return 0.0
        p = np.asarray(poly, dtype=np.float64)
        return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                         np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2.0)

    @staticmethod
    def _centroid(poly):
        """Area centroid (shoelace). GA-30: the vertex mean moved toward whichever side had
        more corners, so the room 'centre' jumped as detail was added."""
        p = np.asarray(poly, dtype=np.float64)
        x, y = p[:, 0], p[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        a = cross.sum() / 2.0
        if abs(a) < 1e-9:
            return float(x.mean()), float(y.mean())
        return float(((x + xn) * cross).sum() / (6.0 * a)), float(((y + yn) * cross).sum() / (6.0 * a))

    @staticmethod
    def _polygon_edges(poly):
        return [[list(poly[i]), list(poly[(i+1) % len(poly)])]
                for i in range(len(poly))] if len(poly) >= 2 else []

    @staticmethod
    def _point_in_polygon(poly, xy, tolerance_m=0.0):
        if len(poly) < 3:
            return False
        distance = cv2.pointPolygonTest(
            np.asarray(poly, dtype=np.float32),
            (float(xy[0]), float(xy[1])), True)
        return distance >= -max(0.0, float(tolerance_m))

    @staticmethod
    def _polygon_iou(poly_a, poly_b, resolution=0.05):
        if len(poly_a) < 3 or len(poly_b) < 3:
            return 0.0
        a, b = np.asarray(poly_a, float), np.asarray(poly_b, float)
        lo, hi = np.minimum(a.min(0), b.min(0)), np.maximum(a.max(0), b.max(0))
        width, height = np.ceil((hi-lo)/resolution).astype(int) + 2
        width, height = max(1, int(width)), max(1, int(height))
        if width * height > 2_000_000:
            resolution *= math.sqrt((width*height)/2_000_000)
            width, height = np.ceil((hi-lo)/resolution).astype(int) + 2
            width, height = max(1, int(width)), max(1, int(height))
        def px(poly):
            return np.rint((np.asarray(poly)-lo)/resolution).astype(np.int32)
        ma = np.zeros((height, width), np.uint8)
        mb = np.zeros((height, width), np.uint8)
        cv2.fillPoly(ma, [px(poly_a)], 255)
        cv2.fillPoly(mb, [px(poly_b)], 255)
        union = np.count_nonzero((ma > 0) | (mb > 0))
        return float(np.count_nonzero((ma > 0) & (mb > 0))/union) if union else 0.0

    @staticmethod
    def _region_shape_metrics(mask, resolution):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'area_m2': 0.0, 'perimeter_m': 0.0, 'aspect_ratio': 0.0, 'width_proxy_m': 0.0}
        contour = max(contours, key=cv2.contourArea)
        area_px = float(cv2.contourArea(contour))
        perimeter_px = float(cv2.arcLength(contour, True))
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        short_side_px = max(1.0, min(rw, rh))
        long_side_px = max(rw, rh)
        return {
            'area_m2': area_px * resolution * resolution,
            'perimeter_m': perimeter_px * resolution,
            'aspect_ratio': float(long_side_px / short_side_px),
            'width_proxy_m': float((2.0 * area_px / max(perimeter_px, 1e-6)) * resolution),
        }

    def _is_corridor_like(self, mask, resolution):
        metrics = self._region_shape_metrics(mask, resolution)
        return metrics['width_proxy_m'] > 0.0 and metrics['aspect_ratio'] >= 2.5

    # ------------------------------------------------------------------
    # 2D GVD construction and room segmentation
    # ------------------------------------------------------------------

    def _binary_free_obstacle(self, grid):
        h, w = int(grid.info.height), int(grid.info.width)
        data = np.asarray(grid.data, np.int16).reshape((h, w))
        free = ((data >= 0) & (data <= self._params['free_threshold'])).astype(np.uint8) * 255
        occupied = ((data >= self._params['occupied_threshold']) & (data <= 100)).astype(np.uint8) * 255
        if self._params['unknown_is_obstacle']:
            occupied[data < 0] = 255
        blur_k = int(self._params.get('map_median_blur_ksize', 0) or 0)
        if blur_k >= 3 and blur_k % 2 == 1:
            occupied = cv2.medianBlur(occupied, blur_k)
        free[occupied > 0] = 0
        return free, occupied

    def _structural_obstacles(self, occupied, resolution, grid=None, cloud_support=None):
        """Return the occupied mask used by the GVD.

        Keep only wall-like occupied components for topology extraction:
        long and thin structures are likely walls, while large compact
        blobs such as beds or tables stay non-structural so they do not
        split a room into fake sub-rooms.
        """
        min_len_px = max(1, int(round(
            self._params['gvd_min_obstacle_length_m'] / max(resolution, 1e-6))))
        max_thickness_px = max(1, int(round(
            self._params['gvd_wall_max_thickness_m'] / max(resolution, 1e-6))))
        min_aspect = float(self._params.get('gvd_wall_min_aspect_ratio', 1.0))
        min_area_px = max(1, int(round(
            self._params['gvd_wall_min_area_m2'] / max(resolution * resolution, 1e-12))))
        min_fill_ratio = float(self._params.get('gvd_wall_min_fill_ratio', 0.0))

        obstacles_u8 = (occupied > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(obstacles_u8, 8)
        structural = np.zeros_like(occupied)
        for i in range(1, count):
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            area_px = int(stats[i, cv2.CC_STAT_AREA])
            long_side = max(bw, bh)
            short_side = min(bw, bh)
            aspect = long_side / max(1.0, short_side)
            bbox_area = max(1, int(bw * bh))
            fill_ratio = float(area_px) / float(bbox_area)
            thin_segment = (
                long_side >= min_len_px
                and short_side <= max_thickness_px
                and aspect >= min_aspect
                and area_px >= min_area_px
                and fill_ratio >= min_fill_ratio
            )
            # A connected wall network (outer boundary plus internal walls) is neither
            # thin nor elongated as one bounding box. Its sparse footprint distinguishes
            # it from a compact furniture blob while preserving connected wall systems.
            sparse_network = (
                long_side >= min_len_px and area_px >= min_area_px and
                fill_ratio <= float(self._params.get('gvd_wall_network_max_fill_ratio', 0.35))
            )
            if thin_segment or sparse_network:
                structural[labels == i] = 255

        if cloud_support is None and self._params.get('enable_3d_structural_filter', False) and grid is not None:
            cloud_support = self._cloud_structural_support(grid)
        if cloud_support is not None:
            observed = self._latest_cloud_observed_mask
            # The 3D cloud may confirm a 2D wall or identify clutter, but it must
            # never veto a structural wall already present in occupancy. Sparse
            # or locally missing cloud returns otherwise open a false passage and
            # let the GVD merge rooms through the wall.
            # cloud_obstacles is evidence that an obstacle exists, not a wall
            # detector.  Real wall reinforcement is supplied separately by
            # detected_wall_support below.  Keeping this out of structural_occ
            # prevents a tall wardrobe/table from becoming a room divider merely
            # because it has a vertical point cloud.
            # Ground is used to estimate the floor datum, not as a negative
            # obstacle vote. At a wall/floor junction both clouds legitimately
            # occupy the same projected cell; treating ground as a veto erases
            # real walls, especially after the support-mask dilation.
        return structural

    def _fill_nonstructural_obstacles(self, free, occupied, structural_occ, resolution):
        topo_free = free.copy()
        nonwall_3d = self._latest_cloud_nonwall_mask
        direct_nonwall = np.zeros_like(occupied, dtype=bool)
        if (nonwall_3d is not None and nonwall_3d.shape == occupied.shape and
                self._params.get('gvd_fill_cloud_nonwall_direct', True)):
            # Do this before connected-component filtering. Furniture touching a
            # wall can share the same 2D occupied component as the wall and would
            # otherwise inherit the wall label instead of being filled into the
            # room floor.
            # Never reopen a cell belonging to a structural wall. The cloud's
            # non-wall classification is allowed to remove clutter only.
            direct_nonwall = ((occupied > 0) & nonwall_3d.astype(bool) &
                              (structural_occ == 0))
            topo_free[direct_nonwall] = 255
        nonstructural = (occupied > 0) & (structural_occ == 0)
        if not np.any(nonstructural):
            self._last_topology_fill_stats = {
                'components': 0, 'pixels': 0, 'confirmed_3d': 0,
                'direct_nonwall_pixels': int(np.count_nonzero(direct_nonwall))}
            return topo_free

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            nonstructural.astype(np.uint8), 8)
        max_area_px = max(1, int(round(
            self._params['gvd_topo_fill_max_area_m2'] / max(resolution * resolution, 1e-12))))
        min_nonwall_ratio = float(self._params.get('gvd_3d_nonwall_component_ratio', 0.05))
        h, w = nonstructural.shape
        filled_components = 0
        filled_pixels = 0
        confirmed_components = 0
        for i in range(1, count):
            area_px = int(stats[i, cv2.CC_STAT_AREA])
            if area_px <= 0:
                continue
            component = labels == i
            confirmed_nonwall = False
            if nonwall_3d is not None and nonwall_3d.shape == component.shape:
                confirmed_nonwall = (
                    np.count_nonzero(component & nonwall_3d) / max(1, area_px)
                    >= min_nonwall_ratio)
            if area_px > max_area_px and not confirmed_nonwall:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            touches_border = (
                x <= 0 or y <= 0 or
                (x + bw) >= w or (y + bh) >= h
            )
            if touches_border:
                continue
            topo_free[component] = 255
            filled_components += 1
            filled_pixels += area_px
            confirmed_components += int(confirmed_nonwall)
        self._last_topology_fill_stats = {
            'components': filled_components,
            'pixels': filled_pixels,
            'confirmed_3d': confirmed_components,
            'direct_nonwall_pixels': int(np.count_nonzero(direct_nonwall)),
        }
        return topo_free

    def _fill_room_holes(self, mask, resolution, protected=None):
        """Fill small clutter holes, but never overwrite protected wall cells."""
        filled = mask.copy()
        hole_limit_px = max(1, int(round(
            self._params['gvd_room_hole_fill_max_area_m2'] / max(resolution * resolution, 1e-12))))
        contours, hierarchy = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            return filled
        hierarchy = hierarchy[0]
        for idx, h in enumerate(hierarchy):
            if h[3] == -1:
                continue
            if cv2.contourArea(contours[idx]) <= hole_limit_px:
                if protected is not None and protected.shape == filled.shape:
                    hole = np.zeros_like(filled)
                    cv2.drawContours(hole, contours, idx, 255, thickness=-1)
                    if np.any((hole > 0) & (protected > 0)):
                        continue
                cv2.drawContours(filled, contours, idx, 255, thickness=-1)
        return filled

    def _navigable_free_component(self, free, grid):
        """Keep all meaningful floor components, including rooms separated by walls.

        The old implementation kept only the robot component (or the largest component).
        That assumption is invalid after a confirmed 3D wall is rasterised: the wall is
        supposed to disconnect adjacent rooms, so this step used to erase every room but
        one.  Only components smaller than the configured room-area floor are treated as
        mapping noise.
        """
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (free > 0).astype(np.uint8), 8)
        if count <= 1:
            self._last_free_component_stats = {
                'before': max(0, count-1), 'kept_label': 0,
                'discarded_pixels': 0,
                'kept_ratio': 0.0,
            }
            return free

        if not self._params.get('gvd_keep_disconnected_rooms', True):
            # Compatibility escape hatch for deployments that explicitly want the old
            # reachable-only behaviour.
            keep = 0
            robot_label = 0
            if self.last_robot_xy is not None:
                pixel = self._world_to_grid(*self.last_robot_xy, grid)
                if pixel is not None:
                    x, y = pixel
                    if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                        robot_label = int(labels[y, x])
                        keep = robot_label
            if keep <= 0:
                keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            result = np.zeros_like(free)
            result[labels == keep] = 255
            self._last_free_component_stats = {
                'before': count - 1, 'kept_components': 1, 'kept_label': keep,
                'kept_pixels': int(stats[keep, cv2.CC_STAT_AREA]),
                'discarded_pixels': int(np.count_nonzero(free) - stats[keep, cv2.CC_STAT_AREA]),
                'kept_ratio': round(float(stats[keep, cv2.CC_STAT_AREA]) /
                                    max(1, int(np.count_nonzero(free))), 4),
                'legacy_reachable_only': True,
            }
            return result

        min_area_px = max(1, int(round(float(self._params.get(
            'gvd_disconnected_component_min_area_m2',
            self._params.get('min_room_area_m2', 1.5))) /
            max(float(grid.info.resolution) ** 2, 1e-12))))
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_labels = np.flatnonzero(areas >= min_area_px) + 1
        # Never return an empty topology just because the map is still fragmentary.
        if keep_labels.size == 0 and areas.size:
            keep_labels = np.asarray([1 + int(np.argmax(areas))])
        result = np.zeros_like(free)
        for label in keep_labels.tolist():
            result[labels == label] = 255
        kept_pixels = int(np.count_nonzero(result))
        self._last_free_component_stats = {
            'before': count - 1,
            'kept_components': int(keep_labels.size),
            'min_component_area_m2': round(min_area_px * float(grid.info.resolution) ** 2, 3),
            'kept_pixels': kept_pixels,
            'discarded_pixels': int(np.count_nonzero(free) - kept_pixels),
            'kept_ratio': round(kept_pixels / max(1, int(np.count_nonzero(free))), 4),
            'legacy_reachable_only': False,
        }
        return result

    def _polygon_boundary_support_ratio(self, polygon, support_mask, grid):
        if support_mask is None or grid is None or len(polygon) < 3:
            return 0.0

        pts = []
        for x, y in polygon:
            ij = self._world_to_grid(x, y, grid)
            if ij is not None:
                pts.append(ij)
        if len(pts) < 3:
            return 0.0

        boundary = np.zeros_like(support_mask, dtype=np.uint8)
        cv2.polylines(boundary, [np.asarray(pts, dtype=np.int32)], True, 255, 1)
        boundary_idx = boundary > 0
        if not np.any(boundary_idx):
            return 0.0

        support_u8 = support_mask.astype(np.uint8)
        support_near = cv2.dilate(support_u8, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
        return float(np.count_nonzero(boundary_idx & support_near)) / float(np.count_nonzero(boundary_idx))

    def _merge_nested_candidates(self, candidates, cloud_support=None, grid=None):
        if len(candidates) < 2:
            return candidates

        ordered = sorted(candidates, key=lambda item: float(item[1]), reverse=True)
        kept = []
        max_nested_area = float(self._params.get('room_nested_merge_max_area_m2', 0.0))
        max_area_ratio = float(self._params.get('room_nested_merge_area_ratio', 0.0))
        max_wall_support = float(self._params.get('room_nested_merge_wall_support_max_ratio', 1.0))

        for polygon, area, centroid, flag in ordered:
            merge_into = None
            if area <= max_nested_area:
                boundary_support = self._polygon_boundary_support_ratio(polygon, cloud_support, grid)
                for parent in kept:
                    parent_polygon, parent_area, _, _ = parent
                    if parent_area <= area:
                        continue
                    if not self._point_in_polygon(parent_polygon, centroid, tolerance_m=0.0):
                        continue
                    if area / max(parent_area, 1e-6) <= max_area_ratio:
                        if boundary_support > max_wall_support:
                            continue
                        merge_into = parent
                        break

            if merge_into is None:
                kept.append((polygon, area, centroid, flag))

        return kept

    def _deduplicate_overlapping_candidates(self, candidates):
        """Enforce one geometric region per room candidate.

        Room masks are disjoint, but polygon extraction intentionally keeps only
        the outer contour.  If a small label becomes a hole in a larger label,
        that conversion can produce nested or duplicate polygons.  Such geometry
        is never a valid room partition, so retain the largest candidate and drop
        the contained/near-duplicate one before temporal tracking sees it.
        """
        if len(candidates) < 2:
            return candidates

        kept = []
        removed_nested = 0
        removed_duplicate = 0
        for candidate in sorted(candidates, key=lambda item: float(item[1]), reverse=True):
            polygon, area, centroid, flag = candidate
            drop = False
            for parent_polygon, parent_area, _, _ in kept:
                ratio = float(area) / max(float(parent_area), 1e-6)
                iou = self._polygon_iou(polygon, parent_polygon)
                contained = self._point_in_polygon(parent_polygon, centroid)
                # A valid room partition cannot contain another room polygon.  The
                # IoU test also catches the same region emitted twice with slightly
                # different contours.
                if contained and ratio <= 0.50:
                    drop = True
                    removed_nested += 1
                    break
                if iou >= 0.55:
                    drop = True
                    removed_duplicate += 1
                    break
            if not drop:
                kept.append(candidate)

        self._last_overlap_cleanup_stats = {
            'input_candidates': len(candidates),
            'output_candidates': len(kept),
            'removed_nested': removed_nested,
            'removed_duplicate': removed_duplicate,
        }
        return kept

    def _detected_wall_support(self, grid):
        """Rasterise confirmed walls as one locally continuous support network.

        Depth views commonly stop a few centimetres before a corner.  Merely
        drawing each finite segment leaves leaks in the wall mask and a later
        GVD pass can join the rooms through that leak.  Confirmed endpoints are
        therefore joined when they are very close and either collinear or form
        a plausible corner.  The maximum gap is intentionally well below the
        doorway width used by segmentation, so doorway-separated pieces remain
        disconnected.
        """
        if not self._params.get('enable_detected_wall_support', True) or grid is None:
            return None
        height, width = int(grid.info.height), int(grid.info.width)
        support = np.zeros((height, width), dtype=np.float32)
        min_obs = max(1, int(self._params.get('detected_wall_min_observations', 1)))
        min_length = float(self._params.get('detected_wall_min_length_m', 0.50))
        min_vertical = float(self._params.get('detected_wall_min_vertical_extent_m', 1.20))
        max_rms = float(self._params.get('detected_wall_max_rms_m', 0.03))
        min_confidence = float(self._params.get('detected_wall_min_confidence', 0.0))
        thickness = max(1, int(round(float(self._params.get(
            'detected_wall_thickness_m', 0.06)) /
            max(float(grid.info.resolution), 1e-6))))
        segments = 0
        rejected = 0
        expired = 0
        qualified = []
        max_age_s = float(self._params.get('detected_wall_max_age_s', 0.0))
        now = time.time()
        for wall in self._detected_wall_map:
            age_s = now - float(wall.get('last_seen', now))
            if max_age_s > 0.0 and age_s > max_age_s:
                expired += 1
                continue
            observations = int(wall.get('observations', 1))
            if observations < min_obs:
                continue
            try:
                p0, p1, _, length = self._wall_geometry(wall)
            except (KeyError, TypeError, ValueError):
                rejected += 1
                continue
            vertical = float(wall.get('z_max', 0.0)) - float(wall.get('z_min', 0.0))
            rms = float(wall.get('inlier_rms_m', float('inf')))
            measured_confidence = float(wall.get('confidence', 1.0))
            if (length < min_length or vertical < min_vertical or rms > max_rms or
                    measured_confidence < min_confidence):
                rejected += 1
                continue
            q0 = self._world_to_grid(float(p0[0]), float(p0[1]), grid)
            q1 = self._world_to_grid(float(p1[0]), float(p1[1]), grid)
            if q0 is None or q1 is None:
                continue
            confidence = measured_confidence * min(1.0, observations / 2.0)
            layer = np.zeros_like(support, dtype=np.uint8)
            cv2.line(layer, q0, q1, 255, thickness)
            support[layer > 0] = np.maximum(support[layer > 0], confidence)
            qualified.append((p0, p1, q0, q1, confidence))
            segments += 1

        # A confirmed doorway is an opening in the structural wall network too.
        # Remove its footprint from the support mask so the GVD/topology and the
        # RViz wall geometry cannot show a wall stripe over the passage.
        doorway_radius = max(0.20, float(self._params.get(
            'gvd_door_max_m', 1.20)) * 0.5)
        for doorway in getattr(self, '_doorway_vlm_state', {}).values():
            if (doorway.get('status') != 'confirmed' or
                    not doorway.get('cuttable') or doorway.get('world') is None):
                continue
            q = self._world_to_grid(*doorway['world'], grid)
            if q is not None:
                cv2.circle(support, q, max(1, int(round(
                    doorway_radius / max(float(grid.info.resolution), 1e-6)))),
                           0.0, -1)

        # Join only pairs of endpoints.  Drawing between arbitrary nearby line
        # interiors would create cross-walls in cluttered areas.  Both straight
        # continuations and near-right-angle corners are accepted; oblique pairs
        # are left untouched because their topology is ambiguous.
        max_gap_m = max(0.0, float(self._params.get(
            'detected_wall_junction_gap_m', 0.30)))
        angle_tol = math.radians(max(0.0, float(self._params.get(
            'detected_wall_junction_angle_deg', 20.0))))
        straight_min = math.cos(angle_tol)
        corner_max = math.sin(angle_tol)
        junctions = 0
        for i, first in enumerate(qualified):
            a0, a1, aq0, aq1, aconf = first
            adir = (a1 - a0) / max(float(np.linalg.norm(a1 - a0)), 1e-9)
            for second in qualified[i + 1:]:
                b0, b1, bq0, bq1, bconf = second
                bdir = (b1 - b0) / max(float(np.linalg.norm(b1 - b0)), 1e-9)
                alignment = abs(float(adir @ bdir))
                if alignment < straight_min and alignment > corner_max:
                    continue
                endpoint_pairs = [
                    (float(np.linalg.norm(ap - bp)), aq, bq)
                    for ap, aq in ((a0, aq0), (a1, aq1))
                    for bp, bq in ((b0, bq0), (b1, bq1))
                ]
                gap_m, qa, qb = min(endpoint_pairs, key=lambda item: item[0])
                if gap_m <= max_gap_m:
                    confidence = min(aconf, bconf)
                    layer = np.zeros_like(support, dtype=np.uint8)
                    cv2.line(layer, qa, qb, 255, thickness)
                    support[layer > 0] = np.maximum(
                        support[layer > 0], confidence)
                    junctions += 1
        self._last_detected_wall_stats = {
            'confirmed_segments': segments,
            'closed_junctions': junctions,
            'rejected_segments': rejected,
            'expired_segments': expired,
            'support_cells': int(np.count_nonzero(support)),
        }
        return support

    def _detected_wall_doorway_cuts(self, free, cut, grid, resolution):
        """Cut doorway-sized gaps between confirmed collinear wall pieces.

        The GVD remains useful for doors inferred only from free-space shape, but
        measured walls are stronger evidence: if two stable pieces lie on the
        same plane and terminate around a free gap, that gap is a doorway.  A
        proposed cut is retained only when it actually separates two regions at
        least as large as ``min_room_area_m2``.  This makes enclosing walls
        dominant without turning short wall fragments into tiny rooms.
        """
        stats = {'proposed': 0, 'accepted': 0, 'accepted_locked': 0,
                 'rejected_not_free': 0, 'rejected_no_split': 0,
                 'rejected_small_partition': 0, 'rejected_vlm_pending': 0}
        if (not self._params.get('detected_wall_direct_door_cuts', True) or
                grid is None or not np.any(cut)):
            self._last_wall_door_cut_stats = stats
            return cut

        min_obs = max(1, int(self._params.get('detected_wall_min_observations', 2)))
        min_length = float(self._params.get('detected_wall_min_length_m', 0.50))
        min_vertical = float(self._params.get('detected_wall_min_vertical_extent_m', 0.90))
        max_rms = float(self._params.get('detected_wall_max_rms_m', 0.05))
        min_confidence = float(self._params.get('detected_wall_min_confidence', 0.0))
        max_age_s = float(self._params.get('detected_wall_max_age_s', 0.0))
        now = time.time()
        walls = []
        for wall in self._detected_wall_map:
            if int(wall.get('observations', 1)) < min_obs:
                continue
            if (max_age_s > 0.0 and
                    now - float(wall.get('last_seen', now)) > max_age_s):
                continue
            try:
                p0, p1, direction, length = self._wall_geometry(wall)
            except (KeyError, TypeError, ValueError):
                continue
            vertical = float(wall.get('z_max', 0.0)) - float(wall.get('z_min', 0.0))
            if (length < min_length or vertical < min_vertical or
                    float(wall.get('inlier_rms_m', float('inf'))) > max_rms or
                    float(wall.get('confidence', 1.0)) < min_confidence):
                continue
            walls.append((p0, p1, direction, length))

        angle_cos = math.cos(math.radians(float(self._params.get(
            'detected_wall_merge_angle_deg', 12.0))))
        plane_tol = float(self._params.get('detected_wall_merge_distance_m', 0.22))
        min_gap = float(self._params.get('detected_wall_door_min_m', 0.55))
        max_gap = float(self._params.get('detected_wall_door_max_m', 2.0))
        min_component_px = max(1, int(round(
            float(self._params['min_room_area_m2']) /
            max(resolution * resolution, 1e-12))))
        thickness = max(1, 2*int(self._params.get('gvd_cut_margin_px', 2)) + 1)
        result = cut.copy()

        proposals = []
        for i, (a0, a1, adir, alen) in enumerate(walls):
            normal = np.array([-adir[1], adir[0]])
            for b0, b1, bdir, blen in walls[i + 1:]:
                if abs(float(adir @ bdir)) < angle_cos:
                    continue
                if abs(float((((b0 + b1) - (a0 + a1))*0.5) @ normal)) > plane_tol:
                    continue
                # Put both finite intervals on the first wall's tangent.  Only
                # disjoint intervals have a doorway between them.
                ai = sorted((0.0, alen))
                bt0, bt1 = float((b0-a0) @ adir), float((b1-a0) @ adir)
                bi = sorted((bt0, bt1))
                if bi[0] > ai[1]:
                    gap, left, right = bi[0]-ai[1], a1, (b0 if bt0 < bt1 else b1)
                elif ai[0] > bi[1]:
                    gap, left, right = ai[0]-bi[1], (b0 if bt0 > bt1 else b1), a0
                else:
                    continue
                if min_gap <= gap <= max_gap:
                    # Keep the direction with the proposal.  Using the loop-scoped
                    # ``adir`` below made every proposal inherit the direction of the
                    # last wall pair examined, producing oblique/wrong doorway cuts.
                    proposals.append((gap, left, right, adir))

        # Narrow, strongly delimited openings first.  Once a cut has separated a
        # room, later candidates are validated only inside their current parent.
        wall_points = []
        height, width = result.shape
        for _gap, left, right, proposal_direction in proposals:
            q0 = self._world_to_grid(float(left[0]), float(left[1]), grid)
            q1 = self._world_to_grid(float(right[0]), float(right[1]), grid)
            if q0 is None or q1 is None:
                continue
            if not (0 <= q0[0] < width and 0 <= q0[1] < height and
                    0 <= q1[0] < width and 0 <= q1[1] < height):
                continue
            midpoint = ((q0[0]+q1[0])//2, (q0[1]+q1[1])//2)
            wall_points.append((midpoint[1], midpoint[0],
                                math.atan2(float(proposal_direction[1]),
                                           float(proposal_direction[0])),
                                max(1.0, _gap / max(2.0 * resolution, 1e-6)), 1.0))
        self._schedule_doorway_confirmations(grid, wall_points)

        for _gap, left, right, proposal_direction in sorted(
                proposals, key=lambda item: item[0]):
            q0 = self._world_to_grid(float(left[0]), float(left[1]), grid)
            q1 = self._world_to_grid(float(right[0]), float(right[1]), grid)
            if q0 is None or q1 is None:
                continue
            height, width = result.shape
            if not (0 <= q0[0] < width and 0 <= q0[1] < height and
                    0 <= q1[0] < width and 0 <= q1[1] < height):
                continue
            stats['proposed'] += 1
            midpoint = ((q0[0]+q1[0])//2, (q0[1]+q1[1])//2)
            world_midpoint = self._grid_to_world(midpoint[0], midpoint[1], grid)
            # The wall-derived proposal is accepted or rejected by free-space occupancy
            # and the two-component validation below. The VLM confirmation is used to
            # lock/recover the opening, while wall geometry remains usable immediately.
            gap_layer = np.zeros_like(free, dtype=np.uint8)
            cv2.line(gap_layer, q0, q1, 255, 1)
            gap_pixels = gap_layer > 0
            if (not np.any(gap_pixels) or
                    np.count_nonzero(gap_pixels & (free > 0)) /
                    max(1, np.count_nonzero(gap_pixels)) < 0.60):
                stats['rejected_not_free'] += 1
                continue
            _, before = cv2.connectedComponents(result, 8)
            parent_label = int(before[midpoint[1], midpoint[0]])
            if parent_label <= 0:
                stats['rejected_no_split'] += 1
                continue
            trial = result.copy()
            cv2.line(trial, q0, q1, 0, thickness)
            _, after = cv2.connectedComponents(trial, 8)
            parent = before == parent_label
            children = np.unique(after[parent])
            children = children[children > 0]
            if children.size < 2:
                stats['rejected_no_split'] += 1
                continue
            areas = [np.count_nonzero(parent & (after == child)) for child in children]
            if min(areas) < min_component_px:
                stats['rejected_small_partition'] += 1
                continue
            result = trial
            stats['accepted'] += 1

        self._last_wall_door_cut_stats = stats
        return result

    def _confirmed_doorway_points(self, grid, free_topo, dist_real, resolution):
        """Recover confirmed openings whose GVD pixel moved between map updates."""
        recovered = []
        max_age = float(self._params.get('doorway_vlm_state_max_age_s', 60.0))
        for item in getattr(self, '_doorway_vlm_state', {}).values():
            if (item.get('status') != 'confirmed' or not item.get('cuttable') or
                    item.get('world') is None):
                continue
            if (item.get('status') != 'confirmed' and max_age > 0.0 and
                    time.time() - float(item.get('updated_at', 0.0)) > max_age):
                continue
            q = self._world_to_grid(*item['world'], grid)
            if q is None:
                continue
            x, y = q
            if not (0 <= x < free_topo.shape[1] and 0 <= y < free_topo.shape[0]):
                continue
            if free_topo[y, x] == 0 or float(dist_real[y, x]) <= 0.0:
                continue
            left_world = item.get('left_world')
            right_world = item.get('right_world')
            point = (y, x, float(item.get('theta', 0.0)),
                     float(dist_real[y, x]), 1.0, left_world, right_world, True)
            if all(math.hypot(y-other[0], x-other[1]) >
                   float(self._params.get('gvd_door_nms_m', 1.20)) /
                   max(resolution, 1e-6) for other in recovered):
                recovered.append(point)
        return recovered

    def _apply_confirmed_door_cuts(self, cut, grid, resolution, wall_mask=None):
        """Apply immutable doorway cuts only when they split free space."""
        applied = 0

        def splits_doorway_component(before_mask, after_mask, a, b):
            """Check that this line splits the free component at this doorway."""
            count_before, before_labels = cv2.connectedComponents(before_mask, 8)
            del count_before
            segment = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
            length = float(np.linalg.norm(segment))
            samples = ([
                (int(round(a[0] + segment[0] * t / max(length, 1.0))),
                 int(round(a[1] + segment[1] * t / max(length, 1.0))))
                for t in range(max(1, int(round(length)) + 1))
            ] if length > 0.0 else [tuple(a)])
            free_samples = [point for point in samples
                            if 0 <= point[1] < before_mask.shape[0] and
                            0 <= point[0] < before_mask.shape[1] and
                            before_mask[point[1], point[0]] > 0]
            if not free_samples:
                return False
            x, y = free_samples[len(free_samples) // 2]
            parent_label = int(before_labels[y, x])
            if parent_label <= 0:
                return False
            parent = before_labels == parent_label
            children = np.unique(after_mask[parent])
            return int(np.count_nonzero(children > 0)) >= 2

        for lock in self._confirmed_door_cuts.values():
            left = lock.get('left_world')
            right = lock.get('right_world')
            if left is None or right is None:
                continue
            q0 = self._world_to_grid(float(left[0]), float(left[1]), grid)
            q1 = self._world_to_grid(float(right[0]), float(right[1]), grid)
            if q0 is None or q1 is None:
                continue
            thickness = max(1, 2 * int(self._params.get('gvd_cut_margin_px', 2)) + 1)
            trial = cut.copy()
            cv2.line(trial, q0, q1, 0, thickness)
            local_split = splits_doorway_component(cut, trial, q0, q1)

            # A metric endpoint can fall one or more cells short of the wall.
            # Extend only along the stored doorway axis, never sideways and never
            # by regenerating the line from the current GVD.
            if not local_split and wall_mask is not None:
                p0 = np.asarray(q0, dtype=float)
                p1 = np.asarray(q1, dtype=float)
                axis = p1 - p0
                length = float(np.linalg.norm(axis))
                if length > 1e-6:
                    axis /= length
                    max_scan = max(1, int(round(float(self._params.get(
                        'detected_wall_door_endpoint_max_extension_m', 0.75)) /
                        max(float(resolution), 1e-6))))

                    def extend_to_wall(start, direction):
                        current = tuple(np.rint(start).astype(int))
                        previous = current
                        h, w = wall_mask.shape
                        for step in range(1, max_scan + 1):
                            probe = np.rint(start + direction * step).astype(int)
                            x, y = int(probe[0]), int(probe[1])
                            if not (0 <= x < w and 0 <= y < h):
                                break
                            if wall_mask[y, x] > 0:
                                return previous
                            previous = (x, y)
                        return current

                    extended_q0 = extend_to_wall(p0, -axis)
                    extended_q1 = extend_to_wall(p1, axis)
                    extended = cut.copy()
                    cv2.line(extended, extended_q0, extended_q1, 0, thickness)
                    if splits_doorway_component(cut, extended,
                                                extended_q0, extended_q1):
                        trial, local_split = extended, True

            if local_split:
                cut = trial
                applied += 1
        self._last_locked_door_cuts = applied
        return cut

    def _floor_height_from_ground(self, now_mono, max_age_s):
        """Estimate this floor's map-frame height from ``cloud_ground``.

        Using base_link.z as floor height shifts all wall-height gates by the
        base mounting offset.  A local robust estimate also prevents points
        belonging to another storey from winning a global median.
        """
        points = self._latest_cloud_ground_points
        received_at = self._latest_cloud_ground_received_at
        fresh = (
            points is not None and points.shape[0] > 0 and
            (max_age_s <= 0.0 or received_at is None or
             now_mono - received_at <= max_age_s))
        robot_z = self._robot_height()
        if not fresh:
            return robot_z, False, 0

        finite = np.isfinite(points[:, 2])
        # First isolate the robot's storey.  This remains valid when the ground
        # cloud contains several floors projected into the same XY area.
        finite &= np.abs(points[:, 2] - robot_z) <= 0.75
        radius = float(self._params.get('gvd_3d_floor_radius_m', 2.0))
        if self.last_robot_xy is not None and radius > 0.0:
            dx = points[:, 0] - float(self.last_robot_xy[0])
            dy = points[:, 1] - float(self.last_robot_xy[1])
            finite &= dx * dx + dy * dy <= radius * radius
        z = points[finite, 2]
        min_points = max(1, int(self._params.get('gvd_3d_floor_min_points', 20)))
        if z.size < min_points:
            return robot_z, False, int(z.size)

        # Median/MAD trimming rejects stair edges and occasional misclassified
        # obstacle points without assuming a perfectly horizontal sensor cloud.
        centre = float(np.median(z))
        mad = float(np.median(np.abs(z - centre)))
        if mad > 1e-6:
            trimmed = z[np.abs(z - centre) <= max(0.03, 3.0 * 1.4826 * mad)]
            if trimmed.size >= min_points:
                centre = float(np.median(trimmed))
        return centre, True, int(z.size)

    def _cloud_structural_support(self, grid):
        # The obstacle cloud is the cleanest source for vertical structure: it
        # does not spend the point budget on the floor and therefore makes the
        # height-band test much less sensitive to furniture/floor imbalance.
        # Keep cloud_map as a fallback for setups where RTAB-Map does not publish
        # cloud_obstacles continuously.
        max_age_s = float(self._params.get('gvd_3d_max_age_s', 0.0))
        now_mono = time.monotonic()
        obstacle_fresh = (
            self._latest_cloud_obstacle_points is not None and
            (max_age_s <= 0.0 or self._latest_cloud_obstacle_received_at is None or
             now_mono - self._latest_cloud_obstacle_received_at <= max_age_s))
        map_fresh = (
            self._latest_cloud_points is not None and
            (max_age_s <= 0.0 or self._latest_cloud_received_at is None or
             now_mono - self._latest_cloud_received_at <= max_age_s))
        # cloud_map provides global coverage, while cloud_obstacles is the cleaner
        # RTAB-Map structural stream for furniture/wall classification. Use both
        # when available: relying only on cloud_map can leave furniture projected
        # into the room boundary, especially close to a wall.
        if map_fresh and obstacle_fresh:
            points = np.concatenate((self._latest_cloud_points,
                                     self._latest_cloud_obstacle_points), axis=0)
            cloud_source = 'cloud_map+cloud_obstacles'
            received_at = max(self._latest_cloud_received_at or 0.0,
                              self._latest_cloud_obstacle_received_at or 0.0)
        elif map_fresh:
            points = self._latest_cloud_points
            cloud_source = 'cloud_map'
            received_at = self._latest_cloud_received_at
        elif obstacle_fresh:
            points = self._latest_cloud_obstacle_points
            cloud_source = 'cloud_obstacles_fallback'
            received_at = self._latest_cloud_obstacle_received_at
        else:
            points = None
            cloud_source = 'none'
            received_at = None
        if points is None or points.shape[0] == 0:
            self._latest_cloud_observed_mask = None
            self._latest_cloud_nonwall_mask = None
            self._latest_cloud_ground_mask = None
            self._last_3d_wall_stats = {'source': 'none', 'reason': 'no_fresh_cloud'}
            return None
        if grid is None or grid.info.resolution <= 0:
            return None

        if received_at is not None and max_age_s > 0.0:
            age_s = now_mono - received_at
            if age_s > max_age_s:
                self._latest_cloud_observed_mask = None
                self._latest_cloud_nonwall_mask = None
                now = time.time()
                if now - self._last_cloud_warn_time > 5.0:
                    self._log('warn', f'Ignoring stale 3D cloud ({age_s:.1f}s old)')
                    self._last_cloud_warn_time = now
                return None

        resolution = float(grid.info.resolution)
        origin = grid.info.origin
        yaw = self._yaw(origin.orientation)
        c = math.cos(yaw)
        s = math.sin(yaw)

        x = points[:, 0] - float(origin.position.x)
        y = points[:, 1] - float(origin.position.y)
        gx = c * x + s * y
        gy = -s * x + c * y
        ix = np.floor(gx / resolution).astype(np.int64)
        iy = np.floor(gy / resolution).astype(np.int64)

        width = int(grid.info.width)
        height = int(grid.info.height)
        valid = (ix >= 0) & (iy >= 0) & (ix < width) & (iy < height)
        if not np.any(valid):
            self._latest_cloud_observed_mask = None
            self._latest_cloud_nonwall_mask = None
            self._last_3d_wall_stats = {
                'source': cloud_source, 'reason': 'cloud_outside_grid'}
            return None

        ix = ix[valid]
        iy = iy[valid]
        z = points[valid, 2].astype(np.float64, copy=False)

        floor_z, ground_fresh, ground_floor_points = self._floor_height_from_ground(
            now_mono, max_age_s)
        relative_z = z - floor_z

        # Keep a separate floor-observation mask.  It is deliberately not mixed
        # into ``observed``: a ground return means "this is floor", whereas an
        # obstacle return means "there is something standing here".
        ground_points = self._latest_cloud_ground_points
        ground_mask = np.zeros((height, width), dtype=bool)
        if ground_fresh:
            gxg = ground_points[:, 0] - float(origin.position.x)
            gyg = ground_points[:, 1] - float(origin.position.y)
            gix = np.floor((c * gxg + s * gyg) / resolution).astype(np.int64)
            giy = np.floor((-s * gxg + c * gyg) / resolution).astype(np.int64)
            gvalid = ((gix >= 0) & (giy >= 0) &
                      (gix < width) & (giy < height))
            if np.any(gvalid):
                ground_flat = gix[gvalid] + giy[gvalid] * width
                ground_counts = np.bincount(ground_flat, minlength=width * height)
                ground_mask = (ground_counts >= 1).reshape((height, width))
                dilation_px = max(0, int(self._params.get(
                    'gvd_3d_support_dilation_px', 0)))
                if dilation_px > 0:
                    kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1),
                                     dtype=np.uint8)
                    ground_mask = cv2.dilate(
                        ground_mask.astype(np.uint8), kernel, iterations=1) > 0
        self._latest_cloud_ground_mask = ground_mask if ground_fresh else None

        min_height = float(self._params.get('gvd_3d_min_height_m', 0.25))
        mid_height = float(self._params.get('gvd_3d_mid_height_m', 0.80))
        high_height = float(self._params.get('gvd_3d_high_height_m', 1.40))
        max_height = float(self._params.get('gvd_3d_max_height_m', 2.20))
        floor_valid = (relative_z >= min_height) & (relative_z <= max_height)
        if not np.any(floor_valid):
            self._latest_cloud_observed_mask = None
            self._latest_cloud_nonwall_mask = None
            self._last_3d_wall_stats = {
                'source': cloud_source,
                'floor_z': round(float(floor_z), 3),
                'floor_points': ground_floor_points,
                'reason': 'no_points_in_height_band',
            }
            return None
        ix = ix[floor_valid]
        iy = iy[floor_valid]
        relative_z = relative_z[floor_valid]

        flat = ix + iy * width
        size = width * height
        counts = np.bincount(flat, minlength=size)
        zmin = np.full(size, np.inf, dtype=np.float64)
        zmax = np.full(size, -np.inf, dtype=np.float64)
        np.minimum.at(zmin, flat, relative_z)
        np.maximum.at(zmax, flat, relative_z)
        span = zmax - zmin

        min_band_points = int(self._params.get('gvd_3d_min_points_per_band', 1))
        band_count = np.zeros(size, dtype=np.uint8)
        for lo, hi in ((min_height, mid_height),
                       (mid_height, high_height),
                       (high_height, max_height)):
            in_band = (relative_z >= lo) & (relative_z < hi)
            if np.any(in_band):
                per_cell = np.bincount(flat[in_band], minlength=size)
                band_count += (per_cell >= min_band_points).astype(np.uint8)

        min_points = int(self._params.get('gvd_3d_min_points_per_cell', 1))
        min_span = float(self._params.get('gvd_3d_min_vertical_span_m', 0.0))
        required_bands = int(self._params.get('gvd_3d_required_height_bands', 2))
        support = ((counts >= min_points) & np.isfinite(span) &
                   (span >= min_span) & (band_count >= required_bands))
        support = support.reshape((height, width))
        observed = (counts >= min_points).reshape((height, width))

        # Height separates walls from low furniture; plan-view shape separates walls from
        # tall compact furniture such as wardrobes and refrigerators. Classify
        # BEFORE dilation: otherwise a one-cell wall becomes three cells thick
        # and can fail its own maximum-thickness gate on coarse maps.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            support.astype(np.uint8), 8)
        wall_support = np.zeros_like(support)
        min_len_px = max(1, int(round(
            self._params['gvd_min_obstacle_length_m'] / max(resolution, 1e-6))))
        max_thickness_px = max(1, int(round(
            self._params['gvd_wall_max_thickness_m'] / max(resolution, 1e-6))))
        min_aspect = float(self._params.get('gvd_wall_min_aspect_ratio', 4.0))
        max_network_fill = float(self._params.get('gvd_wall_network_max_fill_ratio', 0.35))
        for i in range(1, count):
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            long_side, short_side = max(bw, bh), min(bw, bh)
            aspect = long_side / max(1.0, short_side)
            fill = area / max(1.0, float(bw * bh))
            if long_side >= min_len_px and (
                    (short_side <= max_thickness_px and aspect >= min_aspect) or
                    fill <= max_network_fill):
                wall_support[labels == i] = True

        dilation_px = max(0, int(self._params.get('gvd_3d_support_dilation_px', 0)))
        if dilation_px > 0:
            kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), dtype=np.uint8)
            if np.any(wall_support):
                wall_support = cv2.dilate(
                    wall_support.astype(np.uint8), kernel, iterations=1) > 0
            if np.any(observed):
                observed = cv2.dilate(
                    observed.astype(np.uint8), kernel, iterations=1) > 0
        self._latest_cloud_observed_mask = observed
        self._latest_cloud_nonwall_mask = observed & ~wall_support
        self._last_3d_wall_stats = {
            'source': cloud_source,
            'floor_z': round(float(floor_z), 3),
            'floor_points': ground_floor_points,
            'observed_cells': int(np.count_nonzero(observed)),
            'wall_cells': int(np.count_nonzero(wall_support)),
            'nonwall_cells': int(np.count_nonzero(observed & ~wall_support)),
        }
        self._last_3d_wall_stats['ground_cloud_fresh'] = bool(ground_fresh)
        self._last_3d_wall_stats['obstacle_points'] = int(points.shape[0])
        return wall_support

    def _compute_medial_axis(self, free_topo, structural_occupied):
        """One-cell medial-axis approximation of the navigable free space.

        In a sampled occupancy grid the GVD/medial axis is obtained by topology-preserving
        thinning of free space. The Euclidean distance transform supplies the clearance
        function used to locate critical points along that graph. Structural obstacles are
        already reflected in ``free_topo``; the argument is kept to make the GVD backends
        share one interface.
        """
        del structural_occupied
        free = (free_topo > 0).astype(np.uint8) * 255
        if not np.any(free):
            return free, np.zeros(free.shape, dtype=np.float32)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        return self._thin_binary(free), dist

    def _compute_gvd_ridge(self, free_topo, structural_occupied, resolution=1.0):
        """Discrete generalized Voronoi diagram of obstacle-boundary sites.

        OpenCV assigns a distinct label to every occupied pixel (``DIST_LABEL_PIXEL``).
        A free cell is on a Voronoi boundary when an adjacent free cell has a different
        nearest site and the current cell is approximately equidistant from both sites.
        Requiring the sites to be physically separated rejects label changes between
        neighbouring samples of the same locally-flat wall.

        Unlike the legacy connected-component construction, this remains valid when all
        walls touch. Unlike the former local-maxima heuristic, every accepted cell carries
        an explicit pair of distinct generating sites.
        """
        occupied = structural_occupied > 0
        free = free_topo > 0
        skeleton = np.zeros(free.shape, dtype=np.uint8)
        if not np.any(occupied) or not np.any(free):
            return skeleton, np.zeros(free.shape, dtype=np.float32)

        src = np.where(occupied, 0, 255).astype(np.uint8)
        # The labelled variant uses the 5x5 chamfer mask; compute the clearance returned
        # downstream separately with the precise Euclidean transform.
        _, labels = cv2.distanceTransformWithLabels(
            src, cv2.DIST_L2, cv2.DIST_MASK_5,
            labelType=cv2.DIST_LABEL_PIXEL)
        dist = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        obstacle_y, obstacle_x = np.nonzero(occupied)
        obstacle_labels = labels[obstacle_y, obstacle_x].astype(np.int64)
        max_label = int(labels.max())
        site_y = np.full(max_label + 1, -1, dtype=np.int32)
        site_x = np.full(max_label + 1, -1, dtype=np.int32)
        valid_labels = (obstacle_labels > 0) & (obstacle_labels <= max_label)
        site_y[obstacle_labels[valid_labels]] = obstacle_y[valid_labels]
        site_x[obstacle_labels[valid_labels]] = obstacle_x[valid_labels]

        min_sep_px = float(self._params.get('gvd_site_min_separation_m', 0.30)) / max(
            float(resolution), 1e-6)
        tolerance_px = float(self._params.get('gvd_equidistance_tolerance_px', 1.5))
        candidate = np.zeros(free.shape, dtype=bool)

        # Four undirected neighbour pairs cover the full 8-neighbourhood without wrapping.
        h, w = free.shape
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if dx >= 0:
                ays, axs = slice(0, h-dy), slice(0, w-dx)
                bys, bxs = slice(dy, h), slice(dx, w)
            else:
                ays, axs = slice(0, h-dy), slice(-dx, w)
                bys, bxs = slice(dy, h), slice(0, w+dx)

            la = labels[ays, axs]
            lb = labels[bys, bxs]
            pair = free[ays, axs] & free[bys, bxs] & (la > 0) & (lb > 0) & (la != lb)
            if not np.any(pair):
                continue

            ya, xa = np.indices(la.shape)
            ya = ya + (ays.start or 0)
            xa = xa + (axs.start or 0)
            s1y, s1x = site_y[la], site_x[la]
            s2y, s2x = site_y[lb], site_x[lb]
            sites_known = (s1y >= 0) & (s2y >= 0)
            site_sep = np.hypot(s1y-s2y, s1x-s2x)
            d1 = np.hypot(ya-s1y, xa-s1x)
            d2 = np.hypot(ya-s2y, xa-s2x)
            accepted = pair & sites_known & (site_sep >= min_sep_px) & (np.abs(d1-d2) <= tolerance_px)
            candidate[ays, axs] |= accepted

        skeleton[candidate & free] = 255
        skeleton = self._thin_binary(skeleton)
        return skeleton, dist

    @staticmethod
    def _thin_binary(mask):
        """Topology-preserving Zhang-Suen thinning to a one-pixel skeleton.

        Kept local instead of requiring opencv-contrib's ``ximgproc.thinning`` so the room
        manager behaves the same in both the Habitat and robot environments.
        """
        image = (mask > 0).astype(np.uint8)
        if not np.any(image):
            return image * 255

        ximgproc = getattr(cv2, 'ximgproc', None)
        if ximgproc is not None and hasattr(ximgproc, 'thinning'):
            return ximgproc.thinning(image * 255)

        changed = True
        while changed:
            changed = False
            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = p2+p3+p4+p5+p6+p7+p8+p9
            transitions = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            for a, b in ((p3, p4), (p4, p5), (p5, p6), (p6, p7),
                         (p7, p8), (p8, p9), (p9, p2)):
                transitions += ((a == 0) & (b == 1)).astype(np.uint8)
            remove = ((image == 1) & (neighbours >= 2) & (neighbours <= 6) &
                      (transitions == 1) & ((p2*p4*p6) == 0) & ((p4*p6*p8) == 0))
            if np.any(remove):
                image[remove] = 0
                changed = True

            padded = np.pad(image, 1)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = p2+p3+p4+p5+p6+p7+p8+p9
            transitions = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            for a, b in ((p3, p4), (p4, p5), (p5, p6), (p6, p7),
                         (p7, p8), (p8, p9), (p9, p2)):
                transitions += ((a == 0) & (b == 1)).astype(np.uint8)
            remove = ((image == 1) & (neighbours >= 2) & (neighbours <= 6) &
                      (transitions == 1) & ((p2*p4*p8) == 0) & ((p2*p6*p8) == 0))
            if np.any(remove):
                image[remove] = 0
                changed = True

        return image * 255

    @staticmethod
    def _compute_gvd(free_topo, structural_occupied):
        """Discrete brushfire/GVD on the *topological* free/obstacle
        masks: labels every topo-free pixel with the id of its nearest
        structural-obstacle connected component, then marks as skeleton
        every topo-free pixel adjacent to a topo-free pixel with a
        *different* label -- i.e. equidistant from two obstacles."""
        src = np.where(structural_occupied > 0, 0, 255).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(
            src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE,
            labelType=cv2.DIST_LABEL_CCOMP)
        lbl = labels.astype(np.int32)
        freeb = free_topo > 0
        diff = np.zeros(lbl.shape, dtype=bool)
        diff[:, :-1] |= (lbl[:, :-1] != lbl[:, 1:]) & freeb[:, :-1] & freeb[:, 1:]
        diff[:-1, :] |= (lbl[:-1, :] != lbl[1:, :]) & freeb[:-1, :] & freeb[1:, :]
        diff[:-1, :-1] |= (lbl[:-1, :-1] != lbl[1:, 1:]) & freeb[:-1, :-1] & freeb[1:, 1:]
        diff[1:, :-1] |= (lbl[1:, :-1] != lbl[:-1, 1:]) & freeb[1:, :-1] & freeb[:-1, 1:]
        skeleton = np.zeros(lbl.shape, dtype=np.uint8)
        skeleton[diff & freeb] = 255
        return skeleton, dist

    def _prune_skeleton(self, skeleton, resolution):
        """Peels degree-1 spurs shorter than ``gvd_prune_min_branch_m`` off
        the raw GVD skeleton -- grid-discretisation artefacts, not real
        topological branches."""
        skel = skeleton > 0
        min_len_px = max(1, int(round(self._params['gvd_prune_min_branch_m'] / max(resolution, 1e-6))))
        for _ in range(min_len_px):
            deg = self._skeleton_degree(skel)
            endpoints = skel & (deg == 1)
            if not np.any(endpoints):
                break
            skel &= ~endpoints
        return (skel.astype(np.uint8) * 255)

    @staticmethod
    def _skeleton_degree(skel_bool):
        """Degree in the same corner-safe 8-neighbour graph used for tracing."""
        skel = skel_bool.astype(bool)
        h, w = skel.shape
        degree = np.zeros((h, w), dtype=np.uint8)
        ys, xs = np.nonzero(skel)
        for y, x in zip(ys, xs):
            degree[y, x] = len(RoomManager._skeleton_neighbors(skel, int(y), int(x)))
        return degree

    @staticmethod
    def _skeleton_neighbors(skel, y, x):
        """Graph neighbours without diagonal shortcut edges.

        In an 8-connected thinned raster, a one-pixel staircase creates triangles: two
        orthogonal edges plus their diagonal. Those triangles turn ordinary curve pixels
        into false degree-3 junctions. Keep a diagonal only when neither corresponding
        orthogonal bridge pixel exists.
        """
        h, w = skel.shape
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y+dy, x+dx
                if not (0 <= ny < h and 0 <= nx < w and skel[ny, nx]):
                    continue
                if dy != 0 and dx != 0:
                    bridge_a = 0 <= y < h and 0 <= x+dx < w and skel[y, x+dx]
                    bridge_b = 0 <= y+dy < h and 0 <= x < w and skel[y+dy, x]
                    if bridge_a or bridge_b:
                        continue
                out.append((ny, nx))
        return out

    def _trace_branches(self, skel_bool):
        """Splits the skeleton into branches between graph nodes
        (endpoints: degree 1, junctions: degree >= 3). Each branch is the
        pixel chain of degree-2 pixels connecting two nodes (or a direct
        node-node adjacency). Branches may be traced twice (once from each
        end) -- harmless here, since only the per-branch minimum/end-node
        clearances matter, and both traversals compute the same values."""
        deg = self._skeleton_degree(skel_bool)
        ys, xs = np.nonzero(skel_bool)
        is_node = deg[ys, xs] != 2
        node_set = set(zip(ys[is_node].tolist(), xs[is_node].tolist()))
        h, w = skel_bool.shape

        def neighbors(y, x):
            return self._skeleton_neighbors(skel_bool, y, x)

        def edge(a, b):
            return (a, b) if a <= b else (b, a)

        branches = []
        visited = set()
        max_steps = int(h) * int(w) + 8
        for node in node_set:
            for nbr in neighbors(*node):
                if edge(node, nbr) in visited:
                    continue
                path = [node, nbr]
                visited.add(edge(node, nbr))
                prev, curr = node, nbr
                steps = 0
                while curr not in node_set and steps < max_steps:
                    nxts = [n for n in neighbors(*curr)
                            if n != prev and edge(curr, n) not in visited]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    path.append(nxt)
                    visited.add(edge(curr, nxt))
                    prev, curr = curr, nxt
                    steps += 1
                branches.append(path)

        # A pure cycle has no degree != 2 nodes, so seed its one branch from any unvisited
        # edge. The same loop also safely captures a residual edge in malformed input.
        for y, x in zip(ys.tolist(), xs.tolist()):
            start = (y, x)
            for nbr in neighbors(y, x):
                if edge(start, nbr) in visited:
                    continue
                path = [start, nbr]
                visited.add(edge(start, nbr))
                prev, curr = start, nbr
                steps = 0
                while steps < max_steps:
                    nxts = [n for n in neighbors(*curr)
                            if n != prev and edge(curr, n) not in visited]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    path.append(nxt)
                    visited.add(edge(curr, nxt))
                    prev, curr = curr, nxt
                    steps += 1
                branches.append(path)
        return branches

    @staticmethod
    def _branch_direction(path, min_idx):
        """Returns a local branch direction angle in radians.

        The angle is estimated from nearby pixels around the critical point
        so the doorway cut can be drawn orthogonal to the corridor axis.
        """
        if len(path) < 2:
            return 0.0
        left_i = max(0, min_idx - 2)
        right_i = min(len(path) - 1, min_idx + 2)
        if left_i == right_i:
            left_i = max(0, min_idx - 1)
            right_i = min(len(path) - 1, min_idx + 1)
        y1, x1 = path[left_i]
        y2, x2 = path[right_i]
        dy = float(y2 - y1)
        dx = float(x2 - x1)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return 0.0
        return math.atan2(dy, dx)

    def _critical_points(self, skeleton, dist, resolution):
        """For every branch of the (pruned) skeleton, finds the single
        pixel of minimum clearance along that branch and accepts it as a
        doorway cut point only if BOTH hold:

        - absolute: the implied free-space width (2 * clearance) is below
          ``gvd_door_max_m``;
        - relative ("bottleneck"): that minimum clearance is smaller than
          ``gvd_bottleneck_ratio`` times the clearance at the wider of the
          branch's two end nodes.

        The relative test rejects narrow gaps that sit inside an already
        cramped/cluttered area (both ends of the branch are just as narrow
        as its middle -- e.g. a gap between two pieces of furniture), while
        accepting genuine doorways, where the branch narrows sharply
        relative to the open rooms on either side of it.
        """
        skel_bool = skeleton > 0
        if not np.any(skel_bool):
            self._last_critical_stats = {"branches": 0, "reason": "empty skeleton"}
            return []
        door_radius_px = (self._params['gvd_door_max_m'] / 2.0) / max(resolution, 1e-6)
        branches = self._trace_branches(skel_bool)
        points = []

        # GA-196. WHY a branch was rejected, counted. `branches_cut=0` was the only signal
        # this stage produced, and it is the same shape as every unlogged exit this review
        # has found: it says a thing did not happen and nothing about which test refused it.
        # Measured over two runs, 87% of sweeps cut nothing while the skeleton was ~18,600
        # px and NEVER empty -- so the failure is here, among these four filters, and no
        # bundle could say which. Counters only; no behaviour changes.
        rej = {"too_short_px": 0, "too_short_m": 0, "wider_than_door": 0,
               "minimum_at_endpoint": 0, "no_end_clearance": 0,
               "not_a_bottleneck": 0, "wall_supported_recovery": 0, "accepted": 0}
        min_branch_px = max(4, int(round(
            self._params['gvd_prune_min_branch_m'] / max(resolution, 1e-6))))
        endpoint_margin_px = max(1, int(round(
            self._params.get('gvd_critical_endpoint_margin_m', 0.20) /
            max(resolution, 1e-6))))
        narrowest = None

        for path in branches:
            if len(path) < 2:
                rej["too_short_px"] += 1
                continue
            if len(path) < min_branch_px:
                rej["too_short_m"] += 1
                continue
            vals = [float(dist[y, x]) for y, x in path]
            min_idx = int(np.argmin(vals))
            min_val = vals[min_idx]
            end_clearance = max(vals[0], vals[-1])
            # The narrowest clearance any branch offered, in METRES of free width, so the
            # log says how far from a doorway the map actually was rather than only that
            # nothing qualified. A run where this sits just above gvd_door_max_m is a
            # threshold question; one where it sits far above is a map question.
            width_m = 2.0 * min_val * resolution
            if narrowest is None or width_m < narrowest:
                narrowest = width_m

            y, x = path[min_idx]
            theta = float(self._branch_direction(path, min_idx))
            wall_score = self._door_wall_support_score(
                int(y), int(x), theta, min_val, resolution)
            min_wall_support = float(self._params.get('detected_wall_door_min_support', 0.16))
            wall_supported_width = (
                wall_score >= min_wall_support and
                width_m <= float(self._params.get('detected_wall_door_max_m', 2.0)))
            if min_val > door_radius_px and not wall_supported_width:
                rej["wider_than_door"] += 1
                continue
            if min_idx < endpoint_margin_px or min_idx >= len(path)-endpoint_margin_px:
                rej["minimum_at_endpoint"] += 1
                continue
            if end_clearance <= 1e-6:
                rej["no_end_clearance"] += 1
                continue
            base_bottleneck = (
                min_val <= float(self._params['gvd_bottleneck_ratio']) * end_clearance)
            wall_recovery = (
                wall_score >= min_wall_support and
                min_val <= float(self._params.get(
                    'detected_wall_bottleneck_ratio', 0.95)) * end_clearance)
            if not base_bottleneck and not wall_recovery:
                rej["not_a_bottleneck"] += 1
                continue
            if wall_recovery and not base_bottleneck:
                rej["wall_supported_recovery"] += 1
            points.append((int(y), int(x), theta, float(min_val), float(wall_score)))

        # The graph representation can expose the same physical bottleneck on several
        # adjacent branches around a junction. Prefer a representative supported by measured
        # walls; use clearance only as the tie-breaker.
        nms_px = float(self._params.get('gvd_door_nms_m', 0.60)) / max(resolution, 1e-6)
        selected = []
        for point in sorted(points, key=lambda p: (-p[4], p[3])):
            point_world = (point[1], point[0])
            if all(not self._doorway_same_cluster(
                    point_world, point[2], (q[1], q[0]), q[2],
                    radius=nms_px * resolution) for q in selected):
                selected.append(point)
        rej["accepted"] = len(selected)
        min_wall_support = float(self._params.get('detected_wall_door_min_support', 0.16))

        self._last_critical_stats = {
            "branches": len(branches),
            "min_branch_px": min_branch_px,
            "door_max_m": float(self._params['gvd_door_max_m']),
            "narrowest_branch_m": (None if narrowest is None else round(narrowest, 3)),
            "wall_supported_candidates": sum(p[4] >= min_wall_support for p in selected),
            "max_wall_support": round(max((p[4] for p in selected), default=0.0), 3),
            **rej,
        }
        return selected

    def _door_wall_support_score(self, y, x, theta, radius_px, resolution):
        """Return 0..1 when confirmed walls support both ends of a proposed door cut."""
        support = getattr(self, '_active_detected_wall_support', None)
        if support is None or not np.any(support):
            return 0.0
        nx, ny = -math.sin(theta), math.cos(theta)
        search_px = max(1, int(round(float(self._params.get(
            'detected_wall_door_endpoint_radius_m', 0.20)) / max(resolution, 1e-6))))
        h, w = support.shape
        radius_px = max(1.0, float(radius_px))
        # Look for the first wall hit along the proposed opening normal. Merely
        # finding an unrelated wall somewhere in a broad endpoint window is not
        # enough: the free-space ray must actually terminate at a wall on each side.
        max_scan = int(math.ceil(radius_px + search_px))
        min_hit = max(1, int(math.floor(0.35 * radius_px)))
        values = []
        for sign in (-1.0, 1.0):
            hit_value = 0.0
            hit_distance = None
            for step in range(min_hit, max_scan + 1):
                ix = int(round(x + sign * step * nx))
                iy = int(round(y + sign * step * ny))
                if not (0 <= ix < w and 0 <= iy < h):
                    break
                x0, x1 = max(0, ix-search_px), min(w, ix+search_px+1)
                y0, y1 = max(0, iy-search_px), min(h, iy+search_px+1)
                value = (float(np.max(support[y0:y1, x0:x1]))
                         if x0 < x1 and y0 < y1 else 0.0)
                if value > 0.0:
                    hit_value = value
                    hit_distance = step
                    break
            if hit_distance is None:
                return 0.0
            values.append(hit_value)
        # Both rays must terminate on supported walls; one nearby fragment is not
        # a doorway frame. The returned score still reflects the weaker side.
        return min(values)

    def _cut_free_space(self, free, dist_real, critical_points, resolution=None,
                        grid=None, wall_mask=None):
        """Insert critical lines orthogonal to the GVD at doorway minima.

        Each line spans the local clearance diameter and reconnects the two nearest sides
        of the obstacle boundary. This is the standard room partition induced by a critical
        GVD point; the previous disk removed free space in every direction and could erase
        corridor length or merge unrelated nearby cuts.
        """
        cut = free.copy()
        margin_px = max(1, int(self._params['gvd_cut_margin_px']))
        resolution = float(resolution if resolution is not None else
                           getattr(self, '_active_resolution', 1.0))
        min_component_px = max(1, int(round(
            float(self._params['min_room_area_m2']) /
            max(resolution ** 2, 1e-12))))
        accepted = []
        accepted_locked = 0
        rejected_small = 0
        rejected_no_split = 0
        rejected_parent_empty = 0
        rejected_trial_no_split = 0
        # Measured wall support first, then narrowest bottleneck. Once a valid partition
        # exists, later cuts are evaluated against the already partitioned map; ordering is
        # therefore a semantic decision, not merely a performance detail.
        for point in sorted(
                critical_points,
                key=lambda p: (-(p[4] if len(p) >= 5 else 0.0),
                               p[3] if len(p) >= 4 else 0.0)):
            if len(point) >= 4:
                y, x, theta, _ = point[:4]
            else:
                y, x = point[:2]
                theta = 0.0
            locked_doorway = len(point) >= 8 and bool(point[7])
            radius = int(round(float(dist_real[y, x]))) + margin_px
            radius = max(1, radius)
            # Normal to the local GVD tangent.
            if len(point) >= 7 and point[5] is not None and point[6] is not None:
                # A confirmed doorway carries its metric opening segment.  Do not
                # reconstruct it from the current, jittery medial-axis pixel.
                target_grid = grid if grid is not None else getattr(self, 'last_grid', None)
                p0 = self._world_to_grid(float(point[5][0]), float(point[5][1]), target_grid)
                p1 = self._world_to_grid(float(point[6][0]), float(point[6][1]), target_grid)
                if p0 is None or p1 is None:
                    rejected_no_split += 1
                    continue
                # The stored opening endpoints are metric estimates and can land one
                # raster cell short of the reinforced wall. Extend the separator
                # along its own axis, never sideways into the room.
                segment = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
                segment /= max(float(np.linalg.norm(segment)), 1e-9)
                p0 = tuple(np.rint(np.asarray(p0, dtype=float) - margin_px * segment).astype(int))
                p1 = tuple(np.rint(np.asarray(p1, dtype=float) + margin_px * segment).astype(int))
            else:
                nx, ny = -math.sin(theta), math.cos(theta)
                # Formal GVD cut: the branch supplies the local tangent, while
                # the two endpoints are obtained by intersecting its normal with
                # the nearest structural wall on each side. The distance-transform
                # diameter is only a fallback when no wall intersection is visible.
                wall_endpoints = None
                if wall_mask is not None:
                    max_scan = max(radius + margin_px, int(round(
                        float(self._params.get('gvd_door_max_m', 1.20)) /
                        max(resolution, 1e-6))))
                    h, w = wall_mask.shape
                    sides = []
                    for sign in (-1.0, 1.0):
                        previous = (int(round(x)), int(round(y)))
                        hit = None
                        for step in range(1, max_scan + 1):
                            qx = int(round(x + sign * step * nx))
                            qy = int(round(y + sign * step * ny))
                            if not (0 <= qx < w and 0 <= qy < h):
                                break
                            if wall_mask[qy, qx]:
                                hit = previous
                                break
                            previous = (qx, qy)
                        if hit is None:
                            sides = []
                            break
                        sides.append(hit)
                    if len(sides) == 2:
                        wall_endpoints = (sides[0], sides[1])
                if wall_endpoints is not None:
                    p0, p1 = wall_endpoints
                else:
                    p0 = (int(round(x-radius*nx)), int(round(y-radius*ny)))
                    p1 = (int(round(x+radius*nx)), int(round(y+radius*ny)))

            trial = cut.copy()
            cv2.line(trial, p0, p1, 0, max(1, 2*margin_px+1))
            _, before_labels = cv2.connectedComponents(cut, 8)
            _, after_labels = cv2.connectedComponents(trial, 8)
            parent_label = int(before_labels[int(y), int(x)])
            if parent_label <= 0:
                rejected_no_split += 1
                rejected_parent_empty += 1
                continue
            parent = before_labels == parent_label
            children = np.unique(after_labels[parent])
            children = children[children > 0]
            if children.size < 2:
                rejected_no_split += 1
                rejected_trial_no_split += 1
                continue
            areas = np.asarray([
                np.count_nonzero(parent & (after_labels == child)) for child in children
            ])
            if int(areas.min()) < min_component_px:
                rejected_small += 1
                continue
            cut = trial
            accepted.append(point)
            accepted_locked += int(locked_doorway)

        self._last_validated_cuts = accepted
        self._last_cut_stats = {
            'proposed': len(critical_points),
            'accepted': len(accepted),
            'accepted_locked': accepted_locked,
            'accepted_wall_supported': sum(
                len(point) >= 5 and point[4] >= float(self._params.get(
                    'detected_wall_door_min_support', 0.16)) for point in accepted),
            'rejected_no_split': rejected_no_split,
            'rejected_parent_empty': rejected_parent_empty,
            'rejected_trial_no_split': rejected_trial_no_split,
            'rejected_small_partition': rejected_small,
        }
        return cut

    @staticmethod
    def _grow_labels(free, cut, dist_real):
        """Labels the connected components of the cut mask, then grows
        each label back to the full extent of the original (uncut) free
        mask via a distance-transform-guided watershed, restoring the true
        room shape instead of leaving a hole at every doorway."""
        count, labels = cv2.connectedComponents(cut, 8)
        if count <= 1:
            return None
        elevation = (255 - cv2.normalize(dist_real, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
        elevation_bgr = cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)
        markers = labels.astype(np.int32) + 1
        markers[free == 0] = 1
        unknown = (free > 0) & (labels == 0)
        markers[unknown] = 0
        cv2.watershed(elevation_bgr, markers)
        markers[markers == 1] = 0
        markers[markers == -1] = 0
        return markers

    def _merge_small_labels(self, markers, min_pixels, dist_real, resolution):
        count = int(markers.max())
        if count <= 1:
            return markers
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for label in range(1, count + 1):
            mask = (markers == label)
            size = int(mask.sum())
            if size == 0:
                continue
            if size >= min_pixels:
                continue
            dil = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
            neigh = markers[(dil > 0) & (markers != label) & (markers > 0)]
            if neigh.size == 0:
                continue
            vals, counts = np.unique(neigh, return_counts=True)
            best = vals[np.argmax(counts)]
            markers[markers == label] = best
        return markers

    @staticmethod
    def _ear_clip_triangulate(polygon):
        pts = [tuple(p) for p in polygon]
        if len(pts) < 3:
            return []
        area2 = sum(pts[i][0]*pts[(i+1) % len(pts)][1] - pts[(i+1) % len(pts)][0]*pts[i][1]
                    for i in range(len(pts)))
        if area2 < 0:
            pts.reverse()

        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

        def point_in_tri(p, a, b, c):
            d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
            return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))

        idx = list(range(len(pts)))
        triangles = []
        while len(idx) > 3:
            n = len(idx)
            for k in range(n):
                i0, i1, i2 = idx[(k-1) % n], idx[k], idx[(k+1) % n]
                a, b, c = pts[i0], pts[i1], pts[i2]
                if cross(a, b, c) <= 0:
                    continue
                if any(point_in_tri(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                    continue
                triangles.extend([a, b, c])
                idx.pop(k)
                break
            else:
                break
        if len(idx) == 3:
            triangles.extend([pts[i] for i in idx])
        return triangles

    def _segment_regions_gvd(self, grid):
        free, occupied = self._binary_free_obstacle(grid)
        if not np.any(free):
            return []
        resolution = float(grid.info.resolution)

        cloud_support = None
        if self._params.get('enable_3d_structural_filter', False) and grid is not None:
            cloud_support = self._cloud_structural_support(grid)

        detected_wall_support = self._detected_wall_support(grid)
        self._active_detected_wall_support = detected_wall_support
        structural_occ = self._structural_obstacles(occupied, resolution, grid, cloud_support=cloud_support)
        # Keep a copy before injecting measured walls. The latter can close a real
        # doorway in the occupancy raster and make the corresponding GVD branch
        # disappear before it can be used as a geometric partition hypothesis.
        pre_wall_structural_occ = structural_occ.copy()
        reinforced_cells = 0
        closed_free_cells = 0
        topology_wall_mask = None
        if (detected_wall_support is not None and
                self._params.get('detected_wall_reinforce_obstacles', False)):
            confidence_gate = float(self._params.get(
                'detected_wall_topology_confidence', 0.33))
            # Keep the raster OpenCV-compatible: doorway holes are drawn below
            # with cv2.circle(), which does not accept a boolean image.
            topology_wall_mask = (detected_wall_support >= confidence_gate).astype(np.uint8)
            dilation_px = max(0, int(self._params.get(
                'detected_wall_topology_dilation_px', 1)))
            if dilation_px > 0 and np.any(topology_wall_mask):
                kernel = np.ones((2 * dilation_px + 1, 2 * dilation_px + 1), np.uint8)
                topology_wall_mask = cv2.dilate(
                    topology_wall_mask, kernel, iterations=1)
            # Restore confirmed door openings after thickening the wall. The
            # doorway is the only intentional hole in this structural raster.
            doorway_radius = max(0.30, float(self._params.get(
                'gvd_door_max_m', 1.20)) * 0.5)
            for doorway in self._doorway_vlm_state.values():
                if (doorway.get('status') != 'confirmed' or
                        not doorway.get('cuttable') or doorway.get('world') is None):
                    continue
                q = self._world_to_grid(*doorway['world'], grid)
                if q is not None:
                    cv2.circle(topology_wall_mask, q,
                               max(1, int(round(doorway_radius / max(resolution, 1e-6)))),
                               0, -1)
            topology_wall_bool = topology_wall_mask.astype(bool)
            reinforced_cells = int(np.count_nonzero(
                topology_wall_bool & (structural_occ == 0)))
            structural_occ[topology_wall_bool] = 255
        free_topo = self._fill_nonstructural_obstacles(free, occupied, structural_occ, resolution)

        # Preliminary pass on the raw free space. This is only a lazy doorway proposal
        # pass; it prevents wall reinforcement from hiding a door before the VLM sees it.
        pre_points = []
        if topology_wall_mask is not None:
            # This pass is intentionally based on the original occupancy map, not
            # on ``free_topo`` after measured-wall reinforcement.
            pre_free = self._fill_room_holes(
                free.copy(), resolution, protected=pre_wall_structural_occ)
            pre_free = self._navigable_free_component(pre_free, grid)
            pre_method = str(self._params.get('gvd_method', 'label_diff'))
            if pre_method == 'medial_axis':
                pre_raw, pre_dist = self._compute_medial_axis(pre_free, pre_wall_structural_occ)
            elif pre_method in ('boundary_sites', 'ridge'):
                pre_raw, pre_dist = self._compute_gvd_ridge(pre_free, pre_wall_structural_occ, resolution)
            else:
                pre_raw, pre_dist = self._compute_gvd(pre_free, pre_wall_structural_occ)
            pre_points = self._critical_points(
                self._prune_skeleton(pre_raw, resolution), pre_dist, resolution)
            # Same abstain-vs-reject distinction as the main pass below. This copy tested
            # only the flag, so it dropped every pre-pass proposal whenever no wall was
            # confirmed -- and the pre-pass is what keeps a real doorway visible long enough
            # for the VLM to confirm it.
            if (self._params.get('doorway_require_wall_support', True) and
                    self._active_detected_wall_support is not None and
                    np.any(self._active_detected_wall_support)):
                minimum = float(self._params.get('detected_wall_door_min_support', 0.16))
                pre_points = [point for point in pre_points
                              if len(point) >= 5 and point[4] >= minimum]
            self._schedule_doorway_confirmations(grid, pre_points)

        if (topology_wall_mask is not None and
                self._params.get('detected_wall_close_free_space', True)):
            # A wall seen repeatedly in depth is direct surface evidence.  Let it
            # repair short false-free gaps left by the 2D mapper; endpoint joining
            # is capped below door width, so this cannot bridge an actual doorway.
            # Keep a confirmed doorway-sized hole in the reinforced wall. The later
            # GVD cut will close this opening deliberately, but only after it has a
            # visual confirmation and a valid two-component split.
            # Preserve every geometrically supported GVD opening long enough for
            # the actual GVD cut below. VLM confirmation is asynchronous and must
            # not be required merely to keep the geometric opening visible.
            for point in pre_points:
                if len(point) < 2:
                    continue
                q = (int(point[1]), int(point[0]))
                if 0 <= q[0] < topology_wall_mask.shape[1] and 0 <= q[1] < topology_wall_mask.shape[0]:
                    cv2.circle(topology_wall_mask, q,
                               max(1, int(round(0.5 * float(self._params.get(
                                   'gvd_door_max_m', 1.20)) / max(resolution, 1e-6)))),
                               0, -1)
            closed_free_cells = int(np.count_nonzero(
                topology_wall_mask.astype(bool) & (free_topo > 0)))
            free_topo[topology_wall_mask.astype(bool)] = 0
        # Furniture removed from the topology can leave isolated corner pixels after the
        # occupancy median filter. Closed holes below the configured area are clutter, not
        # navigable-space boundaries, and would create dense spurious medial-axis branches.
        free_topo = self._fill_room_holes(
            free_topo, resolution, protected=structural_occ)
        free_topo = self._navigable_free_component(free_topo, grid)
        self._active_resolution = resolution
        # GA-137: `label_diff` is today's behaviour and stays the default -- it is provably
        # always-empty on a connected floorplan, but changing room segmentation changes
        # EVERY room-scoped number, which is a run-design decision, not mine to take.
        _method = str(self._params.get('gvd_method', 'label_diff'))
        if _method == 'medial_axis':
            skeleton_raw, dist_topo = self._compute_medial_axis(
                free_topo, structural_occ)
        elif _method in ('boundary_sites', 'ridge'):
            skeleton_raw, dist_topo = self._compute_gvd_ridge(
                free_topo, structural_occ, resolution)
        elif _method == 'label_diff':
            skeleton_raw, dist_topo = self._compute_gvd(free_topo, structural_occ)
        else:
            raise ValueError(f"unknown rooms.gvd_method {_method!r}; "
                             f"expected 'medial_axis', 'boundary_sites', 'ridge', or 'label_diff'")
        skeleton = self._prune_skeleton(skeleton_raw, resolution)
        critical_points = self._critical_points(skeleton, dist_topo, resolution)

        # A narrow branch in the middle of a room is not a doorway hypothesis.  When
        # depth walls are available, require the GVD bottleneck to terminate against
        # both sides of that wall network before spending a VLM call on it.
        # `np.any`, NOT `is not None`. The guard means "do we have wall evidence to judge
        # against"; `_detected_wall_support` returns None only when the feature is switched off
        # or the grid is missing, and otherwise returns an ALL-ZERO raster when no wall has been
        # confirmed. A zeros array is not None, so this filter always ran, every score was 0.0,
        # and 0.0 < 0.16 deleted every doorway. MEASURED on 20260911_173938_hm3d_00861: 63
        # skeleton branches, 12 bottlenecks accepted by geometry, all 12 dropped, door_cuts=0,
        # one room for the whole house. `_door_wall_support_score` already draws this
        # distinction (it returns 0.0 when the raster is empty); it died here, at the point of
        # use, where 0.0 was read as "rejected" rather than as "cannot say".
        if (self._params.get('doorway_require_wall_support', True) and
                self._active_detected_wall_support is not None and
                np.any(self._active_detected_wall_support)):
            min_support = float(self._params.get('detected_wall_door_min_support', 0.16))
            critical_points = [point for point in critical_points
                               if len(point) >= 5 and point[4] >= min_support]

        # The wall-reinforced pass can no longer see an opening that was closed by
        # the measured wall raster. Retain original GVD candidates and, after a
        # doorway has been confirmed, retain its metric cut even if the current
        # GVD branch temporarily disappears. Before confirmation, VLM never adds
        # anything here.
        for point in pre_points:
            if all(math.hypot(point[0] - other[0], point[1] - other[1]) >
                   float(self._params.get('gvd_door_nms_m', 1.20)) /
                   max(resolution, 1e-6) for other in critical_points):
                critical_points.append(point)
        # A VLM answer belongs to an image captured on a previous map update. Reuse
        # its metric endpoints when the current GVD still has a nearby branch.
        # Confirmed doorways are immutable locks. Do not recover or reposition
        # them from the current GVD; their stored metric line is applied below.
        recovered_points = []
        for recovered in recovered_points:
            distances = [math.hypot(recovered[0] - point[0], recovered[1] - point[1])
                         for point in critical_points]
            if not distances:
                continue
            nearest = int(np.argmin(distances))
            if distances[nearest] <= float(self._params.get('gvd_door_nms_m', 1.20)) / max(resolution, 1e-6):
                point = critical_points[nearest]
                # Preserve the current GVD position/orientation/clearance, while
                # reusing the confirmed metric opening endpoints for stability.
                base = tuple(point[:5]) if len(point) >= 5 else tuple(point[:4]) + (0.0,)
                critical_points[nearest] = base + tuple(recovered[5:7]) + (True,)
        for recovered in recovered_points:
            if all(math.hypot(recovered[0] - point[0], recovered[1] - point[1]) >
                   float(self._params.get('gvd_door_nms_m', 1.20)) /
                   max(resolution, 1e-6) for point in critical_points):
                # A confirmed doorway is a locked part of the accumulated
                # partition, so its cut may be restored if the current GVD branch
                # temporarily disappears.
                critical_points.append(recovered)

        # VLM runs after geometric hypotheses exist. It is deliberately not used to
        # filter them: GVD topology determines the cut, VLM only stabilizes it.
        self._schedule_doorway_confirmations(grid, critical_points)
        vlm_candidate_worlds = [
            self._grid_to_world(point[1], point[0], grid) for point in critical_points]
        self._last_doorway_vlm_stats = {
            'candidates': len(vlm_candidate_worlds),
            'recovered_confirmed': len(recovered_points),
            'cuttable_confirmed': sum(
                self._doorway_cut_confirmed(world) for world in vlm_candidate_worlds),
            'pending': sum(
                self._doorway_key(world) in self._doorway_vlm_pending
                for world in vlm_candidate_worlds),
        }

        cut_wall_mask = (structural_occ > 0)
        cut = self._cut_free_space(
            free_topo, dist_topo, critical_points, resolution, grid=grid,
            wall_mask=cut_wall_mask)
        cut = self._apply_confirmed_door_cuts(
            cut, grid, resolution, wall_mask=structural_occ)
        validated_cuts = getattr(self, '_last_validated_cuts', [])
        cut = self._detected_wall_doorway_cuts(
            free_topo, cut, grid, resolution)
        wall_door_cuts = getattr(self, '_last_wall_door_cut_stats', {})

        if self._params.get('room_use_watershed', False):
            markers = self._grow_labels(free_topo, cut, dist_topo)
        else:
            _, markers = cv2.connectedComponents(cut, 8)
        if markers is None:
            _, markers = cv2.connectedComponents(free_topo, 8)

        # A cut may create tiny connected components around clutter or map noise. Merge
        # them before polygon extraction using the configured minimum room area.
        min_pixels = max(
            int(self._params['min_region_pixels']),
            int(round(float(self._params['min_room_area_m2']) /
                      max(resolution * resolution, 1e-12))))
        markers = self._merge_small_labels(markers, min_pixels, dist_topo, resolution)

        _skel_px = int(np.count_nonzero(skeleton))
        # GA-137: an empty skeleton is a FAILURE, not a measurement, and this line has been
        # printing the fact of its own failure at INFO since the beginning -- `regions=N`
        # comes from the watershed and is non-zero whether or not anything was segmented,
        # so the line reads like a result. skeleton_px=0 means no branches, nothing to cut,
        # and exactly one region per floor: the room gate, the room prior and every
        # room-scoped number downstream are then no-ops over a single room.
        # GA-196: the rejection breakdown travels with the line that reports the cut count,
        # so a bundle can say WHICH test refused every branch instead of only that none
        # survived. Rendered as key=value pairs; an exact-field reader parses it, and a
        # substring grep over `N regions` does not (rule 50 -- that grep is how the branch
        # count was misread as a region count in the first place).
        _cstats = getattr(self, '_last_critical_stats', {}) or {}
        _cdesc = ' '.join(f'{k}={v}' for k, v in _cstats.items())
        _fill_stats = getattr(self, '_last_topology_fill_stats', {}) or {}
        _fill_desc = ' '.join(f'{k}={v}' for k, v in _fill_stats.items())
        _wall_stats = getattr(self, '_last_3d_wall_stats', {}) or {}
        _wall_desc = ' '.join(f'{k}={v}' for k, v in _wall_stats.items())
        _detected_stats = getattr(self, '_last_detected_wall_stats', {}) or {}
        _detected_stats = {**_detected_stats,
                           'reinforced_cells': reinforced_cells,
                           'closed_false_free_cells': closed_free_cells,
                           'door_candidates_recovered': int(
                               _cstats.get('wall_supported_recovery', 0))}
        _detected_desc = ' '.join(f'{k}={v}' for k, v in _detected_stats.items())
        _cut_stats = getattr(self, '_last_cut_stats', {}) or {}
        _cut_desc = ' '.join(f'{k}={v}' for k, v in _cut_stats.items())
        _wall_cut_stats = getattr(self, '_last_wall_door_cut_stats', {}) or {}
        _wall_cut_desc = ' '.join(f'{k}={v}' for k, v in _wall_cut_stats.items())
        _doorway_vlm_desc = ' '.join(
            f'{k}={v}' for k, v in (getattr(self, '_last_doorway_vlm_stats', {}) or {}).items())
        _free_stats = getattr(self, '_last_free_component_stats', {}) or {}
        _free_desc = ' '.join(f'{k}={v}' for k, v in _free_stats.items())
        region_labels = np.unique(markers[markers > 0])
        self._log(
            'warn' if _skel_px == 0 else 'info',
            f'GVD segmentation: skeleton_px={_skel_px} '
            f'branches_cut={len(validated_cuts)} regions={len(region_labels)}'
            + (f' | critical_points: {_cdesc}' if _cdesc else '')
            + (f' | cut_validation: {_cut_desc}' if _cut_desc else '')
            + (f' | wall_door_cuts: {_wall_cut_desc}' if _wall_cut_desc else '')
            + (f' | doorway_vlm: {_doorway_vlm_desc}' if _doorway_vlm_desc else '')
            + (f' | free_component: {_free_desc}' if _free_desc else '')
            + (f' | topology_fill: {_fill_desc}' if _fill_desc else '')
            + (f' | cloud_3d: {_wall_desc}' if _wall_desc else '')
            + (f' | detected_walls: {_detected_desc}' if _detected_desc else '')
            + (' -- SKELETON EMPTY: no segmentation happened, every object will land in one'
               ' room. regions= here is the watershed count, not evidence of a split.'
               if _skel_px == 0 else ''))

        candidates = []
        for label in range(1, int(markers.max()) + 1):
            mask = (markers == label).astype(np.uint8) * 255
            pixels = int(cv2.countNonZero(mask))
            if pixels < min_pixels:
                continue
            area = pixels * resolution * resolution
            if not (self._params['min_room_area_m2'] <= area <= self._params['room_max_area_m2']):
                continue
            if self._params['discard_border_regions']:
                touches_border = bool(
                    np.any(mask[0, :]) or np.any(mask[-1, :]) or
                    np.any(mask[:, 0]) or np.any(mask[:, -1]))
                if touches_border:
                    continue
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            hierarchy = hierarchy[0]
            outer_idx = max(
                (i for i in range(len(contours)) if hierarchy[i][3] == -1),
                key=lambda i: cv2.contourArea(contours[i]), default=None)
            if outer_idx is None:
                continue
            epsilon = max(1.0, self._params['poly_approx_epsilon_m'] / resolution)
            outer_pts = cv2.approxPolyDP(contours[outer_idx], epsilon, True).reshape(-1, 2).astype(np.float64)

            mask = self._fill_room_holes(mask, resolution, protected=structural_occ)
            if np.count_nonzero(mask) < min_pixels:
                continue
            contours2, hierarchy2 = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if contours2 and hierarchy2 is not None:
                hierarchy2 = hierarchy2[0]
                outer_idx2 = max(
                    (i for i in range(len(contours2)) if hierarchy2[i][3] == -1),
                    key=lambda i: cv2.contourArea(contours2[i]), default=None)
                if outer_idx2 is not None:
                    outer_pts = cv2.approxPolyDP(contours2[outer_idx2], epsilon, True).reshape(-1, 2).astype(np.float64)

            polygon = [self._grid_to_world(int(px), int(py), grid) for px, py in outer_pts]
            if len(polygon) >= 3:
                polygon_area = self._polygon_area(polygon)
                if (self._params['min_room_area_m2'] <= polygon_area <=
                        self._params['room_max_area_m2']):
                    candidates.append((polygon, polygon_area, self._centroid(polygon), False))
        merged_candidates = self._merge_nested_candidates(
            candidates, cloud_support=cloud_support, grid=grid)
        merged_candidates = self._deduplicate_overlapping_candidates(merged_candidates)
        overlap_stats = getattr(self, '_last_overlap_cleanup_stats', {}) or {}
        self.last_segmentation_stats = {
            'method': _method,
            'resolution_m': resolution,
            'skeleton_pixels': _skel_px,
            'graph_branches': int(_cstats.get('branches', 0)),
            'door_candidates': len(critical_points),
            'doorway_vlm': dict(getattr(self, '_last_doorway_vlm_stats', {})),
            'door_cuts': len(validated_cuts),
            'locked_door_cuts_applied': int(getattr(self, '_last_locked_door_cuts', 0)),
            'wall_door_cuts': dict(wall_door_cuts),
            'cut_validation': dict(_cut_stats),
            'free_component': dict(_free_stats),
            'regions_after_small_merge': len(region_labels),
            'polygon_candidates_before_nested_merge': len(candidates),
            'polygon_candidates_final': len(merged_candidates),
            'candidate_areas_m2': [round(float(item[1]), 3) for item in merged_candidates],
            'overlap_cleanup': dict(overlap_stats),
            'critical_points': dict(_cstats),
            'topology_fill': dict(_fill_stats),
            'cloud_3d': dict(_wall_stats),
            'detected_walls': dict(_detected_stats),
        }
        return merged_candidates

    def process_grid(self, grid, full_resegment=True, force_resegment=False):
        if grid.header.frame_id and grid.header.frame_id != world_frame():
            self._log('warn', f'Ignoring occupancy grid in {grid.header.frame_id!r}; '
                      f'expected {world_frame()!r}')
            return
        with self._lock:
            self.last_grid = grid
            self.last_robot_xy = self._robot_pose()
            grid_data = np.asarray(grid.data, dtype=np.int16)
            grid_signature = (int(grid.info.width), int(grid.info.height),
                              float(grid.info.resolution),
                              float(grid.info.origin.position.x),
                              float(grid.info.origin.position.y),
                              hash(grid_data.tobytes()))
            if not force_resegment and self._last_grid_signature is not None:
                if grid_signature == self._last_grid_signature:
                    return
                if (self._last_grid_data is not None and
                        self._last_grid_data.shape == grid_data.shape):
                    changed = np.flatnonzero(grid_data != self._last_grid_data)
                    if changed.size:
                        max_cells = int(self._params.get(
                            'room_resegment_local_change_max_cells', 0))
                        max_fraction = float(self._params.get(
                            'room_resegment_local_change_max_bbox_fraction', 0.0))
                        if max_cells > 0 and max_fraction > 0.0:
                            ys, xs = np.unravel_index(
                                changed, (int(grid.info.height), int(grid.info.width)))
                            bbox_area = ((int(xs.max()) - int(xs.min()) + 1) *
                                         (int(ys.max()) - int(ys.min()) + 1))
                            total_area = max(1, int(grid.info.width) * int(grid.info.height))
                            if (changed.size <= max_cells and
                                    bbox_area / total_area <= max_fraction):
                                self._last_grid_signature = grid_signature
                                self._last_grid_data = grid_data.copy()
                                return
            self._last_grid_signature = grid_signature
            self._last_grid_data = grid_data.copy()
            # Windows are detected directly from RGB and do not depend on the
            # presence of GVD doorway candidates.
            self._schedule_window_vlm()
            if full_resegment:
                candidates = self._segment_regions_gvd(grid)
                confirmed_door_cuts = max(
                    int((self.last_segmentation_stats.get('cut_validation', {}) or {}).get(
                        'accepted_locked', 0)),
                    int(self.last_segmentation_stats.get('locked_door_cuts_applied', 0)))
                self.last_segmentation_stats['confirmed_door_cuts'] = confirmed_door_cuts
                strong_split = confirmed_door_cuts > 0
                accept, stability = self._stabilize_partition(
                    candidates, strong_split=strong_split)
                self.last_segmentation_stats['stability'] = stability
                if accept:
                    self._update_regions(candidates)
                    self._update_doorway_room_graph()
                else:
                    self._log(
                        'info',
                        'Room partition held: '
                        f"candidate_regions={stability['candidate_regions']} "
                        f"active_regions={stability['active_regions']} "
                        f"confirmations={stability['confirmations']}/"
                        f"{stability['required_confirmations']}")
                self._assign_detected_walls_to_rooms()
                self.current_room_id = self._room_at(self.last_robot_xy)
                self._publish_geometry(grid)
                self._save_rooms()

    @staticmethod
    def _partition_polygons(partition):
        return [item.polygon if isinstance(item, Region) else item[0] for item in partition]

    def _partitions_compatible(self, left, right, threshold=None):
        """One-to-one polygon compatibility for temporal partition hysteresis."""
        left_polys = self._partition_polygons(left)
        right_polys = self._partition_polygons(right)
        if len(left_polys) != len(right_polys):
            return False
        if not left_polys:
            return True
        if threshold is None:
            threshold = float(self._params.get('region_match_iou_min', 0.20))
        else:
            threshold = float(threshold)
        used = set()
        for polygon in sorted(left_polys, key=self._polygon_area, reverse=True):
            choices = [(self._polygon_iou(polygon, other), i)
                       for i, other in enumerate(right_polys) if i not in used]
            if not choices:
                return False
            score, index = max(choices)
            if score < threshold:
                return False
            used.add(index)
        return True

    def _confirmed_doorway_blocks_merge(self, candidates, active):
        """Prevent robust door/wall evidence from erasing an active room split."""
        if len(candidates) >= len(active):
            return False
        # A confirmed doorway is a persistent topological lock. Do not attempt to
        # infer whether a noisy polygon still contains its two sides: that test can
        # fail exactly when the map has already started collapsing the rooms. Any
        # automatic merge is therefore unsafe while a confirmed passage exists.
        if any(item.get('status') == 'confirmed' and item.get('cuttable')
               for item in getattr(self, '_doorway_vlm_state', {}).values()):
            return True
        candidate_polys = self._partition_polygons(candidates)
        active_polys = self._partition_polygons(active)
        for item in getattr(self, '_doorway_vlm_state', {}).values():
            if (item.get('status') != 'confirmed' or not item.get('cuttable') or
                    item.get('world') is None):
                continue
            world = np.asarray(item['world'], dtype=float)
            theta = float(item.get('theta', 0.0))
            # The GVD cut is orthogonal to its tangent. Sampling along the tangent
            # therefore probes the two rooms on either side of the doorway.
            tangent = np.array([math.cos(theta), math.sin(theta)])
            sample_distance = max(0.25, float(self._params.get(
                'detected_wall_door_endpoint_radius_m', 0.20)) * 1.5)
            samples = (world - tangent * sample_distance,
                       world + tangent * sample_distance)
            owners = []
            for sample in samples:
                matches = [index for index, polygon in enumerate(active_polys)
                           if self._point_in_polygon(polygon, sample, 0.20)]
                owners.append(matches[0] if matches else None)
            if (owners[0] is None or owners[1] is None or owners[0] == owners[1]):
                continue
            # If both sides belonged to different active rooms but now land in
            # the same candidate polygon, this is precisely the forbidden merge.
            if any(self._point_in_polygon(polygon, samples[0], 0.20) and
                   self._point_in_polygon(polygon, samples[1], 0.20)
                   for polygon in candidate_polys):
                return True

        # A doorway need not have been confirmed yet to preserve a split already
        # supported by a persistent structural wall. This is intentionally a
        # merge-only guard: robust walls protect existing room identities but do
        # not invent new rooms when the current partition has only one region.
        min_obs = max(1, int(self._params.get('detected_wall_min_observations', 2)))
        min_length = float(self._params.get('detected_wall_min_length_m', 0.50))
        min_confidence = float(self._params.get('detected_wall_min_confidence', 0.0))
        max_rms = float(self._params.get('detected_wall_max_rms_m', 0.05))
        sample_distance = max(0.25, float(self._params.get(
            'detected_wall_thickness_m', 0.06)) * 3.0)
        for wall in getattr(self, '_detected_wall_map', []):
            if (int(wall.get('observations', 1)) < min_obs or
                    float(wall.get('confidence', 1.0)) < min_confidence or
                    float(wall.get('inlier_rms_m', float('inf'))) > max_rms):
                continue
            try:
                p0, p1, direction, length = self._wall_geometry(wall)
            except (KeyError, TypeError, ValueError):
                continue
            if length < min_length:
                continue
            midpoint = (p0 + p1) * 0.5
            normal = np.array([-direction[1], direction[0]])
            samples = (midpoint - normal * sample_distance,
                       midpoint + normal * sample_distance)
            owners = []
            for sample in samples:
                matches = [index for index, polygon in enumerate(active_polys)
                           if self._point_in_polygon(polygon, sample, 0.20)]
                owners.append(matches[0] if matches else None)
            if (owners[0] is None or owners[1] is None or owners[0] == owners[1]):
                continue
            if any(self._point_in_polygon(polygon, samples[0], 0.20) and
                   self._point_in_polygon(polygon, samples[1], 0.20)
                   for polygon in candidate_polys):
                return True
        return False

    def _stabilize_partition(self, candidates, strong_split=False):
        """Require repeated evidence before replacing the active room topology.

        Split and merge errors are not equivalent.  A false split is reversible;
        a false merge collapses two room identities and reassigns their objects.
        Merges therefore need a longer, separately configurable confirmation run.
        An empty extraction is treated as sensor/SLAM failure once a valid
        partition exists and can never erase all rooms.
        """
        active = [region for region in self.regions.values() if region.misses == 0]
        default_required = max(1, int(self._params.get(
            'room_partition_change_confirmations', 1)))
        change_kind = ('merge' if len(candidates) < len(active) else
                       'split' if len(candidates) > len(active) else
                       'reshape')
        # A confirmed doorway is stronger evidence than the raw count of contour
        # candidates. Polygon cleanup can remove a small/duplicate region in the
        # same update that applies the doorway cut, making a real split look like
        # a merge (e.g. 5 active regions -> 4 candidates). Do not let the merge
        # hysteresis or merge blocker discard that confirmed topological change.
        if strong_split:
            required = 1
        elif change_kind == 'merge':
            required = max(1, int(self._params.get(
                'room_merge_change_confirmations', default_required)))
        elif change_kind == 'split':
            required = (1 if strong_split else max(1, int(self._params.get(
                'room_split_change_confirmations', default_required))))
        else:
            required = default_required
        stable_iou = float(self._params.get(
            'room_partition_stable_iou_min',
            self._params.get('region_match_iou_min', 0.20)))
        base = {
            'candidate_regions': len(candidates),
            'active_regions': len(active),
            'required_confirmations': required,
            'change_kind': change_kind,
        }
        if active and candidates and len(candidates) < len(active):
            active_area = sum(float(region.area_m2) for region in active)
            candidate_area = sum(float(item[1]) for item in candidates)
            retention = candidate_area / max(active_area, 1e-6)
            minimum_retention = float(self._params.get(
                'room_partition_min_area_retention_ratio', 0.75))
            if retention < minimum_retention:
                self._pending_partition = None
                self._pending_partition_count = 0
                return False, {**base, 'accepted': False, 'confirmations': 0,
                               'reason': 'partial_partition_held',
                               'area_retention': round(retention, 3),
                               'minimum_area_retention': minimum_retention}
        if not strong_split and self._confirmed_doorway_blocks_merge(candidates, active):
            self._pending_partition = None
            self._pending_partition_count = 0
            return False, {**base, 'accepted': False, 'confirmations': 0,
                           'reason': 'confirmed_doorway_merge_blocked'}
        if (active and not candidates and
                bool(self._params.get('room_hold_empty_partition', True))):
            self._pending_partition = None
            self._pending_partition_count = 0
            return False, {**base, 'accepted': False, 'confirmations': 0,
                           'reason': 'empty_partition_held'}
        # Bootstrap is immediate.  A genuinely small contour refinement is also
        # immediate; a larger geometry change, split, or merge is debounced.
        if not active or self._partitions_compatible(candidates, active, stable_iou):
            self._pending_partition = None
            self._pending_partition_count = 0
            return True, {**base, 'accepted': True, 'confirmations': 0,
                          'reason': 'bootstrap' if not active else 'compatible_update'}

        if (self._pending_partition is not None and
                self._partitions_compatible(candidates, self._pending_partition,
                                             stable_iou)):
            self._pending_partition_count += 1
        else:
            self._pending_partition_count = 1
        self._pending_partition = candidates
        if self._pending_partition_count >= required:
            confirmations = self._pending_partition_count
            self._pending_partition = None
            self._pending_partition_count = 0
            return True, {**base, 'accepted': True, 'confirmations': confirmations,
                          'reason': 'confirmed_topology_change'}
        return False, {**base, 'accepted': False,
                       'confirmations': self._pending_partition_count,
                       'reason': 'pending_topology_change'}

    def _slow_map_callback(self, msg):
        self.process_grid(msg, True)

    # ------------------------------------------------------------------
    # Region management and compatibility API
    # ------------------------------------------------------------------

    def init_room_node(self, room_id):
        if room_id is None:
            return None
        room_id = str(room_id)
        if room_id not in self.scene_graph:
            self.scene_graph[room_id] = {
                'room_id': room_id, 'region_id': None,
                'semantic_label': 'UnknownRoom', 'description': '',
                'objects': [], 'polygon': [], 'area_m2': 0.0,
                'centroid': [], 'walls': [], 'wall_segments': [], 'detected_walls': [],
                'confirmed': False, 'boundaries': {}, 'last_seen': time.time(),
                'currently_detected': True,
                # GA-185: whether the region backing this room is currently detected. A
                # retired room stays in the registry and stays referenceable; it is simply
                # not the robot's current room any more.
                'active': True, 'retired_at': None,
            }
        return self.scene_graph[room_id]

    def _new_room_id(self):
        room_id = f'room_{self.room_counter}'
        self.room_counter += 1
        return room_id

    def _update_regions(self, candidates):
        now = time.time()
        updated = {}
        observed_room_ids = set()
        old = list(self.regions.values())
        used = set()
        for polygon, area, centroid, _ in candidates:
            best = None
            best_score = -1.0
            best_index = None
            for i, previous in enumerate(old):
                if i in used:
                    continue
                distance = float(np.linalg.norm(np.asarray(centroid)-np.asarray(previous.centroid)))
                if distance > self._params['region_match_distance_m']:
                    continue
                score = self._polygon_iou(polygon, previous.polygon)
                if score >= self._params['region_match_iou_min'] and score > best_score:
                    best, best_score, best_index = previous, score, i
            if best is None:
                region_id = f'region_{self.region_counter}'
                self.region_counter += 1
                best = Region(region_id, polygon, area, centroid)
                best.room_id = self._new_room_id()
                self.init_room_node(best.room_id)
            else:
                used.add(best_index)
                best.polygon, best.area_m2, best.centroid = polygon, area, centroid
                best.misses = 0
            best.last_seen = now
            updated[best.region_id] = best
            observed_room_ids.add(best.room_id)
            room = self.init_room_node(best.room_id)
            room.update({
                'region_id': best.region_id, 'polygon': list(best.polygon),
                'area_m2': float(best.area_m2), 'centroid': list(best.centroid),
                'walls': self._polygon_edges(best.polygon),
                'wall_segments': self._polygon_edges(best.polygon),
                'confirmed': best.area_m2 >= self._params['min_room_area_m2'],
                'last_seen': now,
            })
        for i, previous in enumerate(old):
            if i not in used and previous.region_id not in updated:
                previous.misses += 1
                if previous.misses <= self._params['max_region_misses']:
                    updated[previous.region_id] = previous
        self.regions = updated

        observed_rooms = [self.scene_graph[rid] for rid in observed_room_ids]
        stale_rooms_retired = 0
        for room_id in list(self.scene_graph.keys()):
            if room_id in observed_room_ids:
                self.scene_graph[room_id]['active'] = True
                self.scene_graph[room_id]['currently_detected'] = True
                self.scene_graph[room_id]['retired_at'] = None
                continue
            room = self.scene_graph[room_id]
            room['currently_detected'] = False
            # The occupancy grid is global: an accepted re-segmentation describes
            # the current partition of the mapped floor, not only the robot's local
            # view. Keeping old unmatched regions active caused stale bed-sized
            # rooms and overlapping room IDs to survive indefinitely in room.json.
            # Preserve them in the registry for history, but never expose them as
            # current geometry or use them as assignment fallbacks.
            was_active = bool(room.get('active', True))
            room['active'] = False
            room['retired_at'] = time.time() if was_active else room.get('retired_at')
            stale_rooms_retired += int(was_active)
            if was_active and self.current_room_id == room_id:
                self.current_room_id = None

        self.last_segmentation_stats['stale_rooms_retired'] = stale_rooms_retired

        return list(updated.values())

    def _update_doorway_room_graph(self):
        """Associate confirmed doorways with the two room regions they connect."""
        graph = dict(getattr(self, '_doorway_room_edges', {}))
        active = [region for region in self.regions.values() if region.misses == 0]
        sample_distance = max(0.25, float(self._params.get(
            'detected_wall_door_endpoint_radius_m', 0.20)) * 1.5)
        for key, item in getattr(self, '_doorway_vlm_state', {}).items():
            if (item.get('status') != 'confirmed' or not item.get('cuttable') or
                    item.get('world') is None):
                continue
            centre = np.asarray(item['world'], dtype=float)
            theta = float(item.get('theta', 0.0))
            tangent = np.array([math.cos(theta), math.sin(theta)])
            samples = (centre - sample_distance * tangent,
                       centre + sample_distance * tangent)
            owners = []
            for sample in samples:
                matches = [region.room_id for region in active
                           if self._point_in_polygon(region.polygon, sample, 0.20)]
                owners.append(matches[0] if matches else None)
            previous_edge = graph.get(key, {})
            if owners[0] is None or owners[1] is None or owners[0] == owners[1]:
                # Loop closures or a large contour change can move the doorway
                # sample just outside a polygon. Re-identify the old endpoint
                # rooms geometrically before giving up the topological edge.
                snapshots = (previous_edge.get('room_a_polygon'),
                              previous_edge.get('room_b_polygon'))
                recovered = []
                for snapshot in snapshots:
                    if not snapshot:
                        recovered.append(None)
                        continue
                    best_region, best_score = None, 0.0
                    old_centroid = np.asarray(self._centroid(snapshot), dtype=float)
                    old_area = max(self._polygon_area(snapshot), 1e-6)
                    for region in active:
                        score = self._polygon_iou(snapshot, region.polygon)
                        distance = float(np.linalg.norm(
                            old_centroid - np.asarray(region.centroid, dtype=float)))
                        area_ratio = min(old_area, float(region.area_m2)) / max(
                            old_area, float(region.area_m2), 1e-6)
                        combined = score + 0.15 * area_ratio if distance <= float(
                            self._params.get('region_match_distance_m', 3.0)) else 0.0
                        if combined > best_score:
                            best_region, best_score = region, combined
                    recovered.append(best_region.room_id if best_region is not None else None)
                if owners[0] is None:
                    owners[0] = recovered[0]
                if owners[1] is None:
                    owners[1] = recovered[1]
            if owners[0] is None or owners[1] is None or owners[0] == owners[1]:
                # Never keep presenting a stale edge as a valid room connection.
                # The doorway remains confirmed and protected, but its endpoints
                # are unresolved until the current partition exposes two distinct
                # regions again.
                graph[key] = {
                    **previous_edge,
                    'doorway_key': key,
                    'room_a': None,
                    'room_b': None,
                    'resolved': False,
                    'world': self._json_safe(item.get('world')),
                    'permanent': bool(item.get('permanent_confirmed', True)),
                    'updated_at': time.time(),
                }
                continue
            if owners[0] is not None and owners[1] is not None and owners[0] != owners[1]:
                graph[key] = {
                    'doorway_key': key,
                    'room_a': owners[0], 'room_b': owners[1],
                    'resolved': True,
                    'room_a_polygon': self._json_safe(next(
                        (region.polygon for region in active if region.room_id == owners[0]),
                        previous_edge.get('room_a_polygon', []))),
                    'room_b_polygon': self._json_safe(next(
                        (region.polygon for region in active if region.room_id == owners[1]),
                        previous_edge.get('room_b_polygon', []))),
                    'world': self._json_safe(item.get('world')),
                    'permanent': bool(item.get('permanent_confirmed', True)),
                    'updated_at': time.time(),
                }
        self._doorway_room_edges = graph
        self.last_segmentation_stats['doorway_room_edges'] = self._json_safe(graph)
        return graph

    def _room_at(self, xy):
        if xy is None:
            return None
        tolerance = self._params['room_assignment_tolerance_m']
        matches = [
            r for r in self.regions.values()
            if r.misses == 0 and self._point_in_polygon(r.polygon, xy, tolerance)
        ]
        if matches:
            return min(matches, key=lambda r: r.area_m2).room_id
        historical = [room for room in self.scene_graph.values()
                      if room.get('active', True) and
                      self._point_in_polygon(room.get('polygon', []), xy, tolerance)]
        return (min(historical, key=lambda room: float(room.get('area_m2', float('inf'))))
                .get('room_id')) if historical else None

    def _nearest_room(self, xy):
        # FIX: the guard used to return bare None while every other path
        # returns a (room_id, distance) tuple — callers unpack the result, so
        # the first query before any room existed crashed the node.
        if xy is None or not self.regions:
            return None, float('inf')
        best_room = None
        best_distance = float('inf')
        for region in self.regions.values():
            poly = region.polygon
            if not poly:
                continue
            distance = abs(cv2.pointPolygonTest(
                np.asarray(poly, dtype=np.float32),
                (float(xy[0]), float(xy[1])),
                True,
            ))
            if distance < best_distance:
                best_distance = distance
                best_room = region.room_id
        return best_room, best_distance

    @staticmethod
    def _object_label_key(label):
        return str(label).strip().lower()

    @staticmethod
    def _vlm_room_label(label):
        return str(label).split('#', 1)[0].strip().lower().replace(" ", "_")

    def _effective_room_id(self):
        """The room objects are actually filed under. (GA-350, from GRAPH-API 3a5a818.)

        `current_room_id` is None whenever the robot is not inside a GVD region polygon
        (no map yet, or standing in a doorway). Objects meanwhile fall back to
        rooms.default_room_id (object_services), so returning None here would leave every
        admitted object in a room the room-frame seam never looked at.
        """
        if self.current_room_id is not None:
            return self.current_room_id
        from config import CFG
        return CFG["rooms"]["default_room_id"] or None

    def update_current_room_semantics(self, persistent_objects):
        if self.current_room_id is None:
            return
        room = self.init_room_node(self.current_room_id)
        labels = list(room.get('objects', []) or [])
        for obj in persistent_objects or []:
            if isinstance(obj, dict):
                room_id = obj.get('room_id', self.current_room_id)
                label = obj.get('label', 'unknown')
            else:
                room_id = getattr(obj, 'room_id', self.current_room_id)
                label = getattr(obj, 'label', 'unknown')
            if room_id == self.current_room_id:
                label = str(label).strip()
                if label and all(self._object_label_key(label) != self._object_label_key(existing)
                                 for existing in labels):
                    labels.append(label)
        room['objects'] = sorted(labels)
        room['last_seen'] = time.time()
        if labels:
            encoded_image = self._current_room_frame_base64(self.current_room_id)
            evidence = [self._vlm_room_label(label) for label in labels]
            self._schedule_room_semantics(
                self.current_room_id, evidence, encoded_image=encoded_image)
        self._save_rooms()

    def update_all_rooms_semantics(self, persistent_objects):
        by_room = {}
        for obj in persistent_objects or []:
            if isinstance(obj, dict):
                room_id = obj.get('room_id')
                label = obj.get('label', 'unknown')
            else:
                room_id = getattr(obj, 'room_id', None)
                label = getattr(obj, 'label', 'unknown')
            if room_id is None:
                continue
            label = str(label).strip()
            if not label:
                continue
            by_room.setdefault(room_id, set()).add(label)

        now = time.time()
        for room_id, labels in by_room.items():
            room = self.init_room_node(room_id)
            merged_labels = set(room.get('objects', []) or [])
            merged_labels.update(labels)
            room['objects'] = sorted(merged_labels)
            room['last_seen'] = now
            if merged_labels:
                evidence = sorted(self._vlm_room_label(label) for label in merged_labels)
                previous_evidence = room.get('_semantic_evidence')
                semantic = str(room.get('semantic_label', '')).strip().lower()
                already_labelled = semantic not in ('', 'unknownroom', 'unknown_room')
                last_attempt = float(room.get('_semantic_last_attempt', 0.0) or 0.0)
                if evidence == previous_evidence and (
                        already_labelled or now-last_attempt < 30.0):
                    continue
                room['_semantic_last_attempt'] = now
                self._schedule_room_semantics(room_id, evidence)
        self._save_rooms()

    def _schedule_room_semantics(self, room_id, evidence, encoded_image=None):
        """Run room VLM off the perception callback and reject stale results."""
        if room_id is None or not evidence:
            return
        key = str(room_id)
        evidence = tuple(sorted(str(item) for item in evidence))
        pending = self._room_vlm_pending.get(key)
        if pending is not None and pending[0] == evidence:
            return
        self._room_vlm_pending[key] = (evidence, time.time())
        future = self._room_vlm_executor.submit(
            self.ask_vlm_room_info, list(evidence), encoded_image)
        future.add_done_callback(
            lambda completed: self._on_room_semantics_done(
                key, evidence, completed))

    def _on_room_semantics_done(self, room_id, evidence, future):
        try:
            semantic_name, description = future.result()
            with self._lock:
                room = self.scene_graph.get(str(room_id))
                current_evidence = tuple(sorted(
                    self._vlm_room_label(label)
                    for label in (room or {}).get('objects', []) or []))
                if room is None or current_evidence != evidence:
                    return
                if semantic_name:
                    room['semantic_label'] = semantic_name
                if description:
                    room['description'] = description
                if str(semantic_name).strip().lower() not in (
                        '', 'unknownroom', 'unknown_room'):
                    room['_semantic_evidence'] = list(evidence)
                self._save_rooms()
        except Exception as exc:
            self._log('warn', f'Room VLM failed for {room_id}: {exc}')
        finally:
            pending = self._room_vlm_pending.get(str(room_id))
            if pending is not None and pending[0] == evidence:
                self._room_vlm_pending.pop(str(room_id), None)

    def _current_room_frame_base64(self, room_id):
        """Return only the newest image captured while this exact room was current."""
        if room_id is None or self._room_frame_provider is None:
            return None
        try:
            frames = self._room_frame_provider(str(room_id)) or []
            if not frames:
                return None
            path = frames[-1].get('path') if isinstance(frames[-1], dict) else None
            if not path or not os.path.isfile(path):
                return None
            with open(path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('ascii')
        except Exception as exc:
            self._log('warn', f'Room frame unavailable for {room_id}: {exc}')
            return None

    def update_current_room_geometry(self, room_id, bbox):
        room_id = room_id or self.current_room_id
        if room_id is None or not bbox:
            return
        room = self.init_room_node(room_id)
        room.setdefault('boundaries', {}).update(bbox)
        room['last_seen'] = time.time()

    update_room_geometry = update_current_room_geometry


    def ask_vlm_room_info(self, room_objects_labels, encoded_image=None):
        """
        Asks the VLM for the room's name and description.
        Supporta sia testo che immagini codificate in Base64.
        """
        if not room_objects_labels:
            return "Unknown_Room", "Stanza senza oggetti rilevanti."

        labels_str = ", ".join(set(room_objects_labels))

        try:
            from config import CFG
            model_name = os.environ.get("ROOM_VLM_MODEL", CFG["vlm"]["model"])
            from cv_utils import vlm_call

            text_prompt = (
                f"Analyze these relevant objects: {labels_str}.\n"
                "Identify the room type and describe it.\n"
                "ONLY answer in JSON format: "
                "{\"label\": \"name\", \"description\": \"description\"}. "
                "In the \"label\" field you have to specify one specific room type, "
                "it can't just be something generic like \"room type\""
            )

            raw_content = (vlm_call(
                text_prompt,
                encoded_image=encoded_image,
                timeout=CFG["vlm"]["timeout"],
                model=model_name,
                request_kind="room_semantics",
                image_mime_type="image/jpeg",
            ) or "").strip()
            clean_json = re.sub(r'^```json\s*|```$', '', raw_content, flags=re.MULTILINE).strip()
            data = json.loads(clean_json)
            semantic = str(data.get("label", "Unknown_Room")).strip() or "Unknown_Room"
            desc = str(data.get("description", "")).strip()
            return semantic, desc
        except Exception as e:
            print(f"[VLM ERROR] Fallimento chiamata: {e}")
            return "Unknown_Room", f"Errore: {e}"

    def save_room_to_json(self, room_id, label, description, objects, walls):
        room = self.init_room_node(room_id)
        room["semantic_label"] = label or "Unknown_Room"
        room["description"] = description or ""
        if objects is not None:
            room["objects"] = list(objects)
        if walls is not None:
            room["walls"] = walls
        room["last_seen"] = time.time()
        self._save_rooms(force=True)

    def room_at_bbox(self, bbox):
        """Which room this box is geometrically in, or None when geometry cannot say.

        GA-28 / GA-25: `assign_room_by_geometry` answers `self.current_room_id` — the
        ROBOT's room — when there is no bbox, no room polygon within tolerance, or no
        room at all, and it returns it in the same type as a geometric answer. A caller
        cannot tell "this object is in room 3" from "I do not know, and the robot is in
        room 3". A same-room gate built on that reads two objects in different rooms as
        the same room precisely when the map is least able to separate them.

        This is the honest query. `assign_room_by_geometry` keeps its behaviour and is
        now the one place that applies the fallback, so the two cannot drift.
        """
        if not bbox:
            return None
        candidates = [
            ((bbox['x_min'] + bbox['x_max']) / 2.0, (bbox['y_min'] + bbox['y_max']) / 2.0),
            (bbox['x_min'], bbox['y_min']),
            (bbox['x_min'], bbox['y_max']),
            (bbox['x_max'], bbox['y_min']),
            (bbox['x_max'], bbox['y_max']),
        ]

        for point in candidates:
            room_id = self._room_at(point)
            if room_id is not None:
                return room_id

        nearest = None
        nearest_dist = float('inf')
        for point in candidates:
            room_id, dist = self._nearest_room(point)
            if room_id is not None and dist < nearest_dist:
                nearest = room_id
                nearest_dist = dist

        if nearest is not None and nearest_dist <= float(self._params['room_nearest_fallback_m']):
            return nearest
        return None

    def reassign_objects_by_geometry(self, objects):
        """Re-file objects whose geometric room no longer matches their stored `room_id`.

        GA-28c: this had ONE call site (`object_manager_6.reassign_objects_to_rooms`, run
        after a map resegmentation) and NO definition anywhere in the tree, so the whole
        path raised AttributeError the first time a resegmentation happened.

        Returns ``[(obj, old_room, new_room), ...]`` for the objects that moved, which is
        the shape the caller already unpacks.

        Geometry only. `room_at_bbox` returns None when it cannot say, and an object is
        LEFT WHERE IT IS in that case -- a resegmentation that cannot place an object is
        not evidence the object went anywhere. Using `assign_room_by_geometry` here would
        re-file every unplaceable object into whichever room the robot happens to occupy,
        which is exactly the fallback-as-fact defect GA-28b names.
        """
        changed = []
        for obj in objects or []:
            bbox = getattr(obj, "bbox", None)
            if not bbox:
                continue
            new_room = self.room_at_bbox(bbox)
            if new_room is None:
                continue
            old_room = getattr(obj, "room_id", None)
            if old_room == new_room:
                continue
            obj.room_id = new_room
            if old_room and old_room in self.scene_graph:
                objs = self.scene_graph[old_room].get("objects", [])
                if obj.label in objs:
                    objs.remove(obj.label)
            node = self.init_room_node(new_room)
            if obj.label not in node.get("objects", []):
                node.setdefault("objects", []).append(obj.label)
            changed.append((obj, old_room, new_room))
        return changed

    def assign_room_by_geometry(self, bbox):
        """The room to file this box under, falling back to the robot's room.

        The fallback is deliberate for an assignment — an object must go somewhere —
        and it is wrong for a comparison. Call `room_at_bbox` when the answer matters,
        and treat None as "unknown", never as "same room".
        """
        room_id = self.room_at_bbox(bbox)
        return room_id if room_id is not None else self.current_room_id

    def evaluate_scene(self, descriptions=None, persistent_objects=None):
        self.last_robot_xy = self._robot_pose()
        self.current_room_id = self._room_at(self.last_robot_xy)
        self.update_current_room_semantics(persistent_objects or [])
        return self.current_room_id

    def finalize_current_room(self, persistent_objects):
        room_id = self.current_room_id
        print(f"[FINALISE] finalize called for: {room_id}")
        room_node = self.init_room_node(room_id)

        # 1. Collect the room's objects, merging those already registered
        #    nella room ai label osservati in questa chiamata.
        room_labels = []
        for label in room_node.get("objects", []) or []:
            label = str(label).strip()
            if label and all(self._object_label_key(label) != self._object_label_key(existing)
                             for existing in room_labels):
                room_labels.append(label)

        for o in persistent_objects or []:
            if isinstance(o, dict):
                obj_room_id = o.get('room_id', room_id)
                obj_label = o.get('label', '')
            else:
                obj_room_id = getattr(o, 'room_id', room_id)
                obj_label = getattr(o, 'label', '')
            if obj_room_id == room_id:
                cleaned = str(obj_label or '').strip()
                if cleaned and all(self._object_label_key(cleaned) != self._object_label_key(existing)
                                   for existing in room_labels):
                    room_labels.append(cleaned)

        room_labels_vlm = [self._vlm_room_label(label) for label in room_labels if str(label).strip()]
        
        semantic_name = "Unknown_Room"
        description = "No description (the VLM was unavailable, or the room is empty)."



        if room_labels_vlm:
            vlm_name, vlm_desc = self.ask_vlm_room_info(
                room_labels_vlm,
                encoded_image=self._current_room_frame_base64(room_id),
            )
            if vlm_name and vlm_name != "Unknown_Room":
                semantic_name = vlm_name
            if vlm_desc:
                description = vlm_desc
        
        room_node["semantic_label"] = semantic_name
        room_node["description"] = description
        if room_labels:
            room_node["objects"] = sorted(room_labels)
        objs_to_save = room_node.get("objects", room_labels)



        self.save_room_to_json(
            room_id=room_id, 
            label=semantic_name, 
            description=description, 
            objects=objs_to_save, 
            # `walls` remains the occupancy-derived room boundary. Depth measurements live
            # under `detected_walls` and must not overwrite a different kind of geometry.
            walls=room_node.get("walls", [])
        )


    # ------------------------------------------------------------------
    # Visualization and persistence
    # ------------------------------------------------------------------

    @property
    def current_room_walls(self):
        """Walls of the CURRENT room only. Kept as a property so both existing call sites --
        the append in object_manager_6.walls_callback and the read in
        finalize_current_room -- keep working unchanged while the storage becomes per-room.
        """
        return self._walls_by_room.setdefault(self.current_room_id, [])

    def walls_of(self, room_id):
        return list(self._walls_by_room.get(room_id, []))

    @staticmethod
    def _wall_geometry(wall):
        p0 = np.array([wall["start"]["x"], wall["start"]["y"]], dtype=float)
        p1 = np.array([wall["end"]["x"], wall["end"]["y"]], dtype=float)
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            raise ValueError("wall endpoints coincide")
        return p0, p1, direction / length, length

    def _merge_detected_wall(self, stored, observed):
        """Fuse a repeated observation, or return False when it is a different wall."""
        if "floor_z" in stored and "floor_z" in observed:
            from config import CFG
            floor_tolerance = float(
                CFG.get("walls", {}).get("floor_reset_m", 0.60))
            if abs(float(stored["floor_z"]) - float(observed["floor_z"])) > floor_tolerance:
                return False
        a0, a1, adir, alen = self._wall_geometry(stored)
        b0, b1, bdir, blen = self._wall_geometry(observed)
        angle_deg = float(self._params.get('detected_wall_merge_angle_deg', 8.0))
        if abs(float(adir @ bdir)) < math.cos(math.radians(angle_deg)):
            return False
        amid, bmid = (a0 + a1) * 0.5, (b0 + b1) * 0.5
        normal = np.array([-adir[1], adir[0]])
        if abs(float((bmid - amid) @ normal)) > float(
                self._params.get('detected_wall_merge_distance_m', 0.12)):
            return False

        # Use the actual interval gap, not the midpoint distance.  Midpoints can
        # be close even when two collinear pieces are separated by a doorway;
        # fusing those pieces would rasterise a wall across the opening and merge
        # the two rooms.  Only overlap or a small acquisition gap is interpolated.
        b_t0 = float((b0 - a0) @ adir)
        b_t1 = float((b1 - a0) @ adir)
        b_min, b_max = sorted((b_t0, b_t1))
        interval_gap = max(0.0, max(0.0 - b_max, b_min - alen))
        interpolation_gap = float(self._params.get(
            'detected_wall_interpolation_gap_m', 0.20))
        if interval_gap > max(0.0, interpolation_gap):
            return False

        # Interpolate the line as evidence accumulates. Keeping the first direction
        # forever preserves the angle of the noisiest first view; using the latest
        # direction makes the marker jump. A weighted fit gives a stable canonical
        # line and removes small oblique duplicate strokes.
        support = max(1, int(stored.get("observations", 1)))
        if float(adir @ bdir) < 0:
            bdir = -bdir
        direction = adir * float(support) + bdir
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        normal = np.array([-direction[1], direction[0]])
        # Average the signed offset and midpoint in the new common frame.
        offset = (float(normal @ amid) * support + float(normal @ bmid)) / (support + 1)
        tangent_mid = (float(direction @ amid) * support + float(direction @ bmid)) / (support + 1)
        centre = normal * offset + direction * tangent_mid
        projections = np.array([(p - centre) @ direction for p in (a0, a1, b0, b1)])
        p0, p1 = centre + direction * projections.min(), centre + direction * projections.max()
        stored["start"] = {"x": float(p0[0]), "y": float(p0[1])}
        stored["end"] = {"x": float(p1[0]), "y": float(p1[1])}
        stored["z_min"] = min(float(stored.get("z_min", 0.0)),
                              float(observed.get("z_min", 0.0)))
        stored["z_max"] = max(float(stored.get("z_max", 0.0)),
                              float(observed.get("z_max", 0.0)))
        stored["observations"] = support + 1
        stored["last_seen"] = time.time()
        # A repeated observation can be noisier than the original fit (partial
        # occlusion, grazing angle, or a temporarily sparse depth image).  The
        # persistent marker is a fused wall, so one bad frame must not make an
        # already confirmed wall fail the RMS gate and disappear from RViz.
        stored["n_points"] = max(
            int(stored.get("n_points", 0)), int(observed.get("n_points", 0)))
        old_rms = float(stored.get("inlier_rms_m", float("inf")))
        new_rms = float(observed.get("inlier_rms_m", float("inf")))
        stored["inlier_rms_m"] = min(old_rms, new_rms)
        old_confidence = float(stored.get("confidence", 1.0))
        new_confidence = float(observed.get("confidence", 1.0))
        stored["confidence"] = round(
            (old_confidence * support + new_confidence) / (support + 1), 4)
        if new_confidence >= old_confidence and "evidence" in observed:
            stored["evidence"] = self._json_safe(observed["evidence"])
        stored["temporal_frames"] = max(
            int(stored.get("temporal_frames", 1)),
            int(observed.get("temporal_frames", 1)))
        return True

    def _consolidate_detected_walls(self):
        """Collapse transitive duplicate strokes into one fitted wall segment."""
        walls = list(self._detected_wall_map)
        merges = 0
        changed = True
        while changed and len(walls) > 1:
            changed = False
            for i in range(len(walls)):
                for j in range(i + 1, len(walls)):
                    if self._merge_detected_wall(walls[i], walls[j]):
                        walls.pop(j)
                        merges += 1
                        changed = True
                        break
                if changed:
                    break
        self._detected_wall_map = walls
        self._last_detected_wall_fusion_stats = {
            'segments_after_fusion': len(walls),
            'duplicate_merges': merges,
        }

    def _assign_detected_walls_to_rooms(self):
        """Derive room membership from the current partition without owning persistence."""
        for room in self.scene_graph.values():
            room['detected_walls'] = []
        assigned = 0
        for wall in self._detected_wall_map:
            try:
                p0, p1, _, _ = self._wall_geometry(wall)
            except (KeyError, TypeError, ValueError):
                continue
            midpoint = (p0 + p1) * 0.5
            matching = [room for room in self.scene_graph.values()
                        if room.get("active", True) and
                        self._point_in_polygon(room.get("polygon", []), midpoint, 0.15)]
            for room in matching:
                room['detected_walls'].append(self._json_safe(wall))
                assigned += 1
        return assigned

    def ingest_detected_walls(self, walls):
        """Fuse observations globally, then project them onto the current room partition."""
        with self._lock:
            changed = False
            max_age_s = float(self._params.get('detected_wall_max_age_s', 0.0))
            if max_age_s > 0.0:
                now = time.time()
                before = len(self._detected_wall_map)
                self._detected_wall_map = [
                    wall for wall in self._detected_wall_map
                    if now - float(wall.get('last_seen', now)) <= max_age_s
                ]
                changed = len(self._detected_wall_map) != before
            for observed in walls:
                self._wall_geometry(observed)
                if not any(self._merge_detected_wall(old, observed)
                           for old in self._detected_wall_map):
                    wall = self._json_safe(dict(observed))
                    wall["observations"] = 1
                    wall["last_seen"] = time.time()
                    self._detected_wall_map.append(wall)
                changed = True
            # A new observation may bridge two old partial strokes. Run a second,
            # transitive pass so A+B and B+C become one continuous interpolated wall.
            self._consolidate_detected_walls()
            max_wall_map = int(self._params.get('detected_wall_map_max_segments', 0))
            if max_wall_map > 0 and len(self._detected_wall_map) > max_wall_map:
                def wall_priority(wall):
                    try:
                        _, _, _, length = self._wall_geometry(wall)
                    except (KeyError, TypeError, ValueError):
                        length = 0.0
                    return (int(wall.get('observations', 1)),
                            float(wall.get('confidence', 0.0)),
                            float(length),
                            float(wall.get('last_seen', 0.0)))
                self._detected_wall_map = sorted(
                    self._detected_wall_map, key=wall_priority, reverse=True)[:max_wall_map]
            assigned = self._assign_detected_walls_to_rooms()
            if changed:
                self._save_rooms()
                self._publish_detected_wall_markers()
        return assigned

    def _publish_detected_wall_markers(self):
        """Render only fused walls that pass the structural-support qualification."""
        if self._persistent_wall_marker_pub is None or self.node is None:
            return
        output = MarkerArray()
        clear = Marker()
        clear.header.frame_id = world_frame()
        clear.header.stamp = self.node.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        output.markers.append(clear)

        marker_id = 0
        min_obs = max(1, int(self._params.get('detected_wall_min_observations', 1)))
        min_length = float(self._params.get('detected_wall_min_length_m', 0.50))
        min_vertical = float(self._params.get('detected_wall_min_vertical_extent_m', 1.20))
        max_rms = float(self._params.get('detected_wall_max_rms_m', 0.03))
        min_confidence = float(self._params.get('detected_wall_min_confidence', 0.0))
        max_age_s = float(self._params.get('detected_wall_max_age_s', 0.0))
        now = time.time()

        def wall_parts_without_doors(p0, p1):
            direction = (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-9)
            normal = np.array([-direction[1], direction[0]])
            total = float(np.linalg.norm(p1 - p0))
            intervals = [(0.0, total)]
            for doorway in getattr(self, '_doorway_vlm_state', {}).values():
                if (doorway.get('status') != 'confirmed' or
                        not doorway.get('cuttable') or doorway.get('world') is None):
                    continue
                centre = np.asarray(doorway['world'], dtype=float)
                if abs(float((centre - p0) @ normal)) > float(self._params.get(
                        'detected_wall_endpoint_radius_m', 0.30)):
                    continue
                along = float((centre - p0) @ direction)
                left = doorway.get('left_world')
                right = doorway.get('right_world')
                opening = (float(np.linalg.norm(np.asarray(right) - np.asarray(left)))
                           if left is not None and right is not None else 0.0)
                half = max(0.25, 0.5 * opening + 0.06)
                cut_a, cut_b = max(0.0, along - half), min(total, along + half)
                if cut_b <= cut_a:
                    continue
                updated = []
                for a, b in intervals:
                    if cut_b <= a or cut_a >= b:
                        updated.append((a, b))
                    else:
                        if a < cut_a:
                            updated.append((a, cut_a))
                        if cut_b < b:
                            updated.append((cut_b, b))
                intervals = updated
            return [(p0 + direction * a, p0 + direction * b)
                    for a, b in intervals if b - a >= min_length]

        for wall in self._detected_wall_map:
            observations = max(1, int(wall.get("observations", 1)))
            if observations < min_obs:
                continue
            if (max_age_s > 0.0 and
                    now - float(wall.get('last_seen', now)) > max_age_s):
                continue
            try:
                p0, p1, _, length = self._wall_geometry(wall)
            except (KeyError, TypeError, ValueError):
                continue
            z0 = float(wall.get("z_min", 0.4))
            z1 = float(wall.get("z_max", 2.0))
            if (length < min_length or z1-z0 < min_vertical or
                    float(wall.get("inlier_rms_m", float("inf"))) > max_rms or
                    float(wall.get("confidence", 1.0)) < min_confidence):
                continue
            strength = min(1.0, observations / 6.0)
            for part0, part1 in wall_parts_without_doors(p0, p1):
                part_length = float(np.linalg.norm(part1 - part0))
                yaw = math.atan2(part1[1] - part0[1], part1[0] - part0[0])
                marker = Marker()
                marker.header = clear.header
                marker.ns = "persistent_detected_walls"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose.position.x = float((part0[0] + part1[0]) * 0.5)
                marker.pose.position.y = float((part0[1] + part1[1]) * 0.5)
                marker.pose.position.z = (z0 + z1) * 0.5
                marker.pose.orientation.z = math.sin(yaw * 0.5)
                marker.pose.orientation.w = math.cos(yaw * 0.5)
                marker.scale.x = max(0.01, part_length)
                marker.scale.y = 0.06
                marker.scale.z = max(0.01, z1 - z0)
                marker.color.r = 0.10 * (1.0 - strength)
                marker.color.g = 0.25 + 0.25 * strength
                marker.color.b = 0.55 + 0.45 * strength
                marker.color.a = 0.25 + 0.65 * strength
                output.markers.append(marker)
        self._persistent_wall_marker_pub.publish(output)

    @staticmethod
    def _room_color(room_id):
        try:
            index = int(str(room_id).rsplit('_', 1)[1])
        except Exception:
            index = abs(hash(room_id))
        return RoomManager._ROOM_PALETTE[index % len(RoomManager._ROOM_PALETTE)]

    @staticmethod
    def _fan_triangles(polygon):
        if len(polygon) < 3:
            return []
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)
        triangles = []
        for i in range(len(polygon)):
            a, b = polygon[i], polygon[(i + 1) % len(polygon)]
            triangles.extend([(cx, cy), a, b])
        return triangles

    def _publish_geometry(self, grid):
        if self._marker_pub is None:
            return
        output = MarkerArray()
        current = set()
        for room_id, room in self.scene_graph.items():
            if not room.get('active', True):
                continue
            polygon = room.get('polygon', [])
            if len(polygon) < 3:
                continue
            try:
                marker_id = int(str(room_id).rsplit('_', 1)[1])
            except Exception:
                marker_id = abs(hash(room_id)) % 100000
            current.add(marker_id)
            color_r, color_g, color_b = self._room_color(room_id)

            fill = Marker()
            fill.header = grid.header
            fill.ns = 'room_fill'
            fill.id = marker_id
            fill.type = Marker.TRIANGLE_LIST
            fill.action = Marker.ADD
            fill.pose.orientation.w = 1.0
            fill.scale.x = fill.scale.y = fill.scale.z = 1.0
            fill.color.r, fill.color.g, fill.color.b = color_r, color_g, color_b
            fill.color.a = 0.28
            for x, y in self._ear_clip_triangulate(polygon):
                point = Point()
                point.x = float(x)
                point.y = float(y)
                point.z = 0.02
                fill.points.append(point)
            output.markers.append(fill)

            marker = Marker()
            marker.header = grid.header
            marker.ns = 'rooms'
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.10
            marker.color.r, marker.color.g, marker.color.b = color_r, color_g, color_b
            marker.color.a = 1.0
            for x, y in polygon + [polygon[0]]:
                point = Point()
                point.x = float(x)
                point.y = float(y)
                point.z = 0.05
                marker.points.append(point)
            output.markers.append(marker)

            centroid = room.get('centroid') or self._centroid(polygon)
            area = float(room.get('area_m2', 0.0))
            text_marker = Marker()
            text_marker.header = grid.header
            text_marker.ns = 'room_labels'
            text_marker.id = marker_id
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = float(centroid[0])
            text_marker.pose.position.y = float(centroid[1])
            text_marker.pose.position.z = 0.6
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.30
            text_marker.color.r, text_marker.color.g, text_marker.color.b = color_r, color_g, color_b
            text_marker.color.a = 1.0
            semantic_label = room.get('semantic_label', '').strip()
            if semantic_label and semantic_label != "UnknownRoom":
                text_marker.text = f'{semantic_label}\n{room_id}\n{area:.2f} m2'
            else:
                text_marker.text = f'{room_id}\n{area:.2f} m2'
            output.markers.append(text_marker)

        for marker_id in self._last_marker_ids - current:
            for ns in ('rooms', 'room_labels', 'room_fill'):
                marker = Marker()
                marker.header = grid.header
                marker.ns = ns
                marker.id = marker_id
                marker.action = Marker.DELETE
                output.markers.append(marker)
        self._last_marker_ids = current
        self._marker_pub.publish(output)

        if self._room_areas_pub is not None:
            areas = {rid: round(float(r.get('area_m2', 0.0)), 2) for rid, r in self.scene_graph.items()}
            payload = {'current_room_id': self.current_room_id, 'areas_m2': areas}
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self._room_areas_pub.publish(msg)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {k: RoomManager._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RoomManager._json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return RoomManager._json_safe(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def _save_rooms(self, force=False):
        if not force and time.time()-self._last_save_time < self._params['save_period_s']:
            return
        self._last_save_time = time.time()
        output_dir = (os.environ.get('GRAPH_API_OUTPUT_DIR')
                      or os.environ.get('LOST3DSG_OUTPUT_DIR')
                      or os.path.join(PROJECT_ROOT, 'output'))
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'room.json')
        rooms_payload = self._json_safe(list(self.scene_graph.values()))
        building_payload = {
            'building_id': 'building_0',
            'semantic_label': 'building',
            'description': '',
            'rooms': rooms_payload,
        }
        payload = {
            'current_room_id': self.current_room_id,
            'updated_at': time.time(),
            'segmentation': self._json_safe(self.last_segmentation_stats),
            'detected_walls': self._json_safe(self._detected_wall_map),
            'doorway_room_edges': self._json_safe(
                getattr(self, '_doorway_room_edges', {})),
            'confirmed_door_cuts': self._json_safe(
                getattr(self, '_confirmed_door_cuts', {})),
            'building': building_payload,
            'rooms': rooms_payload,
        }
        tmp = path+'.tmp'
        with open(tmp, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)

    def save_rooms_to_json(self):
        self._save_rooms(force=True)
