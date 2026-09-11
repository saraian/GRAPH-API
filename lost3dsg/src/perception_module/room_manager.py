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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
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

        from config import CFG  # local, as elsewhere in this file
        self._params = {
            # GA-137. Read from CFG so the switch is REACHABLE. `_params` is otherwise a
            # hardcoded dict and CFG["rooms"] is read nowhere in this file -- so adding the
            # key to config.py alone would have created a setting that exists and cannot be
            # reached, which is the defect class this review keeps finding. Checked before
            # shipping it, not after.
            'gvd_method': str(CFG.get('rooms', {}).get('gvd_method', 'medial_axis')),
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
            'detected_wall_close_free_space': True,
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
            # Viewpoint changes perturb a TLS line by several degrees/cell widths.
            # Fuse that jitter instead of restarting the observation counter.
            'detected_wall_merge_angle_deg': 12.0,
            'detected_wall_merge_distance_m': 0.22,
            'detected_wall_merge_gap_m': 0.50,
            'detected_wall_thickness_m': 0.12,
            # Close small acquisition gaps and imperfect corner junctions in the
            # confirmed wall network.  This is deliberately shorter than a door:
            # the network becomes topologically continuous without sealing a real
            # passage between two collinear wall pieces.
            'detected_wall_junction_gap_m': 0.30,
            'detected_wall_junction_angle_deg': 20.0,
            'detected_wall_door_endpoint_radius_m': 0.20,
            'detected_wall_door_min_support': 0.16,
            # Let the confirmed wall layout propose doorway partitions directly,
            # instead of depending entirely on a sometimes unstable GVD branch.
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
            'gvd_door_nms_m': 1.20,
            'gvd_critical_endpoint_margin_m': 0.20,
            'gvd_cut_margin_px': 2,
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
            'room_hold_empty_partition': True,
            # Compatible contour refinements below this IoU are also held;
            # otherwise room polygons visibly breathe at every map callback.
            'room_partition_stable_iou_min': 0.65,
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
        self._grid_sub = node.create_subscription(
            OccupancyGrid, self.map_topic, self._slow_map_callback, qos)
        self._marker_pub = node.create_publisher(MarkerArray, '/room_areas_array', qos)
        self._persistent_wall_marker_pub = node.create_publisher(
            MarkerArray, '/room_detected_wall_markers', qos)
        self._room_pub = node.create_publisher(String, '/current_room', 10)
        self._room_areas_pub = node.create_publisher(String, '/room_areas', 10)
        if self.cloud_map_topic:
            self._cloud_sub = node.create_subscription(
                PointCloud2, self.cloud_map_topic, self._cloud_map_callback, cloud_qos)
            self._log(
                'info',
                f'RoomManager: 3D structural filter enabled from {self.cloud_map_topic}')
        self._cloud_ground_sub = None
        if self.cloud_ground_topic:
            self._cloud_ground_sub = node.create_subscription(
                PointCloud2, self.cloud_ground_topic, self._cloud_ground_callback, cloud_qos)
        self._cloud_obstacles_sub = None
        if self.cloud_obstacles_topic:
            self._cloud_obstacles_sub = node.create_subscription(
                PointCloud2, self.cloud_obstacles_topic,
                self._cloud_obstacles_callback, cloud_qos)
        if self.cloud_ground_topic or self.cloud_obstacles_topic:
            self._log(
                'info',
                f'RoomManager: floor/obstacle clouds enabled from '
                f'{self.cloud_ground_topic}, {self.cloud_obstacles_topic}')
        self._log(
            'info',
            f'RoomManager: subscribed to {self.map_topic} (simple 2D GVD segmentation)')

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
            if observed is not None and observed.shape == structural.shape:
                # A 2D outline can make a bed or table look like a sparse wall network.
                # Where the cloud actually observed the object but found no tall surface,
                # height evidence is a veto on that purely plan-view classification.
                structural[observed & ~cloud_support.astype(bool)] = 0
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
        nonstructural = (occupied > 0) & (structural_occ == 0)
        if not np.any(nonstructural):
            self._last_topology_fill_stats = {'components': 0, 'pixels': 0, 'confirmed_3d': 0}
            return topo_free

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            nonstructural.astype(np.uint8), 8)
        max_area_px = max(1, int(round(
            self._params['gvd_topo_fill_max_area_m2'] / max(resolution * resolution, 1e-12))))
        nonwall_3d = self._latest_cloud_nonwall_mask
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
        }
        return topo_free

    def _fill_room_holes(self, mask, resolution):
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
        thickness = max(1, int(round(float(self._params.get(
            'detected_wall_thickness_m', 0.12)) / max(float(grid.info.resolution), 1e-6))))
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
            if length < min_length or vertical < min_vertical or rms > max_rms:
                rejected += 1
                continue
            q0 = self._world_to_grid(float(p0[0]), float(p0[1]), grid)
            q1 = self._world_to_grid(float(p1[0]), float(p1[1]), grid)
            if q0 is None or q1 is None:
                continue
            confidence = min(1.0, observations / 6.0)
            layer = np.zeros_like(support, dtype=np.uint8)
            cv2.line(layer, q0, q1, 255, thickness)
            support[layer > 0] = np.maximum(support[layer > 0], confidence)
            qualified.append((p0, p1, q0, q1, confidence))
            segments += 1

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
        stats = {'proposed': 0, 'accepted': 0, 'rejected_not_free': 0,
                 'rejected_no_split': 0, 'rejected_small_partition': 0}
        if (not self._params.get('detected_wall_direct_door_cuts', True) or
                grid is None or not np.any(cut)):
            self._last_wall_door_cut_stats = stats
            return cut

        min_obs = max(1, int(self._params.get('detected_wall_min_observations', 2)))
        min_length = float(self._params.get('detected_wall_min_length_m', 0.50))
        min_vertical = float(self._params.get('detected_wall_min_vertical_extent_m', 0.90))
        max_rms = float(self._params.get('detected_wall_max_rms_m', 0.05))
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
                    float(wall.get('inlier_rms_m', float('inf'))) > max_rms):
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
                    proposals.append((gap, left, right))

        # Narrow, strongly delimited openings first.  Once a cut has separated a
        # room, later candidates are validated only inside their current parent.
        for _gap, left, right in sorted(proposals, key=lambda item: item[0]):
            q0 = self._world_to_grid(float(left[0]), float(left[1]), grid)
            q1 = self._world_to_grid(float(right[0]), float(right[1]), grid)
            if q0 is None or q1 is None:
                continue
            height, width = result.shape
            if not (0 <= q0[0] < width and 0 <= q0[1] < height and
                    0 <= q1[0] < width and 0 <= q1[1] < height):
                continue
            stats['proposed'] += 1
            gap_layer = np.zeros_like(free, dtype=np.uint8)
            cv2.line(gap_layer, q0, q1, 255, 1)
            gap_pixels = gap_layer > 0
            if (not np.any(gap_pixels) or
                    np.count_nonzero(gap_pixels & (free > 0)) /
                    max(1, np.count_nonzero(gap_pixels)) < 0.60):
                stats['rejected_not_free'] += 1
                continue
            _, before = cv2.connectedComponents(result, 8)
            midpoint = ((q0[0]+q1[0])//2, (q0[1]+q1[1])//2)
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
        # cloud_map is global and must remain the basis of a global room
        # partition. cloud_obstacles may be only the latest local sensor frame;
        # use it as the clean fallback, never as a reason to discard the global
        # structural evidence while the latter is fresh.
        if map_fresh:
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
            if all(math.hypot(point[0]-q[0], point[1]-q[1]) > nms_px for q in selected):
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
        endpoints = ((x-radius_px*nx, y-radius_px*ny),
                     (x+radius_px*nx, y+radius_px*ny))
        search_px = max(1, int(round(float(self._params.get(
            'detected_wall_door_endpoint_radius_m', 0.20)) / max(resolution, 1e-6))))
        h, w = support.shape
        values = []
        for ex, ey in endpoints:
            ix, iy = int(round(ex)), int(round(ey))
            x0, x1 = max(0, ix-search_px), min(w, ix+search_px+1)
            y0, y1 = max(0, iy-search_px), min(h, iy+search_px+1)
            values.append(float(np.max(support[y0:y1, x0:x1]))
                          if x0 < x1 and y0 < y1 else 0.0)
        # Both sides are required; a cupboard on one side is not a doorway frame.
        return min(values)

    def _cut_free_space(self, free, dist_real, critical_points, resolution=None):
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
        rejected_small = 0
        rejected_no_split = 0
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
            radius = int(round(float(dist_real[y, x]))) + margin_px
            radius = max(1, radius)
            # Normal to the local GVD tangent.
            nx, ny = -math.sin(theta), math.cos(theta)
            p0 = (int(round(x-radius*nx)), int(round(y-radius*ny)))
            p1 = (int(round(x+radius*nx)), int(round(y+radius*ny)))
            trial = cut.copy()
            cv2.line(trial, p0, p1, 0, max(1, 2*margin_px+1))
            _, before_labels = cv2.connectedComponents(cut, 8)
            _, after_labels = cv2.connectedComponents(trial, 8)
            parent_label = int(before_labels[int(y), int(x)])
            if parent_label <= 0:
                rejected_no_split += 1
                continue
            parent = before_labels == parent_label
            children = np.unique(after_labels[parent])
            children = children[children > 0]
            if children.size < 2:
                rejected_no_split += 1
                continue
            areas = np.asarray([
                np.count_nonzero(parent & (after_labels == child)) for child in children
            ])
            if int(areas.min()) < min_component_px:
                rejected_small += 1
                continue
            cut = trial
            accepted.append(point)

        self._last_validated_cuts = accepted
        self._last_cut_stats = {
            'proposed': len(critical_points),
            'accepted': len(accepted),
            'accepted_wall_supported': sum(
                len(point) >= 5 and point[4] >= float(self._params.get(
                    'detected_wall_door_min_support', 0.16)) for point in accepted),
            'rejected_no_split': rejected_no_split,
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
        reinforced_cells = 0
        closed_free_cells = 0
        topology_wall_mask = None
        if (detected_wall_support is not None and
                self._params.get('detected_wall_reinforce_obstacles', False)):
            confidence_gate = float(self._params.get(
                'detected_wall_topology_confidence', 0.33))
            topology_wall_mask = detected_wall_support >= confidence_gate
            reinforced_cells = int(np.count_nonzero(
                topology_wall_mask & (structural_occ == 0)))
            structural_occ[topology_wall_mask] = 255
        free_topo = self._fill_nonstructural_obstacles(free, occupied, structural_occ, resolution)
        if (topology_wall_mask is not None and
                self._params.get('detected_wall_close_free_space', True)):
            # A wall seen repeatedly in depth is direct surface evidence.  Let it
            # repair short false-free gaps left by the 2D mapper; endpoint joining
            # is capped below door width, so this cannot bridge an actual doorway.
            closed_free_cells = int(np.count_nonzero(
                topology_wall_mask & (free_topo > 0)))
            free_topo[topology_wall_mask] = 0
        # Furniture removed from the topology can leave isolated corner pixels after the
        # occupancy median filter. Closed holes below the configured area are clutter, not
        # navigable-space boundaries, and would create dense spurious medial-axis branches.
        free_topo = self._fill_room_holes(free_topo, resolution)
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

        cut = self._cut_free_space(free_topo, dist_topo, critical_points, resolution)
        validated_cuts = getattr(self, '_last_validated_cuts', [])
        cut = self._detected_wall_doorway_cuts(
            free_topo, cut, grid, resolution)
        wall_door_cuts = getattr(self, '_last_wall_door_cut_stats', {})

        markers = self._grow_labels(free_topo, cut, dist_topo)
        if markers is None:
            _, markers = cv2.connectedComponents(free_topo, 8)

        # A cut may create tiny connected components around clutter or map noise. Merge
        # them before polygon extraction using the actual minimum room area, not merely the
        # old 20-pixel implementation floor.
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

            mask = self._fill_room_holes(mask, resolution)
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
            'door_cuts': len(validated_cuts),
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

    def process_grid(self, grid, full_resegment=True):
        if grid.header.frame_id and grid.header.frame_id != world_frame():
            self._log('warn', f'Ignoring occupancy grid in {grid.header.frame_id!r}; '
                      f'expected {world_frame()!r}')
            return
        with self._lock:
            self.last_grid = grid
            self.last_robot_xy = self._robot_pose()
            if full_resegment:
                candidates = self._segment_regions_gvd(grid)
                accept, stability = self._stabilize_partition(candidates)
                self.last_segmentation_stats['stability'] = stability
                if accept:
                    self._update_regions(candidates)
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

    def _stabilize_partition(self, candidates):
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
        if change_kind == 'merge':
            required = max(1, int(self._params.get(
                'room_merge_change_confirmations', default_required)))
        elif change_kind == 'split':
            required = max(1, int(self._params.get(
                'room_split_change_confirmations', default_required)))
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
            semantic_name, description = self.ask_vlm_room_info(
                [self._vlm_room_label(label) for label in labels]
            )
            if semantic_name:
                room['semantic_label'] = semantic_name
            if description:
                room['description'] = description
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
                semantic_name, description = self.ask_vlm_room_info(
                    evidence
                )
                if semantic_name:
                    room['semantic_label'] = semantic_name
                if description:
                    room['description'] = description
                if str(semantic_name).strip().lower() not in ('', 'unknownroom', 'unknown_room'):
                    room['_semantic_evidence'] = evidence
        self._save_rooms()

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
            from openai import OpenAI
            model_name = os.environ.get("ROOM_VLM_MODEL", CFG["vlm"]["model"])
            client = OpenAI(
                base_url=CFG["vlm"]["base_url"],
                api_key=CFG["vlm"]["api_key"] or os.environ.get("OPENAI_API_KEY", "ollama"),
            )

            text_prompt = (
                f"Analyze these relevant objects: {labels_str}.\n"
                "Identify the room type and describe it.\n"
                "ONLY answer in JSON format: "
                "{\"label\": \"name\", \"description\": \"description\"}. "
                "In the \"label\" field you have to specify one specific room type, "
                "it can't just be something generic like \"room type\""
            )

            content = [{"type": "text", "text": text_prompt}]

            if encoded_image:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                })

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                timeout=CFG["vlm"]["timeout"]
            )

            raw_content = (response.choices[0].message.content or "").strip()
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
            vlm_name, vlm_desc = self.ask_vlm_room_info(room_labels_vlm)
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
        a0, a1, adir, alen = self._wall_geometry(stored)
        b0, b1, bdir, blen = self._wall_geometry(observed)
        angle_deg = float(self._params.get('detected_wall_merge_angle_deg', 8.0))
        if abs(float(adir @ bdir)) < math.cos(math.radians(angle_deg)):
            return False
        amid, bmid = (a0 + a1) * 0.5, (b0 + b1) * 0.5
        normal = np.array([-adir[1], adir[0]])
        if abs(float((bmid - amid) @ normal)) > float(
                self._params.get('detected_wall_merge_distance_m', 0.18)):
            return False
        if abs(float((bmid - amid) @ adir)) > (alen + blen) * 0.5 + float(
                self._params.get('detected_wall_merge_gap_m', 0.50)):
            return False

        # Keep one stable line and expand it over the union of both observed intervals.
        support = max(1, int(stored.get("observations", 1)))
        centre = (amid * support + bmid) / (support + 1)
        if float(adir @ bdir) < 0:
            bdir = -bdir
        direction = adir * support + bdir
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
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
        return True

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
            max_age_s = float(self._params.get('detected_wall_max_age_s', 0.0))
            if max_age_s > 0.0:
                now = time.time()
                self._detected_wall_map = [
                    wall for wall in self._detected_wall_map
                    if now - float(wall.get('last_seen', now)) <= max_age_s
                ]
            for observed in walls:
                self._wall_geometry(observed)
                if not any(self._merge_detected_wall(old, observed)
                           for old in self._detected_wall_map):
                    wall = self._json_safe(dict(observed))
                    wall["observations"] = 1
                    wall["last_seen"] = time.time()
                    self._detected_wall_map.append(wall)
            assigned = self._assign_detected_walls_to_rooms()
            if walls:
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
        max_age_s = float(self._params.get('detected_wall_max_age_s', 0.0))
        now = time.time()
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
                    float(wall.get("inlier_rms_m", float("inf"))) > max_rms):
                continue
            yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            strength = min(1.0, observations / 6.0)
            marker = Marker()
            marker.header = clear.header
            marker.ns = "persistent_detected_walls"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = float((p0[0] + p1[0]) * 0.5)
            marker.pose.position.y = float((p0[1] + p1[1]) * 0.5)
            marker.pose.position.z = (z0 + z1) * 0.5
            marker.pose.orientation.z = math.sin(yaw * 0.5)
            marker.pose.orientation.w = math.cos(yaw * 0.5)
            marker.scale.x = max(0.01, length)
            marker.scale.y = 0.06
            marker.scale.z = max(0.01, z1 - z0)
            # Weak confirmations are pale/transparent; repeated support tends to blue.
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
            'building': building_payload,
            'rooms': rooms_payload,
        }
        tmp = path+'.tmp'
        with open(tmp, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)

    def save_rooms_to_json(self):
        self._save_rooms(force=True)
