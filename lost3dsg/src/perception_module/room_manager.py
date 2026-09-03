#!/usr/bin/env python3
"""Simple 2D GVD room manager for ROS 2 and RTAB-Map.

This version keeps the pipeline intentionally small and standard:

1. classify the occupancy grid into free / occupied;
2. compute a 2D GVD on the free space using obstacle labels;
3. prune short skeleton spurs;
4. detect doorway candidates as low-clearance skeleton branches;
5. cut the free space locally at those candidates;
6. watershed the cut free space into room regions;
7. extract polygons and track them over time.

No cloud-map fusion, no corridor-specific heuristics, no extra shape
classifiers. The goal is a clean baseline that is easy to understand and
debug.
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
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

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
                 enable_live_local_grids=False):
        self.w2v = w2v_model
        self.node = None
        self.tf_buffer = None
        self.tf_listener = None
        self.map_topic = map_topic
        self.cloud_map_topic = cloud_map_topic

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
        self.current_room_wall_segments = []
        self.last_grid = None
        self.last_robot_xy = None
        self._last_save_time = 0.0
        self._last_marker_ids = set()
        self._lock = threading.RLock()
        self._grid_sub = None
        self._marker_pub = None
        self._room_pub = None
        self._room_areas_pub = None
        self._cloud_sub = None
        self._latest_cloud_points = None
        self._latest_cloud_frame = None
        self._latest_cloud_stamp = None
        self._last_cloud_warn_time = 0.0

        from config import CFG  # local, as elsewhere in this file
        self._params = {
            # GA-137. Read from CFG so the switch is REACHABLE. `_params` is otherwise a
            # hardcoded dict and CFG["rooms"] is read nowhere in this file -- so adding the
            # key to config.py alone would have created a setting that exists and cannot be
            # reached, which is the defect class this review keeps finding. Checked before
            # shipping it, not after.
            'gvd_method': str(CFG.get('rooms', {}).get('gvd_method', 'label_diff')),
            # 2D map classification
            'free_threshold': 20,
            'occupied_threshold': 50,
            'unknown_is_obstacle': True,
            'map_median_blur_ksize': 3,
            'min_room_area_m2': 0.5,
            'room_max_area_m2': 100.0,
            # Clutter filtering for topology extraction only
            'gvd_min_obstacle_length_m': 0.4,
            'gvd_wall_max_thickness_m': 0.18,
            'gvd_wall_min_aspect_ratio': 4.0,
            'gvd_wall_min_area_m2': 0.08,
            'gvd_wall_min_fill_ratio': 0.35,
            # Optional 3D support for structural obstacle filtering
            'enable_3d_structural_filter': bool(cloud_map_topic),
            'gvd_3d_min_points_per_cell': 4,
            'gvd_3d_min_vertical_span_m': 0.75,
            'gvd_3d_min_support_ratio': 0.05,
            'gvd_3d_support_dilation_px': 1,
            'gvd_topo_fill_max_area_m2': 3.0,
            'gvd_room_hole_fill_max_area_m2': 2.0,
            'room_nested_merge_max_area_m2': 6.0,
            'room_nested_merge_area_ratio': 0.20,
            'room_nested_merge_wall_support_max_ratio': 0.20,
            # GVD construction / pruning
            'gvd_prune_min_branch_m': 0.35,
            'gvd_door_max_m': 1.20,
            'gvd_bottleneck_ratio': 0.80,
            'gvd_cut_margin_px': 2,
            # Segmentation / region bookkeeping
            'min_region_pixels': 20,
            'region_match_distance_m': 3.0,
            'region_match_iou_min': 0.20,
            'max_region_misses': 8,
            'poly_approx_epsilon_m': 0.08,
            'room_assignment_tolerance_m': 0.12,
            # Persistence/output
            'save_period_s': 1.0,
            'room_stale_prune_s': 10.0,
            'discard_border_regions': False,
        }

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
        self._grid_sub = node.create_subscription(
            OccupancyGrid, self.map_topic, self._slow_map_callback, qos)
        self._marker_pub = node.create_publisher(MarkerArray, '/room_areas_array', qos)
        self._room_pub = node.create_publisher(String, '/current_room', 10)
        self._room_areas_pub = node.create_publisher(String, '/room_areas', 10)
        if self.cloud_map_topic:
            self._cloud_sub = node.create_subscription(
                PointCloud2, self.cloud_map_topic, self._cloud_map_callback, qos)
            self._log(
                'info',
                f'RoomManager: 3D structural filter enabled from {self.cloud_map_topic}')
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
                    'map', frame, rclpy.time.Time(),
                    timeout=Duration(seconds=0.3))
                return float(tf.transform.translation.x), float(tf.transform.translation.y)
            except Exception:
                continue
        return None

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
        if msg.header.frame_id and msg.header.frame_id != 'map':
            now = time.time()
            if now - self._last_cloud_warn_time > 5.0:
                self._log(
                    'warn',
                    f'Ignoring 3D cloud in frame "{msg.header.frame_id}" because only map-frame clouds are supported')
                self._last_cloud_warn_time = now
            return

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
            self._latest_cloud_points = None
            self._latest_cloud_frame = msg.header.frame_id or 'map'
            self._latest_cloud_stamp = msg.header.stamp
            return

        if points.ndim == 1:
            points = points.reshape(1, -1)
        self._latest_cloud_points = points[:, :3].astype(np.float64, copy=False)
        self._latest_cloud_frame = msg.header.frame_id or 'map'
        self._latest_cloud_stamp = msg.header.stamp

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
        p = np.asarray(poly, dtype=np.float64)
        return float(p[:, 0].mean()), float(p[:, 1].mean())

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
            if (
                long_side >= min_len_px
                and short_side <= max_thickness_px
                and aspect >= min_aspect
                and area_px >= min_area_px
                and fill_ratio >= min_fill_ratio
            ):
                structural[labels == i] = 255

        if cloud_support is None and self._params.get('enable_3d_structural_filter', False) and grid is not None:
            cloud_support = self._cloud_structural_support(grid)
            if cloud_support is not None:
                support_dilation_px = max(0, int(self._params.get('gvd_3d_support_dilation_px', 0)))
                if support_dilation_px > 0:
                    kernel = np.ones((2 * support_dilation_px + 1, 2 * support_dilation_px + 1), dtype=np.uint8)
                    cloud_support = cv2.dilate(cloud_support.astype(np.uint8), kernel, iterations=1) > 0
                structural = np.maximum(structural, cloud_support.astype(np.uint8) * 255)
        return structural

    def _fill_nonstructural_obstacles(self, free, occupied, structural_occ, resolution):
        topo_free = free.copy()
        nonstructural = (occupied > 0) & (structural_occ == 0)
        if not np.any(nonstructural):
            return topo_free

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            nonstructural.astype(np.uint8), 8)
        max_area_px = max(1, int(round(
            self._params['gvd_topo_fill_max_area_m2'] / max(resolution * resolution, 1e-12))))
        h, w = nonstructural.shape
        for i in range(1, count):
            area_px = int(stats[i, cv2.CC_STAT_AREA])
            if area_px <= 0 or area_px > max_area_px:
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
            topo_free[labels == i] = 255
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

    def _cloud_structural_support(self, grid):
        points = self._latest_cloud_points
        if points is None or points.shape[0] == 0:
            return None
        if grid is None or grid.info.resolution <= 0:
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
            return None

        ix = ix[valid]
        iy = iy[valid]
        z = points[valid, 2].astype(np.float64, copy=False)
        flat = ix + iy * width
        size = width * height
        counts = np.bincount(flat, minlength=size)
        zmin = np.full(size, np.inf, dtype=np.float64)
        zmax = np.full(size, -np.inf, dtype=np.float64)
        np.minimum.at(zmin, flat, z)
        np.maximum.at(zmax, flat, z)
        span = zmax - zmin

        min_points = int(self._params.get('gvd_3d_min_points_per_cell', 1))
        min_span = float(self._params.get('gvd_3d_min_vertical_span_m', 0.0))
        support = (counts >= min_points) & np.isfinite(span) & (span >= min_span)
        return support.reshape((height, width))

    @staticmethod
    def _compute_gvd_ridge(free_topo, structural_occupied):
        """Medial axis by RIDGE DETECTION on the distance transform.

        GA-137. The label-difference method below cannot produce a skeleton on a normal
        floorplan, and the reason is structural rather than a matter of tuning: it marks a
        pixel only where two adjacent free pixels have DIFFERENT nearest-obstacle CONNECTED
        COMPONENT ids. In any ordinary building the outer walls and the interior walls
        touch, so there is exactly ONE obstacle component, every free pixel carries the same
        label, and the skeleton is empty at every resolution with every threshold correct.

        MEASURED on a synthetic two-room floorplan (outer walls + an interior wall with a
        doorway):
            connected walls   -> 1 obstacle component  -> skeleton_px = 0
            interior wall detached from the outer wall -> 3 components -> skeleton_px = 432
            two separate blobs -> 2 components -> skeleton_px = 120
        The method only works when the obstacles are already disconnected, which is the one
        case a floorplan is not.

        The true GVD is the set of points equidistant from two nearest obstacle POINTS --
        a ridge of the distance transform, which exists regardless of connectivity. On the
        same connected-wall map this yields skeleton_px = 22278, and the doorway shows as a
        clear clearance minimum (5.00 px against an open-room median of 16.00), which is
        exactly the local minimum `_critical_points` cuts on.
        """
        occ = structural_occupied > 0
        src = np.where(occ, 0, 255).astype(np.uint8)
        dist = cv2.distanceTransform(src, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        freeb = free_topo > 0
        d = np.where(freeb, dist, -1.0)
        ridge = np.zeros(d.shape, dtype=bool)
        # On the axis if the clearance is a local maximum along ANY direction.
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = np.roll(np.roll(d, dy, 0), dx, 1)
            b = np.roll(np.roll(d, -dy, 0), -dx, 1)
            ridge |= (d >= a) & (d >= b) & (d > 0)
        skeleton = np.zeros(d.shape, dtype=np.uint8)
        skeleton[ridge & freeb] = 255
        return skeleton, dist

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
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        for _ in range(min_len_px):
            deg = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
            endpoints = skel & (deg == 1)
            if not np.any(endpoints):
                break
            skel &= ~endpoints
        return (skel.astype(np.uint8) * 255)

    @staticmethod
    def _skeleton_degree(skel_bool):
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        deg = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        deg[~skel_bool] = 0
        return deg

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
            out = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and skel_bool[ny, nx]:
                        out.append((ny, nx))
            return out

        branches = []
        max_steps = int(h) * int(w) + 8
        for node in node_set:
            for nbr in neighbors(*node):
                path = [node, nbr]
                prev, curr = node, nbr
                steps = 0
                while curr not in node_set and steps < max_steps:
                    nxts = [n for n in neighbors(*curr) if n != prev]
                    if not nxts:
                        break
                    nxt = nxts[0]
                    path.append(nxt)
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
        points = set()

        # GA-196. WHY a branch was rejected, counted. `branches_cut=0` was the only signal
        # this stage produced, and it is the same shape as every unlogged exit this review
        # has found: it says a thing did not happen and nothing about which test refused it.
        # Measured over two runs, 87% of sweeps cut nothing while the skeleton was ~18,600
        # px and NEVER empty -- so the failure is here, among these four filters, and no
        # bundle could say which. Counters only; no behaviour changes.
        rej = {"too_short_px": 0, "too_short_m": 0, "wider_than_door": 0,
               "no_end_clearance": 0, "not_a_bottleneck": 0, "accepted": 0}
        min_branch_px = max(4, int(round(
            self._params['gvd_prune_min_branch_m'] / max(resolution, 1e-6))))
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

            if min_val > door_radius_px:
                rej["wider_than_door"] += 1
                continue
            if end_clearance <= 1e-6:
                rej["no_end_clearance"] += 1
                continue
            if min_val > float(self._params['gvd_bottleneck_ratio']) * end_clearance:
                rej["not_a_bottleneck"] += 1
                continue
            y, x = path[min_idx]
            points.add((int(y), int(x), float(self._branch_direction(path, min_idx)), float(min_val)))
            rej["accepted"] += 1

        self._last_critical_stats = {
            "branches": len(branches),
            "min_branch_px": min_branch_px,
            "door_max_m": float(self._params['gvd_door_max_m']),
            "narrowest_branch_m": (None if narrowest is None else round(narrowest, 3)),
            **rej,
        }
        return list(points)

    def _cut_free_space(self, free, dist_real, critical_points):
        """Zeroes a disk sized to the true local corridor half-width (from
        the distance transform of the REAL, unfiltered free mask) at every
        critical point, severing free-space connectivity across each
        doorway."""
        cut = free.copy()
        margin_px = max(1, int(self._params['gvd_cut_margin_px']))
        for point in critical_points:
            if len(point) >= 4:
                y, x, theta, _ = point  # theta unused below; kept only by the unpacking
            else:
                y, x = point[:2]
                # unused — nothing below reads theta
                # theta = 0.0
            radius = int(round(float(dist_real[y, x]))) + margin_px
            radius = max(1, radius)
            cv2.circle(cut, (int(x), int(y)), radius, 0, -1)

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

        structural_occ = self._structural_obstacles(occupied, resolution, grid, cloud_support=cloud_support)
        free_topo = self._fill_nonstructural_obstacles(free, occupied, structural_occ, resolution)
        # GA-137: `label_diff` is today's behaviour and stays the default -- it is provably
        # always-empty on a connected floorplan, but changing room segmentation changes
        # EVERY room-scoped number, which is a run-design decision, not mine to take.
        _method = str(self._params.get('gvd_method', 'label_diff'))
        if _method == 'ridge':
            skeleton_raw, dist_topo = self._compute_gvd_ridge(free_topo, structural_occ)
        elif _method == 'label_diff':
            skeleton_raw, dist_topo = self._compute_gvd(free_topo, structural_occ)
        else:
            raise ValueError(f"unknown rooms.gvd_method {_method!r}; "
                             f"expected 'label_diff' or 'ridge'")
        skeleton = self._prune_skeleton(skeleton_raw, resolution)
        critical_points = self._critical_points(skeleton, dist_topo, resolution)

        cut = self._cut_free_space(free_topo, dist_topo, critical_points)

        markers = self._grow_labels(free_topo, cut, dist_topo)
        if markers is None:
            _, markers = cv2.connectedComponents(free_topo, 8)

        min_pixels = self._params['min_region_pixels']
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
        self._log(
            'warn' if _skel_px == 0 else 'info',
            f'GVD segmentation: skeleton_px={_skel_px} '
            f'branches_cut={len(critical_points)} regions={int(markers.max())}'
            + (f' | critical_points: {_cdesc}' if _cdesc else '')
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
                candidates.append((polygon, self._polygon_area(polygon), self._centroid(polygon), False))
        return self._merge_nested_candidates(candidates, cloud_support=cloud_support, grid=grid)

    def process_grid(self, grid, full_resegment=True):
        with self._lock:
            self.last_grid = grid
            self.last_robot_xy = self._robot_pose()
            if full_resegment:
                candidates = self._segment_regions_gvd(grid)
                self._update_regions(candidates)
                self.current_room_id = self._room_at(self.last_robot_xy)
                self._publish_geometry(grid)
                self._save_rooms()

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
                'centroid': [], 'walls': [], 'wall_segments': [],
                'confirmed': False, 'boundaries': {}, 'last_seen': time.time(),
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

        active_room_ids = {r.room_id for r in updated.values()}
        now2 = time.time()
        stale_s = self._params['room_stale_prune_s']
        for room_id in list(self.scene_graph.keys()):
            if room_id in active_room_ids:
                self.scene_graph[room_id]['active'] = True
                continue
            last_seen = self.scene_graph[room_id].get('last_seen', 0)
            if now2 - last_seen > stale_s:
                # GA-185: RETIRED, NOT DELETED. This used to `del` the room, and
                # `room_stale_prune_s` is 10 SECONDS while a room's `last_seen` is refreshed
                # only while the robot is IN it -- so every room the tour left for more than
                # ten seconds was erased from the registry within one sweep.
                #
                # MEASURED in run 20260901_055513: room.json listed ONE room while the
                # objects carried two, `room_0` holding 41 of the 186. The objects outlived
                # the room they point at, and the published room count became "how many
                # rooms were visible in the last ten seconds" rather than "how many rooms
                # were mapped" -- which is also why run 044225 reported 4 and this one 1.
                #
                # A mapped room is part of the building whether or not it is in view. The
                # pruning intent -- stop treating it as CURRENT -- is kept by the flag; the
                # record is kept because objects still reference it and a dangling reference
                # is worse than a stale one.
                room = self.scene_graph[room_id]
                if room.get('active', True):
                    room['retired_at'] = now2
                room['active'] = False
                if self.current_room_id == room_id:
                    self.current_room_id = None

        return list(updated.values())

    def _room_at(self, xy):
        if xy is None:
            return None
        tolerance = self._params['room_assignment_tolerance_m']
        matches = [
            r for r in self.regions.values()
            if self._point_in_polygon(r.polygon, xy, tolerance)
        ]
        return min(matches, key=lambda r: r.area_m2).room_id if matches else None

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
                semantic_name, description = self.ask_vlm_room_info(
                    [self._vlm_room_label(label) for label in sorted(merged_labels)]
                )
                if semantic_name:
                    room['semantic_label'] = semantic_name
                if description:
                    room['description'] = description
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

        tolerance = float(self._params['room_assignment_tolerance_m'])
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

        if nearest is not None and nearest_dist <= max(tolerance, 0.45):
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
            walls=self.current_room_walls
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
        output_dir = os.path.join(PROJECT_ROOT, 'output')
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
            'building': building_payload,
            'rooms': rooms_payload,
        }
        tmp = path+'.tmp'
        with open(tmp, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)

    def save_rooms_to_json(self):
        self._save_rooms(force=True)
