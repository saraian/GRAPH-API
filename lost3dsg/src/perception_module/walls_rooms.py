import json

import numpy as np
from rclpy.qos import QoSProfile, DurabilityPolicy
from scipy.ndimage import binary_dilation, label
from skimage.measure import find_contours
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from cv_utils import create_pointcloud2_msg


_GVD_SKELETON_DTYPE = np.dtype([
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("intensity", np.float32),
])


class WallsRoomMixin:
    def init_gvd_room_state(self):
        self.current_room_id = getattr(self, "current_room_id", "room_0")
        self.room_area_pubs = getattr(self, "room_area_pubs", {})
        self.wall_pubs = getattr(self, "wall_pubs", {})
        self.room_walls = getattr(self, "room_walls", {})
        self.accumulated_walls = getattr(self, "accumulated_walls", {})
        self.gvd_room_polygons = getattr(self, "gvd_room_polygons", {})
        self.gvd_room_skeletons = getattr(self, "gvd_room_skeletons", {})
        self.gvd_room_clouds = getattr(self, "gvd_room_clouds", {})
        self.gvd_room_stats = getattr(self, "gvd_room_stats", {})

        self.latest_gvd_graph = None
        self.latest_gvd_skeleton = None

        self.room_grid_resolution = float(getattr(self, "room_grid_resolution", 0.10))
        self.room_grid_margin = float(getattr(self, "room_grid_margin", 0.80))
        self.room_gvd_polygon_padding_voxels = int(getattr(self, "room_gvd_polygon_padding_voxels", 5))
        self.room_gvd_keep_largest_component_only = bool(
            getattr(self, "room_gvd_keep_largest_component_only", True)
        )

    def gvd_graph_callback(self, msg: MarkerArray):
        self.latest_gvd_graph = msg

    def gvd_skeleton_callback(self, msg: PointCloud2):
        self.latest_gvd_skeleton = msg

    def room_callback(self, msg):
        new_room = msg.data
        if new_room == self.current_room_id:
            return

        self.get_logger().info(f"Room changed from {self.current_room_id} to {new_room}")
        self.refresh_current_room_geometry(self.current_room_id)
        self.publish_room_area(self.current_room_id)
        self.current_room_id = new_room

    def process_walls_and_publish(self):
        return self.process_gvd_and_publish()

    def process_gvd_and_publish(self):
        geometry = self.refresh_current_room_geometry(self.current_room_id)
        if geometry is None:
            self.get_logger().warn("No GVD data yet: waiting for /gvd_graph and /gvd_skeleton")
            return

        polygon, skeleton_points, stats = geometry
        self._publish_room_pointcloud(self.current_room_id, skeleton_points)
        self._publish_wall_segments(self._polygon_to_segments(polygon))
        self.publish_room_area(self.current_room_id)

    def refresh_current_room_geometry(self, room_id=None):
        room_id = room_id or self.current_room_id
        geometry = self._compute_room_geometry()
        if geometry is None:
            return None

        polygon, skeleton_points, stats = geometry
        self.gvd_room_polygons[room_id] = polygon
        self.gvd_room_skeletons[room_id] = skeleton_points
        self.gvd_room_clouds[room_id] = skeleton_points
        self.gvd_room_stats[room_id] = stats
        return geometry

    def _compute_room_geometry(self):
        points = self._collect_latest_gvd_points()
        if points.shape[0] < 3:
            return None

        xy = points[:, :2]
        min_xy = xy.min(axis=0) - self.room_grid_margin
        max_xy = xy.max(axis=0) + self.room_grid_margin
        extent = np.maximum(max_xy - min_xy, self.room_grid_resolution)
        grid_shape = np.maximum(np.ceil(extent / self.room_grid_resolution).astype(int), 1)

        occupancy = np.zeros(tuple(grid_shape), dtype=bool)
        idx = np.floor((xy - min_xy) / self.room_grid_resolution).astype(np.int64)
        valid = np.all((idx >= 0) & (idx < grid_shape), axis=1)
        idx = idx[valid]
        if idx.shape[0] == 0:
            return None

        occupancy[idx[:, 0], idx[:, 1]] = True
        if self.room_gvd_keep_largest_component_only:
            occupancy = self._keep_largest_component_2d(occupancy)

        padded = binary_dilation(
            occupancy,
            structure=np.ones((3, 3), dtype=bool),
            iterations=max(1, self.room_gvd_polygon_padding_voxels),
        )
        polygon = self._extract_polygon_from_mask(padded, min_xy)
        if polygon is None:
            polygon = self._fallback_convex_hull(xy)
        if polygon is None or len(polygon) < 3:
            return None

        skeleton_points = points.astype(np.float64)
        stats = {
            "point_count": int(points.shape[0]),
            "occupied_cells": int(occupancy.sum()),
            "polygon_cells": int(padded.sum()),
        }
        return polygon, skeleton_points, stats

    def _collect_latest_gvd_points(self):
        clouds = []

        if self.latest_gvd_skeleton is not None:
            try:
                clouds.append(self._parse_gvd_skeleton_cloud(self.latest_gvd_skeleton))
            except Exception as exc:
                self.get_logger().warn(f"Failed to parse /gvd_skeleton: {exc}")

        if self.latest_gvd_graph is not None:
            try:
                clouds.append(self._parse_gvd_graph_markers(self.latest_gvd_graph))
            except Exception as exc:
                self.get_logger().warn(f"Failed to parse /gvd_graph: {exc}")

        if not clouds:
            return np.empty((0, 3), dtype=np.float64)

        points = np.concatenate([c for c in clouds if c.size > 0], axis=0) if any(c.size > 0 for c in clouds) else np.empty((0, 3), dtype=np.float64)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        return points.astype(np.float64)

    def _parse_gvd_skeleton_cloud(self, msg: PointCloud2):
        if msg.point_step != _GVD_SKELETON_DTYPE.itemsize:
            raise ValueError(
                f"Unexpected point_step={msg.point_step}, expected {_GVD_SKELETON_DTYPE.itemsize}"
            )

        arr = np.frombuffer(msg.data, dtype=_GVD_SKELETON_DTYPE, count=msg.width * msg.height)
        return np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float64)

    def _parse_gvd_graph_markers(self, msg: MarkerArray):
        points = []
        for marker in msg.markers:
            if marker.type not in (Marker.SPHERE_LIST, Marker.LINE_LIST, Marker.LINE_STRIP):
                continue
            for pt in marker.points:
                points.append([float(pt.x), float(pt.y), float(pt.z)])
        if not points:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(points, dtype=np.float64)

    def _keep_largest_component_2d(self, occupancy):
        labeled, num = label(occupancy, structure=np.ones((3, 3), dtype=int))
        if num <= 1:
            return occupancy
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        if sizes.size == 0 or sizes.max() <= 0:
            return occupancy
        largest = int(np.argmax(sizes))
        return labeled == largest

    def _extract_polygon_from_mask(self, mask, min_xy):
        contours = find_contours(mask.astype(np.float32).T, 0.5)
        if not contours:
            return None

        contour = max(contours, key=self._contour_area)
        if contour.shape[0] < 3:
            return None

        polygon = []
        for row, col in contour:
            x = float(min_xy[0] + col * self.room_grid_resolution)
            y = float(min_xy[1] + row * self.room_grid_resolution)
            polygon.append([x, y])
        return polygon

    @staticmethod
    def _contour_area(contour):
        if contour.shape[0] < 3:
            return 0.0
        x = contour[:, 1]
        y = contour[:, 0]
        return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    def _fallback_convex_hull(self, points):
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(points)
            return [[float(points[idx, 0]), float(points[idx, 1])] for idx in hull.vertices]
        except Exception:
            return None

    def _polygon_to_segments(self, polygon):
        if not polygon or len(polygon) < 2:
            return []

        segments = []
        for idx in range(len(polygon)):
            p1 = polygon[idx]
            p2 = polygon[(idx + 1) % len(polygon)]
            segments.append({
                "start": {"x": float(p1[0]), "y": float(p1[1])},
                "end": {"x": float(p2[0]), "y": float(p2[1])},
            })
        return segments

    def _publish_room_pointcloud(self, room_id, points_xyz):
        if points_xyz is None or len(points_xyz) == 0:
            return

        self.room_walls.setdefault(room_id, [])
        room_points = [[float(x), float(y), float(z), 150, 150, 150] for x, y, z in points_xyz]
        self.room_walls[room_id] = room_points

        if room_id not in self.wall_pubs:
            qos_persistent = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.wall_pubs[room_id] = self.create_publisher(PointCloud2, f"/walls/{room_id}", qos_persistent)

        try:
            pc2_msg = create_pointcloud2_msg(self.room_walls[room_id], "map")
            self.wall_pubs[room_id].publish(pc2_msg)
        except Exception as exc:
            self.get_logger().error(f"PointCloud2 room publish error: {exc}")

    def _publish_wall_segments(self, wall_segments):
        if not wall_segments:
            return
        msg = String()
        msg.data = json.dumps(wall_segments)
        self.wall_segments_pub.publish(msg)

    def publish_room_area(self, room_id):
        polygon = self.gvd_room_polygons.get(room_id)
        if polygon is None or len(polygon) < 3:
            geometry = self.refresh_current_room_geometry(room_id)
            polygon = geometry[0] if geometry is not None else None

        if polygon is None or len(polygon) < 3:
            self.get_logger().warn(f"Too few GVD points to compute room geometry for {room_id}")
            return

        if room_id not in self.room_area_pubs:
            qos_latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.room_area_pubs[room_id] = self.create_publisher(Marker, f"/room_areas/{room_id}", qos_latch)

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "room_areas"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.5
        marker.color.a = 1.0

        for x, y in polygon:
            marker.points.append(Point(x=float(x), y=float(y), z=0.05))
        marker.points.append(Point(x=float(polygon[0][0]), y=float(polygon[0][1]), z=0.05))

        self.room_area_pubs[room_id].publish(marker)
