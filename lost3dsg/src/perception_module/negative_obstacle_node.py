#!/usr/bin/env python3
# *****
# Date: 2026-11-23
# Editor: WangZX
# Remark: Negative obstacle (drop-off) detection for the unknown-scene tasks (Exploration / ObjectNav).
#         rtabmap builds its grid with Grid/NormalsSegmentation false and Grid/MaxGroundHeight 0.25, so every
#         point below 0.25 m -- including the floor below, seen down a stairwell -- is "ground" and its cell is
#         marked free, and ray tracing also clears the cells over the void. Frontiers and candidates then
#         appear downstairs. This node finds those cells from the rtabmap cloud and republishes the map:
#           /exploration/map          drop-off cells = 100 (occupied) for frontier / info gain / planner
#           /exploration/map_display  drop-off cells = -1 (unknown) for the BEV, so they are not drawn
#           /exploration/negative_obstacles  JSON statistics
#         PreferFilteredMap is the subscription helper the consumers use: the raw map until the first
#         filtered one arrives, so a missing or failed node falls back to the old behaviour.
#         Building mode is untouched: the node idles and the consumers keep /rtabmap/map.
# *****
"""Negative obstacle node.

Subscribes : /rtabmap/map (OccupancyGrid, latched), <cloud_topic> (PointCloud2, default /rtabmap/cloud_map)
TF         : <world_frame> <- <bev.base_frame> (the robot's height gives the current floor)
Publishes  : /exploration/map, /exploration/map_display (OccupancyGrid, latched)
             /exploration/negative_obstacles (std_msgs/String JSON, latched)

A launch-file component that exits shuts the whole run down, so this node never exits on its own.
"""

import array
import json
import math
import sys
import time
import traceback

import numpy as np

FILTERED_MAP_TOPIC = "/exploration/map"
DISPLAY_MAP_TOPIC = "/exploration/map_display"
STATS_TOPIC = "/exploration/negative_obstacles"

DEFAULT_NEG_PARAMS = {
    "drop_m": 0.15,          # a point this far below the current floor is negative evidence
    "floor_band_m": 0.10,    # |z - floor| within this is floor evidence
    "min_points": 2,         # negative evidence needed to seed a cell
    "floor_ratio": 0.5,      # a seed needs floor evidence <= floor_ratio * negative evidence
    "grow_max_m": 1.5,       # growth from the seeds over free cells without floor evidence
    "close_radius_m": 0.10,  # morphological closing of the mask
    "keep_radius_m": 0.6,    # never mask cells this close to the robot
    "free_thresh": 25,       # same free threshold as frontier_extractor
    # *****
    # Date: 2026-11-23
    # Editor: WangZX
    # Remark: False-positive guards (see negative_mask): seed components smaller than min_seed_area_m2 are noise,
    #         and growth stops at empty cells with at least floor_neighbors floor-supported 8-neighbours.
    # *****
    "min_seed_area_m2": 0.05,
    "floor_neighbors": 4,
    # *****
    # Date: 2026-11-23
    # Editor: WangZX
    # Remark: Slope tolerance (continuous_ground): neighbouring ground cells whose heights differ by at most
    #         max_step_m are one continuous surface; gaps up to gap_bridge_m without points are bridged.
    # *****
    "max_step_m": 0.07,
    "gap_bridge_m": 0.10,
    # *****
    # Date: 2026-11-23
    # Editor: WangZX
    # Remark: Overhead obstacles (clear_overhead): rtabmap counts every point between Grid/MaxGroundHeight (0.25 m)
    #         and Grid/MaxObstacleHeight (1.8 m) as an obstacle, so a low door lintel (hm3d_00337: lintel points at
    #         1.5-1.8 m) draws an occupied line across the doorway. The robot is 1.5 m tall (agent_height).
    # *****
    "clear_overhead": True,
    "ground_max_m": 0.25,     # rtabmap Grid/MaxGroundHeight: below this a point is ground
    "overhead_min_m": 1.5,    # robot height: points at or above this do not block the robot
    "occ_thresh": 65,         # same occupied threshold as frontier_extractor
}


def neg_params(params=None):
    p = dict(DEFAULT_NEG_PARAMS)
    p.update({k: v for k, v in (params or {}).items() if v is not None and k in DEFAULT_NEG_PARAMS})
    return p


def points_to_cells(points, shape, res, origin_x, origin_y, yaw=0.0):
    """(N,3) map-frame points -> (rows, cols, z, inside) on the grid of the given shape."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    dx, dy = pts[:, 0] - origin_x, pts[:, 1] - origin_y
    c, s = math.cos(yaw), math.sin(yaw)
    cols = np.floor((c * dx + s * dy) / res).astype(np.int64)
    rows = np.floor((-s * dx + c * dy) / res).astype(np.int64)
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1]) & np.isfinite(pts[:, 2])
    return rows, cols, pts[:, 2], inside


def height_evidence(points, shape, res, origin_x, origin_y, floor_z, params=None, yaw=0.0):
    """Per-cell counts (neg, floor) of points below the floor and points at floor height."""
    p = neg_params(params)
    h, w = shape
    neg = np.zeros(h * w, np.int32)
    flo = np.zeros(h * w, np.int32)
    if points is None or len(points) == 0:
        return neg.reshape(h, w), flo.reshape(h, w)
    rows, cols, z, inside = points_to_cells(points, shape, res, origin_x, origin_y, yaw)
    idx = rows[inside] * w + cols[inside]
    z = z[inside] - float(floor_z)
    below = z < -float(p["drop_m"])
    # *****
    # Date: 2026-11-23
    # Editor: WangZX
    # Remark: Floor evidence reaches up to ground_max_m (rtabmap's own ground limit), not just +floor_band_m: rooms
    #         on one storey differ by ~0.1 m (live 00861) and a slightly higher floor is walkable, never a drop.
    #         With the band at +-0.10 m the neighbouring room had no floor evidence and the stairwell mask grew into it.
    # *****
    at = (z >= -float(p["floor_band_m"])) & (z <= float(p["ground_max_m"]))
    neg += np.bincount(idx[below], minlength=h * w).astype(np.int32)
    flo += np.bincount(idx[at], minlength=h * w).astype(np.int32)
    return neg.reshape(h, w), flo.reshape(h, w)


# *****
# Date: 2026-11-23
# Editor: WangZX
# Remark: Ramps were masked as drop-offs: the low part of a gentle downward ramp lies more than drop_m below the
#         floor. A drop-off is a height DISCONTINUITY (stairwell lip, stair riser), a ramp is not. This finds the
#         ground cells joined to the current floor by a chain of neighbouring cells with height steps
#         <= max_step_m; those cells are continuous ground and are never masked, however low they are.
#         Ground height per cell = highest point at or below floor_z + floor_band_m (so a stair lip cell keeps
#         the upper floor's height). Empty cells up to gap_bridge_m wide are bridged with the neighbouring
#         height, but only cells that hold points can be returned as continuous ground.
# *****
def continuous_ground(points, shape, res, origin_x, origin_y, floor_z, params=None, yaw=0.0):
    """Boolean mask of ground cells connected to the current floor by gentle height steps."""
    import cv2
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    p = neg_params(params)
    h, w = shape
    out = np.zeros((h, w), bool)
    if points is None or len(points) == 0:
        return out
    rows, cols, z, inside = points_to_cells(points, shape, res, origin_x, origin_y, yaw)
    band = float(p["floor_band_m"])
    top = float(p["ground_max_m"])          # floor evidence up to rtabmap's ground limit (see height_evidence)
    sel = inside & (z <= float(floor_z) + top)
    if not sel.any():
        return out
    lo = -1e6
    ground = np.full(h * w, lo, np.float64)
    np.maximum.at(ground, rows[sel] * w + cols[sel], z[sel])
    ground = ground.reshape(h, w)
    valid = ground > lo / 2
    hb, vb = ground.astype(np.float32), valid.copy()
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(int(round(float(p["gap_bridge_m"]) / res))):
        d = cv2.dilate(np.where(vb, hb, np.float32(lo)), k3)
        fill = ~vb & (d > lo / 2)
        if not fill.any():
            break
        hb[fill] = d[fill]
        vb |= fill
    step = float(p["max_step_m"])
    ids = np.arange(h * w).reshape(h, w)
    src, dst = [], []
    for (a_r, a_c, b_r, b_c) in ((slice(None), slice(0, -1), slice(None), slice(1, None)),
                                 (slice(0, -1), slice(None), slice(1, None), slice(None)),
                                 (slice(0, -1), slice(0, -1), slice(1, None), slice(1, None)),
                                 (slice(0, -1), slice(1, None), slice(1, None), slice(0, -1))):
        ok = vb[a_r, a_c] & vb[b_r, b_c] & (np.abs(hb[a_r, a_c] - hb[b_r, b_c]) <= step)
        src.append(ids[a_r, a_c][ok])
        dst.append(ids[b_r, b_c][ok])
    src, dst = np.concatenate(src), np.concatenate(dst)
    graph = coo_matrix((np.ones(len(src), np.int8), (src, dst)), shape=(h * w, h * w))
    _, lab = connected_components(graph, directed=False)
    lab = lab.reshape(h, w)
    floor_cells = valid & (ground >= float(floor_z) - band) & (ground <= float(floor_z) + top)
    if not floor_cells.any():
        return out
    good = np.zeros(lab.max() + 1, bool)
    good[np.unique(lab[floor_cells])] = True
    return good[lab] & valid


# *****
# Date: 2026-11-23
# Editor: WangZX
# Remark: Floor = the most populated height (histogram mode), not the median. Live 00337: the floor under and
#         around the robot is in the camera blind zone, so within 1.5 m the band held mostly wall-base and furniture
#         points and the median drifted from -2.77 to -2.63; the real floor then fell outside floor_band_m (no
#         overhead clearing) and below drop_m (false drop-offs). The floor is the largest horizontal surface, so
#         it is the histogram peak; the search radius is 3 m to reach past the blind zone.
# *****
def estimate_floor(points, robot_xyz, radius_m=(1.5, 3.0), band_m=0.3, bin_m=0.02, fallback=None, min_points=50,
                   peak_share=0.2):
    """Floor height near the robot: the densest height bin within band_m of the robot height, else `fallback`
    (the last good estimate; the robot z when None -- the habitat base sits ~0.17 m above the mesh floor)."""
    # *****
    # Date: 2026-11-23
    # Editor: WangZX
    # Remark: Nearest radius first. Live 00861: rooms on one storey differ by ~0.13 m and at 3 m the larger
    #         neighbouring room won (1.175 vs 1.057 under the robot), close to drop_m. The search now tries 1.5 m
    #         and widens to 3 m when the near band has fewer than min_points points (blind zone) or no sharp peak
    #         (the floor is a plane, a sharp peak; scattered wall-base / furniture points are flat): the peak bin
    #         must hold at least peak_share of the band points.
    # *****
    rz = float(robot_xyz[2])
    fallback = rz if fallback is None else float(fallback)
    if points is None or len(points) == 0:
        return fallback
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    d = np.hypot(pts[:, 0] - robot_xyz[0], pts[:, 1] - robot_xyz[1])
    in_band = np.abs(pts[:, 2] - rz) <= band_m
    radii = radius_m if isinstance(radius_m, (tuple, list)) else (radius_m,)
    for rad in radii:
        near = in_band & (d <= float(rad))
        if int(near.sum()) < min_points:
            continue
        z = pts[near, 2]
        edges = np.arange(rz - band_m, rz + band_m + bin_m, bin_m)
        hist, edges = np.histogram(z, bins=edges)
        k = int(np.argmax(hist))
        peak = z[(z >= edges[k] - bin_m) & (z <= edges[k + 1] + bin_m)]
        if len(peak) and hist[k] >= float(peak_share) * len(z):
            return float(np.median(peak))
    return fallback


# *****
# Date: 2026-11-23
# Editor: WangZX
# Remark: Overhead-only occupied cells (a low door lintel, a shelf above head height). A cell is cleared to free
#         when it is occupied in the rtabmap grid, holds a floor point (the floor under it was seen, so nothing at
#         body height hid it), holds NO point between ground_max_m and overhead_min_m above the floor, and holds
#         at least one point at or above overhead_min_m. A wall has points at body height and stays; a cell whose
#         floor was never seen stays as rtabmap made it.
# *****
def overhead_cells(grid, points, res, origin_x, origin_y, floor_z, flo, params=None, yaw=0.0):
    """Boolean mask of occupied cells whose only obstacle points are above the robot."""
    p = neg_params(params)
    g = np.asarray(grid, dtype=np.int16)
    h, w = g.shape
    if points is None or len(points) == 0:
        return np.zeros((h, w), bool)
    rows, cols, z, inside = points_to_cells(points, g.shape, res, origin_x, origin_y, yaw)
    idx = rows[inside] * w + cols[inside]
    dz = z[inside] - float(floor_z)
    body = (dz > float(p["ground_max_m"])) & (dz < float(p["overhead_min_m"]))
    high = dz >= float(p["overhead_min_m"])
    n_body = np.bincount(idx[body], minlength=h * w).reshape(h, w)
    n_high = np.bincount(idx[high], minlength=h * w).reshape(h, w)
    # *****
    # Date: 2026-11-23
    # Editor: WangZX
    # Remark: Body / overhead evidence over the 3x3 neighbourhood: the rtabmap grid and its cloud are offset by a
    #         cell here and there (live 00337: lintel points one cell beside the occupied cell), and a cell next to
    #         a wall end or door jamb must keep its obstacle.
    # *****
    import cv2
    box = lambda a: cv2.boxFilter(a.astype(np.float32), -1, (3, 3), normalize=False,  # noqa: E731
                                  borderType=cv2.BORDER_CONSTANT)
    return (g >= int(p["occ_thresh"])) & (flo >= 1) & (box(n_body) == 0) & (box(n_high) >= 1)


def negative_mask(grid, neg, flo, res, params=None, robot_rc=None, ground_ok=None):
    """Boolean mask of drop-off cells (see module remark). ground_ok: continuous_ground cells, never masked."""
    import cv2
    p = neg_params(params)
    g = np.asarray(grid, dtype=np.int16)
    # *****
    # Date: 2026-11-23
    # Editor: WangZX
    # Remark: False-positive fixes (re-applied). /rtabmap/cloud_map is voxelised at the grid cell size, so a cell
    #         holds about ONE point per surface: per-cell counts with min_points >= 2 seeded only from noise or
    #         vertical structure, a lone noise point below the floor could seed, and real floor cells that hold no
    #         point (voxel / cell misalignment, grazing view) let the growth run across the floor. Now:
    #         (1) a seed cell holds a below-floor point and its 3x3 neighbourhood holds >= min_points of them;
    #         (2) seed connected components smaller than min_seed_area_m2 are dropped;
    #         (3) growth does not enter an empty cell with >= floor_neighbors floor-supported 8-neighbours (a
    #             hole inside the floor, not the void over a stairwell);
    #         (4) any cell holding a floor point and not a seed is protected (was: >= min_points floor points).
    # *****
    min_pts = int(p["min_points"])
    neg3 = cv2.boxFilter(neg.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    seed = (neg >= 1) & (neg3 >= min_pts) & (flo <= float(p["floor_ratio"]) * neg)
    min_area = int(round(float(p["min_seed_area_m2"]) / (res * res)))
    if min_area > 1 and seed.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        seed = keep[lab]
    protected = (flo >= 1) & ~seed
    # *****
    # Date: 2026-11-23
    # Editor: WangZX
    # Remark: Continuous ground (a ramp) neither seeds, nor is grown into, nor stays masked (continuous_ground).
    # *****
    if ground_ok is not None:
        seed &= ~ground_ok
        protected |= ground_ok
    if not seed.any():
        return np.zeros(g.shape, bool)
    free = (g >= 0) & (g < int(p["free_thresh"]))
    floor_nb = cv2.boxFilter((flo >= 1).astype(np.float32), -1, (3, 3), normalize=False,
                             borderType=cv2.BORDER_CONSTANT)
    grow_ok = free & (flo == 0) & (floor_nb < float(p["floor_neighbors"]))
    if ground_ok is not None:
        grow_ok &= ~ground_ok
    mask = seed.copy()
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(int(round(float(p["grow_max_m"]) / res))):
        nxt = mask | (cv2.dilate(mask.astype(np.uint8), k3).astype(bool) & grow_ok)
        if nxt.sum() == mask.sum():
            break
        mask = nxt
    r = int(round(float(p["close_radius_m"]) / res))
    if r > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ker).astype(bool)
    mask &= ~protected
    if robot_rc is not None:
        kr = float(p["keep_radius_m"]) / res
        rr, cc = np.ogrid[:g.shape[0], :g.shape[1]]
        mask &= (rr - robot_rc[0]) ** 2 + (cc - robot_rc[1]) ** 2 > kr * kr
    return mask


def apply_mask(grid, mask, value):
    out = np.array(grid, dtype=np.int16, copy=True)
    out[mask] = int(value)
    return out


# *****
# Date: 2026-11-23
# Editor: WangZX
# Remark: Subscription helper for the map consumers (frontier_node, info_gain_node, exploration_controller_node,
#         ros_bev). Subscribes the raw map and the filtered one; forwards the raw map only until the first
#         filtered map arrives, then only the filtered one.
# *****
class PreferFilteredMap:
    def __init__(self, node, raw_topic, filtered_topic, callback, qos):
        from nav_msgs.msg import OccupancyGrid
        self.callback = callback
        self.filtered_seen = False
        self.raw_sub = node.create_subscription(OccupancyGrid, raw_topic, self._on_raw, qos)
        self.filtered_sub = node.create_subscription(OccupancyGrid, filtered_topic, self._on_filtered, qos)

    def _on_raw(self, msg):
        if not self.filtered_seen:
            self.callback(msg)

    def _on_filtered(self, msg):
        self.filtered_seen = True
        self.callback(msg)


def filter_enabled(cfg):
    """True when the consumers should read the filtered maps: an unknown-scene task with the filter on."""
    mode = str((cfg.get("task") or {}).get("mode") or "Building")
    ncfg = ((cfg.get("exploration") or {}).get("negative_obstacles") or {})
    return mode != "Building" and bool(ncfg.get("enabled", True))


def _load_cfg():
    try:
        from config import CFG
        return CFG
    except Exception:
        return {}


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def _node_class():
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    class NegativeObstacleNode(Node):
        def __init__(self):
            super().__init__("negative_obstacle_node")
            cfg = _load_cfg()
            self.task_mode = str((cfg.get("task") or {}).get("mode") or "Building")
            ncfg = ((cfg.get("exploration") or {}).get("negative_obstacles") or {})
            self.active = filter_enabled(cfg)
            if not self.active:
                self.get_logger().info(
                    f"negative obstacle filter idle (task={self.task_mode}, enabled={ncfg.get('enabled', True)})")
                return
            self.params = neg_params(ncfg)
            self.period = 1.0 / max(0.05, float(ncfg.get("rate_hz", 1.0)))
            self.world = (cfg.get("tf") or {}).get("world_frame", "map")
            bev = cfg.get("bev") or {}
            self.base = bev.get("base_frame", "base_link")
            self.map_topic = bev.get("map_topic", "/rtabmap/map")
            self.cloud_topic = str(ncfg.get("cloud_topic", "/rtabmap/cloud_map"))
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            cloud_qos = QoSProfile(depth=1, durability=DurabilityPolicy.VOLATILE,
                                   reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub_map = self.create_publisher(OccupancyGrid, FILTERED_MAP_TOPIC, latched)
            self.pub_display = self.create_publisher(OccupancyGrid, DISPLAY_MAP_TOPIC, latched)
            self.pub_stats = self.create_publisher(String, STATS_TOPIC, latched)
            self.buffer = Buffer()
            self.listener = TransformListener(self.buffer, self)
            self.point_cloud2 = point_cloud2
            self.map = None
            self.cloud = None          # (N,3) float64 in the world frame
            self.dirty = False
            self.last_run = 0.0
            self.create_subscription(OccupancyGrid, self.map_topic, self.on_map, latched)
            self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, cloud_qos)
            self.create_timer(0.2, self.tick)
            self.get_logger().info(
                f"negative obstacle filter active (task={self.task_mode}): {self.map_topic} + {self.cloud_topic} "
                f"-> {FILTERED_MAP_TOPIC}, {DISPLAY_MAP_TOPIC}; params={self.params}")

        def on_map(self, msg):
            self.map = msg
            self.dirty = True
            if self.cloud is None:
                # Pass-through until a cloud arrives, so the consumers always have a map.
                self.pub_map.publish(msg)
                self.pub_display.publish(msg)

        def on_cloud(self, msg):
            try:
                pts = self.point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
                pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
                src = (msg.header.frame_id or self.world).lstrip("/")
                if src != self.world and len(pts):
                    tf = self.buffer.lookup_transform(self.world, src, Time())
                    t, q = tf.transform.translation, tf.transform.rotation
                    x, y, z, w = q.x, q.y, q.z, q.w
                    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                    pts = pts @ rot.T + np.array([t.x, t.y, t.z])
                self.cloud = pts
                self.dirty = True
            except TransformException as exc:
                self.get_logger().warn(f"cloud frame transform failed: {exc}", throttle_duration_sec=10.0)
            except Exception:
                self.get_logger().error("cloud decode failed:\n" + traceback.format_exc(),
                                        throttle_duration_sec=10.0)

        def _robot(self):
            try:
                tf = self.buffer.lookup_transform(self.world, self.base, Time())
            except TransformException:
                return None
            t = tf.transform.translation
            return (t.x, t.y, t.z)

        def tick(self):
            if not self.dirty or self.map is None or self.cloud is None:
                return
            if time.monotonic() - self.last_run < self.period:
                return
            robot = self._robot()
            self.last_run = time.monotonic()
            self.dirty = False
            if robot is None:
                # No floor reference yet: keep the consumers on the current map rather than a stale one.
                self.pub_map.publish(self.map)
                self.pub_display.publish(self.map)
                return
            try:
                self.process(self.map, self.cloud, robot)
            except Exception:
                self.get_logger().error("negative obstacle filter failed:\n" + traceback.format_exc())
                self.pub_map.publish(self.map)
                self.pub_display.publish(self.map)

        def process(self, msg, cloud, robot):
            info = msg.info
            if info.width <= 0 or info.height <= 0 or len(msg.data) != info.width * info.height:
                return
            t0 = time.perf_counter()
            grid = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
            ox, oy = info.origin.position.x, info.origin.position.y
            yaw = _yaw(info.origin.orientation)
            # *****
            # Date: 2026-11-23
            # Editor: WangZX
            # Remark: Fall back to the last good floor estimate, not the robot z (0.17 m above the floor in habitat).
            # *****
            floor_z = estimate_floor(cloud, robot, fallback=getattr(self, "last_floor", None))
            self.last_floor = floor_z
            neg, flo = height_evidence(cloud, grid.shape, info.resolution, ox, oy, floor_z, self.params, yaw)
            # *****
            # Date: 2026-11-23
            # Editor: WangZX
            # Remark: Clear overhead-only occupied cells (a low door lintel) before anything else (overhead_cells).
            # *****
            overhead = np.zeros(grid.shape, bool)
            if self.params["clear_overhead"]:
                overhead = overhead_cells(grid, cloud, info.resolution, ox, oy, floor_z, flo, self.params, yaw)
                if overhead.any():
                    grid = grid.copy()
                    grid[overhead] = 0
            rr, rc, _, ins = points_to_cells(np.array([[robot[0], robot[1], 0.0]]), grid.shape,
                                             info.resolution, ox, oy, yaw)
            robot_rc = (int(rr[0]), int(rc[0])) if bool(ins[0]) else None
            # *****
            # Date: 2026-11-23
            # Editor: WangZX
            # Remark: Ramps: cells joined to the floor by gentle height steps are never masked.
            # *****
            ground_ok = continuous_ground(cloud, grid.shape, info.resolution, ox, oy, floor_z, self.params, yaw)
            mask = negative_mask(grid, neg, flo, info.resolution, self.params, robot_rc, ground_ok)
            for pub, value in ((self.pub_map, 100), (self.pub_display, -1)):
                out = OccupancyGrid()
                out.header = msg.header
                out.info = msg.info
                out.data = array.array("b", apply_mask(grid, mask, value).astype(np.int8).tobytes())
                pub.publish(out)
            ms = (time.perf_counter() - t0) * 1000.0
            n = int(mask.sum())
            stats = {"stamp": time.time(), "floor_z": round(floor_z, 3), "robot_z": round(float(robot[2]), 3),
                     "masked_cells": n, "masked_area_m2": round(n * info.resolution ** 2, 2),
                     "overhead_cleared_cells": int(overhead.sum()),
                     "cloud_points": int(len(cloud)), "ms": round(ms, 1)}
            self.pub_stats.publish(String(data=json.dumps(stats)))
            self.get_logger().info(
                f"negative obstacles: {n} cells ({stats['masked_area_m2']} m2), overhead cleared "
                f"{stats['overhead_cleared_cells']} cells, floor_z={floor_z:.2f} "
                f"points={len(cloud)} {ms:.0f} ms", throttle_duration_sec=10.0)

    return NegativeObstacleNode


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    try:
        node = _node_class()()
    except Exception:
        traceback.print_exc()
        from rclpy.node import Node
        node = Node("negative_obstacle_node")
        node.get_logger().error("negative obstacle filter disabled for this run: start-up failed (traceback above)")
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
