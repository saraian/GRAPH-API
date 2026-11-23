#!/usr/bin/env python3
# *****
# Date: 2026-11-19
# Editor: WangZX
# Remark: Frontier extraction from a 2D occupancy grid (rtabmap /rtabmap/map). Pure numpy/cv2/skimage,
#         no ROS import, so it can be unit-tested without a ROS environment.
# *****
"""Frontier extraction on a 2D occupancy grid.

Pipeline (grid values follow nav_msgs/OccupancyGrid: -1 unknown, 0..100 occupancy %):
  1. classify cells into free / occupied / unknown with two thresholds;
  2. treat the camera blind zone around visited poses as seen; open the free mask to drop isolated
     free speckles, and ignore small unknown holes inside
     known space (both create fake frontiers);
  3. inflate occupied cells by the robot radius -> non-traversable band;
  4. frontier cell = traversable free cell with at least one unknown 8-neighbour;
  5. 8-connected clustering, clusters shorter than min_frontier_size_m are dropped;
  6. clusters longer than max_frontier_length_m are split along their principal axis;
  7. geodesic distance from the robot over traversable cells (skimage MCP); the viewpoint of a
     cluster is its reachable cell closest to the cluster centroid;
  8. every frontier is returned in world (map-frame) metres.

Grid layout: row index grows with map +y, column index with map +x, cell (r, c) centre is
origin + R(yaw) * ((c + 0.5) * res, (r + 0.5) * res).
"""

import math
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

DEFAULT_PARAMS = {
    "free_thresh": 25,              # 0 <= v < free_thresh -> free
    "occ_thresh": 65,               # v >= occ_thresh -> occupied
    "robot_radius_m": 0.20,         # obstacle inflation radius
    "min_frontier_size_m": 0.3,     # clusters shorter than this are noise
    "max_frontier_length_m": 2.0,   # longer clusters are split into equal parts
    "max_cells_per_frontier": 50,   # downsampling of the cell list in the output
    # *****
    # Date: 2026-11-19
    # Editor: WangZX
    # Remark: Live test fix. The RGB-D camera does not see the floor within ~1.2 m of the robot, so the robot
    #         stands in unknown cells: the start search radius grows to 1.5 m (the snap distance is added to
    #         the path length). rtabmap ray tracing leaves 1-3 cell unknown seams inside seen space: gaps up to
    #         fill_gap_m between known cells are filled before extraction.
    # *****
    "start_search_radius_m": 1.5,   # robot cell snapping when the robot sits in unknown / inflated cells
    "min_unknown_area_m2": 0.10,    # unknown blobs smaller than this (holes inside known space) are ignored
    "fill_gap_m": 0.15,             # unknown seams up to this width between known cells count as free
    # *****
    # Date: 2026-11-20
    # Editor: WangZX
    # Remark: Camera blind zone. The floor within ~0.7-1.0 m of a pose is never seen, so a ring of unknown
    #         cells surrounds every place the robot stood and its border read as a fake frontier. Unknown
    #         cells within blind_radius_m of a visited pose, in line of sight (no occupied cell in between),
    #         are treated as free. 0 disables it.
    # *****
    "blind_radius_m": 1.0,
}


@dataclass
class GridInfo:
    resolution: float
    origin_x: float
    origin_y: float
    yaw: float = 0.0


@dataclass
class Frontier:
    id: int
    cells: list                     # [[x, y], ...] world metres, downsampled
    centroid: list                  # [x, y]
    viewpoint: list                 # [x, y] reachable cell nearest to the centroid (or centroid)
    size_m: float                   # approx. length = cell count * resolution
    n_cells: int
    path_dist_m: float = -1.0       # geodesic distance robot -> viewpoint, -1 if unreachable
    reachable: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _params(params):
    out = dict(DEFAULT_PARAMS)
    out.update({k: v for k, v in (params or {}).items() if v is not None})
    return out


def cells_to_world(rows, cols, info):
    """Cell indices -> map-frame metres (cell centres)."""
    lx = (np.asarray(cols, dtype=np.float64) + 0.5) * info.resolution
    ly = (np.asarray(rows, dtype=np.float64) + 0.5) * info.resolution
    c, s = math.cos(info.yaw), math.sin(info.yaw)
    return info.origin_x + c * lx - s * ly, info.origin_y + s * lx + c * ly


def world_to_cell(x, y, info):
    """Map-frame metres -> (row, col), not clipped."""
    dx, dy = x - info.origin_x, y - info.origin_y
    c, s = math.cos(info.yaw), math.sin(info.yaw)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return int(math.floor(ly / info.resolution)), int(math.floor(lx / info.resolution))


def classify(grid, p):
    """Return (free, occupied, unknown) boolean masks."""
    g = np.asarray(grid, dtype=np.int16)
    unknown = g < 0
    occupied = g >= int(p["occ_thresh"])
    free = (g >= 0) & (g < int(p["free_thresh"]))
    return free, occupied, unknown


def _disk(radius_cells):
    r = max(0, int(radius_cells))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# *****
# Date: 2026-11-20
# Editor: WangZX
# Remark: Mask of unknown cells inside the camera blind zone of the visited poses (see blind_radius_m).
#         A cell is cleared only if the straight line from the pose to it crosses no occupied cell, so the
#         disk never reaches through a wall.
# *****
def blind_zone_mask(unknown, occupied, info, poses_xy, radius_m):
    h, w = unknown.shape
    out = np.zeros((h, w), bool)
    rad = int(math.ceil(radius_m / info.resolution))
    if rad <= 0 or not poses_xy:
        return out
    dr, dc = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    keep = (dr ** 2 + dc ** 2) <= rad * rad
    dr, dc = dr[keep], dc[keep]
    t = np.linspace(0.0, 1.0, rad + 1)[:-1]                     # samples strictly before the cell
    ray_r = np.rint(dr[:, None] * t[None, :]).astype(np.int64)  # (K, n) offsets along each ray
    ray_c = np.rint(dc[:, None] * t[None, :]).astype(np.int64)
    for x, y in poses_xy:
        r0, c0 = world_to_cell(x, y, info)
        if not (-rad <= r0 < h + rad and -rad <= c0 < w + rad):
            continue
        rr, cc = r0 + ray_r, c0 + ray_c
        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        hit = np.zeros(rr.shape, bool)
        hit[inside] = occupied[rr[inside], cc[inside]]
        tr, tc = r0 + dr, c0 + dc
        ok = ~hit.any(axis=1) & (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
        out[tr[ok], tc[ok]] = True
    return out & unknown


def frontier_mask(grid, info, params=None, visited_xy=None):
    """Steps 1-4. Returns (frontier, traversable) boolean masks."""
    p = _params(params)
    free, occupied, unknown = classify(grid, p)
    k3 = np.ones((3, 3), np.uint8)
    # *****
    # Date: 2026-11-20
    # Editor: WangZX
    # Remark: Clear the camera blind zone around the visited poses before anything else.
    # *****
    if visited_xy and p["blind_radius_m"] > 0:
        seen = blind_zone_mask(unknown, occupied, info, visited_xy, p["blind_radius_m"])
        free = free | seen
        unknown = unknown & ~seen
    # *****
    # Date: 2026-11-19
    # Editor: WangZX
    # Remark: Close the known mask to fill thin unknown seams left by ray tracing (see fill_gap_m).
    # *****
    gap = int(round(p["fill_gap_m"] / info.resolution))
    if gap > 0:
        kgap = np.ones((gap + 1, gap + 1), np.uint8)
        known_closed = cv2.morphologyEx((~unknown).astype(np.uint8), cv2.MORPH_CLOSE, kgap).astype(bool)
        filled = unknown & known_closed
        free = free | (filled & ~occupied)
        unknown = unknown & ~filled
    min_area = int(round(p["min_unknown_area_m2"] / (info.resolution ** 2)))
    if min_area > 1:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(unknown.astype(np.uint8), connectivity=8)
        small = np.zeros(n, bool)
        small[1:] = stats[1:, cv2.CC_STAT_AREA] < min_area
        unknown = unknown & ~small[lab]
    free = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_OPEN, k3).astype(bool)
    inflate = _disk(math.ceil(p["robot_radius_m"] / info.resolution))
    blocked = cv2.dilate(occupied.astype(np.uint8), inflate).astype(bool)
    traversable = free & ~blocked
    near_unknown = cv2.dilate(unknown.astype(np.uint8), k3).astype(bool)
    return traversable & near_unknown, traversable


def _split_cluster(rows, cols, max_cells):
    """Split one cluster into equal parts along its principal axis."""
    n = len(rows)
    parts = int(math.ceil(n / float(max_cells))) if max_cells > 0 else 1
    if parts <= 1:
        return [(rows, cols)]
    pts = np.stack([cols, rows], axis=1).astype(np.float64)
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    proj = centred @ vt[0]
    order = np.argsort(proj)
    return [(rows[idx], cols[idx]) for idx in np.array_split(order, parts) if len(idx)]


def _geodesic_from(traversable, start_rc, p, res):
    """Geodesic cost (metres) over traversable cells; inf where unreachable. None without a start."""
    from skimage.graph import MCP_Geometric

    h, w = traversable.shape
    r0, c0 = start_rc
    snap_m = 0.0
    if not (0 <= r0 < h and 0 <= c0 < w) or not traversable[r0, c0]:
        # The robot often sits inside the inflated band next to a wall, or in the unknown blind zone in
        # front of the camera: snap to the nearest traversable cell within start_search_radius_m.
        rad = int(math.ceil(p["start_search_radius_m"] / res))
        r_lo, r_hi = max(0, r0 - rad), min(h, r0 + rad + 1)
        c_lo, c_hi = max(0, c0 - rad), min(w, c0 + rad + 1)
        if r_lo >= r_hi or c_lo >= c_hi:
            return None
        rr, cc = np.nonzero(traversable[r_lo:r_hi, c_lo:c_hi])
        if len(rr) == 0:
            return None
        rr, cc = rr + r_lo, cc + c_lo
        d2 = (rr - r0) ** 2 + (cc - c0) ** 2
        k = int(np.argmin(d2))
        snap_m = math.sqrt(float(d2[k])) * res
        r0, c0 = int(rr[k]), int(cc[k])
    costs = np.where(traversable, 1.0, np.inf)
    cum, _ = MCP_Geometric(costs).find_costs([(r0, c0)])
    return cum * res + snap_m


# *****
# Date: 2026-11-20
# Editor: WangZX
# Remark: visited_xy (poses the robot stood at, map-frame metres) feeds the blind-zone clearing; the current
#         robot pose is always included.
# *****
def extract_frontiers(grid, info, robot_xy=None, params=None, visited_xy=None):
    """Steps 1-8. `grid` is an (H, W) int array, `info` a GridInfo, `robot_xy` map-frame metres."""
    p = _params(params)
    res = float(info.resolution)
    grid = np.asarray(grid, dtype=np.int16)
    poses = list(visited_xy or [])
    if robot_xy is not None:
        poses.append(tuple(robot_xy))
    fmask, traversable = frontier_mask(grid, info, p, poses)

    _, labels = cv2.connectedComponents(fmask.astype(np.uint8), connectivity=8)
    min_cells = max(1, int(round(p["min_frontier_size_m"] / res)))
    max_cells = int(round(p["max_frontier_length_m"] / res))

    dist = None
    if robot_xy is not None:
        dist = _geodesic_from(traversable, world_to_cell(robot_xy[0], robot_xy[1], info), p, res)

    frontiers = []
    rr, cc = np.nonzero(labels)
    lab_of = labels[rr, cc]
    order = np.argsort(lab_of, kind="stable")
    rr, cc, lab_of = rr[order], cc[order], lab_of[order]
    bounds = np.flatnonzero(np.diff(lab_of)) + 1
    for rows, cols in zip(np.split(rr, bounds), np.split(cc, bounds)):
        if len(rows) < min_cells:
            continue
        for prow, pcol in _split_cluster(rows, cols, max_cells):
            if len(prow) < min_cells:
                continue
            xs, ys = cells_to_world(prow, pcol, info)
            cx, cy = float(xs.mean()), float(ys.mean())
            # Viewpoint: reachable cell nearest to the centroid; else the cell nearest to it.
            d2 = (xs - cx) ** 2 + (ys - cy) ** 2
            reachable, path = False, -1.0
            pick = int(np.argmin(d2))
            if dist is not None:
                dcell = dist[prow, pcol]
                ok = np.isfinite(dcell)
                if ok.any():
                    reachable = True
                    pick = int(np.argmin(np.where(ok, d2, np.inf)))
                    path = float(dcell[pick])
            step = max(1, int(math.ceil(len(xs) / float(p["max_cells_per_frontier"]))))
            frontiers.append(Frontier(
                id=len(frontiers),
                cells=[[round(float(x), 3), round(float(y), 3)] for x, y in zip(xs[::step], ys[::step])],
                centroid=[round(cx, 3), round(cy, 3)],
                viewpoint=[round(float(xs[pick]), 3), round(float(ys[pick]), 3)],
                size_m=round(len(prow) * res, 3),
                n_cells=int(len(prow)),
                path_dist_m=round(path, 3),
                reachable=reachable,
            ))
    return frontiers
