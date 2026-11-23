# *****
# Date: 2026-11-23
# Editor: WangZX
# Remark: Unit tests of the negative obstacle (drop-off) filter, negative_obstacle_node.py, on a synthetic
#         stairwell: upper floor, a ray-traced void over the stairwell, the floor below. No ROS needed.
# *****
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "perception_module"))

from frontier_extractor import GridInfo, extract_frontiers  # noqa: E402
from negative_obstacle_node import (DEFAULT_NEG_PARAMS, apply_mask, estimate_floor, height_evidence,  # noqa: E402
                                    negative_mask)

RES = 0.05
FREE, OCC, UNK = 0, 100, -1


def _scene(floor_z=0.0):
    """11 x 6 m grid. x 0..6 upper floor (floor points at floor_z), x 6..7 void (free, no points, as ray tracing
    leaves it), x 7..10 the floor below (points at floor_z - 3), x 10..11 unknown. Walls on y = 0 and y = 6."""
    g = np.full((120, 220), FREE, np.int16)
    g[:, 200:] = UNK
    g[0:2, :200] = OCC
    g[-2:, :200] = OCC
    pts = []
    for r in range(2, 118):
        for c in range(0, 200):
            x, y = (c + 0.5) * RES, (r + 0.5) * RES
            if x < 6.0:
                pts.append((x, y, floor_z))
            elif x >= 7.0:
                pts += [(x, y, floor_z - 3.0)] * 2
    # The lip: the last upper-floor column also sees two points of the floor below.
    for r in range(2, 118):
        pts += [((119 + 0.5) * RES, (r + 0.5) * RES, floor_z - 3.0)]
    return g, np.asarray(pts, float)


def _mask(g, pts, floor_z=0.0, robot_rc=(60, 40)):
    neg, flo = height_evidence(pts, g.shape, RES, 0.0, 0.0, floor_z)
    return negative_mask(g, neg, flo, RES, robot_rc=robot_rc)


def test_mask_covers_the_floor_below_and_the_void_but_not_the_upper_floor():
    g, pts = _scene()
    m = _mask(g, pts)
    xs = (np.arange(220) + 0.5) * RES
    inner = slice(5, 115)
    assert m[inner][:, xs >= 7.0].mean() > 0.99 or m[inner][:, (xs >= 7.0) & (xs < 10.0)].all()
    assert m[inner][:, (xs >= 6.0) & (xs < 7.0)].all()          # ray-traced void over the stairwell
    assert not m[:, xs < 6.0].any()                             # upper floor, including the lip column
    assert not m[:, xs >= 10.0].any()                           # unknown stays as it is


def test_robot_on_an_upper_storey():
    g, pts = _scene(floor_z=2.9)
    assert abs(estimate_floor(pts, (2.0, 3.0, 2.9)) - 2.9) < 1e-6
    m = _mask(g, pts, floor_z=2.9)
    xs = (np.arange(220) + 0.5) * RES
    assert m[5:115][:, (xs >= 7.0) & (xs < 10.0)].all() and not m[:, xs < 6.0].any()


def test_output_values_and_frontiers_downstairs_are_gone():
    g, pts = _scene()
    m = _mask(g, pts)
    algo, disp = apply_mask(g, m, 100), apply_mask(g, m, -1)
    assert (algo[m] == 100).all() and (disp[m] == -1).all() and (algo[~m] == g[~m]).all()
    info = GridInfo(RES, 0.0, 0.0)
    raw = extract_frontiers(g, info, (2.0, 3.0))
    assert any(f.centroid[0] > 9.0 for f in raw)                 # the downstairs edge before filtering
    filt = extract_frontiers(algo, info, (2.0, 3.0))
    assert not any(f.centroid[0] > 5.5 for f in filt), [f.centroid for f in filt]


def test_no_points_below_the_floor_means_no_mask():
    g, pts = _scene()
    flat = pts[pts[:, 2] > -1.0]
    assert not _mask(g, flat).any()


def test_cells_near_the_robot_are_never_masked():
    g, pts = _scene()
    m = _mask(g, pts, robot_rc=(60, 150))                      # robot standing over the floor below (x = 7.5)
    rr, cc = np.ogrid[:120, :220]
    # *****
    # Date: 2026-11-23
    # Editor: WangZX
    # Remark: Use the configured default keep radius instead of a hard-coded 0.6 m.
    # *****
    kr = float(DEFAULT_NEG_PARAMS["keep_radius_m"]) / RES
    assert not m[(rr - 60) ** 2 + (cc - 150) ** 2 <= kr ** 2].any()


def test_timing_400x400_200k_points():
    rng = np.random.default_rng(0)
    g = np.zeros((400, 400), np.int16)
    pts = np.column_stack([rng.uniform(0, 20, 200000), rng.uniform(0, 20, 200000),
                           np.where(rng.random(200000) < 0.2, -3.0, 0.0)])
    t0 = time.perf_counter()
    neg, flo = height_evidence(pts, g.shape, RES, 0.0, 0.0, 0.0)
    negative_mask(g, neg, flo, RES, robot_rc=(200, 200))
    dt = time.perf_counter() - t0
    print(f"evidence + mask: {dt * 1000:.0f} ms")
    assert dt < 0.2


# *****
# Date: 2026-11-23
# Editor: WangZX
# Remark: False-positive regression tests: a voxelised cloud (one point per cell, as /rtabmap/cloud_map), lone
#         noise points below the floor, and holes inside the floor next to a single noise seed.
# *****
def _sparse_scene():
    g, pts = _scene()
    _, idx = np.unique(np.floor(pts / [RES, RES, 1.0]).astype(np.int64), axis=0, return_index=True)
    return g, pts[np.sort(idx)]


def test_voxelised_cloud_still_masks_the_stairwell():
    g, pts = _sparse_scene()
    m = _mask(g, pts)
    xs = (np.arange(220) + 0.5) * RES
    assert m[5:115][:, (xs >= 6.0) & (xs < 10.0)].mean() > 0.97
    assert not m[:, xs < 6.0].any()


def test_lone_noise_points_below_the_floor_are_ignored():
    g = np.zeros((120, 120), np.int16)
    rng = np.random.default_rng(1)
    floor = [((c + 0.5) * RES, (r + 0.5) * RES, 0.0) for r in range(120) for c in range(120)]
    noise = np.column_stack([rng.uniform(0.5, 5.5, 15), rng.uniform(0.5, 5.5, 15), np.full(15, -0.8)])
    pts = np.vstack([np.asarray(floor), noise, noise + [0.001, 0.0, 0.0]])  # two noise points per cell
    assert not _mask(g, pts, robot_rc=(10, 10)).any()


def test_growth_does_not_run_through_holes_in_the_floor():
    g = np.zeros((120, 120), np.int16)
    rng = np.random.default_rng(2)
    floor = [((c + 0.5) * RES, (r + 0.5) * RES, 0.0) for r in range(120) for c in range(120) if rng.random() > 0.3]
    blob = [((c + 0.5) * RES, (r + 0.5) * RES, -1.0) for r in range(58, 63) for c in range(58, 63)]
    m = _mask(g, np.asarray(floor + blob), robot_rc=(10, 10))
    assert m.sum() < 60, m.sum()      # the 5x5 blob (+ closing), not the 30 % of empty floor cells around it
