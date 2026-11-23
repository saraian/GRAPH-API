# *****
# Date: 2026-11-19
# Editor: WangZX
# Remark: Unit tests of frontier_extractor on synthetic occupancy grids (no ROS needed).
# *****
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "perception_module"))

from frontier_extractor import GridInfo, extract_frontiers, world_to_cell  # noqa: E402

RES = 0.05
INFO = GridInfo(RES, 0.0, 0.0)
FREE, OCC, UNK = 0, 100, -1


def corridor():
    """Free corridor x in [0.5, 3.0) m, y in [1.0, 2.0) m, walls on y sides, unknown beyond x = 3.0."""
    g = np.full((60, 100), UNK, np.int16)
    g[18:42, 8:60] = OCC          # walls band
    g[20:40, 10:60] = FREE        # corridor interior (open towards +x where it meets unknown)
    return g


def test_corridor_end_gives_one_reachable_frontier():
    fr = extract_frontiers(corridor(), INFO, robot_xy=(0.8, 1.5))
    assert len(fr) == 1
    f = fr[0]
    assert f.reachable and f.path_dist_m > 0
    # *****
    # Date: 2026-11-19
    # Editor: WangZX
    # Remark: Seam filling (fill_gap_m) may close one extra cell at the corridor mouth, so allow one cell past 3.0 m.
    # *****
    # the frontier sits at the open end of the corridor (x close to 3.0 m)
    assert 2.8 < f.centroid[0] < 3.1
    assert 1.0 < f.centroid[1] < 2.0


def test_cells_next_to_walls_are_excluded():
    fr = extract_frontiers(corridor(), INFO, robot_xy=(0.8, 1.5), params={"robot_radius_m": 0.2})
    ys = [y for f in fr for _, y in f.cells]
    # walls at y = 0.9..1.0 and 2.0..2.1, inflation 0.2 m -> frontier y inside (1.2, 1.8)
    assert min(ys) > 1.15 and max(ys) < 1.85


def test_small_noise_cluster_is_dropped():
    g = np.full((60, 60), FREE, np.int16)
    g[30, 30] = UNK                # single unknown hole -> tiny ring of frontier cells
    fr = extract_frontiers(g, INFO, robot_xy=(0.5, 0.5), params={"min_frontier_size_m": 0.5})
    assert fr == []


def test_frontier_behind_wall_is_unreachable():
    g = corridor()
    g[20:40, 30:33] = OCC         # wall across the corridor at x = 1.5 m
    fr = extract_frontiers(g, INFO, robot_xy=(0.8, 1.5))
    assert len(fr) == 1 and not fr[0].reachable and fr[0].path_dist_m == -1.0


def test_long_frontier_is_split():
    g = np.full((200, 100), UNK, np.int16)
    g[:, :50] = FREE               # 10 m long straight free/unknown border at x = 2.5 m
    fr = extract_frontiers(g, INFO, robot_xy=(1.0, 5.0), params={"max_frontier_length_m": 2.0})
    assert len(fr) >= 5
    assert all(f.size_m <= 2.0 + 1e-6 for f in fr)
    assert all(f.reachable for f in fr)


# *****
# Date: 2026-11-19
# Editor: WangZX
# Remark: Live-test regressions: robot standing in the unknown camera blind zone, and thin ray-tracing seams.
# *****
def test_robot_in_blind_zone_still_reaches_frontier():
    g = corridor()
    g[20:40, 10:30] = UNK          # first 1.0 m in front of the robot never seen
    fr = extract_frontiers(g, INFO, robot_xy=(0.8, 1.5))
    # the blind block's own border is also a (real) frontier; the corridor end must still be reachable
    end = [f for f in fr if f.centroid[0] > 2.8]
    assert len(end) == 1 and end[0].reachable and end[0].path_dist_m > 0.5


def test_ray_tracing_seams_do_not_make_frontiers():
    g = corridor()
    for c in range(14, 56, 6):
        g[20:40, c] = UNK          # 1-cell unknown seams across the corridor
    fr = extract_frontiers(g, INFO, robot_xy=(0.8, 1.5))
    assert len(fr) == 1 and fr[0].centroid[0] > 2.8


def test_world_cell_roundtrip():
    r, c = world_to_cell(1.23, 0.47, INFO)
    assert (r, c) == (9, 24)


# *****
# Date: 2026-11-20
# Editor: WangZX
# Remark: Camera blind zone: the unknown ring around a visited pose must not produce a frontier, and the
#         clearing must not reach through a wall.
# *****
def room_with_blind_ring():
    """6 x 6 m room fully seen except a 0.8 m unknown disk around the robot at (3, 3)."""
    g = np.full((120, 120), FREE, np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = OCC
    rr, cc = np.mgrid[0:120, 0:120]
    g[(rr - 60) ** 2 + (cc - 60) ** 2 <= 16 ** 2] = UNK
    return g


def test_blind_ring_around_robot_is_not_a_frontier():
    g = room_with_blind_ring()
    assert len(extract_frontiers(g, INFO, robot_xy=(3.0, 3.0), params={"blind_radius_m": 0.0})) >= 1
    assert extract_frontiers(g, INFO, robot_xy=(3.0, 3.0)) == []


def test_blind_ring_of_an_earlier_pose_is_cleared():
    g = room_with_blind_ring()
    assert extract_frontiers(g, INFO, robot_xy=(1.0, 1.0), visited_xy=[(3.0, 3.0)]) == []


def test_blind_zone_does_not_clear_through_a_wall():
    g = np.full((60, 120), UNK, np.int16)
    g[:, :60] = FREE
    g[:, 60] = OCC                 # wall at x = 3.0 m, unknown room behind it
    g[:, 61:63] = FREE             # a sliver seen behind the wall keeps a frontier there
    fr = extract_frontiers(g, INFO, robot_xy=(2.6, 1.5), params={"robot_radius_m": 0.0})
    assert fr and all(f.centroid[0] > 3.0 for f in fr)
    for f in fr:
        assert any(x > 3.1 for x, _ in f.cells)
