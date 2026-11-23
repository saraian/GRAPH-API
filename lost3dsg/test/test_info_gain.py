# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: Unit tests of the CRMI geometric gain (info_gain.py) on synthetic occupancy grids (no ROS needed).
# *****
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "perception_module"))

from frontier_extractor import GridInfo  # noqa: E402
from info_gain import (DEFAULT_GAIN_PARAMS, UNKNOWN_ROW, GeometricGain, SemanticGain,  # noqa: E402
                       beam_mi, belief_table, entropy_bits, evaluate_candidates, outcome_likelihood,
                       pool_grid, utility)

RES = 0.05
INFO = GridInfo(RES, 0.0, 0.0)
FREE, OCC, UNK = 0, 100, -1


def test_unknown_belief_has_max_entropy_and_known_is_lower():
    centers, table = belief_table(10, 20.0)
    assert math.isclose(entropy_bits(table[UNKNOWN_ROW]), math.log2(10), rel_tol=1e-9)
    assert entropy_bits(table[0]) < 0.5 * math.log2(10)
    assert entropy_bits(table[100]) < 0.5 * math.log2(10)
    assert np.allclose(table.sum(axis=1), 1.0)


def test_outcome_distribution_sums_to_one():
    p = dict(DEFAULT_GAIN_PARAMS)
    centers, table = belief_table()
    rows = np.array([[UNKNOWN_ROW] * 8 + [100] * 4])           # unknown cells then a wall
    pmf = table[rows]
    dist = (np.arange(rows.shape[1]) + 1) * 0.2
    I, p_out = beam_mi(pmf, np.ones(rows.shape, bool), dist, centers, p)
    assert math.isclose(p_out.sum(), 1.0, rel_tol=1e-9)
    lik, none = outcome_likelihood(dist[None, :], p)
    assert np.allclose(lik.sum(-1), 1.0) and math.isclose(none.sum(), 1.0)
    assert (I >= 0).all()


def test_known_free_cells_carry_little_information():
    p = dict(DEFAULT_GAIN_PARAMS)
    centers, table = belief_table()
    dist = (np.arange(10) + 1) * 0.2
    free = table[np.zeros((1, 10), np.int16)]
    unk = table[np.full((1, 10), UNKNOWN_ROW, np.int16)]
    I_free, _ = beam_mi(free, np.ones((1, 10), bool), dist, centers, p)
    I_unk, _ = beam_mi(unk, np.ones((1, 10), bool), dist, centers, p)
    assert I_unk.sum() > 3 * I_free.sum()


def _room(open_side):
    """8 x 8 m map: a known free 4 x 4 m room in the middle, walls on 3 sides, unknown elsewhere."""
    g = np.full((160, 160), UNK, np.int16)
    g[40:120, 40:120] = FREE
    g[40:42, 40:120] = OCC
    g[118:120, 40:120] = OCC
    g[40:120, 40:42] = OCC
    if not open_side:
        g[40:120, 118:120] = OCC                                   # closed room
    return g


def test_viewpoint_facing_unknown_beats_closed_room():
    geo = GeometricGain()
    geo.set_grid(_room(open_side=True), INFO)
    g_open = geo.evaluate((5.7, 4.0))                              # by the open (east) side
    geo.set_grid(_room(open_side=False), INFO)
    g_closed = geo.evaluate((5.7, 4.0))
    assert g_open > 2 * g_closed


def test_known_wall_blocks_unknown_behind_it():
    # 10 x 10 m, known free except the east strip; the 4 m scan from (5, 5) stays inside the map,
    # so the east strip is the only unknown space it can see.
    g = np.full((200, 200), FREE, np.int16)
    g[:, 120:] = UNK
    with_wall = g.copy()
    with_wall[:, 116:119] = OCC                                    # wall between robot and unknown
    all_known = np.full((200, 200), FREE, np.int16)
    all_known[:, 116:119] = OCC
    geo = GeometricGain()
    gains = []
    for grid in (g, with_wall, all_known):
        geo.set_grid(grid, INFO)
        gains.append(geo.evaluate((5.0, 5.0)))
    open_gain, walled_gain, base = gains
    # Known cells keep a little uncertainty (finite confidence), so compare above that baseline.
    assert walled_gain - base < 0.2 * (open_gain - base)


def test_pooling_keeps_thin_walls_and_unknown_majority():
    g = np.zeros((8, 8), np.int16)
    g[0, 0] = OCC                                                  # one occupied cell in block (0,0)
    g[4:8, 4:8] = UNK                                              # block (1,1) fully unknown
    pooled, f = pool_grid(g, 0.05, 0.2)
    assert f == 4 and pooled[0, 0] == 100 and pooled[1, 1] == -1 and pooled[0, 1] == 0


def test_utility_discounts_distance_and_unreachable():
    assert utility(10, 0, 0.0, True) > utility(10, 0, 4.0, True)
    assert math.isclose(utility(10, 0, 2.0, False), 0.5 * utility(10, 0, 2.0, True))
    # semantic weight 0: the semantic gain does not change the utility
    assert utility(10, 99, 1.0, True) == utility(10, 0, 1.0, True)
    assert SemanticGain().evaluate({"id": 0}) == 0.0


def test_evaluate_candidates_ranks_and_uses_path_or_straight_distance():
    g = _room(open_side=True)
    payload = {"stamp": 1.0, "robot": [4.0, 4.0], "frontiers": [
        {"id": 0, "viewpoint": [5.7, 4.0], "reachable": True, "path_dist_m": 1.7},
        {"id": 1, "viewpoint": [3.0, 3.0], "reachable": False, "path_dist_m": -1.0},
    ]}
    out = evaluate_candidates(g, INFO, payload)
    assert out["best_id"] == 0
    by_id = {c["id"]: c for c in out["candidates"]}
    assert by_id[0]["distance_m"] == 1.7
    assert math.isclose(by_id[1]["distance_m"], math.hypot(1.0, 1.0), rel_tol=1e-3)
    assert [c["rank"] for c in out["candidates"]] == [0, 1]


def test_timing_25_candidates_on_400x400_grid():
    rng = np.random.default_rng(0)
    g = np.full((400, 400), UNK, np.int16)
    g[100:300, 100:300] = FREE
    g[rng.random(g.shape) < 0.01] = OCC
    payload = {"robot": [10.0, 10.0], "frontiers": [
        {"id": i, "viewpoint": [5.0 + 0.4 * i, 5.0 + 0.3 * i], "reachable": True, "path_dist_m": 1.0}
        for i in range(25)]}
    t0 = time.perf_counter()
    out = evaluate_candidates(g, INFO, payload)
    dt = time.perf_counter() - t0
    print(f"25 candidates on 400x400: {dt * 1000:.0f} ms (reported {out['compute_ms']} ms)")
    assert dt < 3.0
