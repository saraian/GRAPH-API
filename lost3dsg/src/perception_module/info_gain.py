#!/usr/bin/env python3
# *****
# Date: 2026-11-19
# Editor: WangZX
# Remark: Interface skeleton for the information gain module (geometric ray-casting entropy reduction,
#         GP + BKIO estimator, weighted sum with the semantic gain). Not implemented yet.
# *****
# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: Geometric gain implemented as CRMI (Xu et al., RA-L 2021) with the sensor cause model (SCM):
#         360 deg virtual ray casting from each candidate viewpoint, per-cell mutual information between the
#         confidence-rich belief and the virtual measurement. The CR map is approximated from the rtabmap
#         occupancy grid (belief_from_grid, the replaceable layer). Semantic gain is a reserved interface
#         (weight 0). Pure numpy, no ROS import, so it is unit-testable.
# *****
"""Information gain = w_geo * geometric gain + w_sem * semantic gain.

Inputs are the frontiers produced by frontier_extractor.extract_frontiers (published as JSON on
/exploration/frontiers) and the occupancy grid they were extracted from.

Geometric gain (CRMI with SCM), per candidate viewpoint:
  * the occupancy grid is pooled to `cell_m` cells and every cell gets a discrete belief over its
    occupancy level m (n_bins values); unknown cells are uniform, cells rtabmap has observed are a
    discretised Beta with mean = occupancy/100 and concentration `known_confidence`;
  * `n_beams` virtual beams over 360 deg, each up to `max_range_m`;
  * along a beam with cells 1..n, the cause of the reading is the first hit:
        F_j = prod_{l<j} (1 - m^_l) * m^_j,   F_none = prod_l (1 - m^_l)
    and the reading o (z discretised by `z_step_m`, plus a max-range outcome) has the beam mixture
    likelihood p(o | cause) (hit gaussian + short exponential + random + max);
  * the posterior of cell j after reading o is b_j(m) * L_j(m, o), with the likelihood ratio
        L_j(m, o) = C_before + c_j * m / m^_j + S_after * (1 - m) / (1 - m^_j)
    where c_l = p(o | l) F_l / p(o), C_before = sum_{l<j} c_l, S_after = sum_{l>j} c_l (+ none);
    E_b[L_j] = 1;
  * I_j = H(b_j) - sum_o p(o) H(b_j | o) in bits, and the candidate gain sums I_j over the cells the
    scan reaches, each cell counted once with its best beam (`dedup_cells`; off = the paper's Eq. 3).
"""

import math
import time

import numpy as np

# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: Defaults of the CRMI geometric gain and of the utility; the node merges exploration.info_gain
#         (sensor / belief sub-sections flattened) over these.
# *****
DEFAULT_GAIN_PARAMS = {
    # utility
    "w_geo": 1.0,
    "w_sem": 0.0,                  # semantic gain reserved, disabled
    "distance_decay_per_m": 0.2,  # utility *= exp(-decay * distance)
    "unreachable_factor": 0.5,     # utility *= factor for a frontier not yet reachable
    # sensor (virtual 360 deg scan, beam mixture model)
    "max_range_m": 4.0,
    "n_beams": 180,
    "z_step_m": 0.1,
    "sigma_hit_m": 0.1,
    "z_hit": 0.7,
    "z_short": 0.1,
    "z_max": 0.1,
    "z_rand": 0.1,
    "lambda_short": 0.2,
    # belief (approximate confidence-rich map)
    "cell_m": 0.2,                 # gain grid resolution; rtabmap cells are pooled to this size
    "n_bins": 10,
    "known_confidence": 20.0,      # Beta concentration for cells rtabmap has observed
    "free_thresh": 25,             # same thresholds as frontier_extractor
    "occ_thresh": 65,
    "stop_prob": 1e-3,             # cells the beam reaches with lower probability are skipped
    "dedup_cells": True,
    "beam_batch": 30,
}

UNKNOWN_ROW = 101                  # belief table row for unknown cells (rows 0..100 = occupancy %)


def _params(params):
    out = dict(DEFAULT_GAIN_PARAMS)
    out.update({k: v for k, v in (params or {}).items() if v is not None})
    return out


def entropy_bits(pmf, axis=-1):
    """Discrete entropy in bits along `axis`."""
    p = np.clip(pmf, 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=axis)


# ------------------------------------------------------------------------------------------------
# Approximate confidence-rich map (the replaceable layer)
# ------------------------------------------------------------------------------------------------
def belief_table(n_bins=10, known_confidence=20.0):
    """(centers, table): table[v] is the belief for occupancy v% (0..100), table[101] for unknown."""
    centers = (np.arange(n_bins) + 0.5) / n_bins
    table = np.empty((102, n_bins))
    for v in range(101):
        mean = min(max(v / 100.0, 0.02), 0.98)
        a, b = mean * known_confidence, (1.0 - mean) * known_confidence
        logp = (a - 1.0) * np.log(centers) + (b - 1.0) * np.log(1.0 - centers)
        p = np.exp(logp - logp.max())
        table[v] = p / p.sum()
    table[UNKNOWN_ROW] = 1.0 / n_bins
    return centers, table


def pool_grid(grid, res, cell_m, free_thresh=25, occ_thresh=65):
    """Pool an occupancy grid to `cell_m` cells. Returns (pooled int16 grid, factor).

    A pooled cell is occupied (100) if any source cell is occupied, unknown (-1) if more than half of
    its source cells are unknown, otherwise free (0). Conservative: a thin wall survives pooling.
    """
    g = np.asarray(grid, dtype=np.int16)
    f = max(1, int(round(cell_m / res)))
    if f == 1:
        return g, 1
    h, w = g.shape
    H, W = -(-h // f), -(-w // f)
    pad = np.full((H * f, W * f), -1, np.int16)
    pad[:h, :w] = g
    blocks = pad.reshape(H, f, W, f).swapaxes(1, 2).reshape(H, W, f * f)
    occ = (blocks >= occ_thresh).any(axis=2)
    unk = (blocks < 0).mean(axis=2) > 0.5
    out = np.zeros((H, W), np.int16)
    out[unk] = -1
    out[occ] = 100
    return out, f


def belief_rows(grid):
    """Belief-table row for every cell of `grid`: occupancy % for known cells, UNKNOWN_ROW for -1."""
    g = np.asarray(grid, dtype=np.int16)
    return np.where(g < 0, UNKNOWN_ROW, np.clip(g, 0, 100)).astype(np.int16)


# ------------------------------------------------------------------------------------------------
# Ray casting
# ------------------------------------------------------------------------------------------------
def ray_cells(origin_rc, angle, n_steps, step_cells):
    """Ordered distinct cells (rows, cols, distance in cells) along one ray from a fractional origin.

    Cells outside the grid are kept: the caller treats them as unknown, so a candidate at the map edge
    is not penalised for looking into space rtabmap has not allocated yet.
    """
    t = np.arange(1, n_steps + 1) * step_cells
    r = np.floor(origin_rc[0] + t * math.sin(angle)).astype(np.int64)
    c = np.floor(origin_rc[1] + t * math.cos(angle)).astype(np.int64)
    keep = np.ones(len(t), bool)
    keep[1:] = (r[1:] != r[:-1]) | (c[1:] != c[:-1])
    r0, c0 = int(math.floor(origin_rc[0])), int(math.floor(origin_rc[1]))
    keep &= ~((r == r0) & (c == c0))          # the viewpoint's own cell is where the robot stands
    return r[keep], c[keep], t[keep]


# ------------------------------------------------------------------------------------------------
# SCM likelihood and per-beam CRMI
# ------------------------------------------------------------------------------------------------
def outcome_likelihood(dists_m, p):
    """p(o | cause) for causes at `dists_m` (shape (..., L)) plus the no-hit cause.

    Outcomes: n_z range bins of width z_step_m over [0, R) and one max-range outcome (index n_z).
    Returns (lik_hit (..., L, n_z + 1), lik_none (n_z + 1,)), each row normalised to 1.
    """
    R, dz = float(p["max_range_m"]), float(p["z_step_m"])
    n_z = int(math.ceil(R / dz))
    z = (np.arange(n_z) + 0.5) * dz
    sig, lam = float(p["sigma_hit_m"]), float(p["lambda_short"])
    d = np.asarray(dists_m, dtype=np.float64)[..., None]
    hit = np.exp(-0.5 * ((z - d) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
    short = np.where(z < d, lam * np.exp(-lam * z), 0.0)
    norm_short = 1.0 - np.exp(-lam * np.clip(d, 1e-6, None))
    short = short / norm_short
    rand = 1.0 / R
    dens = p["z_hit"] * hit + p["z_short"] * short + p["z_rand"] * rand
    lik = np.concatenate([dens * dz, np.full(d.shape[:-1] + (1,), 0.0) + p["z_max"]], axis=-1)
    lik = lik / lik.sum(axis=-1, keepdims=True)
    # no hit inside the range: the true reading is max range; short and random readings still possible
    short_n = lam * np.exp(-lam * z) / (1.0 - math.exp(-lam * R))
    none = np.concatenate([(p["z_short"] * short_n + p["z_rand"] * rand) * dz, [p["z_hit"] + p["z_max"]]])
    return lik, none / none.sum()


def beam_mi(pmf, valid, dists_m, centers, p):
    """CRMI of a batch of beams.

    pmf    (B, L, K) prior beliefs of the cells along each beam (padding rows arbitrary)
    valid  (B, L)    True for real cells
    dists_m (B, L)   distance of each cell from the viewpoint
    Returns (I (B, L) per-cell mutual information in bits, p_out (B, O) outcome distribution).
    """
    B, L, K = pmf.shape
    mhat = (pmf * centers).sum(-1)
    mhat = np.where(valid, np.clip(mhat, 1e-6, 1 - 1e-6), 0.0)
    passed = np.cumprod(np.concatenate([np.ones((B, 1)), 1.0 - mhat], axis=1), axis=1)
    pass_before, pass_all = passed[:, :L], passed[:, L]
    F = pass_before * mhat                                        # first-hit probability, (B, L)
    lik, lik_none = outcome_likelihood(dists_m, p)                # (B, L, O), (O,)
    joint = lik * F[..., None]                                    # p(o, cause = j)
    joint_none = pass_all[:, None] * lik_none[None, :]            # p(o, no hit)
    p_out = joint.sum(1) + joint_none                             # (B, O)
    c = joint / np.clip(p_out[:, None, :], 1e-300, None)          # posterior cause probability
    c_none = joint_none / np.clip(p_out, 1e-300, None)
    after = np.cumsum(c[:, ::-1, :], axis=1)[:, ::-1, :] - c + c_none[:, None, :]
    before = 1.0 - c - after
    m = centers[None, None, None, :]
    mh = mhat[..., None, None]
    ratio = (before[..., None] + c[..., None] * m / np.clip(mh, 1e-6, None)
             + after[..., None] * (1.0 - m) / np.clip(1.0 - mh, 1e-6, None))   # (B, L, O, K)
    post = pmf[:, :, None, :] * np.clip(ratio, 0.0, None)
    post = post / np.clip(post.sum(-1, keepdims=True), 1e-300, None)
    h_post = (p_out[:, None, :] * entropy_bits(post)).sum(-1)    # E_o[H(b | o)]
    I = entropy_bits(pmf) - h_post
    reach = valid & (pass_before >= p["stop_prob"])
    return np.where(reach, np.clip(I, 0.0, None), 0.0), p_out


# ------------------------------------------------------------------------------------------------
# Candidate-level gains
# ------------------------------------------------------------------------------------------------
class GeometricGain:
    """CRMI geometric gain of a 360 deg virtual scan at a candidate viewpoint (see module docstring)."""

    def __init__(self, params=None):
        self.p = _params(params)
        self.centers, self.table = belief_table(int(self.p["n_bins"]), float(self.p["known_confidence"]))
        self._grid_key = None

    def set_grid(self, grid, info):
        """Pool the occupancy grid once per map; `info` has resolution, origin_x, origin_y, yaw."""
        pooled, f = pool_grid(grid, info.resolution, self.p["cell_m"],
                              self.p["free_thresh"], self.p["occ_thresh"])
        self.rows = belief_rows(pooled)
        self.cell = info.resolution * f
        self.info = info

    def _origin_rc(self, xy):
        dx, dy = xy[0] - self.info.origin_x, xy[1] - self.info.origin_y
        c, s = math.cos(self.info.yaw), math.sin(self.info.yaw)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        return ly / self.cell, lx / self.cell

    def evaluate(self, viewpoint_xy):
        """Gain in bits at a viewpoint (map-frame metres). Requires set_grid() first."""
        p = self.p
        H, W = self.rows.shape
        o_rc = self._origin_rc(viewpoint_xy)
        n_steps = int(math.ceil(p["max_range_m"] / self.cell * 2))
        angles = np.linspace(0.0, 2 * math.pi, int(p["n_beams"]), endpoint=False)
        rays = [ray_cells(o_rc, a, n_steps, 0.5) for a in angles]
        L = max(1, max(len(r[0]) for r in rays))
        keys_all, mi_all = [], []
        batch = max(1, int(p["beam_batch"]))
        for s in range(0, len(rays), batch):
            chunk = rays[s:s + batch]
            B = len(chunk)
            rr = np.zeros((B, L), np.int64)
            cc = np.zeros((B, L), np.int64)
            dist = np.full((B, L), p["max_range_m"])
            valid = np.zeros((B, L), bool)
            for i, (r, c, t) in enumerate(chunk):
                n = len(r)
                rr[i, :n], cc[i, :n], dist[i, :n], valid[i, :n] = r, c, t * self.cell, True
            inside = valid & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            rows = np.full((B, L), UNKNOWN_ROW, np.int16)
            rows[inside] = self.rows[rr[inside], cc[inside]]
            I, _ = beam_mi(self.table[rows], valid, dist, self.centers, p)
            keep = I > 0
            keys_all.append((rr[keep] + 100000) * 1000003 + (cc[keep] + 100000))
            mi_all.append(I[keep])
        keys = np.concatenate(keys_all) if keys_all else np.zeros(0, np.int64)
        mis = np.concatenate(mi_all) if mi_all else np.zeros(0)
        if not len(mis):
            return 0.0
        if p["dedup_cells"]:
            uniq, inv = np.unique(keys, return_inverse=True)
            best = np.zeros(len(uniq))
            np.maximum.at(best, inv, mis)
            return float(best.sum())
        return float(mis.sum())


class SemanticGain:
    """Reserved semantic gain (LOST-3DSG similarity). Disabled: returns 0.

    `context` is where the semantic part will get its inputs: the ObjectNav target
    (CFG.task.objectnav.target), the object manager's objects near the candidate, and the
    LOST-3DSG similarity entry point (nlp_utils.lost_similarity).
    """

    def __init__(self, params=None):
        self.p = dict(params or {})

    def evaluate(self, candidate, context=None):
        return 0.0


class BKIEstimator:
    """GP + BKIO (Bayesian Kernel Inference Optimization) surrogate of the geometric gain.

    TODO: evaluate the exact gain only on a selected subset of candidates, estimate mean/variance of
    the others with a sparse BKI kernel over the evaluated samples, and pick the next exact samples
    with an acquisition function (e.g. UCB).
    """

    def __init__(self, length_scale_m=1.0, exact_budget=8):
        self.length_scale_m = length_scale_m
        self.exact_budget = exact_budget

    def select_exact(self, candidates):
        raise NotImplementedError("BKIEstimator.select_exact not implemented yet")

    def update(self, samples):
        raise NotImplementedError("BKIEstimator.update not implemented yet")

    def predict(self, candidates):
        raise NotImplementedError("BKIEstimator.predict not implemented yet")


# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: Default weights follow the config (geometric only, semantic weight 0).
# *****
def combine(geometric, semantic, w_geo=1.0, w_sem=0.0):
    """Weighted sum of the two gains (semantic gain comes from the LOST-3DSG similarity scores)."""
    return w_geo * float(geometric) + w_sem * float(semantic)


# *****
# Date: 2026-11-22
# Editor: WangZX
# Remark: Utility of a candidate: combined gain discounted by distance and, for a frontier not yet
#         reachable through known free space, by unreachable_factor.
# *****
def candidate_distance(frontier, robot_xy):
    """Path length when reachable, else the straight-line distance (0 without a robot pose)."""
    if frontier.get("reachable") and frontier.get("path_dist_m", -1) >= 0:
        return float(frontier["path_dist_m"])
    if robot_xy is None:
        return 0.0
    vp = frontier["viewpoint"]
    return float(math.hypot(vp[0] - robot_xy[0], vp[1] - robot_xy[1]))


def utility(geo, sem, distance_m, reachable, params=None):
    p = _params(params)
    u = combine(geo, sem, p["w_geo"], p["w_sem"]) * math.exp(-p["distance_decay_per_m"] * distance_m)
    return u if reachable else u * p["unreachable_factor"]


def evaluate_candidates(grid, info, frontier_payload, params=None, semantic=None, context=None):
    """Score every frontier of a /exploration/frontiers payload. Returns the candidates payload."""
    p = _params(params)
    t0 = time.perf_counter()
    geo = GeometricGain(p)
    geo.set_grid(grid, info)
    robot = frontier_payload.get("robot")
    use_sem = semantic is not None and p["w_sem"] != 0.0
    out = []
    for fr in frontier_payload.get("frontiers") or []:
        g = geo.evaluate(fr["viewpoint"])
        s = semantic.evaluate(fr, context) if use_sem else 0.0
        d = candidate_distance(fr, robot)
        out.append({"id": fr["id"], "viewpoint": fr["viewpoint"], "reachable": bool(fr.get("reachable")),
                    "distance_m": round(d, 3), "geo_gain_bits": round(g, 3), "sem_gain": round(s, 3),
                    "utility": round(utility(g, s, d, bool(fr.get("reachable")), p), 4)})
    out.sort(key=lambda c: c["utility"], reverse=True)
    for rank, c in enumerate(out):
        c["rank"] = rank
    return {"candidates": out, "best_id": out[0]["id"] if out else None,
            "compute_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "frontier_stamp": frontier_payload.get("stamp"),
            "weights": {"w_geo": p["w_geo"], "w_sem": p["w_sem"]}}
