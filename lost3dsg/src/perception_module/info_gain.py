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
#         occupancy grid (belief_from_grid, the replaceable layer). Semantic gain is enabled for
#         ObjectNav and remains off by default for other tasks. Pure numpy, no ROS import.
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
    "w_sem": 0.0,                  # off by default; ObjectNav supplies task.objectnav.semantic_weight
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
    """LOST-3DSG similarity between an ObjectNav target and visible graph objects."""

    def __init__(self, params=None):
        self.p = dict(params or {})
        self.target = None

    def set_target(self, target):
        self.target = dict(target or {})

    def score_visible_objects(self, visible, context=None):
        """Return the best LOST score and per-object evidence for a set of visible nodes."""
        context = context or {}
        target = self.target or context.get("target_fields") or {}
        target_label = str(target.get("label") or "").strip()
        compare = context.get("lost_similarity_detailed")
        label_compare = context.get("semantic_similarity")
        embed = context.get("get_embedding")
        model = context.get("embedding_model")
        if not target_label or compare is None or embed is None or model is None:
            return 0.0, []

        target_description = target.get("description")
        target_embedding = target.get("description_embedding")
        if target_embedding is None and target_description:
            target_embedding = embed(model, target_description)
            target["description_embedding"] = target_embedding

        scored, best = [], 0.0
        for row in visible or []:
            obj = row.get("_node") if isinstance(row, dict) else None
            if not isinstance(obj, dict):
                continue
            object_label = str(obj.get("label") or "").split("#", 1)[0].strip()
            if not object_label or object_label.lower() in {"unknown", "none", "n/a"}:
                continue
            description = obj.get("description")
            cache = context.setdefault("object_embeddings", {})
            cache_key = (str(obj.get("object_id") or object_label), str(description or ""))
            if cache_key not in cache:
                cache[cache_key] = embed(model, description) if description else None
            score, evidence = compare(
                model,
                target_label, object_label,
                target.get("color"), obj.get("color"),
                target.get("material"), obj.get("material"),
                target_embedding, cache[cache_key],
            )
            score = float(score)
            label_score = (float(label_compare(model, target_label, object_label))
                           if label_compare is not None else 0.0)
            best = max(best, score)
            scored.append({"object_id": obj.get("object_id"), "label": object_label,
                           "bbox": obj.get("bbox"),
                           "color": obj.get("color"), "material": obj.get("material"),
                           "description": description,
                           "room_id": obj.get("room_id", "unknown"),
                           "distance_m": row.get("distance_m"),
                           "lost_similarity": round(score, 4),
                           "label_similarity": round(label_score, 4), "evidence": evidence})
        return best, scored

    def evaluate(self, candidate, context=None):
        context = context or {}
        rows = objects_in_camera_view(candidate.get("viewpoint", ()), context.get("objects"),
                                      context.get("camera"), include_nodes=True)
        return self.score_visible_objects(rows, context)[0]


def _ray_box_hit(origin, target, box, eps=1e-6):
    """Whether a segment from camera origin to target enters an AABB before its end."""
    try:
        t0, t1 = 0.0, 1.0 - eps
        for axis in "xyz":
            lo, hi = float(box[f"{axis}_min"]), float(box[f"{axis}_max"])
            d = target["xyz".index(axis)] - origin["xyz".index(axis)]
            o = origin["xyz".index(axis)]
            if abs(d) < eps:
                if o < lo or o > hi:
                    return False
                continue
            a, b = (lo - o) / d, (hi - o) / d
            if a > b:
                a, b = b, a
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1:
                return False
        return t1 >= t0
    except (KeyError, TypeError, ValueError):
        return False


def _map_occludes(origin, point, box, grid, info, threshold=65):
    """Check known occupied map cells between the camera and the near XY face of a target."""
    if grid is None or info is None:
        return False
    try:
        res = float(info.resolution)
        dx, dy = point[0] - origin[0], point[1] - origin[1]
        t0, t1 = 0.0, 1.0
        for o, d, lo, hi in ((origin[0], dx, box[0][0], box[0][1]),
                             (origin[1], dy, box[1][0], box[1][1])):
            if abs(d) < 1e-9:
                if o < lo or o > hi:
                    return False
                continue
            a, b = (lo - o) / d, (hi - o) / d
            if a > b:
                a, b = b, a
            t0, t1 = max(t0, a), min(t1, b)
            if t0 > t1:
                return False
        stop = max(0.0, t0 - res / max(math.hypot(dx, dy), res))
        steps = max(1, int(math.ceil(stop * math.hypot(dx, dy) / max(res * 0.5, 1e-4))))
        c, s = math.cos(float(getattr(info, "yaw", 0.0))), math.sin(float(getattr(info, "yaw", 0.0)))
        g = np.asarray(grid)
        for i in range(1, steps + 1):
            t = stop * i / steps
            wx, wy = origin[0] + t * dx, origin[1] + t * dy
            lx, ly = c * (wx - info.origin_x) + s * (wy - info.origin_y), \
                     -s * (wx - info.origin_x) + c * (wy - info.origin_y)
            col, row = int(math.floor(lx / res)), int(math.floor(ly / res))
            if 0 <= row < g.shape[0] and 0 <= col < g.shape[1] and g[row, col] >= threshold:
                return True
        return False
    except (AttributeError, TypeError, ValueError, IndexError):
        return False


def objects_in_camera_view(viewpoint, objects, camera=None, include_nodes=False):
    """Return persistent object nodes visible in at least one pose of a 360-degree scan.

    The camera is modelled with its configured HFOV/pitch and a panoramic sequence of yaw poses.
    A target is included when part of its 3D AABB projects inside the image and a ray to that
    visible part is not blocked by known occupied map cells or another scene-graph object's AABB.
    This is a geometric first pass; the camera pans through a full 360-degree scan.
    """
    camera = camera or {}
    try:
        x, y = float(viewpoint[0]), float(viewpoint[1])
        height = float(camera.get("height_m", 1.5))
        hfov = math.radians(float(camera.get("hfov_deg", 90.0)))
        vfov = math.radians(float(camera.get("vfov_deg", 60.0)))
        pitch = math.radians(float(camera.get("pitch_deg", -30.0)))
        max_range = float(camera.get("max_range_m", 4.0))
        yaw_step = max(math.radians(5), min(hfov / 2.0, math.radians(30)))
    except (TypeError, ValueError, IndexError):
        return []
    if hfov <= 0 or vfov <= 0 or max_range <= 0:
        return []
    origin = (x, y, height)
    rows = []
    for obj in objects or []:
        if not isinstance(obj, dict) or not isinstance(obj.get("bbox"), dict):
            continue
        box = obj["bbox"]
        try:
            bounds = [(float(box[f"{a}_min"]), float(box[f"{a}_max"])) for a in "xyz"]
            if any(lo > hi for lo, hi in bounds):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        corners = [(a, b, c) for a in bounds[0] for b in bounds[1] for c in bounds[2]]
        center = tuple((lo + hi) / 2.0 for lo, hi in bounds)
        # Face-centres help with broad boxes whose corners all fall outside a narrow image.
        points = corners + [center]
        points += [((bounds[0][0] + bounds[0][1]) / 2, yy, zz)
                   for yy in bounds[1] for zz in bounds[2]]
        points += [(xx, (bounds[1][0] + bounds[1][1]) / 2, zz)
                   for xx in bounds[0] for zz in bounds[2]]
        points += [(xx, yy, (bounds[2][0] + bounds[2][1]) / 2)
                   for xx in bounds[0] for yy in bounds[1]]
        distance = math.dist(origin, center)
        rows.append((obj, bounds, points, distance))

    visible = []
    for obj, bounds, points, distance in rows:
        if distance > max_range:
            continue
        seen = False
        for point in points:
            ray = tuple(point[i] - origin[i] for i in range(3))
            rng = math.sqrt(sum(v * v for v in ray))
            if rng <= 1e-6 or rng > max_range:
                continue
            if _map_occludes(origin, point, bounds, camera.get("grid"), camera.get("grid_info"),
                             camera.get("occ_thresh", 65)):
                continue
            for yaw in (i * yaw_step for i in range(int(math.ceil(2 * math.pi / yaw_step)))):
                # Camera optical axes: forward is horizontal yaw with the configured down pitch.
                forward = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw),
                           math.sin(pitch))
                right = (-math.sin(yaw), math.cos(yaw), 0.0)
                up = (-math.sin(pitch) * math.cos(yaw), -math.sin(pitch) * math.sin(yaw),
                      math.cos(pitch))
                z = sum(ray[i] * forward[i] for i in range(3))
                if z <= 0:
                    continue
                px = sum(ray[i] * right[i] for i in range(3)) / z
                py = sum(ray[i] * up[i] for i in range(3)) / z
                if abs(px) > math.tan(hfov / 2) or abs(py) > math.tan(vfov / 2):
                    continue
                blocked = any(other is not obj and
                              _ray_box_hit(origin, point, other_box)
                              for other, other_box, _, _ in rows)
                if not blocked:
                    seen = True
                    break
            if seen:
                break
        if seen:
            row = {"object_id": obj.get("object_id"), "label": obj.get("label", ""),
                   "room_id": obj.get("room_id", "unknown"),
                   "distance_m": round(distance, 3)}
            if include_nodes:
                row["_node"] = obj
            visible.append(row)
    return visible


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
# Remark: Default weights follow the config (geometric only outside ObjectNav).
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
    semantic_context = context or {}
    out = []
    for fr in frontier_payload.get("frontiers") or []:
        g = geo.evaluate(fr["viewpoint"])
        s = 0.0
        d = candidate_distance(fr, robot)
        candidate = {"id": fr["id"], "viewpoint": fr["viewpoint"], "reachable": bool(fr.get("reachable")),
                     "distance_m": round(d, 3), "geo_gain_bits": round(g, 3), "sem_gain": round(s, 3),
                     "utility": round(utility(g, s, d, bool(fr.get("reachable")), p), 4)}
        if semantic_context.get("objects") is not None:
            camera = dict(semantic_context.get("camera") or {})
            camera["grid"], camera["grid_info"] = grid, info
            visible = objects_in_camera_view(fr["viewpoint"], semantic_context["objects"], camera,
                                             include_nodes=use_sem)
            if use_sem:
                s, candidate["visible_objects"] = semantic.score_visible_objects(visible, semantic_context)
            else:
                candidate["visible_objects"] = visible
        candidate["sem_gain"] = round(s, 4)
        candidate["utility"] = round(utility(g, s, d, bool(fr.get("reachable")), p), 4)
        out.append(candidate)
    out.sort(key=lambda c: c["utility"], reverse=True)
    for rank, c in enumerate(out):
        c["rank"] = rank
    return {"candidates": out, "best_id": out[0]["id"] if out else None,
            "compute_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "frontier_stamp": frontier_payload.get("stamp"),
            "weights": {"w_geo": p["w_geo"], "w_sem": p["w_sem"]}}
