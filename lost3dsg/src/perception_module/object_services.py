#!/usr/bin/env python3

import hashlib
import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from types import SimpleNamespace

import association as assoc
import numpy as np
import rclpy
from bbox_fusion import (
    apply_request_fusion,
    fusion_summary,
    merge_bbox_fusion,
    reset_bbox_fusion,
)
from box_view import box_corners_map, enclosing_box_from_points
from builtin_interfaces.msg import Time as TimeMsg
from config import CFG, world_frame
from cv_utils import publish_persistent_centroids, publish_pov_volume, set_marker_from_bbox
from detection_index import DetectionIndex
from detection_types import observation_dict_from_msg
from hooks import DecisionLog, load_store
from map_database import MapDatabase
from nlp_utils import _known, get_embedding, lost_similarity_detailed, world2vec
from object_info import Object
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from room_manager import RoomManager
from tf2_ros import Buffer, TransformListener

# Explicit, not `import *`. Only the names this file does NOT define itself:
# publish_persistent_bboxes is defined BELOW and also in cv_utils with a different body,
# so importing it here would swap a 34-line implementation for a 5-line wrapper (GA-77).
from utils import compute_iou_3d
from visualization_msgs.msg import Marker, MarkerArray
from world_model import wm

from lost3dsg.srv import (
    AddObject,
    DeleteObjects,
    MergeObjects,
    QueryObjects,
    RemoveObject,
    UpdateObject,
)

# =============  EXPLORATION PARAMETERS (config.yaml: association) =============
EXPLORATION_IOU_THRESHOLD = CFG["association"]["exploration_iou_threshold"]
SIM_THRESHOLD = CFG["association"]["sim_threshold"]
TRACKING_IOU_THRESHOLD = CFG["association"]["tracking_iou_threshold"]
VOLUME_EXPANSION_RATIO = CFG["association"]["volume_expansion_ratio"]
EXPLORATION_FRAME_LIMIT = CFG["association"]["exploration_frame_limit"]
OBJECT_STABILITY_TIMEOUT = CFG["association"]["object_stability_timeout"]
POV_SCALE_FACTOR = CFG["association"]["pov_scale_factor"]
# GA-26: consecutive in-view misses before an object is deleted from the map. Was the bare
# literal 5 at the one site that used it -- a destructive threshold that could not be
# changed without editing source, in a file whose other six thresholds are config.
MAX_MISSES_BEFORE_DELETE = CFG["association"].get("max_misses_before_delete", 5)
# GA-26: the remaining literals of the destructive branches. A scene with wardrobes could
# not be run without editing source: every wardrobe box was "suspicious" forever.
SUSPICIOUS_MAX_EXTENT_M = CFG["association"].get("suspicious_max_extent_m", 3.0)
SUSPICIOUS_MAX_VOLUME_M3 = CFG["association"].get("suspicious_max_volume_m3", 1.5)
UPDATE_IN_PLACE_DISTANCE_M = CFG["association"].get("update_in_place_distance_m", 0.5)
UNCERTAIN_MOVE_DISTANCE_M = CFG["association"].get("uncertain_move_distance_m", 0.8)

# GA-06. Merge is the most destructive operation in the system -- it ends one object's
# identity -- and it ran on the LOOSEST gate: two hardcoded literals, 0.8 m and 0.75,
# against a match threshold of 0.85. So the association loop refused a pair at 0.80,
# correctly keeping two identities, and merge fused them anyway on the next callback.
#
# Measured on the mp3d bundle: 283 merges against 334 admissions on a 51-object scene, an
# 85% applied merge rate, with the similarity spread bottoming out at 0.754 -- just above
# the 0.75 literal, which is where a too-loose gate shows.
#
# A fusion must demand MORE evidence than a match, not less. The default sits midway
# between the match bar and certainty rather than at an invented constant: at
# sim_threshold 0.85 that is 0.925. It is a knob, and it is the one to turn if the merge
# rate is still high.
MERGE_MAX_DISTANCE = CFG["association"].get("merge_max_distance_m", 0.8)
# A very close, substantially overlapping re-detection of the same base label is stronger
# identity evidence than a fluctuating VLM description. This is a separate geometry-
# dominant lane rather than a lower global similarity threshold: distant or merely nearby
# objects still have to satisfy the ordinary semantic gate.
MERGE_NEAR_DISTANCE = float(CFG["association"].get("merge_near_distance_m", 0.02))
MERGE_NEAR_IOU_THRESHOLD = float(CFG["association"].get(
    "merge_near_iou_threshold", 0.25))
if MERGE_NEAR_DISTANCE < 0.0:
    raise ValueError(
        f"association.merge_near_distance_m ({MERGE_NEAR_DISTANCE}) must be >= 0")
if not 0.0 <= MERGE_NEAR_IOU_THRESHOLD <= 1.0:
    raise ValueError(
        f"association.merge_near_iou_threshold ({MERGE_NEAR_IOU_THRESHOLD}) must be in [0, 1]")
# AABB broad-phase radius for the legacy merge criterion.  It is deliberately at least the
# criterion's centre-distance threshold: a smaller margin could hide a valid merge before the
# exact distance/similarity checks get to evaluate it.  This is a candidate-generation value,
# not a second merge criterion.
MERGE_AABB_MARGIN_M = float(CFG["association"].get(
    "merge_aabb_margin_m", MERGE_MAX_DISTANCE))
if MERGE_AABB_MARGIN_M < MERGE_MAX_DISTANCE:
    raise ValueError(
        f"association.merge_aabb_margin_m ({MERGE_AABB_MARGIN_M}) must be >= "
        f"merge_max_distance_m ({MERGE_MAX_DISTANCE}) so the AABB broad phase cannot hide "
        "a pair that the legacy merge criterion would evaluate")
# GA-101: how many of the three OPTIONAL terms (colour, material, description) must have
# been comparable for a merge to be allowed. 0 restores the old behaviour, where a pair
# with nothing measurable scored 1.0000 on label agreement alone and merged.
MERGE_MIN_EVIDENCE = CFG["association"].get("merge_min_evidence", 1)
MERGE_MIN_SIMILARITY = CFG["association"].get(
    "merge_min_similarity", SIM_THRESHOLD + (1.0 - SIM_THRESHOLD) / 2.0)

# Asserted at load, not trusted. Two config values that must stay ordered will not, and an
# inversion is invisible: it looks exactly like the behaviour this fix removes.
if MERGE_MIN_SIMILARITY <= SIM_THRESHOLD:
    raise ValueError(
        f"association.merge_min_similarity ({MERGE_MIN_SIMILARITY}) must be STRICTLY greater "
        f"than association.sim_threshold ({SIM_THRESHOLD}): merging two objects destroys an "
        f"identity and must demand more evidence than matching them, never less")


def _merge_base_label(label):
    """Return the identity label used by the geometry-dominant merge lane."""
    return str(label or "").split("#", 1)[0].strip().casefold()


def _near_geometry_duplicate(a, b, distance):
    """Whether a close same-label pair has enough 3D overlap to bypass VLM similarity.

    The distance check alone is intentionally insufficient: two small, adjacent objects can
    have nearby centres without being the same object. Requiring AABB overlap keeps this
    exception tied to the same measured volume rather than turning it into a broad distance
    relaxation. Invalid geometry follows the normal gate.
    """
    if distance > MERGE_NEAR_DISTANCE:
        return False, None
    if _merge_base_label(getattr(a, "label", None)) != _merge_base_label(
            getattr(b, "label", None)):
        return False, None
    try:
        iou = float(compute_iou_3d(a.bbox, b.bbox))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False, None
    if not math.isfinite(iou):
        return False, iou
    return iou >= MERGE_NEAR_IOU_THRESHOLD, iou

# GA-186. Which association engine decides a merge.
#
#   "legacy"   -- the hard-gate cascade: AABB candidates, then room-inequality -> similarity
#                 < MERGE_MIN_SIMILARITY -> evidence -> distance > MERGE_MAX_DISTANCE, each
#                 a refusal on a calibrated constant. The AABB stage is a broad phase and
#                 does not change any of those exact merge criteria.
#   "evidence" -- association.py: kNN candidates from each object's OWN covariance shell,
#                 then fused log-odds over overlap / separation / co-visibility / ontology
#                 / appearance / room, committed against log(cost_ratio). No calibrated
#                 similarity threshold, and abstention is a first-class outcome.
#
# DEFAULT IS "legacy" ON PURPOSE: flipping the engine changes every decision the system
# makes, and it must be an explicit act in a run config, not something a reader of this
# file discovers afterwards in a bundle. Set association.merge_engine or MERGE_ENGINE=evidence.
MERGE_ENGINE = (os.environ.get("MERGE_ENGINE")
                or CFG["association"].get("merge_engine", "legacy")).strip().lower()
if MERGE_ENGINE not in ("legacy", "evidence"):
    raise ValueError(f"association.merge_engine must be 'legacy' or 'evidence', got {MERGE_ENGINE!r}")

# The cost of a FALSE MERGE relative to a MISSED MERGE. The only judgement call in the
# evidence engine, and deliberately a config value rather than a threshold: a duplicate is
# visible and repairable, a wrong merge destroys an identity, so it sits well above 1.
# commit_threshold() turns it into log-odds -- there is no similarity constant to tune.
MERGE_COST_RATIO = float(CFG["association"].get("merge_cost_ratio", 20.0))
# 2026-09-14 owner ruling (merge-algorithm lane): ontology channel off for this paper; the
# attribute score in as a bounded channel (association.channel_attributes). Both declared in
# config.yaml beside merge_engine.
MERGE_ONTOLOGY_CHANNEL = bool(CFG["association"].get("merge_ontology_channel", True))
MERGE_ATTRIBUTE_MAX_LOG_ODDS = float(CFG["association"].get("merge_attribute_max_log_odds", 2.0))
# Asserted at load, not trusted (review 2026-09-14): "attributes alone can never commit a merge"
# holds only while the cap is below log(cost_ratio), and two config values that must stay
# ordered will not by themselves.
if MERGE_ATTRIBUTE_MAX_LOG_ODDS >= math.log(MERGE_COST_RATIO):
    raise ValueError(
        f"association.merge_attribute_max_log_odds ({MERGE_ATTRIBUTE_MAX_LOG_ODDS}) must be below "
        f"log(merge_cost_ratio) = {math.log(MERGE_COST_RATIO):.3f}: attributes alone must not commit")
# The one locality test both stages apply: boxes intersect or their largest per-axis gap is at
# most this. It is the association broad-phase margin, not a new constant.
LOCALITY_GAP_M = float(CFG["association"].get("association_margin_m", 0.3))
# How many nearest neighbours per object survive candidate generation. None = no cap, keep
# everything inside the covariance shell. A cap that silently drops a pair is the defect
# that lost the air-conditioner pair, so generate_candidates reports what the cap removed.
_knn_k = CFG["association"].get("merge_knn_k", None)
MERGE_KNN_K = None if _knn_k in (None, 0, "", "none") else int(_knn_k)
# GA-188: how many CONSECUTIVE sweeps a pair must clear the commit threshold before the
# merge is applied. 1 restores commit-on-first-sighting. Above 1, a transient geometry
# error has to survive being re-measured before it can destroy an identity -- which is the
# design's answer to "one lucky frame must not commit a merge", and it is enforced by
# persistence rather than by inflating a score through repetition.
# Env overrides config, so an ablation changes ONE variable per run instead of editing a
# shared file between two runs that must otherwise be identical. Recorded into
# run_metadata by run_sim.sh, so a bundle states which arm produced it.
MERGE_MIN_CONSECUTIVE = int(os.environ.get(
    "MERGE_MIN_CONSECUTIVE", CFG["association"].get("merge_min_consecutive", 2)))

file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))

# world2vec is imported explicitly above -- loaded once in nlp_utils.
OPERATIONS_LOG = CFG["paths"]["operations_log"]

def resolve_output_root():
    return (os.environ.get("GRAPH_API_OUTPUT_DIR")
            or os.environ.get("LOST3DSG_OUTPUT_DIR")
            or os.path.join(PROJECT_ROOT, "output"))


log_dir = resolve_output_root()
os.makedirs(log_dir, exist_ok=True)
SYNTHETIC_LOG_FILE = os.path.join(log_dir, "operations.txt")


def _apply_orientation(bbox, request):
    """GA-312. Both service handlers used to rebuild a six-key box, so an object written through
    the API lost its tilt while one written in-process kept it, and the persisted map could not
    say which geometry a stored box was. Every stored box now carries `has_orientation`
    explicitly -- False is a statement, not an absence -- and the tilt when there is one."""
    if getattr(request, "has_orientation", False):
        bbox["has_orientation"] = True
        bbox["yaw"] = float(request.yaw)
        bbox["oriented_center"] = [float(v) for v in request.oriented_center]
        bbox["oriented_extents"] = [float(v) for v in request.oriented_extents]
    else:
        bbox["has_orientation"] = False


def _axial_delta(a, b):
    """Smallest angle between two box AXES (period pi), radians, >= 0."""
    d = (a - b) % math.pi
    return min(d, math.pi - d)


def _is_oriented(bbox):
    return bool(bbox) and "yaw" in bbox and bool(bbox.get("oriented_extents"))


def _bbox_points(bbox):
    """Return a bounded point representation that preserves the whole stored box.

    An oriented box is not guaranteed to contain every corner of the separate AABB when the
    two were produced by different stages or by an older build, so where the AABB carries
    extent the OBB does not, BOTH corner sets are kept and the next exact enclosure is a
    superset of both representations instead of letting the OBB path shrink the merge geometry.

    A REDUNDANT AABB IS DROPPED, and that is not a refinement -- it is the difference between
    fusion keeping the yaw and losing it. For a correctly produced box the AABB simply IS the
    axis-aligned bound of the OBB, so its eight corners add no extent and instead surround the
    OBB with an axis-aligned copy of itself. The union footprint is then close to square,
    ``pca_oriented_box`` refuses it for want of anisotropy, and ``fuse_orientation`` returns a
    box with no ``yaw`` at all. MEASURED on a 2.0 x 1.0 box: the oriented corners alone give an
    anisotropy of 2.00 and the PCA accepts at every yaw; unioned with a derived AABB the PCA
    REFUSES across roughly 30 to 60 degrees, so every object whose axis falls in that band
    silently loses its orientation on the next fused view.
    """
    b = bbox or {}
    points = []
    corners, oriented = box_corners_map(b)
    has_aabb = all(k in b for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"))
    aabb_corners = [[x, y, z]
                    for x in (b["x_min"], b["x_max"])
                    for y in (b["y_min"], b["y_max"])
                    for z in (b["z_min"], b["z_max"])] if has_aabb else []
    if corners:
        points.extend(np.asarray(corners, dtype=np.float64))
    if aabb_corners and not (oriented and _aabb_is_the_bound_of(aabb_corners, corners)):
        points.extend(np.asarray(aabb_corners, dtype=np.float64))
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _aabb_is_the_bound_of(aabb_corners, obb_corners, tol=1e-6):
    """Is this AABB just the axis-aligned bound of those oriented corners?

    If so it is derivable and carries no extent of its own. If it reaches outside that bound
    even on one axis it holds real geometry -- the case ``_bbox_points`` keeps both sets for.
    """
    a = np.asarray(aabb_corners, dtype=np.float64)
    o = np.asarray(obb_corners, dtype=np.float64)
    if a.size == 0 or o.size == 0:
        return False
    return bool(np.allclose(a.min(axis=0), o.min(axis=0), atol=tol)
                and np.allclose(a.max(axis=0), o.max(axis=0), atol=tol))


def _object_geometry_points(obj):
    """Return bounded-size geometry retained for a persistent object.

    New accumulators keep the corners of the current enclosing box, which represents all prior
    points without growing with every frame. Older accumulators did not have ``points``; their
    persisted box is the only geometry available after an upgrade.
    """
    acc = getattr(obj, "_yaw_acc", None) if obj is not None else None
    if isinstance(acc, dict) and acc.get("points"):
        try:
            points = np.asarray(acc["points"], dtype=np.float64)
            if points.ndim == 2 and points.shape[1] == 3 and len(points):
                return points
        except (TypeError, ValueError):
            pass
    return _bbox_points(getattr(obj, "bbox", None) if obj is not None else None)


def _geometry_accumulator(box, oriented_views=0, axial=None, view=None):
    """Build a bounded accumulator for one already coherent box.

    `axial` is the running (cos 2yaw, sin 2yaw) SUM over the accepted views and `view` the
    representative. Both are carried in, never re-derived: an accumulator that recomputes them
    from the box it is describing reports the current yaw wearing the name of an average.
    """
    points = _bbox_points(box)
    oriented = _is_oriented(box)
    yaw = float(box.get("yaw", 0.0)) if oriented else 0.0
    if view is None:
        view = ({"yaw": yaw,
                 "oriented_center": list(box["oriented_center"]),
                 "oriented_extents": list(box["oriented_extents"])}
                if oriented else None)
    if axial is None:
        axial = (oriented_views * math.cos(2.0 * yaw), oriented_views * math.sin(2.0 * yaw))
    return {
        # c/s are the AXIAL SUM over the accepted views, so 0.5*atan2(s, c) is the axial mean
        # (GA-315 part 2). A box axis has period pi, which is why the doubled angle is summed
        # rather than the yaw: a scalar mean of views either side of +-90 degrees lands on the
        # wrong axis entirely.
        "n": int(oriented_views),
        "c": float(axial[0]),
        "s": float(axial[1]),
        "view": view,
        "points": points.tolist(),
    }


def _enclose_geometry(points, include_orientation):
    """Fit one exact AABB and, when stable, one exact PCA OBB to a point/corner union."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        return None
    return enclosing_box_from_points(points, include_orientation=include_orientation)


def _combine_object_geometry(a, b):
    """Return a box and accumulator that enclose both object geometries."""
    pa = _object_geometry_points(a)
    pb = _object_geometry_points(b)
    points = (np.concatenate((pa, pb), axis=0) if len(pa) and len(pb)
              else (pa if len(pa) else pb))
    include_orientation = (_is_oriented(getattr(a, "bbox", None))
                           or _is_oriented(getattr(b, "bbox", None))
                           or int((getattr(a, "_yaw_acc", None) or {}).get("n", 0)) > 0
                           or int((getattr(b, "_yaw_acc", None) or {}).get("n", 0)) > 0)
    merged = _enclose_geometry(points, include_orientation=include_orientation)
    if merged is None:
        # Malformed geometry must not turn a valid identity merge into a service failure.
        merged = dict(getattr(a, "bbox", None) or getattr(b, "bbox", None) or {})
    elif include_orientation and not _is_oriented(merged):
        # The combined AABB can be nearly isotropic even when one input has a valid PCA box.
        # Do not erase that measured orientation merely because the enclosing union cannot
        # choose a stable new axis; keep an actual input OBB for RViz and for the next update.
        source = (getattr(a, "bbox", None) if _is_oriented(getattr(a, "bbox", None))
                  else getattr(b, "bbox", None))
        if _is_oriented(source):
            for key in ("yaw", "oriented_center", "oriented_extents"):
                merged[key] = source[key]
            merged["has_orientation"] = True

    def count(obj):
        acc = getattr(obj, "_yaw_acc", None)
        old = int(acc.get("n", 0)) if isinstance(acc, dict) else 0
        return old if old else int(_is_oriented(getattr(obj, "bbox", None)))

    return merged, _geometry_accumulator(merged, count(a) + count(b))


def _legacy_fuse_orientation(obj, bbox):
    """GA-315 part 2. -> (the box to store, the accumulator to store beside it as `_yaw_acc`).

    Replaces last-write-wins on the yaw. Measured over 20260903_230232: of 40 objects with an
    early and a final yaw, 11 differed by more than 20 degrees, every one by whole-dict
    replacement (`obj.bbox = bbox`), so the LAST close-range view -- the clipped wedge whose
    PCA axis is its hypotenuse (GA-315) -- overwrote every good far view before it. With part 1
    a clipped view arrives with NO yaw, and the same replacement then erased a good yaw with a
    yaw-less dict instead of a 45-degree one (rule 15, named by the ontology lane).

    The fused yaw is the AXIAL mean of every accepted view: the mean of (cos 2yaw, sin 2yaw),
    halved. A box axis has period pi, and a scalar mean of yaws either side of +-90 degrees
    lands on the wrong axis (53 degrees on the audit's synthetic sequence; 0 for views at +85
    and -85). `oriented_center` / `oriented_extents` stay those of ONE measured view -- never a
    synthesised box (GA-20's rule): the representative is replaced by an arriving view when
    that view lies at least as close to the fused axis as the representative does AT THAT
    MOMENT, so it is the view nearest the axis as the views came, not the nearest in
    hindsight (on the audit's four far views it keeps the view 3.3 degrees off the axis while
    a later one sits 1.2 degrees off). `yaw_view` is that view's own yaw, so a reader can see
    by how many degrees the stored extents were measured off the stored axis; `yaw_views` is
    the count fused.

    A view without a yaw updates the AABB and leaves the axis where the accepted views put it.
    `has_orientation` says whether ANY accepted view exists. Keys are added, never removed.
    ponytail: one representative view is kept, greedily; keep every view's box if the
    representative ever needs re-choosing after a merge. A merge keeps the keeper's accumulator
    and drops the discard's views (the keeper's box is the one kept anyway, GA-20).
    """
    acc = getattr(obj, "_yaw_acc", None) if obj is not None else None
    if acc is None:
        acc = {"n": 0, "c": 0.0, "s": 0.0, "view": None}
        prior = getattr(obj, "bbox", None) if obj is not None else None
        if _is_oriented(prior):
            acc = _acc_add(acc, prior)
    out = dict(bbox)
    if _is_oriented(bbox):
        acc = _acc_add(acc, bbox)
    else:
        acc = dict(acc)
    if acc["n"] == 0:
        return out, acc
    fused = 0.5 * math.atan2(acc["s"], acc["c"])
    view = acc["view"]
    if _is_oriented(bbox) and (view is None
                               or _axial_delta(float(bbox["yaw"]), fused)
                               <= _axial_delta(float(view["yaw"]), fused)):
        view = {"yaw": float(bbox["yaw"]),
                "oriented_center": [float(v) for v in bbox["oriented_center"]],
                "oriented_extents": [float(v) for v in bbox["oriented_extents"]]}
    if view is None:
        # A LEGACY accumulator: n/c/s written by an older build, no representative. The object's
        # own persisted box is a REAL measured view, so seed from it rather than inventing extents
        # or dropping an orientation the object legitimately has. If there is no prior either, the
        # count has no subject: keep the axis out of the box instead of fabricating one.
        prior_box = getattr(obj, "bbox", None) if obj is not None else None
        if not _is_oriented(prior_box):
            acc["view"] = None
            return out, acc
        view = {"yaw": float(prior_box["yaw"]),
                "oriented_center": [float(v) for v in prior_box["oriented_center"]],
                "oriented_extents": [float(v) for v in prior_box["oriented_extents"]]}
    acc["view"] = view
    out["yaw"] = float(fused)
    out["oriented_center"] = list(view["oriented_center"])
    out["oriented_extents"] = list(view["oriented_extents"])
    out["yaw_view"] = view["yaw"]
    out["yaw_views"] = acc["n"]
    out["has_orientation"] = True
    return out, acc


def _acc_add(acc, bbox):
    """Add one ORIENTED view to the axial accumulator.

    The representative is seeded HERE when there is none. Seeding the accumulator from an
    object's persisted box incremented `n` without ever storing a view, so the next CLIPPED
    (yaw-less) arrival found `n > 0` and `view is None` and subscripted None. A count and a
    representative are one fact and must move together.
    """
    th = 2.0 * float(bbox["yaw"])
    view = acc["view"]
    if view is None:
        view = {"yaw": float(bbox["yaw"]),
                "oriented_center": [float(v) for v in bbox["oriented_center"]],
                "oriented_extents": [float(v) for v in bbox["oriented_extents"]]}
    return {"n": acc["n"] + 1, "c": acc["c"] + math.cos(th), "s": acc["s"] + math.sin(th),
            "view": view}


def _rotate_z(points, angle):
    """Rotate (N,3) points about the world z axis. Positive angle is world -> frame-of-yaw."""
    c, s = math.cos(angle), math.sin(angle)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    out = np.empty_like(pts)
    out[:, 0] = c * pts[:, 0] - s * pts[:, 1]
    out[:, 1] = s * pts[:, 0] + c * pts[:, 1]
    out[:, 2] = pts[:, 2]
    return out


def _merge_in_object_frame(points, yaw):
    """Union geometry IN THE OBJECT'S OWN FRAME, then rotate the result back to the world.

    Owner design, 2026-09-14: keep both the rotated and the unrotated representation, merge the
    unrotated one, and re-rotate afterwards.

    WHY IT MATTERS, and it is not only about the yaw. A world-axis-aligned box around a ROTATED
    object is larger than the object -- that is what an AABB of an oriented box is. Unioning those
    world AABBs across views compounds the excess every time, so the stored box grows with the
    number of observations rather than converging on the object. Rotating into the object frame
    first makes the union exact: the merged extents are the true extents along the object's own
    axes, and re-rotating restores the oriented box with no inflation.

    It also removes the PCA from this path entirely. `pca_oriented_box` declines a fit whose
    footprint is close to square, which is exactly what a multi-view union looks like, and a
    declined fit silently dropped the orientation. Here the axis is the axial mean that is already
    being accumulated, so no fit can decline and no orientation can be lost.
    """
    local = _rotate_z(points, -yaw)
    lo, hi = local.min(axis=0), local.max(axis=0)
    centre_local = (lo + hi) / 2.0
    extents = hi - lo
    centre_world = _rotate_z(centre_local.reshape(1, 3), yaw)[0]
    # The world AABB is the bound of the re-rotated corners, so the two representations stay
    # consistent with each other rather than being measured independently.
    corners_local = np.array([[x, y, z] for x in (lo[0], hi[0])
                              for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=np.float64)
    corners_world = _rotate_z(corners_local, yaw)
    w_lo, w_hi = corners_world.min(axis=0), corners_world.max(axis=0)
    return {
        "x_min": float(w_lo[0]), "x_max": float(w_hi[0]),
        "y_min": float(w_lo[1]), "y_max": float(w_hi[1]),
        "z_min": float(w_lo[2]), "z_max": float(w_hi[2]),
        "yaw": float(yaw),
        "oriented_center": [float(v) for v in centre_world],
        "oriented_extents": [float(v) for v in extents],
    }


def fuse_orientation(obj, bbox):
    """Return one coherent enclosing box for the object's retained observations.

    The former implementation averaged yaw independently from the dimensions of one
    representative view. That produced a box whose extents were measured in one orientation
    but rendered at another. We now union the previous and incoming box corners, fit PCA once to
    that union, and use exact projections. The accumulator retains only the current eight
    corners, so its memory use is bounded.
    """
    acc = getattr(obj, "_yaw_acc", None) if obj is not None else None
    previous = getattr(obj, "bbox", None) if obj is not None else None
    prior_points = _object_geometry_points(obj) if obj is not None else np.empty((0, 3))
    incoming_points = _bbox_points(bbox)
    points = (np.concatenate((prior_points, incoming_points), axis=0)
              if len(prior_points) and len(incoming_points)
              else (prior_points if len(prior_points) else incoming_points))
    old_n = int(acc.get("n", 0)) if isinstance(acc, dict) else int(_is_oriented(previous))
    n = old_n + int(_is_oriented(bbox))

    # The axial sum over every accepted view, carried forward rather than re-derived.
    if isinstance(acc, dict) and ("c" in acc or "s" in acc):
        axial = [float(acc.get("c", 0.0)), float(acc.get("s", 0.0))]
        rep = acc.get("view")
    elif _is_oriented(previous):
        py = float(previous["yaw"])
        axial = [math.cos(2.0 * py), math.sin(2.0 * py)]
        rep = {"yaw": py,
               "oriented_center": list(previous["oriented_center"]),
               "oriented_extents": list(previous["oriented_extents"])}
    else:
        axial, rep = [0.0, 0.0], None
    if _is_oriented(bbox):
        by = float(bbox["yaw"])
        axial = [axial[0] + math.cos(2.0 * by), axial[1] + math.sin(2.0 * by)]
    fused_yaw = 0.5 * math.atan2(axial[1], axial[0]) if n else None
    # The representative is a REAL measured view, replaced when an arriving one sits nearer the
    # fused axis (GA-20: the stored extents are always some view's, never synthesised).
    if _is_oriented(bbox) and fused_yaw is not None and (
            rep is None or _axial_delta(float(bbox["yaw"]), fused_yaw)
            <= _axial_delta(float(rep["yaw"]), fused_yaw)):
        rep = {"yaw": float(bbox["yaw"]),
               "oriented_center": [float(v) for v in bbox["oriented_center"]],
               "oriented_extents": [float(v) for v in bbox["oriented_extents"]]}

    # Merge unrotated, then re-rotate. An oriented input can no longer produce an unoriented
    # output: the axis is the axial mean above, not a fit that may decline.
    if len(points) == 0:
        out = dict(bbox or previous or {})
    elif fused_yaw is not None:
        out = _merge_in_object_frame(points, fused_yaw)
    else:
        out = _enclose_geometry(points, include_orientation=False) or dict(bbox or previous or {})

    new_acc = _geometry_accumulator(out, n, axial=axial, view=rep)
    if _is_oriented(out):
        # yaw_view names the REPRESENTATIVE view's own yaw. When the PCA fit succeeded the two
        # agree; when it declined and the axial mean answered, they differ, and that difference
        # is exactly the thing a reader needs to see rather than have hidden.
        out["yaw_view"] = float(rep["yaw"]) if rep is not None else float(out["yaw"])
        out["yaw_views"] = n
        out["has_orientation"] = True
    else:
        out["has_orientation"] = False
    return out, new_acc


# GA-314. Credibility order of the admission grades for the merge survivor rule. An ungraded
# object (no hook wrote a verdict) ranks with no_grounds: both mean "nothing checked", and
# inventing a fifth level for it would be a constant nobody measured.
GRADE_RANK = {"admit": 0, "hold": 1, "no_grounds": 2, "decline": 3}


def merge_rank(o):
    """Which of two merging objects survives: the LOWER tuple is the keeper.

    GA-314. Measured over 20260904_192014: in 3 of the 5 merges the DELETED side was graded
    `admit` and the survivor `decline`, because the rank read only the description and the
    age -- and the grade never reached the object at all. It does now (object_manager_6
    copies it beside `entity`), and it ranks FIRST: a declined box is the one the envelope
    says is not the object it claims to be, so it must not absorb a credible neighbour.

    GA-372. Equal grades are broken on `admission_filled` (owner ruling 2026-09-08): how many of
    the verdict's eight property slots the proposal carried. Measured on 152446's 24
    equal-grade merges: the alignment score ties on 23 (same class, same score), the
    admission count is 1 on every object, sightings live only in memory; `filled` decides
    17 and reverses 4 age picks. An object without the field ranks as 0 filled -- nothing
    checked -- the same reading as an ungraded object.

    After that the pre-GA-314 rule stands unchanged (GA-25): a described object
    outranks an undescribed one; then a KNOWN age ranks ahead of an unknown one, so an
    unstamped object cannot claim seniority it has no evidence for (reading absent as 0 is
    the GA-12 defect: the newcomer always wins); then the ESTABLISHED identity outlives the
    newcomer, D14's prefer-strict reading; then object_id, so two objects created in the
    same tick still resolve deterministically.
    """
    grade = GRADE_RANK.get(getattr(o, "admission_grade", None), GRADE_RANK["no_grounds"])
    filled = getattr(o, "admission_filled", None)
    filled = -int(filled) if filled is not None else 0
    described = 0 if str(o.description).strip().lower() == 'unknown' else -1
    ct = getattr(o, "creation_time", None)
    return (grade, filled, described, 1 if ct is None else 0, ct if ct is not None else 0.0,
            str(getattr(o, "object_id", "") or o.label))


def _box_row(bbox):
    """A bbox as six rounded numbers for a decision row, or None. Compact on purpose: a dict of
    six keys is twice the bytes, and GA-232's lesson is that per-pair rows are what took
    hook_decisions.jsonl to 1.71 GB. Order is the one every reader here already uses."""
    if not bbox:
        return None
    try:
        return [round(float(bbox[k]), 3) for k in
                ("x_min", "y_min", "z_min", "x_max", "y_max", "z_max")]
    except (KeyError, TypeError, ValueError):
        return None


def pair_attribute_score(a, b):
    """-> (score, evidence) of lost_similarity_detailed for two world-model objects.

    Embeddings cache on the object, so each description is encoded once. GA-101: the score
    alone cannot say whether four axes agreed or nothing was comparable, and those two produce
    the SAME number when the labels match, so the evidence dict travels with it. Shared by the
    merge loop and the evidence engine's attribute channel (2026-09-14 owner ruling).
    """
    if getattr(a, "embedding", None) is None:
        a.embedding = get_embedding(world2vec, a.description)
    if getattr(b, "embedding", None) is None:
        b.embedding = get_embedding(world2vec, b.description)
    a_label = a.label.split('#')[0] if '#' in a.label else a.label
    b_label = b.label.split('#')[0] if '#' in b.label else b.label
    return lost_similarity_detailed(world2vec, a_label, b_label, a.color, b.color,
                                    a.material, b.material, a.embedding, b.embedding)


def _attribute_channel_input(aa, bb):
    """AssocContext.attribute_score_fn: (score, optional_count) from the two AssocObjects'
    world-model sources; None when a side has no source (channel abstains)."""
    if getattr(aa, "source", None) is None or getattr(bb, "source", None) is None:
        return None
    score, ev = pair_attribute_score(aa.source, bb.source)
    la = str(aa.source.label).split('#')[0].strip().lower()
    lb = str(bb.source.label).split('#')[0].strip().lower()
    # same_kind is EXACT base-label equality (see association.channel_attributes for why a
    # label-similarity floor was rejected). It is what lets the attribute channel stand in for
    # the uncollected co-visibility on a genuine re-observation (GA-328 guard).
    return score, int(ev["optional_count"]), (la == lb and la != "")


def _centroid_from_bbox(bbox):
    """GA-296. The centroid a world-model Object is created WITHOUT.

    Both `Object(...)` sites here passed `None` for it and nothing downstream ever assigned
    it, so `obj.centroid` was None for the life of every object in the map. That is not
    cosmetic: `_record_sighting` returns early on `centroid is None`, so NO OBJECT HAS EVER
    RECORDED A SIGHTING -- `observations` is empty everywhere, `position_covariance` returns
    None for every object, and `search_radius` therefore falls back to extent-only on BOTH
    the merge sweep and the tracking scan. Measured consequence: GA-289's `reach_fallback`
    is 14801/14801 and 3014/3014 in the two runs that carry the counter, i.e. 100%.

    The covariance shell, co-visibility and appearance re-identification are all functions of
    `observations`, so all three have been abstaining by construction since GA-186 rather
    than because the evidence was absent. Deriving it from the box the object already has
    is the whole fix; the box is the only position information the object carries.
    """
    b = assoc._as_bounds(bbox)
    if b is None:
        return None
    # A plain list, not the numpy array box_centroid returns: this value is stored on a
    # long-lived object that is serialised by `save_persistent_perceptions` and tested for
    # truth in several readers, and a numpy array raises on both. Every consumer that wants
    # an array (AssocObject, Observation) calls np.asarray on it anyway.
    return [float(v) for v in assoc.box_centroid(b)]


MAX_OBSERVATIONS_PER_OBJECT = int(CFG["association"].get("max_observations_per_object", 64))


def _absorb_into_keeper(keeper, discard, merged_bbox):
    """What the SURVIVOR of a merge must inherit besides the geometry.

    THE MOVE PATH ALREADY DOES THIS and the merge path never did. When the merge removed the
    discarded object from the map it dropped that object's `observations` with it -- and
    `observations` is the SOLE input of three things: `association.position_covariance` (so
    the survivor's position uncertainty is measured from half the evidence),
    `association.channel_covisibility` (so a frame that would veto a later wrong merge is
    forgotten), and the appearance descriptors (so a re-identification loses half its views).
    A merge made the map LESS able to reason about the object it had just consolidated.

    The centroid is rewritten for the same reason: `Object.centroid` is assigned once, when
    the object is created, and by no later code, so without this line the survivor's centre
    stays at its pre-merge position for the rest of the run while its box grows.

    Dedup on `frame_id`: the keeper and the discard detected in ONE frame is one view of one
    object, not two. The cap and the drop-oldest rule are the ones `_record_sighting` in
    object_manager_6 already applies, reading the same config key, so no second constant
    exists to drift.
    """
    keeper.centroid = _centroid_from_bbox(merged_bbox) or keeper.centroid
    kept = list(getattr(keeper, "observations", None) or [])
    seen = {getattr(o, "frame_id", None) for o in kept}
    inherited = [o for o in (getattr(discard, "observations", None) or [])
                 if getattr(o, "frame_id", None) not in seen]
    if inherited:
        kept.extend(inherited)
        kept.sort(key=lambda o: getattr(o, "stamp", 0) or 0)
        keeper.observations = kept[-MAX_OBSERVATIONS_PER_OBJECT:]


def synchronized_world_model(callback):
    """Serialize callbacks that read/write the shared world model."""
    @wraps(callback)
    def wrapper(*args, **kwargs):
        with wm.lock:
            return callback(*args, **kwargs)
    return wrapper

def normalise_embedding(raw):
    """-> a float32 vector, or None. THE ONE PLACE AN EMBEDDING BECOMES ABSENT OR PRESENT.

    GA-171. `_serialize_embedding` encodes None as `[]` to cross the Graph API, and the
    decoder turned that back into `np.asarray([])` -- an array of shape (0,). AN EMPTY ARRAY
    IS NOT None: it passes every `is not None` guard in the tree and then fails inside the
    dot product, which is where it surfaced:

        ValueError: shapes (384,) and (0,) not aligned

    The ADD path already handled it (object_services :1097-1105 maps size 0 back to None).
    THE UPDATE PATH DID NOT -- it assigned the raw value straight onto the new object, so an
    object that had been REPLACED carried an empty array where every other object carried
    None or a vector. That is why it only appeared once the association stage was finally
    running long enough to update and then merge the same object.

    Absence has to survive a round trip. Encoding it as `[]` and decoding it as a value is
    the same defect as `unknown` meaning both "measured" and "not reported" -- and the fix
    is the same: one function decides, and it decides the same way for every caller.
    """
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32).flatten()
    if arr.size == 0:
        return None
    return arr


def normalise_clip_embedding(raw):
    """-> a finite runtime CLIP image vector as a list, or ``None``.

    Runtime appearance vectors cross two typed boundaries (HTTP JSON and a ROS
    ``float32[]``).  Keep their validation separate from the 300-D text embedding:
    the two spaces have different dimensions and different consumers.
    """
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32).flatten()
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return [float(value) for value in arr]


def inside_area(o, bounds):
    """Is this object's box wholly inside `bounds` = (xmin, xmax, ymin, ymax, zmin, zmax)?

    GA-22. This was broken twice over and had never returned True. It read `xmin`..`zmax`
    as module globals -- they are locals of the ONE caller and exist nowhere else -- and it
    unpacked `bbox`, which is a dict, so the six names took the dict's KEYS and the first
    comparison would have raised TypeError anyway. The caller's bare `except Exception`
    turned both into an empty result identical to "no objects match", so the area filter
    has never worked and nothing could have revealed it.
    """
    bbox = getattr(o, "bbox", None)
    if not bbox:
        return False
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    return (
        bbox["x_min"] >= xmin and bbox["x_max"] <= xmax and
        bbox["y_min"] >= ymin and bbox["y_max"] <= ymax and
        bbox["z_min"] >= zmin and bbox["z_max"] <= zmax
    )

def create_object_key(label, material, color, description):
    """Create a unique key for an object based on attributes."""
    key_dict = {
        "label": label if label else "",
        "material": material if material else "",
        "color": color if color else "",
        "description": description if description else ""
    }
    return json.dumps(key_dict, sort_keys=True)

def create_object_id(label, material, color, description):
    key = create_object_key(label, material, color, description)
    return hashlib.sha256(key.encode("utf-8")).hexdigest() 


def bbox_volume(bbox):
    if not bbox:
        return 0.0
    return max(0.0, bbox["x_max"] - bbox["x_min"]) * \
        max(0.0, bbox["y_max"] - bbox["y_min"]) * \
        max(0.0, bbox["z_max"] - bbox["z_min"])


def bbox_is_suspicious(bbox, reference_bbox=None):
    if not bbox:
        return True

    sizes = [
        bbox["x_max"] - bbox["x_min"],
        bbox["y_max"] - bbox["y_min"],
        bbox["z_max"] - bbox["z_min"],
    ]
    if any(size <= 0.0 for size in sizes):
        return True

    if max(sizes) > SUSPICIOUS_MAX_EXTENT_M:
        return True

    volume = bbox_volume(bbox)
    if volume > SUSPICIOUS_MAX_VOLUME_M3:
        return True

    if reference_bbox:
        ref_volume = bbox_volume(reference_bbox)
        if ref_volume > 1e-6 and volume > max(3.0 * ref_volume, ref_volume + 0.75):
            return True

    return False

def save_uncertain_objects(node):
    """Write the uncertain-object pool to output/uncertain_objects.txt.

    GA-22. This lived in `object_manager_6` and was called from BOTH modules, but
    `object_services` never imports it -- and cannot, because `object_manager_6` imports
    `object_services`, so a top-level import would be circular. Every call from here raised
    NameError, the handler's broad `except Exception` returned `success=False`, and the
    uncertain objects had already been removed two lines earlier: the caller saw a failure,
    retried, and failed identically over an empty list.

    Moved here because this is where `uncertain_objects` lives. `object_manager_6` already
    imports from this module, so its own call site keeps working with no cycle.
    """
    # The run's OWN output directory, which is the bundle: `GRAPH_API_OUTPUT_DIR` is what the
    # launcher bind-mounts, and `graph_api_bridge` reads this file from there (`_graph_version`
    # watches it and the on-hold / rejected / abstained audit loads it). Building the path from
    # PROJECT_ROOT instead made the two agree only when the run's output directory happened to
    # BE the source tree's `output/`, so on any other host the audit was silently empty and the
    # graph fingerprint missed a file it believed it was watching. It also wrote INSIDE the tree
    # under test, which the container copies at start, so a run changed what the next launch
    # copied. PROJECT_ROOT stays as the fallback for a bare source-tree run.
    output_dir = os.environ.get("GRAPH_API_OUTPUT_DIR") or os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "uncertain_objects.txt")

    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"UNCERTAIN OBJECTS - Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        if not node.uncertain_objects:
            f.write("No uncertain objects at the moment.\n")
        else:
            f.write(f"Total uncertain objects: {len(node.uncertain_objects)}\n\n")
            for i, obj in enumerate(node.uncertain_objects, 1):
                f.write(f"{i}. {obj.label}\n")
                if obj.bbox:
                    x_center = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
                    y_center = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
                    z_center = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
                    f.write(f"   Center position: X={x_center:.3f}, Y={y_center:.3f}, Z={z_center:.3f}\n")


def save_persistent_perceptions(node):
    output_dir = resolve_output_root()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "persistent_perception.json")

    # 1. Carica lo stato attuale dal file (unica fonte di verità)
    existing_by_id = {}
    if os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                existing_data = json.load(f)
            existing_by_id = {e["object_id"]: e for e in existing_data}
        except (json.JSONDecodeError, OSError):
            existing_by_id = {}

    current_ids = set()
    changed = False

    # One SNAPSHOT for both the iteration and `current_ids` below. Iterating the LIVE
    # list while a merge on the HTTP thread removes from it makes Python's list
    # iterator silently skip an object -- which then falls out of `current_ids` and is
    # DELETED from the stored JSON by the `removed_ids` pass below while still being on
    # the map: a live object vanishing from the artifact. The snapshot is one copy of
    # tens of pointers; the dump then describes one consistent world.
    for obj in wm.snapshot():
        if not getattr(obj, "object_id", None):
            obj.object_id = f"obj_{uuid.uuid4().hex}"

        ensure_relations(obj)
        obj_id = obj.object_id
        current_ids.add(obj_id)

        # Keep the runtime CLIP appearance vector with the object's bbox.  The
        # evaluator works per instance; a label-keyed sidecar can associate
        # the wrong vector when two objects share a label.  The offline HOV-SG
        # vector is added later as `hovsg_embedding` by the evaluation tool.
        bbox = dict(obj.bbox) if isinstance(obj.bbox, dict) else obj.bbox
        fused_bbox = (
            dict(obj.fused_bbox)
            if isinstance(getattr(obj, "fused_bbox", None), dict)
            else None
        )
        clip_embedding = getattr(obj, "clip_embedding", None)
        clip_values = None
        if clip_embedding is not None:
            try:
                candidate = np.asarray(clip_embedding, dtype=np.float32).flatten()
            except (TypeError, ValueError):
                candidate = np.asarray([])
            if candidate.size > 0 and np.all(np.isfinite(candidate)):
                clip_values = [float(value) for value in candidate]
                if isinstance(bbox, dict):
                    bbox["clip_embedding"] = clip_values

        # The graph's semantic matcher also maintains a text/description
        # embedding on the persistent object.  It used to be kept only in
        # memory, so the run bundle could not be replayed with the same
        # semantic evidence.  Persist it under an explicit name: it is not
        # the runtime image CLIP vector and it is not HOV-SG's offline vector.
        description_embedding = getattr(obj, "embedding", None)
        if description_embedding is not None:
            try:
                description_values = np.asarray(
                    description_embedding, dtype=np.float32
                ).flatten()
            except (TypeError, ValueError):
                description_values = np.asarray([])
            if (description_values.size > 0
                    and np.all(np.isfinite(description_values))):
                description_embedding = [
                    float(value) for value in description_values
                ]
            else:
                description_embedding = None

        new_entry = {
            "object_id": obj_id,
            "label": obj.label,
            "description": obj.description,
            "color": obj.color,
            "material": obj.material,
            "shape": obj.shape,
            "bbox": bbox,
            # The measured-view box above remains the association geometry.
            # This separate box is reconstructed from several accepted views.
            "fused_bbox": fused_bbox,
            "bbox_fusion": fusion_summary(obj),
            "room_id": getattr(obj, "room_id", "unknown"),
            "relations": {k: sorted(list(v)) for k, v in obj.relations.items()},
            # Added 2026-08-31. This was in-memory only, so GA-12's invariant -- every
            # object carries an age, and a moved object keeps its predecessor's -- could not
            # be checked from a bundle at all, and no analysis could ask how old an object
            # was. `last_perception_timestamp` below records when it was LAST SEEN, which is
            # a different question. Additive: a reader that does not look for this key
            # cannot break on it.
            "creation_time": getattr(obj, "creation_time", None),
            # GA-314. The admission grade the survivor rule ranks on, so a bundle can show
            # which side of a merge was the credible one. Additive; None when ungraded.
            "admission_grade": getattr(obj, "admission_grade", None),
            "admission_filled": getattr(obj, "admission_filled", None),
            "last_perception_timestamp": getattr(obj, "last_perception_time", None),
            "last_perception_datetime": (
                datetime.fromtimestamp(obj.last_perception_time, tz=timezone.utc).isoformat()
                if getattr(obj, "last_perception_time", None) else None
            ),
        }
        # Keep a top-level copy as well as the bbox copy.  The bbox copy is the
        # association input used by the runtime; the top-level field makes the
        # per-object appearance feature directly visible to offline consumers and
        # preserves compatibility with cloud-backend archives.
        if clip_values is not None:
            new_entry["clip_embedding"] = clip_values
        if description_embedding is not None:
            new_entry["description_embedding"] = description_embedding

        old_entry = existing_by_id.get(obj_id)
        if old_entry != new_entry:
            existing_by_id[obj_id] = new_entry
            changed = True

    removed_ids = set(existing_by_id.keys()) - current_ids
    for rid in removed_ids:
        del existing_by_id[rid]
        changed = True

    if not changed:
        return

    data = list(existing_by_id.values())
    tmp_path = save_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, save_path)

    msg = f"Saved {len(data)} objects to persistent_perception.json"
    node.log_both('info', msg)


def _stamp_from_seconds(timestamp_sec):
    stamp = TimeMsg()
    sec = int(timestamp_sec)
    nanosec = int(round((timestamp_sec - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    stamp.sec = sec
    stamp.nanosec = nanosec
    return stamp


def publish_persistent_bboxes(node, wm, pub):
    marker_array = MarkerArray()
    for i, obj in enumerate(wm.persistent_perceptions):
        if obj.bbox is None or "door" in obj.label.lower():
             continue
        display_bbox = getattr(obj, "fused_bbox", None) or obj.bbox
        marker = Marker()
        marker.header.frame_id = world_frame()
        obj_stamp = getattr(obj, "last_perception_time", None)
        marker.header.stamp = _stamp_from_seconds(obj_stamp) if obj_stamp else node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        # display_bbox, not obj.bbox: the fused multi-view box is the one the evaluator
        # scores (bbox_source), so RViz must show the same geometry the numbers come from.
        if not set_marker_from_bbox(marker, display_bbox):
            continue
        marker.color.a = 0.5
        marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        marker_array.markers.append(marker)
    pub.publish(marker_array)
    
def bbox_centroid_in_volume(bbox, volume):
    """Check whether the centroid of a bounding box lies inside a volume."""
    if bbox is None:
        return False

    centroid_x = (bbox["x_min"] + bbox["x_max"]) / 2.0
    centroid_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
    centroid_z = (bbox["z_min"] + bbox["z_max"]) / 2.0

    is_inside = (
        volume["x_min"] <= centroid_x <= volume["x_max"] and
        volume["y_min"] <= centroid_y <= volume["y_max"] and
        volume["z_min"] <= centroid_z <= volume["z_max"]
    )

    return is_inside


def ensure_relations(obj):
    if not hasattr(obj, "relations") or obj.relations is None:
        obj.relations = {
            "isIn": set(),
            "isOn": set(),
            "isNextTo": set(),
            "isAbove": set(),
            "isUnder": set(),
        }
    else:
        for key in ["isIn", "isOn", "isNextTo", "isAbove", "isUnder"]:
            if key not in obj.relations:
                obj.relations[key] = set()


def bbox_center(b):
    return {
        "x": (b["x_min"] + b["x_max"]) / 2.0,
        "y": (b["y_min"] + b["y_max"]) / 2.0,
        "z": (b["z_min"] + b["z_max"]) / 2.0,
    }


def overlap_1d(a_min, a_max, b_min, b_max):
    inter = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    a_len = max(1e-6, a_max - a_min)
    b_len = max(1e-6, b_max - b_min)
    return inter / min(a_len, b_len)


def overlap_xy(b1, b2):
    ox = overlap_1d(b1["x_min"], b1["x_max"], b2["x_min"], b2["x_max"])
    oy = overlap_1d(b1["y_min"], b1["y_max"], b2["y_min"], b2["y_max"])
    return ox * oy


def horizontal_distance(c1, c2):
    dx = c1["x"] - c2["x"]
    dy = c1["y"] - c2["y"]
    return (dx * dx + dy * dy) ** 0.5


def infer_spatial_relations(obj_a, obj_b):
    rels = []

    b1 = getattr(obj_a, "bbox", None)
    b2 = getattr(obj_b, "bbox", None)
    if b1 is None or b2 is None:
        return rels

    if not getattr(obj_a, "object_id", None) or not getattr(obj_b, "object_id", None):
        return rels

    c1 = bbox_center(b1)
    c2 = bbox_center(b2)

    xy_overlap = overlap_xy(b1, b2)
    dz_top = b1["z_min"] - b2["z_max"]
    dz_bottom = b2["z_min"] - b1["z_max"]
    hd = horizontal_distance(c1, c2)
    z_center_diff = abs(c1["z"] - c2["z"])

    size_a_x = b1["x_max"] - b1["x_min"]
    size_a_y = b1["y_max"] - b1["y_min"]
    size_b_x = b2["x_max"] - b2["x_min"]
    size_b_y = b2["y_max"] - b2["y_min"]

    max_xy_size = max(size_a_x, size_a_y, size_b_x, size_b_y)

    on_gap_thresh = 0.10
    above_gap_thresh = 0.20
    next_to_dist_thresh = 1.2 * max_xy_size
    next_to_z_thresh = 0.35

    if xy_overlap > 0.3 and 0.0 <= dz_top <= on_gap_thresh:
        rels.append((obj_a.object_id, "isOn", obj_b.object_id))

    if c1["z"] > c2["z"] and xy_overlap > 0.2 and dz_top > above_gap_thresh:
        rels.append((obj_a.object_id, "isAbove", obj_b.object_id))

    if c1["z"] < c2["z"] and xy_overlap > 0.2 and dz_bottom > above_gap_thresh:
        rels.append((obj_a.object_id, "isUnder", obj_b.object_id))

    is_vertical_relation = (
        (xy_overlap > 0.3 and 0.0 <= dz_top <= on_gap_thresh) or
        (c1["z"] > c2["z"] and xy_overlap > 0.2 and dz_top > above_gap_thresh) or
        (c1["z"] < c2["z"] and xy_overlap > 0.2 and dz_bottom > above_gap_thresh)
    )

    # Conservative containment: A is "in" B if the centroid of A lies inside B
    # and A is smaller than B on every axis.
    a_smaller_than_b = (
        (b1["x_max"] - b1["x_min"]) <= (b2["x_max"] - b2["x_min"]) and
        (b1["y_max"] - b1["y_min"]) <= (b2["y_max"] - b2["y_min"]) and
        (b1["z_max"] - b1["z_min"]) <= (b2["z_max"] - b2["z_min"])
    )
    if a_smaller_than_b and bbox_centroid_in_volume(b1, b2):
        rels.append((obj_a.object_id, "isIn", obj_b.object_id))

    if (
        not is_vertical_relation and
        hd <= next_to_dist_thresh and
        xy_overlap < 0.2 and
        z_center_diff <= next_to_z_thresh
    ):
        rels.append((obj_a.object_id, "isNextTo", obj_b.object_id))

    return rels

    

class ObjectServices(Node):
    def __init__(self, room_manager):
        super().__init__('object_services_node')
        self.room_manager=room_manager
        self.get_logger().info("=== ObjectServices Initialized ===")
        self.get_logger().info(f"Log sintetico operazioni: {SYNTHETIC_LOG_FILE}")

        self.tracking_step_counter = 0
        self.exploration_step_counter = 0
        
        # Persistence adapter (config `hooks.store`); default = the SQLite temporal map
        self.db = load_store(CFG, lambda: MapDatabase(db_path=os.path.join(log_dir, "tiago_temporal_map_5.db")))
        qos_latch = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.persistent_bbox_pub = self.create_publisher(MarkerArray, '/persistent_bbox', qos_latch)
        self.persistent_centroids_pub = self.create_publisher(MarkerArray, '/persistent_centroids', qos_latch)
        self.considered_volume_pub = self.create_publisher(MarkerArray, '/considered_volume', qos_latch)
        self.uncertain_bboxes_pub = self.create_publisher(MarkerArray, '/uncertain_object', qos_latch)
        self.uncertain_centroids_pub = self.create_publisher(MarkerArray, '/uncertain_centroids', qos_latch)
        self.uncertain_objects = []

        # Merge, delete and update were recorded only as prose in operations.txt, with no
        # object named and no reason given, so none of them could be joined to anything.
        #
        # The requirement this satisfies: the decision history of one object must be
        # reconstructable. That means every record names the object it concerns, and a merge
        # names BOTH sides -- otherwise the discarded object's history ends without a successor
        # and the keeper's history begins without its inheritance.
        self.decision_log = DecisionLog(
            CFG["hooks"]["decisions_log"] or os.path.join(log_dir, "hook_decisions.jsonl"))

        # GA-188. Beliefs about pairs, carried ACROSS merge sweeps -- this is the state that
        # makes `merge_min_consecutive` mean anything. Keyed by the sorted object-id pair,
        # garbage-collected each sweep so it cannot grow for the life of the process.
        # Unused by the legacy engine, which decides from a single scoring and keeps nothing.
        self._hypotheses = {}
        self._dry_hypotheses = {}   # dry runs never touch the live store (review 2026-09-14)
        self._merge_sweep = 0
        self._mutation_sequence = 0
        self._provenance_incomplete = False
        self._mutation_ledger_path = os.path.join(
            os.environ.get("GRAPH_API_OUTPUT_DIR") or log_dir,
            "mutation_receipts.jsonl",
        )

        with open(SYNTHETIC_LOG_FILE, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"NEW RUN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n")
        
        # --- INIT ROOM MANAGER ---
        
       
        self.create_service(AddObject,        '/graph/add_object',          self._cb_add_object)
        self.create_service(RemoveObject,     '/graph/remove_object',       self._cb_remove_object)
        self.create_service(DeleteObjects,     '/graph/delete_objects',       self._cb_delete_unseen_objects)
        self.create_service(UpdateObject,     '/graph/update_object',       self._cb_update_object)
        self.create_service(MergeObjects,     '/graph/merge_objects',       self._cb_merge_objects)
        self.create_service(QueryObjects,     '/graph/query_objects',      self._cb_query_objects)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_logger().info('Object Tracking Service ready')

    
    def log_operation(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        with open(SYNTHETIC_LOG_FILE, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def log_both(self, level, message):
        if level == 'info':
            self.get_logger().info(message)
        elif level == 'warn':
            self.get_logger().warn(message)
        elif level == 'error':
            self.get_logger().error(message)
        elif level == 'debug':
            self.get_logger().debug(message)

        if level in ['info', 'warn']:
            prefix = "[INFO] " if level == 'info' else "[WARN] "
            print(f"{prefix}{message}")

    @staticmethod
    def _prepare_receipt(response):
        response.mutation_event_id = ""
        response.mutation_state = "none"
        response.object_revision = 0
        response.geometry_epoch = 0

    def _write_mutation_receipt(self, event_id, request, state, operation,
                                obj=None, reason="", fusion_applied=False):
        """Append one ordered mutation stage; a write failure invalidates provenance."""
        self._mutation_sequence += 1
        record = {
            "schema_version": 1,
            "mutation_sequence": self._mutation_sequence,
            "mutation_event_id": event_id,
            "observation_attempt_id": str(
                getattr(request, "observation_attempt_id", "") or ""),
            "observation": observation_dict_from_msg(
                getattr(request, "observation", None)),
            "description_observation": observation_dict_from_msg(
                getattr(request, "description_observation", None)),
            "operation": operation,
            "mutation_state": state,
            "reason": reason,
            "object_id": (str(getattr(obj, "object_id", "") or "")
                          if obj is not None else None),
            "object_revision": (int(getattr(obj, "object_revision", 0))
                                if obj is not None else None),
            "geometry_epoch": (int(getattr(obj, "geometry_epoch", 0))
                               if obj is not None else None),
            "fusion_applied": bool(fusion_applied),
            "recorded_at": time.time(),
        }
        try:
            os.makedirs(os.path.dirname(self._mutation_ledger_path), exist_ok=True)
            with open(self._mutation_ledger_path, "a", encoding="utf-8") as ledger:
                ledger.write(json.dumps(record, sort_keys=True) + "\n")
                ledger.flush()
                os.fsync(ledger.fileno())
        except OSError:
            self._provenance_incomplete = True
            raise
        return record

    @staticmethod
    def _set_receipt_response(response, event_id, state, obj=None):
        response.mutation_event_id = event_id
        response.mutation_state = state
        response.object_revision = int(getattr(obj, "object_revision", 0)) if obj else 0
        response.geometry_epoch = int(getattr(obj, "geometry_epoch", 0)) if obj else 0

    def record_direct_mutation(self, event_id, obj, observation,
                               description_observation, state,
                               fusion_applied=False):
        """Record a manager-side mutation that does not cross the ROS service."""
        request = SimpleNamespace(
            observation=observation,
            description_observation=description_observation,
            observation_attempt_id=event_id,
        )
        return self._write_mutation_receipt(
            event_id, request, state, "direct_exploration_update",
            obj=obj, fusion_applied=fusion_applied)


    @synchronized_world_model
    def _cb_delete_unseen_objects(self, request, response):
        try:
            import json

            flat = list(request.pov_volume_flat)
            if len(flat) == 6:
                pov_volume = {
                    'x_min': flat[0], 'x_max': flat[1],
                    'y_min': flat[2], 'y_max': flat[3],
                    'z_min': flat[4], 'z_max': flat[5],
                }
            else:
                pov_volume = None

            check_uncertain = getattr(request, 'check_uncertain', False)
            current_labels  = set(request.current_labels)

            deleted_labels           = []
            uncertain_removed_count  = 0

            if not check_uncertain:
                if not pov_volume:
                    print("❌ CRITICAL TF ERROR: pov_volume is empty! "
                        "Il robot non sa dove sta guardando "
                        "(Controlla il frame in lookup_transform). "
                        "Cancellazione annullata.")
                    response.success = False
                    response.message = "pov_volume vuoto, cancellazione annullata"
                    return response

                try:
                    publish_pov_volume(self, pov_volume, self.considered_volume_pub)
                except Exception as e:
                    print(f"⚠️ Could not publish the view volume: {e}")

                objects_to_remove = []

                for obj in list(wm.persistent_perceptions):

                    # Visto in questo frame → azzera contatore. GA-26: keyed on identity --
                    # the caller sends object_id (label only when there is none), so one
                    # chair in view no longer clears every chair's tally.
                    if (getattr(obj, 'object_id', None) or obj.label) in current_labels:
                        obj.not_seen_in_pov_frames = 0
                        continue

                    if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                        if not hasattr(obj, 'not_seen_in_pov_frames'):
                            obj.not_seen_in_pov_frames = 0
                        obj.not_seen_in_pov_frames += 1

                        if obj.not_seen_in_pov_frames >= MAX_MISSES_BEFORE_DELETE:
                            objects_to_remove.append(obj)
                    else:
                        # GA-26: leaving the field of view RESETS the counter, it does not
                        # freeze it. The increment above is already correctly conditional on
                        # being in view, but nothing cleared the tally on the way out -- so
                        # three misses, an absence of any length, then two more added to five
                        # and deleted an object that had been looked at twice.
                        #
                        # Consecutive means consecutive. An object out of view is not being
                        # missed; it is not being looked at.
                        obj.not_seen_in_pov_frames = 0

                for obj in objects_to_remove:
                    print(f"🗑️ [DELETED] Object '{obj.label}' is no longer "
                        f"presente nel volume osservato! RIMOSSO.")

                    if obj in wm.persistent_perceptions:
                        wm.persistent_perceptions.remove(obj)

                    # GA-47: the neighbours of a deleted object get a second look. om6 sets
                    # this to its trigger; the removed object itself cannot be queued.
                    hook = getattr(self, "on_object_removed", None)
                    if hook:
                        try:
                            hook(obj)
                        except Exception as e:
                            self.get_logger().error(f"on_object_removed failed for {obj.label}: {e}")

                    room_id = getattr(obj, 'room_id', None)
                    if room_id and room_id in self.room_manager.scene_graph:
                        objs = self.room_manager.scene_graph[room_id]["objects"]
                        if obj.label in objs:
                            objs.remove(obj.label)

                    if hasattr(self, 'db'):
                        self.db.on_object_deleted(
                            obj,
                            reason="not seen in POV",
                            step=self.tracking_step_counter
                        )

                    deleted_labels.append(obj.label)
                    # A deletion ends an object's history. Record which object, and why, or the
                    # history stops with no explanation. 341 deletions against 30 additions on
                    # the 26 Aug run is a ratio nobody can examine without this.
                    try:
                        self.decision_log.write(
                            "delete", getattr(obj, "object_id", obj.label),
                            label=obj.label, reason="not seen in POV",
                            step=self.tracking_step_counter)
                    except Exception as e:
                        self.get_logger().error(f"decision_log delete failed: {e}")

                if deleted_labels:
                    save_persistent_perceptions(self)

                    try:
                        with open(OPERATIONS_LOG, 'a') as f:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            for lbl in deleted_labels:
                                f.write(f"[{timestamp}] 🗑️ CANCELLATO (non visto): {lbl}\n")
                    except Exception as e:
                        self.get_logger().error(f"Could not write to operations.txt: {e}")

            if check_uncertain:
                uncertain_to_remove = []

                if pov_volume:
                    for uncertain_obj in self.uncertain_objects:
                        if (uncertain_obj.bbox and
                                bbox_centroid_in_volume(uncertain_obj.bbox, pov_volume)):
                            uncertain_to_remove.append(uncertain_obj)

                for uncertain_obj in uncertain_to_remove:
                    self.uncertain_objects.remove(uncertain_obj)
                    print(f"🗑️ [UNCERTAIN REMOVED] '{uncertain_obj.label}'")

                    try:
                        with open(OPERATIONS_LOG, 'a') as f:
                            timestamp = datetime.now().strftime('%H:%M:%S')
                            f.write(f"[{timestamp}] ⚠️ UNCERTAIN RIMOSSO: {uncertain_obj.label}\n")
                    except Exception as e:
                        self.get_logger().error(f"Could not write to operations.txt: {e}")

                uncertain_removed_count = len(uncertain_to_remove)

            response.success                 = True
            response.deleted_count           = len(deleted_labels)
            response.uncertain_removed_count = uncertain_removed_count
            response.deleted_labels_json     = json.dumps(deleted_labels)
            response.message                 = (
                f"{len(deleted_labels)} oggetti cancellati, "
                f"{uncertain_removed_count} uncertain rimossi"
            )

        except Exception as e:
            self.get_logger().error(f"_cb_delete_unseen_objects failed: {e}")
            response.success                 = False
            response.message                 = str(e)
            response.deleted_count           = 0
            response.uncertain_removed_count = 0
            response.deleted_labels_json     = "[]"

        return response
    
    @synchronized_world_model
    def _publish_merge_pending(self):
        """Write the pending-merge state where the FEED HOST can read it. GA-258.

        WHY THIS EXISTS. A hypothesis commits only after `merge_min_consecutive` consecutive
        sweeps over the threshold. MEASURED on 20260901_174810_hm3d_00861: of 265,944 held
        pairs the fused log-odds reached p90 7.77 against a commit threshold of 3.0 -- more
        than a tenth of them had ENOUGH evidence and were held anyway, because the agent kept
        turning and the pair left view before a second consecutive sweep could confirm it.
        A fixed dwell either wastes time when nothing is pending or leaves too early when
        something is; the agent should look for exactly as long as there is something to
        confirm.

        The feed host runs on the HOST and this runs in the container, so a file is the
        channel: `/ws/output` is bind-mounted to the run directory, and this needs no port,
        no service and no ordering between two processes that start independently.

        Written atomically -- GA-257 was a 0-byte knowledge_graph.ttl caused by a
        non-atomic serialise being interrupted, and a reader polling this file mid-write
        would see a truncated one for the same reason.
        """
        out = os.environ.get("GRAPH_API_OUTPUT_DIR")
        if not out:
            return
        try:
            thr = assoc.commit_threshold(MERGE_COST_RATIO)
            pending = []
            for key, h in self._hypotheses.items():
                if h.committed or h.vetoed_by:
                    continue
                streak = getattr(h, "_streak", 0)
                total = h.total
                # PENDING means "over the bar, short of the streak" -- the population that
                # more looking would convert. A pair below threshold is not pending; waiting
                # for it is waiting for evidence that is not accumulating.
                if total >= thr and streak < MERGE_MIN_CONSECUTIVE:
                    pending.append({"a": key[0], "b": key[1],
                                    "log_odds": round(float(total), 3),
                                    "streak": int(streak),
                                    "needs": MERGE_MIN_CONSECUTIVE})
            pending.sort(key=lambda d: -d["log_odds"])
            blob = {
                "t": time.time(),
                "sweep": self._merge_sweep,
                # GA-339 (a). The most sweeps any pending pair still needs, so the feed host
                # can bound a hold without scanning `pairs`, which is capped at 40 below.
                # 0 when nothing is pending: measured, not absent.
                "needs_max": max((d["needs"] - d["streak"] for d in pending), default=0),
                # Additive, NOT a rename: found/dashboard/replay_server.py:1297 renders
                # `d.threshold` from this blob, and a renamed key reads as undefined there
                # with no error. Same log-odds unit as the evidence engine's row.
                "threshold_log_odds": round(float(thr), 3),
                "threshold": round(float(thr), 3),
                "min_consecutive": MERGE_MIN_CONSECUTIVE,
                # GA-341 follow-up (orchestrator, 2026-09-08). The EFFECTIVE merge floor and
                # reach of the arm that actually ran, so a bundle can attest them. Run 1's
                # criterion "no merge row with threshold_similarity <= sim_threshold" read 0
                # and that zero was VACUOUS: the similarity floor belongs to the legacy arm
                # and the evidence arm emits `threshold_log_odds`, so NO row carried the key
                # the criterion named. A check whose subject is absent asserts nothing. These
                # keys are what a reader should test instead (rule 6, additive).
                "engine": MERGE_ENGINE,
                "min_similarity": MERGE_MIN_SIMILARITY,
                "max_distance_m": MERGE_MAX_DISTANCE,
                "sim_threshold": SIM_THRESHOLD,
                "live_hypotheses": len(self._hypotheses),
                "pending": len(pending),
                # Capped: the feed host only needs the COUNT, and the list is for the viewer.
                # An unbounded list would rewrite megabytes every sweep.
                "pairs": pending[:40],
            }
            path = os.path.join(out, "merge_pending.json")
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(blob, fh)
            os.replace(tmp, path)
        except OSError as exc:
            # GA-339 (b), rule 14. Only the write itself may fail softly: the feed host reads
            # an absent or stale file as UNKNOWN and holds to its cap, so a disk error costs
            # frames, not a merge. A non-serialisable value (TypeError / ValueError from
            # json.dump) is a bug in this function and crashes -- it used to be swallowed by
            # `except Exception`, which made the feed hold to cap at every stop with no trace.
            self.get_logger().warn(f"[ASSOC] could not publish merge_pending: {exc}")

    def _interrupt_hypothesis(self, a, b, reason, dry_run=False):
        """A gate refused an OFFERED pair before scoring: break its streak (review 2026-09-14).
        The pair stays live for `_hypothesis_gc` (it was offered), so without this the next
        passing sweep would commit as "2 consecutive" after only one."""
        key = tuple(sorted((str(getattr(a, "object_id", None) or a.label),
                            str(getattr(b, "object_id", None) or b.label))))
        h = (self._dry_hypotheses if dry_run else self._hypotheses).get(key)
        if h is not None:
            h.interrupt(reason, frame_id=self._merge_sweep)

    def _hypothesis_gc(self, live_keys):
        """Drop hypotheses whose pair no longer exists. GA-188.

        Without this the store grows for the life of the process and keeps a belief about
        objects that were merged away sweeps ago -- and a stale hypothesis whose object_id
        is later reused would carry a streak nobody measured. Committed ones go too: the
        merge is applied and its provenance is already in the decision record.
        """
        stale = [k for k, h in self._hypotheses.items() if h.committed or k not in live_keys]
        for k in stale:
            del self._hypotheses[k]
        return len(stale)

    def _assoc_build(self, objects):
        """-> (AssocContext, {id(world_model_object): AssocObject}). GA-186.

        The adapter between the world model's `Object` and `association.AssocObject`.
        Everything it cannot measure is left as None so the corresponding channel ABSTAINS;
        nothing here invents a value to keep a channel running.

        `map_volume_m3` is the axis-aligned hull of the objects themselves, not the metric
        map. It is the volume the overlap and separation priors are taken over -- "how
        surprising is this much overlap in a space this big" -- and the objects' own extent
        is the honest answer to that when no map volume is published. A tiny scene must not
        read as a huge one, so it is floored rather than allowed to reach zero on a
        single-object map.
        """
        boxes = [o.bbox for o in objects if getattr(o, "bbox", None)]
        if boxes:
            xs = [b["x_min"] for b in boxes] + [b["x_max"] for b in boxes]
            ys = [b["y_min"] for b in boxes] + [b["y_max"] for b in boxes]
            zs = [b["z_min"] for b in boxes] + [b["z_max"] for b in boxes]
            map_volume = max((max(xs) - min(xs)) * (max(ys) - min(ys)) * (max(zs) - min(zs)), 1.0)
        else:
            map_volume = 1.0

        rooms = {getattr(o, "room_id", None) for o in objects}
        rooms.discard(None)
        # GA-192: how many DISTINCT aligned types this scene actually contains. The channel
        # needs it as the null model -- "how surprising is it that two objects share a type"
        # depends on how many types were available to differ. MEASURED from the scene, never
        # a guessed vocabulary size: a fabricated n_types turns an abstention into a
        # confident prior, and the channel is explicitly built to abstain instead.
        types = {getattr(o, "onto_type", None) for o in objects
                 if getattr(o, "onto_aligned", False)}
        types.discard(None)
        n_types = len(types) if len(types) >= 2 else None
        # GA-309. The ontology veto is supplied by the admission hook, through ONE optional
        # attribute: `disjoint(type_a, type_b) -> bool`. An absent attribute is "not supplied"
        # -- exactly the arm every bundle before 2026-09-07 ran, where channel_ontology never
        # vetoed -- and the record then carries no `disjoint_source`. The hook is the object
        # manager's; it hands it over at startup (`self.object_services.filter_hook`). The
        # callable raises on an unknown class name by contract (rule 14); nothing here
        # substitutes a False for it.
        hook = getattr(self, "filter_hook", None)
        disjoint_fn = getattr(hook, "disjoint", None)
        ctx = assoc.AssocContext(
            map_volume_m3=map_volume,
            n_rooms=max(len(rooms), 1),
            n_types=n_types,
            cost_ratio=MERGE_COST_RATIO,
            # 2026-09-14 owner ruling: ontology off for this paper, attributes in, bounded.
            use_ontology=MERGE_ONTOLOGY_CHANNEL,
            attribute_score_fn=_attribute_channel_input,
            attribute_reference=SIM_THRESHOLD,
            attribute_max_log_odds=MERGE_ATTRIBUTE_MAX_LOG_ODDS,
            locality_gap_m=LOCALITY_GAP_M,
            disjoint_fn=disjoint_fn,
            disjoint_source=(getattr(hook, "name", type(hook).__name__)
                             if disjoint_fn is not None else None),
            # GA-186: the detector's 2D boxes now reach the object through Bbox3d.msg, so
            # co-visibility can tell a duplicate detection of one object from two objects in
            # one frame -- and vetoes only on a MEASURED disjoint overlap. The function
            # returns None when a shared frame has no box on either side, which keeps the
            # abstention for the case that was never measurable.
            overlap_2d_fn=assoc.shared_frame_overlap_2d,
        )

        built = {}
        for o in objects:
            if getattr(o, "bbox", None) is None:
                continue
            try:
                built[id(o)] = assoc.AssocObject(
                    object_id=getattr(o, "object_id", None) or o.label,
                    label=o.label,
                    # Review 2026-09-14: ONE box for candidate generation, the overlap
                    # channel and the gates -- the fused box when there is one. Before, the
                    # shell and the gap test read the measured box while the gates read the
                    # fused one, so a pair the gate would accept could never be offered.
                    bbox=assoc.locality_box(o),
                    # E3, 2026-09-14: `centroid=` is NOT passed, so AssocObject derives it
                    # from the box on the line above -- the one box this sweep scores.
                    # `o.centroid` is written once when the object is created and by no later
                    # code, so passing it meant the candidate shell and the separation channel
                    # measured from where the object was FIRST seen while every other channel
                    # measured the current box. MEASURED on the GA-493 bundle: the two differ
                    # by more than 1 cm on 7 of 40 scored pairs, by up to 0.18 m.
                    observations=getattr(o, "observations", None) or [],
                    room_id=getattr(o, "room_id", None),
                    # GA-192: the type the extension aligned this object to, and whether that
                    # alignment held. Absent stays absent -- channel_ontology abstains on an
                    # unaligned side rather than comparing labels the ontology never endorsed.
                    onto_type=getattr(o, "onto_type", None),
                    onto_aligned=bool(getattr(o, "onto_aligned", False)),
                    source=o,
                )
            except (TypeError, ValueError, KeyError) as e:
                # A malformed object must not be silently dropped from candidate
                # generation -- that is the unlogged-skip defect this design exists to end.
                self.log_both('error', f"[ASSOC] AssocObject non costruito per "
                                       f"'{getattr(o, 'label', '?')}': {e}")
        return ctx, built

    def _merge_candidates(self, objects, _refused, max_distance_m=None):
        """-> (pairs, ctx, assoc_objects). Which pairs are even offered to a decision.

        The legacy engine uses the old merge gates, but its all-pairs enumeration is now
        narrowed by the cleanup branch's binary-search AABB index.  The index is a broad
        phase only: every candidate still goes through the unchanged room, distance,
        similarity, and evidence checks below.  Objects with malformed/missing boxes are
        retained as explicit fallback pairs so they still reach the existing
        ``bbox_absent`` diagnostic instead of disappearing silently.

        The evidence engine retains its covariance-shell candidate generator, because its
        spatial shell is part of that engine's decision model rather than the legacy AABB
        broad phase.
        """
        if MERGE_ENGINE == "legacy":
            # The load-time guard at the top of this module asserts the margin is at least the
            # CONFIG distance, so the broad phase "cannot hide a pair that the legacy merge
            # criterion would evaluate". It never sees `request.max_distance`, which is what
            # `_cb_merge_objects` actually compares against, so a request asking for MORE than
            # the config default was narrowed to the fixed margin and the pairs between the two
            # were dropped here, uncompared. MEASURED before this fix: two objects 2.0 m apart
            # with a 1.0 m box gap and `request.max_distance=3.0` were never compared.
            #
            # WIDEN, NEVER NARROW. `max` cannot drop a pair the fixed margin would have kept, so
            # this changes no decision that the config path already makes. `max_distance_m` is
            # positive by the time it arrives: `_cb_merge_objects` refuses a request at or below
            # zero before reaching here. A non-numeric value raises rather than being muted --
            # silently falling back to the config margin is the bug this removes.
            margin = MERGE_AABB_MARGIN_M
            if max_distance_m is not None:
                margin = max(margin, float(max_distance_m))
            index = DetectionIndex()
            indexed = {}
            valid = []
            invalid = []
            examined = 0
            for i, obj in enumerate(objects):
                key = f"merge:{i}"
                try:
                    index.upsert(key, obj.bbox)
                    indexed[key] = i
                    valid.append(i)
                except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
                    invalid.append(i)

            pair_indices = set()
            for i in valid:
                key = f"merge:{i}"
                hits = index.query(objects[i].bbox, margin)
                # `last_examined` is per QUERY. Summing it once per hit multiplied the count by
                # the hit count (every object hits at least itself), so the logged `examined`
                # was inflated on every sweep.
                examined += index.last_examined
                for hit in hits:
                    j = indexed[hit]
                    if i < j:
                        pair_indices.add((i, j))

            # A missing/invalid box cannot be safely pruned by geometry.  Keep all pairs
            # involving it so the exact merge loop logs ``bbox_absent`` as before.
            for i in invalid:
                for j in range(i + 1, len(objects)):
                    pair_indices.add((i, j))
                for j in range(i):
                    pair_indices.add((j, i))

            pairs = [(objects[i], objects[j], {})
                     for i, j in sorted(pair_indices)]
            all_pairs = len(objects) * max(0, len(objects) - 1) // 2
            self.log_both(
                'info',
                f"[MERGE AABB] objects={len(objects)} all_pairs={all_pairs} "
                f"candidates={len(pairs)} examined={examined} "
                f"invalid_bbox={len(invalid)} margin={margin:.3f}m "
                f"(config {MERGE_AABB_MARGIN_M:.3f}m, request {max_distance_m})",
            )
            # GA-232's rule, applied to this engine too: COUNTS, never per-pair rows -- the
            # per-pair population is what took hook_decisions.jsonl to 1.71 GB. Without this
            # row the broad phase reintroduces the exact defect `_refused` exists to prevent:
            # a pair dropped here never reaches the distance check, so no `merge_refused` is
            # written, and "the gate refused N pairs" and "association produced no pairs"
            # become one observation again. Its only other trace is the log line above, which
            # goes to the container log -- a property of the launch, not of the bundle.
            not_offered = all_pairs - len(pairs)
            if not_offered > 0:
                try:
                    self.decision_log.write(
                        "not_offered_summary", "<sweep>",
                        sweep=self._merge_sweep, n_pairs=not_offered,
                        by_reason={"aabb_margin": not_offered})
                except Exception as e:
                    self.get_logger().error(
                        f"decision_log not_offered_summary failed: {e}")
            return pairs, None, {}

        ctx, built = self._assoc_build(objects)
        ordered = [o for o in objects if id(o) in built]
        offered, excluded = assoc.generate_candidates(
            [built[id(o)] for o in ordered], ctx, k=MERGE_KNN_K)

        back = {id(built[id(o)]): o for o in ordered}

        # GA-232. COUNTED, NOT WRITTEN ONE BY ONE.
        #
        # A `not_offered` pair is one the kNN candidate generator never put forward, so it
        # was never compared and it HAS NO SIMILARITY -- the field is null on every one of
        # them. MEASURED on 20260901_174810_hm3d_00861: 2,350,062 of the 2,617,640
        # merge_refused records are not_offered, 89.8% of the file, and 2,351,696 records
        # carry no similarity at all. Those two numbers are the same population.
        #
        # That matters because of what the per-pair record is FOR. The docstring below
        # defends writing full records rather than counters, and the argument it defends is
        # about the similarity band -- "the 0.85-0.925 band was EMPTY, so the old gate would
        # have refused them too". A record with no similarity cannot participate in that
        # argument. Every record that CAN still gets written in full: 267,578 of them,
        # including all 120,635 at or above 0.85.
        #
        # The cost of writing them was not small. hook_decisions.jsonl reached 1.71 GB on a
        # 61-minute run, /graph_data could not parse it inside 120 s, and the disk filled.
        # Dropping this one population takes the file to roughly a tenth with no loss to any
        # question the bundle is asked -- and the COUNT is kept exactly, by exclusion reason,
        # so "how many pairs were never offered, and why" is still answerable.
        excluded_tally = {}
        for aa, bb, meta in excluded:
            a, b = back.get(id(aa)), back.get(id(bb))
            if a is not None and b is not None:
                why = str((meta or {}).get("reason"))
                excluded_tally[why] = excluded_tally.get(why, 0) + 1
        if excluded_tally:
            try:
                self.decision_log.write(
                    "not_offered_summary", "<sweep>",
                    sweep=self._merge_sweep, n_pairs=sum(excluded_tally.values()),
                    by_reason=excluded_tally)
            except Exception as e:
                self.get_logger().error(f"decision_log not_offered_summary failed: {e}")

        pairs = []
        for aa, bb, meta in offered:
            a, b = back.get(id(aa)), back.get(id(bb))
            if a is not None and b is not None:
                pairs.append((a, b, meta))
        # One sweep = one hypothesis update per offered pair. The counter is the frame_id
        # in each hypothesis's history, so the provenance says WHICH sweep saw what.
        self._merge_sweep += 1
        live = {tuple(sorted((str(getattr(a, "object_id", None) or a.label),
                              str(getattr(b, "object_id", None) or b.label))))
                for a, b, _m in pairs}
        dropped = self._hypothesis_gc(live)
        self.log_both('info', f"[ASSOC] sweep {self._merge_sweep}: {len(pairs)} offered, "
                              f"{len(excluded)} excluded, of {len(ordered)} objects "
                              f"(all-pairs would be {len(ordered) * (len(ordered) - 1) // 2}); "
                              f"live hypotheses {len(self._hypotheses)}, dropped {dropped}")
        self._publish_merge_pending()
        return pairs, ctx, built

    def _cb_merge_objects(self, request, response):
        try:
            import json

            # GA-25, first residual. This used to read `<= 0` as "use the configured value",
            # the OPPOSITE of config.py's `0 = off`, so a caller sending 0 to DISABLE the
            # similarity requirement silently got the loosest live gate in the system. A
            # request can no longer configure zero at all: both gates must be positive, and
            # a zero or negative value is refused with the value the caller must send. Every
            # in-tree caller (the bridge, merge_duplicate_objects) already sends explicit
            # positive values, so nothing relies on the old default-by-zero.
            if request.max_distance <= 0.0 or request.min_similarity <= 0.0:
                response.success = False
                response.merged_count = 0
                response.merge_log_json = "[]"
                response.message = (
                    f"refused: max_distance={request.max_distance} min_similarity="
                    f"{request.min_similarity}; both must be > 0. A merge request cannot "
                    f"disable a gate -- send the values you mean (config defaults are "
                    f"{MERGE_MAX_DISTANCE} m and {MERGE_MIN_SIMILARITY}).")
                return response
            # GA-341. The load-time assertion above forbids a CONFIG merge floor at or below the
            # match gate, but a REQUEST could still carry one: the bridge's body default was
            # 0.75 against sim_threshold 0.85, so `POST /merge {}` merged pairs the association
            # loop had just refused. Refuse it here, naming the bound, rather than clamp it.
            if request.min_similarity <= SIM_THRESHOLD:
                response.success = False
                response.merged_count = 0
                response.merge_log_json = "[]"
                response.message = (
                    f"refused: min_similarity={request.min_similarity} must be STRICTLY greater "
                    f"than association.sim_threshold ({SIM_THRESHOLD}); the config merge floor "
                    f"is {MERGE_MIN_SIMILARITY} (GA-341).")
                return response
            MAX_DISTANCE   = request.max_distance
            MIN_SIMILARITY = request.min_similarity
            dry_run        = getattr(request, 'dry_run', False)

            # Take the comparison snapshot under the world-model lock.  The legacy path
            # intentionally performs its expensive similarity work outside the lock, but
            # copying the live list without the lock allowed an admission/removal callback
            # to race the iterator and made a scan-complete merge see a partially changed map.
            # `wm.snapshot()` keeps the short critical section at the snapshot boundary;
            # the existing membership checks below still protect the later write-back.
            objects = wm.snapshot()
            to_remove       = set()
            # One dict per pair, not a tuple. It was a 3-tuple, then a 4-tuple when the
            # similarity had to reach the decision record, and GA-80 needs the rooms there
            # too -- each widening touching four unpack sites, any one of which could be
            # missed for a ValueError inside a service callback that neither py_compile nor
            # ruff can see. A dict ends that: adding a field never breaks a reader.
            to_remove_pairs = []   # each carries its own "log" entry for merge_log_json

            def _pair_similarity(a, b, a_label, b_label):
                """-> (score, evidence). Embeddings cache on the object, so once per object.

                GA-101: the score alone cannot say whether four axes agreed or nothing was
                comparable, and those two produce the SAME number when the labels match.
                """
                return pair_attribute_score(a, b)

            def _refused(a, b, reason, similarity, **extra):
                """A refusal is a decision and belongs beside the admissions.

                Every refusal path here used to be print-then-continue, and only an APPLIED
                merge reached hook_decisions.jsonl -- so from the bundle alone "the gate
                refused twenty pairs" and "association produced no pairs" were THE SAME
                OBSERVATION. Run 19's real numbers (20 pairs compared, 19 refused on
                similarity, ceiling 0.797) were recoverable only because stdout happened to
                be captured by how that run was launched, which is a property of the launch
                and not of the bundle format.

                Those numbers are what struck a claim already circulating as evidence:
                "0/5 merges under 0.925 vs 87% under the old gate" reads as a threshold
                effect and is not one -- the 0.85-0.925 band was EMPTY, so the old gate
                would have refused them too. The run measured the scene, not the knob.

                `similarity` is recorded on ALL THREE reasons, including room and distance.
                Counters alone would not have carried that argument: a run that refuses
                everything on locality must still show whether those pairs would have
                passed on attributes, or the next reader repeats the same mistake.

                `kind` is "merge_refused", never "merge" -- a reader counting merges must
                separate these BY FIELD, not by inferring it from a payload key.
                """
                # The terms that EXISTED WHEN THE PAIR WAS COMPARED, per side.
                # persistent_perception.json holds the final values of SURVIVING objects
                # only, so a merge destroys its own diagnostic evidence -- 11 of run A's 14
                # high-scoring pairs could not be resolved afterwards because they had been
                # merged away, which makes every term-absence count computed from the
                # bundle a LOWER BOUND. Recorded here, at the moment of comparison, it is
                # exact and survives whatever happens to the objects afterwards.
                try:
                    self.decision_log.write(
                        "merge_refused", getattr(a, "object_id", a.label),
                        candidate=getattr(b, "object_id", b.label),
                        reason=reason,
                        a_label=a.label, b_label=b.label,
                        a_has_color=_known(a.color), b_has_color=_known(b.color),
                        a_has_material=_known(a.material), b_has_material=_known(b.material),
                        a_has_description=_known(a.description),
                        b_has_description=_known(b.description),
                        # WHICH SWEEP, AND BOTH SIDES' GEOMETRY. Measured by the DGX lane on the
                        # 144-stop tour 20260914_172628, and both make a question unanswerable
                        # rather than merely harder:
                        #   * no refusal row carried a sweep number (0 of 19,892), and
                        #     `not_offered_summary` is an aggregate with no pair identities, so a
                        #     pair's streak cannot be reconstructed at all -- which is exactly
                        #     what `merge_min_consecutive` has to be judged on. Time-binning gave
                        #     59 clusters against the engine's own 62.
                        #   * of the 68 pairs held at 1/2 consecutive, BOTH sides survived to the
                        #     final state in 0 cases: a held pair is by definition one the run
                        #     later resolved, so persistent_perception.json has no box for one
                        #     side and no assignment rule can score it against ground truth.
                        #     c17e38c fixed this for applied merges; the interesting pairs are
                        #     the ones that did NOT commit.
                        # Compact on purpose (six rounded numbers, not a dict): GA-232's lesson is
                        # that per-pair rows are what took this file to 1.71 GB. Added keys only.
                        sweep=self._merge_sweep,
                        a_box=_box_row(a.bbox), b_box=_box_row(b.bbox),
                        a_fused=_box_row(getattr(a, "fused_bbox", None)),
                        b_fused=_box_row(getattr(b, "fused_bbox", None)),
                        similarity=similarity, dry_run=bool(dry_run), **extra)
                except Exception as e:
                    self.get_logger().error(f"decision_log merge_refused failed: {e}")

            print("══════════════════════════════════════════════")
            print(f"🔍 MERGE CHECK: {len(objects)} objects in memory")
            print("══════════════════════════════════════════════")

            # GA-186. Candidate generation is a NAMED STEP now, and it is where the two
            # engines differ first. "legacy" used to offer EVERY pair, which is what every
            # run before the dev/lost3dsg-cleanup merge measured; it now applies an AABB
            # broad phase at `merge_aabb_margin_m` first, so that sentence no longer
            # describes it. "evidence" offers only pairs inside the two objects' own
            # covariance shells. BOTH engines now report what they excluded, as a counted
            # `not_offered_summary` row, so a pair that was never compared is visible in the
            # bundle instead of being invisible the way refusals were before they were
            # logged.
            pair_iter, assoc_ctx, assoc_objs = self._merge_candidates(
                objects, _refused, MAX_DISTANCE)

            for a, b, pair_meta in pair_iter:
                    # GA-23: `a` is re-tested on every pair rather than once per outer
                    # iteration. A pair resolving with keeper = b condemns `a` while later
                    # pairs still hold it, and it could then be chosen keeper again -- so the
                    # apply block would write keeper.bbox onto an object already removed from
                    # the world model. The flat iteration makes this a plain per-pair check.
                    if a in to_remove:
                        continue
                    if b in to_remove:
                        # Also logged: a pair skipped because one side is already condemned
                        # is a pair that was never judged on its own evidence, and the count
                        # of those is how you tell "the gate refused it" from "the gate
                        # never saw it".
                        _refused(a, b, "already_condemned", None,
                                 decision_reason="one side was condemned earlier in this sweep",
                                 offered_by=pair_meta.get("offered_by"))
                        continue

                    near_geometry_override = False
                    near_geometry_iou = None

                    if a.bbox is None or b.bbox is None:
                        # THE LAST UNLOGGED EXIT IN THE SELECTION PATH, and it matters more
                        # than its two lines suggest. MEASURED in run 20260831_184822: a
                        # single service call saw 118 objects and logged 121 pairs, where
                        # all-pairs is 6903 -- so 98% of pairs left the loop before reaching
                        # any gate, and every refusal reason IS logged (similarity 5996,
                        # distance 285, room 9). The shortfall is therefore entirely in the
                        # exits that record nothing, and this is one of the two.
                        #
                        # The consequence is not abstract: ALL EIGHT air-conditioner pairs
                        # within 1.5 m of each other in that run were never offered to the
                        # comparison at all -- including a 0.385 m pair whose volumes are
                        # 0.0016 m3 and 0.2338 m3, a 146x ratio and the exact fragment/whole
                        # shape the redesign exists to catch. Nobody could say why, because
                        # the skip left no trace.
                        #
                        # A pair that is never compared is invisible in precisely the way
                        # refusals were before they were logged. So it is logged now.
                        _refused(a, b, "bbox_absent", None,
                                 decision_reason="a side has no bbox, so locality cannot be evaluated",
                                 offered_by=pair_meta.get("offered_by"),
                                 a_has_bbox=a.bbox is not None, b_has_bbox=b.bbox is not None)
                        continue

                    # GA-25: locality before attributes. Geometry, never the room-type
                    # belief. `room_at_bbox` returns None when it cannot say, and None is
                    # UNKNOWN -- never "same room". Refusing only on positive disagreement
                    # keeps the gate strict without inventing separation the map cannot
                    # support; the similarity and distance gates still apply beneath it.
                    #
                    # assign_room_by_geometry is NOT usable here: it falls back to the
                    # robot's own room, in the same type as a real answer, so two objects in
                    # different rooms would read as the same room exactly when the map is
                    # least able to separate them (GA-28).
                    a_label = a.label.split('#')[0] if '#' in a.label else a.label
                    b_label = b.label.split('#')[0] if '#' in b.label else b.label

                    room_a = self.room_manager.room_at_bbox(a.bbox)
                    room_b = self.room_manager.room_at_bbox(b.bbox)

                    if MERGE_ENGINE == "evidence":
                        # 2026-09-14 owner ruling, in this order: room gate, then geometry,
                        # then the evidence decision. Both gates read the fused box when there
                        # is one. The legacy arm below keeps its own gates untouched, so an
                        # ablation against it compares the ruling with the pre-ruling system.
                        ba, bb_ = assoc.locality_bounds(a), assoc.locality_bounds(b)
                        overlapping = (ba is not None and bb_ is not None
                                       and assoc.intersection_volume(ba, bb_) > 0.0)
                        if (room_a is not None and room_b is not None and room_a != room_b
                                and not overlapping):
                            # Two boxes that OVERLAP are one thing whichever side of a room
                            # boundary each centre fell on. MEASURED on GA-493: 56 of 57
                            # distinct room refusals were one kitchen boundary (room_4/room_5)
                            # and 7 of them scored above the merge gate. Only NON-overlapping
                            # boxes in different rooms are refused here; the similarity is
                            # computed solely to be recorded.
                            print(f"   ❌ DIFFERENT ROOMS ({room_a} != {room_b}), boxes apart")
                            _room_sim, _room_ev = _pair_similarity(a, b, a_label, b_label)
                            _refused(a, b, "room", _room_sim,
                                     evidence_count=_room_ev["optional_count"],
                                     decision_reason="rooms differ and the boxes do not intersect",
                                     offered_by=pair_meta.get("offered_by"),
                                     room_a=room_a, room_b=room_b)
                            self._interrupt_hypothesis(a, b, "room", dry_run)
                            continue
                        if assoc.geometry_compatible(ba, bb_, LOCALITY_GAP_M) is False:
                            # The one locality test both stages apply: intersect, or a largest
                            # per-axis gap at most LOCALITY_GAP_M. A pair that fails it is
                            # never scored on attributes (GA-25's order, kept).
                            _refused(a, b, "geometry", None,
                                     decision_reason="largest per-axis gap exceeds the locality gap",
                                     gap_m=round(assoc.box_gap(ba, bb_), 3),
                                     threshold_gap_m=LOCALITY_GAP_M, room_a=room_a, room_b=room_b,
                                     offered_by=pair_meta.get("offered_by"))
                            self._interrupt_hypothesis(a, b, "geometry", dry_run)
                            continue
                        # GA-186. No gate cascade and no similarity constant: every channel
                        # runs, the log-odds are fused, and the pair commits only if the
                        # total clears log(cost_ratio). Three outcomes, not two -- a HOLD is
                        # not a refusal, and recording them as the same thing is what made
                        # "the gate refused it" and "the gate could not tell" indistinguishable.
                        aa = assoc_objs.get(id(a))
                        bb = assoc_objs.get(id(b))
                        if aa is None or bb is None:
                            _refused(a, b, "assoc_object_missing", None,
                                     decision_reason="no AssocObject was built for a side",
                                     offered_by=pair_meta.get("offered_by"))
                            continue
                        # Sorted, so the same pair keys identically whichever side is `a`
                        # this sweep -- candidate order is not stable between sweeps and an
                        # order-dependent key would start a fresh hypothesis every time,
                        # silently disabling the persistence requirement below.
                        hyp_key = tuple(sorted((str(getattr(a, "object_id", None) or a.label),
                                                str(getattr(b, "object_id", None) or b.label))))
                        ps = assoc.score_pair(aa, bb, assoc_ctx)
                        threshold = assoc.commit_threshold(assoc_ctx.cost_ratio)

                        # GA-188. The verdict comes from a HYPOTHESIS THAT PERSISTS ACROSS
                        # SWEEPS, not from this one scoring. Scoring once and committing let
                        # a single frame's geometry error destroy an identity; the design
                        # says a decision must survive being re-measured. Hypothesis also
                        # owns two things this branch was reimplementing badly: a veto is
                        # PERMANENT for the pair (co-visibility is a fact, not evidence to
                        # be outweighed later), and the full per-frame history is kept, so
                        # the merge is explainable and reversible afterwards.
                        # A dry run must not advance a LIVE pair's streak (review 2026-09-14),
                        # but it must still be able to say "would merge on the second look":
                        # dry runs accumulate in their own store. ponytail: that store is not
                        # garbage-collected; it grows with the pairs dry runs ever scored.
                        store = self._hypotheses if not dry_run else self._dry_hypotheses
                        h = store.get(hyp_key)
                        if h is None:
                            h = assoc.Hypothesis(hyp_key)
                            store[hyp_key] = h
                        h.update(ps, frame_id=self._merge_sweep)
                        decision, why = h.decide(threshold,
                                                 min_evidence=MERGE_MIN_EVIDENCE,
                                                 min_consecutive=MERGE_MIN_CONSECUTIVE)

                        rec = ps.as_record()
                        rec.update(distance=pair_meta.get("distance_m"),
                                   reach_m=pair_meta.get("reach_m"),
                                   offered_by=pair_meta.get("offered_by"),
                                   room_a=room_a, room_b=room_b, engine="evidence",
                                   # The THIRD unit of a key called `threshold`: log-odds,
                                   # not the similarity or the metres the 2026-09-06 rename
                                   # split apart. Named rather than retired -- this is the
                                   # arm that actually runs, so every existing reader of
                                   # `threshold` on a merge_refused row is reading THIS,
                                   # and the legacy key stays until those readers move.
                                   threshold_log_odds=round(threshold, 4),
                                   threshold=round(threshold, 4),
                                   hypothesis_total=(None if h.vetoed_by else round(h.total, 4)),
                                   updates=len(h.history), decision_reason=why)

                        if decision != "merge":
                            # "reject" (vetoed), "abstain" (nothing measured yet) and "hold"
                            # (below threshold, or not yet persistent, or containment
                            # unchecked) are DIFFERENT ANSWERS and are recorded as such. A
                            # hold is not a refusal: the pair is still live and will be
                            # re-decided on the next sweep.
                            _refused(a, b, decision, None if h.vetoed_by else h.total, **rec)
                            continue
                        rec["provenance"] = h.provenance()
                        h.committed = True

                        # Committed. The apply block below is shared with the legacy engine
                        # and reads `sim`, `dist` and `ev`, so they are filled from the
                        # evidence result -- `sim` is LOG-ODDS here, not a 0..1 similarity,
                        # which is why every record carries `engine`.
                        sim = h.total
                        ev = {"optional_count": ps.evidence_count}
                        dist = pair_meta.get("distance_m")
                        if dist is None:
                            dist = float(np.linalg.norm(
                                np.asarray(aa.centroid) - np.asarray(bb.centroid)))
                    if (MERGE_ENGINE == "legacy"
                            and room_a is not None and room_b is not None and room_a != room_b):
                        # LEGACY ARM ONLY, unchanged: refuses on room whether or not the boxes
                        # overlap. The evidence arm's room gate stands above the engine switch
                        # (2026-09-14). The similarity is computed HERE, on the refusal path
                        # only, and solely to be recorded (GA-25's order is kept).
                        print(f"   ❌ DIFFERENT ROOMS ({room_a} != {room_b})")
                        _room_sim, _room_ev = _pair_similarity(a, b, a_label, b_label)
                        _refused(a, b, "room", _room_sim,
                                 evidence_count=_room_ev["optional_count"],
                                 decision_reason="different rooms (legacy arm)",
                                 offered_by=pair_meta.get("offered_by"),
                                 room_a=room_a, room_b=room_b)
                        continue

                    ax = (a.bbox['x_min'] + a.bbox['x_max']) / 2.0
                    ay = (a.bbox['y_min'] + a.bbox['y_max']) / 2.0
                    az = (a.bbox['z_min'] + a.bbox['z_max']) / 2.0
                    bx = (b.bbox['x_min'] + b.bbox['x_max']) / 2.0
                    by = (b.bbox['y_min'] + b.bbox['y_max']) / 2.0
                    bz = (b.bbox['z_min'] + b.bbox['z_max']) / 2.0

                    # GA-186: the loop is flat now, so there are no `i`/`j` indices to print.
                    # They were the enumeration of a nested loop that no longer exists, and
                    # printing them here would have been a NameError on the legacy path.
                    print(f"\n📐 COMPARISON {a_label} vs {b_label}:")
                    print(f"   Pos A: ({ax:.2f}, {ay:.2f}, {az:.2f})")
                    print(f"   Pos B: ({bx:.2f}, {by:.2f}, {bz:.2f})")

                    if MERGE_ENGINE == "legacy":
                        dist = np.sqrt((ax - bx)**2 + (ay - by)**2 + (az - bz)**2)
                    print(f"   Distance: {dist:.3f}m (threshold: {MAX_DISTANCE}m)")

                    if MERGE_ENGINE == "legacy" and dist > MAX_DISTANCE:
                        # GA-25, second residual. This gate used to run LAST, after
                        # `_pair_similarity` had already embedded both descriptions -- the
                        # expensive comparison on every pair the cheap one was about to
                        # reject. On the legacy engine every pair is offered, so that was
                        # every pair in the map. It now runs first, and the refusal record
                        # carries no similarity because none was measured.
                        print(f"   ❌ TOO FAR APART ({dist:.2f}m > {MAX_DISTANCE}m)")
                        # Same joint rename: this path's threshold is METRES, the
                        # similarity path's is unitless -- the typed key says which.
                        _refused(a, b, "distance", None,
                                 evidence_count=None,
                                 decision_reason="centre distance exceeds the legacy gate",
                                 offered_by=pair_meta.get("offered_by"),
                                 distance=dist, threshold_distance_m=MAX_DISTANCE,
                                 room_a=room_a, room_b=room_b)
                        continue

                    if MERGE_ENGINE == "legacy":
                        near_geometry_override, near_geometry_iou = _near_geometry_duplicate(
                            a, b, dist)

                    if MERGE_ENGINE == "legacy":
                        # Embedding lazy; missing description embeddings are absent evidence,
                        # not a reason to skip the pair (lost_similarity renormalises).
                        # GUARDED: in evidence mode `sim` and `ev` are already the fused
                        # log-odds and the channel count, and recomputing them here would
                        # silently overwrite the decision that was just made.
                        sim, ev = _pair_similarity(a, b, a_label, b_label)

                    print("   Sim semantiche:")
                    print(f"     Label: '{a_label}' vs '{b_label}'")
                    print(f"     Colore: '{a.color}' vs '{b.color}'")
                    print(f"     Materiale: '{a.material}' vs '{b.material}'")
                    print(f"     Similarity: {sim:.3f} (threshold: {MIN_SIMILARITY})")

                    # GA-21: the `forzo merge` bypass that stood here is deleted. It fired
                    # ONLY when the evidence had already said do not merge, and overrode that
                    # with label equality plus a hardcoded 0.5 overlap -- so a red hardback
                    # and a blue notebook, both labelled "book" at the same spot, scored ~0.55,
                    # were correctly refused, and were then fused anyway. Overlap is locality,
                    # not similarity; the same confusion as GA-05, in the one operation that
                    # destroys an identity.
                    if (MERGE_ENGINE == "legacy" and sim < MIN_SIMILARITY
                            and not near_geometry_override):
                        print(f"   ❌ LOW SIMILARITY ({sim:.2f} < {MIN_SIMILARITY})")
                        # Unit-typed key (joint rename with the ontology lane, their
                        # inbox 00002/00004/00005): `threshold` was unit-polymorphic -- 0.925
                        # cosine here, 0.8 METRES on the distance path below -- and was
                        # misread once by a reader and once by a test. TRANSITION CLOSED
                        # 2026-09-06 on the owner's authorisation: the legacy key is retired
                        # here, ontology's reader prefers the typed key and keeps a fallback
                        # for rows written before d76f997. Rows from the EVIDENCE engine still
                        # carry `threshold` -- a third unit, log-odds -- and are named by
                        # `threshold_log_odds` beside it rather than swept into this rename.
                        _refused(a, b, "similarity", sim,
                                 evidence_count=ev["optional_count"],
                                 decision_reason="attribute score below the legacy merge floor",
                                 offered_by=pair_meta.get("offered_by"),
                                 threshold_similarity=MIN_SIMILARITY,
                                 room_a=room_a, room_b=room_b)
                        continue

                    decision_reason = (
                        "near_geometry_label_iou"
                        if near_geometry_override else
                        ("similarity" if MERGE_ENGINE == "legacy" else "evidence"))
                    if near_geometry_override:
                        print(
                            f"   ✅ GEOMETRY OVERRIDE: same label, centre distance {dist:.3f}m, "
                            f"AABB IoU {near_geometry_iou:.3f}; similarity {sim:.3f} is non-authoritative"
                        )

                    # GA-101: a score that passed the gate on NOTHING must not merge.
                    # With colour, material and description all absent the divisor is the
                    # label weight alone, so two identical label strings score exactly
                    # 1.0000 -- above any threshold, from zero measured evidence. Raising
                    # MIN_SIMILARITY cannot reach this: the pairs it is meant to catch sit
                    # ABOVE the ones that actually agreed on something.
                    #
                    # Placed AFTER the similarity gate on purpose. Running it first would
                    # relabel every genuine low-score refusal as "evidence_absent" and hide
                    # a real signal -- a different-label pair with nothing else measured
                    # scored 0.0000 because the labels WERE compared and disagreed. This
                    # only refuses pairs that would otherwise have been merged.
                    if MERGE_ENGINE == "legacy" and ev["optional_count"] < MERGE_MIN_EVIDENCE:
                        print(f"   ❌ NO EVIDENCE ({ev['optional_count']} optional "
                              f"terms < {MERGE_MIN_EVIDENCE}; sim {sim:.3f} on the label alone)")
                        _refused(a, b, "evidence_absent", sim,
                                 evidence_count=ev["optional_count"],
                                 decision_reason="no optional attribute term was comparable",
                                 offered_by=pair_meta.get("offered_by"),
                                 required=MERGE_MIN_EVIDENCE, room_a=room_a, room_b=room_b)
                        continue

                    # GA-25 / GA-314: which identity survives follows the evidence, not list
                    # order -- credibility first, then the rule `merge_rank` documents.
                    keeper, discard = (a, b) if merge_rank(a) <= merge_rank(b) else (b, a)

                    # The survivor's geometry must enclose both measurements. Keeping only the
                    # keeper's box left a merged duplicate represented by whichever partial view
                    # happened to win the identity tie-break. `_combine_object_geometry` fits one
                    # coherent PCA box over the two measured boxes and keeps the AABB exact too.
                    merged_bbox, merged_geometry_acc = _combine_object_geometry(keeper, discard)

                    vol_a = ((a.bbox['x_max']-a.bbox['x_min']) *
                            (a.bbox['y_max']-a.bbox['y_min']) *
                            (a.bbox['z_max']-a.bbox['z_min']))
                    vol_b = ((b.bbox['x_max']-b.bbox['x_min']) *
                            (b.bbox['y_max']-b.bbox['y_min']) *
                            (b.bbox['z_max']-b.bbox['z_min']))
                    vol_m = ((merged_bbox['x_max']-merged_bbox['x_min']) *
                            (merged_bbox['y_max']-merged_bbox['y_min']) *
                            (merged_bbox['z_max']-merged_bbox['z_min']))

                    print("   ✅ MERGE!")
                    print(f"     Volume A: {vol_a:.3f}m³ | Volume B: {vol_b:.3f}m³ → Kept: {vol_m:.3f}m³")
                    print(f"     Kept: '{keeper.label}' | Removed: '{discard.label}'")
                    print(f"     Keeper desc: '{keeper.description[:40]}...'")
                    print(f"     Bbox kept: x[{merged_bbox['x_min']:.2f},{merged_bbox['x_max']:.2f}] "
                        f"y[{merged_bbox['y_min']:.2f},{merged_bbox['y_max']:.2f}] "
                        f"z[{merged_bbox['z_min']:.2f},{merged_bbox['z_max']:.2f}]")

                    to_remove.add(discard)
                    to_remove_pairs.append({
                        "keeper": keeper, "discard": discard, "bbox": merged_bbox,
                        "geometry_acc": merged_geometry_acc,
                        "similarity": sim,
                        "distance": dist,
                        "decision_reason": decision_reason,
                        "near_geometry_override": near_geometry_override,
                        "near_geometry_iou": near_geometry_iou,
                        "keeper_room": room_a if keeper is a else room_b,
                        "discard_room": room_b if keeper is a else room_a,
                    })
                    to_remove_pairs[-1]["log"] = {
                        "keeper":      keeper.label,
                        "keeper_id":   getattr(keeper, "object_id", None),
                        "discarded":   discard.label,
                        # GA-25's room gate refuses a pair only when BOTH rooms are known
                        # and differ. Recording both -- including None for "geometry cannot
                        # say" -- is what makes that rule checkable from a bundle: without
                        # it, a merge that should have been refused and one that was
                        # correctly allowed are identical in the log.
                        "keeper_room":    room_a if keeper is a else room_b,
                        "discarded_room": room_b if keeper is a else room_a,
                        "distance":    round(dist, 3),
                        "similarity":  round(sim, 3),
                        "decision_reason": decision_reason,
                        "near_geometry_override": near_geometry_override,
                        "near_geometry_iou": (
                            round(near_geometry_iou, 3)
                            if near_geometry_iou is not None else None),
                        # Keep the same evidence metadata on accepted merges that the
                        # refusal path already records. A similarity of 1.0 is not enough
                        # to calibrate this gate: it can come from a label-only/identical
                        # attribute comparison, or from several genuinely agreeing fields.
                        "evidence_count": ev.get("optional_count"),
                        "engine": MERGE_ENGINE,
                        "score_type": "log_odds" if MERGE_ENGINE == "evidence" else "similarity",
                        "threshold_similarity": MIN_SIMILARITY if MERGE_ENGINE == "legacy" else None,
                        "threshold_distance_m": MAX_DISTANCE if MERGE_ENGINE == "legacy" else None,
                        "threshold_near_distance_m": (
                            MERGE_NEAR_DISTANCE if MERGE_ENGINE == "legacy" else None),
                        "threshold_near_iou": (
                            MERGE_NEAR_IOU_THRESHOLD if MERGE_ENGINE == "legacy" else None),
                        "threshold_log_odds": threshold if MERGE_ENGINE == "evidence" else None,
                        "decision_details": rec if MERGE_ENGINE == "evidence" else None,
                        "merged_bbox": merged_bbox,
                        # CORRECTED 2026-09-14. This read "keeper", which was true when the
                        # key was added on 2026-09-01 and the merge really did keep the
                        # keeper's own box. It is not true now: the box written onto the
                        # keeper is the AABB enclosing BOTH objects' corner clouds
                        # (`_combine_object_geometry` above). A row that names the wrong
                        # source is worse than no row, because the next reader acts on it.
                        # `bbox_from_object_id` below stays the keeper's id, and correctly:
                        # the hull IS written onto the keeper. No reader consumes this value
                        # (the only other `bbox_source` in the tree is an unrelated key in
                        # build_hm3d_eval_manifest.py), so this corrects a value, not a
                        # meaning.
                        "bbox_source": "union_hull",
                        "bbox_from_object_id": getattr(keeper, "object_id", None) or keeper.label,
                        # THE DISCARDED SIDE'S GEOMETRY, BECAUSE THE MERGE DESTROYS IT.
                        # MEASURED by the DGX lane on the 144-stop tour 20260914_172628: of 51
                        # applied merges, the keeper survives in persistent_perception.json 51
                        # times and the DISCARD 0 times, so no assignment rule can score an
                        # applied merge against ground truth from the bundle -- merge PRECISION
                        # was uncomputable by construction, while every refusal was scoreable.
                        # Recorded here, at the moment of the decision, so it survives the
                        # removal (the same reason `_refused` records each side's terms).
                        # Added keys only (working rule 6).
                        "discarded_bbox": dict(discard.bbox) if discard.bbox else None,
                        "discarded_fused_bbox": (dict(discard.fused_bbox)
                                                 if getattr(discard, "fused_bbox", None) else None),
                        "keeper_fused_bbox": (dict(keeper.fused_bbox)
                                              if getattr(keeper, "fused_bbox", None) else None),
                        "discarded_label_full": discard.label,
                        "discarded_color": discard.color,
                        "discarded_material": discard.material,
                        "discarded_description": discard.description,
                    }

            # GA-197. THE MERGE RECORDS ARE WRITTEN WHETHER OR NOT THIS IS A DRY RUN, and
            # they used to sit inside the `not dry_run` guard below. A dry run therefore
            # decided merges, printed the "MERGE!" banner, filled `to_remove_pairs` -- and
            # wrote NOTHING, while every REFUSAL logged normally. The bundle then showed a
            # full refusal stream and zero merges, which reads as "the gate refused
            # everything": exactly the ambiguity refusal-logging was introduced to end,
            # reintroduced from the other side. The `dry_run` field on the record was dead
            # by construction -- it could only ever be written as False.
            #
            # A dry run exists to say what the gate WOULD do. Its decisions are the output.
            # (Review 2026-09-14: the rows are written AFTER the locked apply below, over
            # `applied_pairs`, so a live pair skipped as stale leaves no `merge` row, no
            # merge_log entry and no operations.txt line. A dry run records every decided
            # pair, as before.)

            # `merged_count` and the merge records report APPLIED merges. A dry run applies
            # nothing and its decisions ARE its output, so it keeps the decided pairs; the
            # live path replaces this with what survived the locked re-check below.
            applied_pairs = list(to_remove_pairs)
            if to_remove_pairs and not dry_run:
                print(f"\n🗑️ REMOVING: {len(to_remove_pairs)} duplicate objects:")

                # GA-393, NARROWED per the owner's ruling ("keep the fix; reduce the guard to
                # the part that writes"). The comparison sweep above runs on `objects`, a
                # snapshot taken at entry, so holding the lock across it blocked the other
                # executor threads for a median 569 ms and up to 2.8 s of a 7.4 s cycle. THIS
                # is where the world model is actually mutated, and it was the only service
                # callback in this file taking no lock at all while add, update, delete and
                # query all take one.
                #
                # Because the sweep ran unlocked, a pair decided there may no longer be valid:
                # another thread can have removed either side in between. BOTH are re-checked
                # inside the lock -- the keeper was never checked before -- and a stale pair is
                # skipped and COUNTED, never applied to an object that has left the map.
                stale = 0
                applied_pairs = []
                with wm.lock:
                    for pair in to_remove_pairs:
                        keeper, discard, merged_bbox = pair["keeper"], pair["discard"], pair["bbox"]

                        if keeper not in wm.persistent_perceptions:
                            stale += 1
                            continue
                        if discard not in wm.persistent_perceptions:
                            # Same race as the keeper: the discard left the map between the
                            # unlocked sweep and this write. Nothing is merged, so nothing
                            # is counted. Before, this case fell through to the room-graph
                            # update and was still reported as a merge.
                            stale += 1
                            continue
                        applied_pairs.append(pair)
                        # Merge geometry while both tracks still exist. The
                        # operation preserves distinct view IDs, so two
                        # tracks containing the same cycle do not count it twice.
                        merge_bbox_fusion(keeper, discard)
                        # Recompute against the current keeper because several accepted
                        # pairs in one sweep can share the same keeper. This makes a merge
                        # chain enclose every member instead of letting the last pair shrink
                        # the box back to its first two observations. (shared tree, 2026-09-14)
                        merged_bbox, merged_geometry_acc = _combine_object_geometry(keeper, discard)
                        keeper.bbox = merged_bbox
                        keeper._yaw_acc = merged_geometry_acc
                        # BEFORE the discard leaves the map below: the survivor inherits the
                        # sightings, or the merge destroys the evidence it was made from.
                        _absorb_into_keeper(keeper, discard, merged_bbox)
                        cx = (discard.bbox['x_min'] + discard.bbox['x_max']) / 2.0
                        cy = (discard.bbox['y_min'] + discard.bbox['y_max']) / 2.0
                        print(f"   - {discard.label} @ ({cx:.2f}, {cy:.2f})")
                        wm.persistent_perceptions.remove(discard)
                        # GA-26: the store learns about the merge, or its row stays active
                        # forever and the database disagrees with the map by one object.
                        if hasattr(self, 'db'):
                            try:
                                self.db.on_object_merged(keeper, discard, step=self.tracking_step_counter)
                            except Exception as e:
                                self.get_logger().error(f"db.on_object_merged failed: {e}")

                        discard_room = getattr(discard, 'room_id', None)
                        if discard_room and discard_room in self.room_manager.scene_graph:
                            objs = self.room_manager.scene_graph[discard_room]["objects"]
                            if discard.label in objs:
                                objs.remove(discard.label)

                        self.room_manager.update_room_geometry(
                            getattr(keeper, 'room_id', self.room_manager.current_room_id),
                            merged_bbox
                        )

                    save_persistent_perceptions(self)
                    publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
                    publish_persistent_centroids(self, wm, self.persistent_centroids_pub)

                if stale:
                    self.log_both("warn", f"[MERGE] {stale} pair(s) skipped: an object left "
                                          f"the map between the unlocked sweep and the write")

                try:
                    with open(OPERATIONS_LOG, 'a') as f:
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        for pair in applied_pairs:
                            keeper, discard = pair["keeper"], pair["discard"]
                            f.write(f"[{timestamp}] 🔗 MERGE: '{discard.label}' → '{keeper.label}'\n")
                except Exception as e:
                    self.get_logger().error(f"Could not write to operations.txt: {e}")

                # A merge is the one decision that RE-ROUTES history: the discarded object stops
                # existing and its past belongs to the keeper. Record both ids, so a reader
                # reconstructing the keeper knows to follow the discarded one backwards, and a
                # reader looking up the discarded id learns where it went instead of finding a
                # history that simply stops.
            elif not to_remove_pairs:
                print("\n✅ NO duplicates found.")

            # GA-197 + review 2026-09-14: one `merge` row per RECORDED pair -- every decided
            # pair on a dry run, every APPLIED pair on a live run.
            for pair in applied_pairs:
                keeper, discard = pair["keeper"], pair["discard"]
                try:
                    self.decision_log.write(
                        "merge", getattr(keeper, "object_id", keeper.label),
                        merged_from=getattr(discard, "object_id", discard.label),
                        keeper_label=keeper.label, discarded_label=discard.label,
                        keeper_room=pair["keeper_room"], discard_room=pair["discard_room"],
                        sweep=self._merge_sweep,
                        similarity=pair["similarity"],
                        # the shared tree's legacy-arm near-geometry fields, carried verbatim
                        decision_reason=pair.get("decision_reason"),
                        near_geometry_override=pair.get("near_geometry_override", False),
                        near_geometry_iou=pair.get("near_geometry_iou"),
                        distance=pair.get("distance"),
                        threshold_near_distance_m=(
                            MERGE_NEAR_DISTANCE if pair.get("near_geometry_override") else None),
                        threshold_near_iou=(
                            MERGE_NEAR_IOU_THRESHOLD if pair.get("near_geometry_override") else None),
                        dry_run=bool(dry_run))
                except Exception as e:
                    self.get_logger().error(f"decision_log merge failed: {e}")

            print("══════════════════════════════════════════════\n")

            # ── Risposta ──────────────────────────────────────────────────────
            response.success        = True
            response.merged_count   = len(applied_pairs)
            response.merge_log_json = json.dumps([p["log"] for p in applied_pairs])
            response.message        = (
                f"{'[DRY RUN] ' if dry_run else ''}"
                f"{len(applied_pairs)} merge(s) "
                f"{'simulati' if dry_run else 'eseguiti'}"
            )

        except Exception as e:
            self.get_logger().error(f"_cb_merge_objects failed: {e}")
            response.success        = False
            response.message        = str(e)
            response.merged_count   = 0
            response.merge_log_json = "[]"

        return response
        

    @synchronized_world_model
    def _cb_add_object(self, request, response):
        self._prepare_receipt(response)
        mutation_event_id = uuid.uuid4().hex
        new_obj = None
        applied = False
        try:
            bbox = {
                "x_min": request.x_min, "x_max": request.x_max,
                "y_min": request.y_min, "y_max": request.y_max,
                "z_min": request.z_min, "z_max": request.z_max,
            }
            _apply_orientation(bbox, request)   # GA-312
            label       = request.label
            description = request.description
            color       = request.color
            material    = request.material

            raw_clip_embedding = getattr(request, "clip_embedding", None)
            sent_clip_embedding = bool(getattr(request, "has_clip_embedding", True))
            clip_embedding = (
                normalise_clip_embedding(raw_clip_embedding)
                if sent_clip_embedding else None
            )
            if clip_embedding is not None:
                bbox["clip_embedding"] = clip_embedding

            new_obj = Object(label, _centroid_from_bbox(bbox), bbox, description, color, material)
            new_obj.clip_embedding = clip_embedding
            # Identity remains stable when visual attributes are refined.
            new_obj.object_id = f"obj_{uuid.uuid4().hex}"
            new_obj.object_revision = 1
            new_obj.geometry_epoch = 1
            new_obj.creation_time = time.time()
            new_obj.relations = {
                "isIn": set(),
                "isOn": set(),
                "isNextTo": set(),
                "isAbove": set(),
                "isUnder": set(),
            }
            apply_request_fusion(new_obj, request, label=label)

            raw_embedding = getattr(request, 'description_embedding', None)
            # GA-184. The flag, not the length, says whether an embedding was SENT. A
            # `float32[]` cannot carry null, so absence and emptiness are identical bytes on
            # this boundary and `size == 0` had to stand in for both -- which is why one
            # 58-minute run logged 190 "empty embedding serialised" warnings for objects
            # that simply had no description to embed. That was the message type reporting
            # its own limitation, not the producer misbehaving.
            #
            # `getattr(..., True)` is the compatibility default and it is the SAFE direction:
            # against a service built before the flag existed it falls back to reading the
            # length, which is exactly today's behaviour. Defaulting False would silently
            # discard every real embedding the moment the two sides were out of step.
            sent_embedding = bool(getattr(request, 'has_description_embedding', True))

            if raw_embedding is None or not sent_embedding:
                # Absent, and SAID to be absent -- an ordinary state for an object with no
                # usable description, so this is debug rather than a warning. It was logged
                # at warn level and became 190 lines of noise that read like a defect.
                self.log_both(
                    'debug',
                    f"[EMBEDDING] no embedding for '{label}' "
                    f"(descrizione='{description}')"
                )
                new_obj.embedding = None
            else:
                embedding = np.asarray(raw_embedding, dtype=np.float32).flatten()
                if embedding.size == 0:
                    # The flag SAYS one was sent and it is empty. Now that absence has its
                    # own channel, this is a genuine contradiction between the two fields
                    # and stays a warning.
                    self.log_both(
                        'warn',
                        f"[EMBEDDING] has_description_embedding=True ma array VUOTO per "
                        f"'{label}' (descrizione='{description}') -- i due campi si "
                        f"contraddicono"
                    )
                    new_obj.embedding = None
                else:
                    if embedding.size != 300:
                        self.log_both(
                            'warn',
                            f"[EMBEDDING] Embedding con dimensione inattesa per '{label}': "
                            f"{embedding.size}"
                        )
                    new_obj.embedding = embedding.tolist()

            if hasattr(request, 'room_id') and request.room_id:
                assigned_room = request.room_id
            else:
                assigned_room = self.room_manager.current_room_id

            if not assigned_room:
                # config seam: without SLAM-derived room polygons no room can
                # ever be known, which would reject every object forever
                assigned_room = CFG["rooms"]["default_room_id"]
            if not assigned_room:
                raise ValueError(
                    f"cannot assign object '{label}': no current room "
                    f"is known (the robot is outside every polygon, or its pose is not available yet)."
                )

            new_obj.room_id = assigned_room
            self.room_manager.update_room_geometry(assigned_room, bbox)

            room_entry = self.room_manager.scene_graph.get(assigned_room)
            if room_entry is None:
                # difesa extra: se per qualche motivo la entry non esiste ancora, creala
                room_entry = self.room_manager.init_room_node(assigned_room)

            if label not in room_entry["objects"]:
                room_entry["objects"].append(label)

            new_obj.room_id = assigned_room
            self.room_manager.update_room_geometry(assigned_room, bbox)

            if label not in self.room_manager.scene_graph[assigned_room]["objects"]:
                self.room_manager.scene_graph[assigned_room]["objects"].append(label)

            wm.persistent_perceptions.append(new_obj)
            applied = True
            self._write_mutation_receipt(
                mutation_event_id, request, "applied_in_memory", "add",
                obj=new_obj,
                fusion_applied=bool(getattr(request, "has_fusion_voxels", False)),
            )

            in_exploration = getattr(request, 'in_exploration', False)
            phase = "exploration" if in_exploration else "tracking"
            step  = self.exploration_step_counter if in_exploration else self.tracking_step_counter
            self.db.on_new_object(new_obj, phase=phase, step=step)

            x_size = bbox["x_max"] - bbox["x_min"]
            y_size = bbox["y_max"] - bbox["y_min"]
            z_size = bbox["z_max"] - bbox["z_min"]
            volume = x_size * y_size * z_size

            mode_tag = "[EXPLORATION]" if in_exploration else f"[TRACKING STEP {self.tracking_step_counter}]"
            self.log_both('info', f"{mode_tag} New object '{label}' in {assigned_room} (vol: {volume:.3f} m³)")

            save_persistent_perceptions(self)
            self._write_mutation_receipt(
                mutation_event_id, request, "applied_and_checkpointed", "add",
                obj=new_obj,
                fusion_applied=bool(getattr(request, "has_fusion_voxels", False)),
            )

            if in_exploration:
                self.exploration_step_counter += 1

            try:
                with open(OPERATIONS_LOG, 'a') as f:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
                    cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
                    cz = (bbox["z_min"] + bbox["z_max"]) / 2.0
                    f.write(f"[{timestamp}] 🟢 AGGIUNTO: {label} in {assigned_room} "
                            f"a pos({cx:.2f}, {cy:.2f}, {cz:.2f})\n")
            except Exception as e:
                self.get_logger().error(f"Could not write to operations.txt: {e}")

            response.success   = True
            response.message   = f"Object '{label}' added to {assigned_room}"
            response.object_id = new_obj.object_id
            self._set_receipt_response(
                response, mutation_event_id, "applied_and_checkpointed", new_obj)

        except Exception as e:
            self.get_logger().error(f"_cb_add_object failed: {e}")
            response.success = False
            response.message = str(e)
            response.object_id = ""
            self._set_receipt_response(
                response, mutation_event_id,
                "incomplete" if applied else "refused", new_obj)

        return response
    

    def _cb_remove_object(self, request, response):
        try:
            pov_volume = {
                "x_min": request.pov_x_min,
                "x_max": request.pov_x_max,
                "y_min": request.pov_y_min,
                "y_max": request.pov_y_max,
                "z_min": request.pov_z_min,
                "z_max": request.pov_z_max,
            }

            uncertain_to_remove = []

            if pov_volume:
                for obj in self.uncertain_objects:
                    if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                        uncertain_to_remove.append(obj)

            for obj in uncertain_to_remove:
                self.uncertain_objects.remove(obj)

            save_uncertain_objects(self)

            response.success = True
            response.message = f"Removed {len(uncertain_to_remove)} uncertain objects"

        except Exception as e:
            response.success = False
            response.message = str(e)

        return response


    @synchronized_world_model
    def _cb_update_object(self, request, response):
        self._prepare_receipt(response)
        mutation_event_id = uuid.uuid4().hex
        mutation_applied = False
        updated_obj = None
        try:
            obj_id = request.object_id
            all_ids = [getattr(o, "object_id", None) or o.label for o in wm.persistent_perceptions]
            self.get_logger().info(f"[DEBUG UPDATE] Cerco '{obj_id}' tra: {all_ids}")
            best_match = next(
                (o for o in wm.persistent_perceptions if getattr(o, "object_id", None) == obj_id),
                None,
            )
            if best_match is None:
                response.success = False
                response.message = f"Object {obj_id} not found"
                response.object_id = ""
                response.distance = 0.0
                response.iou = 0.0
                response.replaced = False
                self._write_mutation_receipt(
                    mutation_event_id, request, "refused", "update",
                    reason="object_not_found")
                self._set_receipt_response(response, mutation_event_id, "refused")
                return response

            bbox = {
                "x_min": request.x_min,
                "x_max": request.x_max,
                "y_min": request.y_min,
                "y_max": request.y_max,
                "z_min": request.z_min,
                "z_max": request.z_max,
            }
            _apply_orientation(bbox, request)   # GA-312
            # GA-315 part 2. The view as it arrived, and the box to STORE for an in-place
            # update: AABB from this view, axis fused over every accepted view. A real move
            # (the rebuild branch below) starts from the raw view again.
            raw_bbox = dict(bbox)
            bbox, yaw_acc = fuse_orientation(best_match, raw_bbox)

            description_embedding = getattr(request, "description_embedding", None)
            raw_clip_embedding = getattr(request, "clip_embedding", None)
            sent_clip_embedding = bool(getattr(request, "has_clip_embedding", True))
            incoming_clip_embedding = (
                normalise_clip_embedding(raw_clip_embedding)
                if sent_clip_embedding else None
            )
            previous_clip_embedding = normalise_clip_embedding(
                getattr(best_match, "clip_embedding", None)
            )
            if previous_clip_embedding is None and isinstance(best_match.bbox, dict):
                previous_clip_embedding = normalise_clip_embedding(
                    best_match.bbox.get("clip_embedding")
                )
            clip_embedding = incoming_clip_embedding or previous_clip_embedding
            if clip_embedding is not None:
                # fuse_orientation returns a newly constructed geometry dict and therefore
                # intentionally drops arbitrary metadata. Put the appearance measurement
                # back on both the fused box and the raw move box before any update branch
                # can store either one.
                raw_bbox["clip_embedding"] = clip_embedding
                bbox["clip_embedding"] = clip_embedding

            updated_obj = best_match
            updated_obj.object_id = getattr(best_match, "object_id", None) or f"obj_{uuid.uuid4().hex}"
            distance = 0.0
            iou = 0.0

            if getattr(request, "update_bbox", False):
                old_bbox = best_match.bbox
                if old_bbox is None:
                    best_match.bbox = bbox
                    best_match._yaw_acc = yaw_acc   # GA-315 part 2
                    apply_request_fusion(best_match, request, label=best_match.label)
                    if clip_embedding is not None:
                        best_match.clip_embedding = clip_embedding
                    wm.refresh_spatial(best_match)
                    best_match.object_revision = int(
                        getattr(best_match, "object_revision", 0)) + 1
                    best_match.geometry_epoch = max(
                        1, int(getattr(best_match, "geometry_epoch", 0)))
                    mutation_applied = True
                    self._write_mutation_receipt(
                        mutation_event_id, request, "applied_in_memory", "update",
                        obj=best_match,
                        fusion_applied=bool(getattr(request, "has_fusion_voxels", False)))
                    save_persistent_perceptions(self)
                    self._write_mutation_receipt(
                        mutation_event_id, request, "applied_and_checkpointed", "update",
                        obj=best_match,
                        fusion_applied=bool(getattr(request, "has_fusion_voxels", False)))
                    response.success = True
                    response.message = "bbox initialized"
                    response.object_id = best_match.object_id
                    response.distance = 0.0
                    response.iou = 0.0
                    response.replaced = False
                    self._set_receipt_response(
                        response, mutation_event_id, "applied_and_checkpointed", best_match)
                    return response
                iou = compute_iou_3d(bbox, old_bbox)

                old_x = (old_bbox["x_min"] + old_bbox["x_max"]) / 2.0
                old_y = (old_bbox["y_min"] + old_bbox["y_max"]) / 2.0
                old_z = (old_bbox["z_min"] + old_bbox["z_max"]) / 2.0
                new_x = (bbox["x_min"] + bbox["x_max"]) / 2.0
                new_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
                new_z = (bbox["z_min"] + bbox["z_max"]) / 2.0
                distance = np.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2 + (new_z - old_z) ** 2)

                if "door" in best_match.label.lower():
                    # Doors bypass the plausibility gate (flat and tall). GA-26: they
                    # used to be ASSIGNED distance=0.0, iou=1.0 -- two measurements
                    # replaced by constants, so every door read as stationary.
                    best_match.bbox = bbox
                    best_match._yaw_acc = yaw_acc   # GA-315 part 2
                    updated_obj = best_match

                # THE GATE JUDGES THE INCOMING VIEW, NOT THE UNION. `bbox` above is
                # fuse_orientation(best_match, raw_bbox) -- the stored box UNIONED with this
                # view -- so asking "is it more than 3x the old box, or over 1.5 m3, or over
                # 3 m across?" of the union is a question that answers itself for any object
                # already near a limit. MEASURED with the real function on 20260914_180343's
                # 47 detector boxes: fusing a box WITH ITSELF inflates its volume by a median
                # 2.83x (max 4.14x), and 15 of the 47 would be refused if re-observed
                # identically. Of that run's 8 update attempts, 6 were refused
                # `implausible_bbox` and became duplicate objects -- and ground truth
                # (evaluator-only, read afterwards) says all 8 had paired the detection with
                # the RIGHT instance. The association verdict was right 8 of 8; this gate was
                # wrong 6 of 6. `raw_bbox` is what the sensor actually reported.
                elif bbox_is_suspicious(raw_bbox, old_bbox):
                    # GA-26: this used to set a rejection message and fall through to
                    # the success tail, which overwrote it with success=True -- byte-
                    # identical to an attribute-only update. The detection was NOT
                    # absorbed; per GA-10 the caller offers it to admission instead.
                    self.get_logger().warn(
                        f"[UPDATE] Bbox sospetta per '{best_match.label}', mantengo quella precedente"
                    )
                    response.success = False
                    response.message = f"bbox rejected as implausible for {obj_id}"
                    response.object_id = getattr(best_match, "object_id", "") or ""
                    response.distance = float(distance)
                    response.iou = float(iou)
                    response.replaced = False
                    self._write_mutation_receipt(
                        mutation_event_id, request, "refused", "update",
                        obj=best_match, reason="implausible_bbox")
                    self._set_receipt_response(
                        response, mutation_event_id, "refused", best_match)
                    return response

                elif (distance < UPDATE_IN_PLACE_DISTANCE_M or iou >= TRACKING_IOU_THRESHOLD
                      # 2026-09-14: the SAME locality test that admitted the detection. A view
                      # the association loop accepted because it intersects (or lies within
                      # LOCALITY_GAP_M of) the object's fused box is a re-observation, not a
                      # move. Review measured 33 of 205 tracking detections falling into the
                      # rebuild branch below under the new gate with the old rule here.
                      or assoc.geometry_compatible(assoc._as_bounds(bbox),
                                                   assoc.locality_bounds(best_match),
                                                   LOCALITY_GAP_M) is True):
                    best_match.bbox = bbox
                    best_match._yaw_acc = yaw_acc   # GA-315 part 2
                    self.room_manager.update_room_geometry(
                        getattr(best_match, "room_id", self.room_manager.current_room_id),
                        bbox
                    )
                    updated_obj = best_match

                # GA-12: an unstamped object is of unknown age, which is NOT yet stable.
                elif (time.time() - (time.time() if getattr(best_match, "creation_time", None) is None
                                     else best_match.creation_time)) < OBJECT_STABILITY_TIMEOUT:
                    best_match.bbox = bbox
                    best_match._yaw_acc = yaw_acc   # GA-315 part 2
                    updated_obj = best_match

                else:
                    # GA-24: the room is resolved and the replacement is built COMPLETELY
                    # before the object leaves the world model, and the swap has nothing
                    # fallible between the remove and the append.
                    #
                    # Before, the object was removed first and re-appended only after
                    # `scene_graph[new_room]` -- a bare dict index on a key that can be
                    # None, because the old fallback was `current_room_id`, which is itself
                    # None when the robot stands outside every room polygon. The KeyError
                    # was caught by the handler's broad except, which returned success=False
                    # and left the object gone from the map and from disk permanently.
                    #
                    # A move that cannot resolve a room is now REFUSED, not completed: the
                    # object stays where it is. Per GA-10 the caller offers the refused
                    # detection to admission rather than dropping it, so the failure mode
                    # is a visible duplicate instead of a silently lost object -- D14's
                    # direction. A clean move needs room identity to be trustworthy, which
                    # is GA-28.
                    # GA-315 part 2: a moved object is a fresh sighting; its axis restarts
                    # from this view alone, not from views of where it used to stand.
                    bbox = raw_bbox
                    new_room = self.room_manager.assign_room_by_geometry(bbox)
                    if not new_room or new_room not in self.room_manager.scene_graph:
                        self.log_both(
                            "warn",
                            f"[MOVE REFUSED] '{best_match.label}': no room "
                            f"resolvable for the new position (room={new_room!r}); "
                            f"the object stays where it was")
                        response.success = False
                        response.message = f"move refused: no resolvable room for {obj_id}"
                        response.object_id = getattr(best_match, "object_id", "") or ""
                        response.distance = float(distance)
                        response.iou = float(iou)
                        response.replaced = False
                        self._write_mutation_receipt(
                            mutation_event_id, request, "refused", "move",
                            obj=best_match, reason="room_unresolved")
                        self._set_receipt_response(
                            response, mutation_event_id, "refused", best_match)
                        return response

                    updated_obj = Object(
                        best_match.label,
                        _centroid_from_bbox(bbox),
                        bbox,
                        best_match.description,
                        best_match.color,
                        best_match.material
                    )
                    # A physical move starts a new geometry history. Keep this
                    # explicit so a later state-copy change cannot mix positions.
                    reset_bbox_fusion(updated_obj)
                    updated_obj.object_id = getattr(best_match, "object_id", None) or f"obj_{uuid.uuid4().hex}"
                    # Preserve the logical track revision across the replacement object.
                    # The common tail increments it after the new geometry is visible.
                    updated_obj.object_revision = int(
                        getattr(best_match, "object_revision", 0))
                    updated_obj.geometry_epoch = int(
                        getattr(best_match, "geometry_epoch", 0)) + 1
                    _, updated_obj._yaw_acc = fuse_orientation(None, raw_bbox)   # GA-315 part 2
                    # Reviewed 2026-09-07: a moved object was rebuilt WITHOUT its sightings, so
                    # co-visibility and the late-description join (GA-108) lost every object
                    # that ever moved. Carry the identity-bearing state across.
                    for _attr in ("observations", "shape", "provisional", "ontologically_usable",
                                  "onto_aligned", "onto_type", "not_seen_in_pov_frames", "creation_time",
                                  "clip_embedding", "_cycle_bbox_2d", "admission_grade",
                                  "admission_filled"):
                        if hasattr(best_match, _attr):
                            setattr(updated_obj, _attr, getattr(best_match, _attr))
                    # GA-171: normalised, exactly as the add path does. This line used
                    # to assign the raw request value, so a replaced object could carry
                    # an empty array that every `is not None` guard downstream accepted.
                    updated_obj.embedding = normalise_embedding(description_embedding)
                    updated_obj.relations = getattr(best_match, "relations", {
                        "isIn": set(),
                        "isOn": set(),
                        "isNextTo": set(),
                        "isAbove": set(),
                        "isUnder": set(),
                    })
                    # GA-12: creation_time is written at exactly ONE site in this module --
                    # the add path -- and was NOT carried across here, so every object that
                    # had ever moved read as ~1.8 billion seconds old. That made the
                    # stability branch above unreachable for it (once moved, always
                    # replaced), removed check_tracking_transition's stability protection,
                    # and armed the uncertain-cleanup expiry. Absence of a timestamp is
                    # unknown age, never maximal age.
                    updated_obj.creation_time = getattr(best_match, "creation_time", None) or time.time()
                    updated_obj.room_id = new_room

                    # The swap itself: two list operations, nothing between them that can
                    # raise. Every fallible call -- the two db events, the room geometry
                    # update, the scene-graph append -- happens AFTER the world model is
                    # whole again, so a failure in any of them leaves the object present
                    # rather than deleted. That is the whole point of GA-24; leaving the
                    # db calls inside the window would have reproduced it with a smaller
                    # aperture.
                    if best_match in wm.persistent_perceptions:
                        wm.persistent_perceptions.remove(best_match)
                        wm.persistent_perceptions.append(updated_obj)
                        moved_from_map = True
                    else:
                        wm.persistent_perceptions.append(updated_obj)
                        moved_from_map = False
                    mutation_applied = True

                    if moved_from_map:
                        self.db.on_object_moved(
                            best_match,
                            old_bbox=best_match.bbox,
                            new_bbox=bbox,
                            distance=distance,
                            iou=iou,
                            step=self.tracking_step_counter
                        )

                    if distance > UNCERTAIN_MOVE_DISTANCE_M:
                        if best_match not in self.uncertain_objects:
                            self.uncertain_objects.append(best_match)
                            self.db.on_uncertain_added(best_match, step=self.tracking_step_counter)

                    self.room_manager.update_room_geometry(new_room, bbox)
                    if updated_obj.label not in self.room_manager.scene_graph[new_room]["objects"]:
                        self.room_manager.scene_graph[new_room]["objects"].append(updated_obj.label)
                    self.log_operation(f"[MOVED] '{best_match.label}' moved by {distance:.2f}m")
                    # The same event, joinable. The prose line above names a label, carries no
                    # identifier and no date, and cannot be joined to anything; it stays for
                    # a human reading the console.
                    try:
                        self.decision_log.write(
                            "update", getattr(best_match, "object_id", best_match.label),
                            label=best_match.label, change="moved",
                            distance_m=round(float(distance), 3),
                            step=self.tracking_step_counter)
                    except Exception as e:
                        self.get_logger().error(f"decision_log update failed: {e}")

            # In-place updates accumulate this view. A confirmed physical move
            # built a new Object above, so its state starts here with the new
            # position and never mixes old and new geometry.
            apply_request_fusion(updated_obj, request, label=updated_obj.label)

            # GA-26: attributes were written BEFORE the box check and never rolled back, so
            # a mis-associated detection whose box was refused still left its description,
            # colour and material on the object. Applied here, past every refusal.
            if hasattr(request, "description") and request.description:
                if request.description != updated_obj.description:
                    # GA-26: the vector was refreshed only when missing, so a new description
                    # kept the old text's embedding. (The request's own embedding is the
                    # DETECTION's, not this text's -- om6 sends the stored description back.)
                    updated_obj.embedding = get_embedding(world2vec, request.description)
                updated_obj.description = request.description
            if hasattr(request, "color") and request.color:
                updated_obj.color = request.color
            if hasattr(request, "material") and request.material:
                updated_obj.material = request.material
            if clip_embedding is not None:
                updated_obj.clip_embedding = clip_embedding

            # Several legacy update branches write ``best_match.bbox`` in place.  Keep the
            # derived AABB index synchronized before the next detection or scan-complete
            # merge uses it.  Replacement moves already invalidated the index through the
            # tracked world-model list, so refresh_spatial is harmless there too.
            wm.refresh_spatial(updated_obj)
            updated_obj.object_revision = int(
                getattr(updated_obj, "object_revision", 0)) + 1
            updated_obj.geometry_epoch = max(
                1, int(getattr(updated_obj, "geometry_epoch", 0)))
            mutation_applied = True
            self._write_mutation_receipt(
                mutation_event_id, request, "applied_in_memory", "update",
                obj=updated_obj,
                fusion_applied=bool(getattr(request, "has_fusion_voxels", False)))
            save_persistent_perceptions(self)
            self._write_mutation_receipt(
                mutation_event_id, request, "applied_and_checkpointed", "update",
                obj=updated_obj,
                fusion_applied=bool(getattr(request, "has_fusion_voxels", False)))

            replaced = updated_obj is not best_match

            self.log_both("info", f"UPDATE {obj_id} dist={distance:.2f}m iou={iou:.2f} replaced={replaced}")

            try:
                with open(OPERATIONS_LOG, "a") as f:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
                    cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
                    cz = (bbox["z_min"] + bbox["z_max"]) / 2.0
                    tag = "SPOSTATO" if replaced else "AGGIORNATO"
                    f.write(f"{timestamp} {tag} {obj_id} a pos=({cx:.2f}, {cy:.2f}, {cz:.2f}) dist={distance:.2f}m iou={iou:.2f}\n")
            except Exception as e:
                self.get_logger().error(f"Could not write to operations.txt: {e}")

            response.success = True
            response.message = f"Object {obj_id} updated dist={distance:.2f}m, iou={iou:.2f}"
            response.object_id = getattr(updated_obj, "object_id", None) or updated_obj.label
            response.distance = float(distance)
            response.iou = float(iou)
            response.replaced = replaced
            self._set_receipt_response(
                response, mutation_event_id, "applied_and_checkpointed", updated_obj)
            return response

        except Exception as e:
            self.get_logger().error(f"_cb_update_object failed: {e}")
            response.success = False
            response.message = str(e)
            response.object_id = ""
            response.distance = 0.0
            response.iou = 0.0
            response.replaced = False
            self._set_receipt_response(
                response, mutation_event_id,
                "incomplete" if mutation_applied else "refused", updated_obj)
            return response

    @synchronized_world_model
    def _cb_query_objects(self, request, response):
        try:
            pool = self.uncertain_objects if request.uncertain_only else wm.persistent_perceptions

            results = list(pool)
            if request.room_id:
                results = [o for o in results
                        if getattr(o, 'room_id', '') == request.room_id]
            if request.label_filter:
                results = [o for o in results
                        if request.label_filter.lower() in o.label.lower()]

            if len(request.area_filter) == 6:
                bounds = tuple(request.area_filter)
                results = [o for o in results if inside_area(o, bounds)]

            response.object_ids = [getattr(o, "object_id", None) or o.label for o in results]
            response.serialized_json = json.dumps([{
                "object_id":   getattr(o, "object_id", None) or o.label,
                "label":       o.label,
                "description": o.description,
                "color":       o.color,
                "material":    o.material,
                "bbox":        o.bbox,
                "fused_bbox":  getattr(o, "fused_bbox", None),
                "bbox_fusion": fusion_summary(o),
                "room_id":     getattr(o, 'room_id', 'unknown')
            } for o in results])
            response.success = True
        except Exception as e:
            # GA-22 remainder. The bare `except Exception:` here had no binding and no
            # log, so a NameError or TypeError inside the area filter produced an empty
            # result set with no diagnosis anywhere. `success` does separate it from a
            # genuine no-match, but nothing recorded WHAT failed -- which is why the
            # area filter could be broken twice over and never be noticed.
            #
            # NOT removed outright: this is a service callback, and QueryObjects.srv has
            # no message field to carry the reason to the caller. Adding one is a .srv
            # change and a rebuild. Binding and logging makes the failure loud and
            # diagnosable now; the field is the better fix and is not tonight's.
            self.get_logger().error(f"_cb_query_objects failed: {e!r}")
            response.success = False
            response.serialized_json = "[]"
        return response

        
def main(args=None):
    rclpy.init(args=args)
    room_manager = RoomManager(
        w2v_model=world2vec,
        map_topic='/rtabmap/map',
        cloud_map_topic='/rtabmap/cloud_map',
    )
    service_node = ObjectServices(room_manager)

    try:
        rclpy.spin(service_node)
    except KeyboardInterrupt:
        from datetime import datetime
        print(f"\nOBJECT SERVICES chiuso ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        
        # Force one last global save of the perceptions
        save_persistent_perceptions(service_node)

    finally:
        service_node.destroy_node()
        # --- MODIFICA QUESTA PARTE ---
        if rclpy.ok():
            rclpy.shutdown()
        # -----------------------------

if __name__ == '__main__':
    main()
