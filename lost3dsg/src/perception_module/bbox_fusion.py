"""Multi-observation object geometry from map-frame mask points.

Each detection already carries the filtered map-frame points used to build its
single-view box.  This module voxelises those points and retains only the amount
of per-view evidence needed by the measured reconstruction policy:

* one to three views: use the voxel union;
* four or more views: keep voxels seen from at least two distinct views;
* if the agreement filter is empty: use the union.

The state keeps view identity.  This makes repeated delivery and later object
merges idempotent.  A plain Counter cannot do that when two duplicate tracks
contain the same observation.
"""
from __future__ import annotations

import numpy as np
from config import CFG

_ASSOCIATION = CFG.get("association", {}) or {}
VOXEL_SIZE_M = float(_ASSOCIATION.get("bbox_fusion_voxel_m", 0.03))
MIN_AGREEING_VIEWS = int(_ASSOCIATION.get("bbox_fusion_min_views", 2))
AGREEMENT_START_VIEWS = int(
    _ASSOCIATION.get("bbox_fusion_agreement_start_views", 4)
)

if VOXEL_SIZE_M <= 0.0:
    raise ValueError("association.bbox_fusion_voxel_m must be positive")
if MIN_AGREEING_VIEWS < 1:
    raise ValueError("association.bbox_fusion_min_views must be at least 1")
if AGREEMENT_START_VIEWS < MIN_AGREEING_VIEWS:
    raise ValueError(
        "association.bbox_fusion_agreement_start_views must be at least "
        "bbox_fusion_min_views"
    )
STRUCTURAL_LABELS = frozenset({
    "wall", "shower wall", "floor", "ceiling", "door", "doorway",
    "door frame", "window", "window frame", "stairs", "staircase",
    "railing", "beam", "column", "pillar", "ledge", "roof",
})
EXCLUDED_LABELS = frozenset({"", "unknown"})
_INT32_MIN = np.iinfo(np.int32).min
_INT32_MAX = np.iinfo(np.int32).max


def fusion_eligible_label(label):
    """Return whether the measured object-fusion policy applies to ``label``."""
    base = str(label or "").split("#", 1)[0].strip().lower()
    return base not in STRUCTURAL_LABELS and base not in EXCLUDED_LABELS


def voxel_keys_from_points(points, voxel_m=VOXEL_SIZE_M):
    """Return unique int32 XYZ voxel keys for finite map-frame points."""
    voxel_m = float(voxel_m)
    if voxel_m <= 0.0:
        raise ValueError("voxel_m must be positive")
    if points is None:
        return np.empty((0, 3), dtype=np.int32)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N, 3+)")
    pts = pts[:, :3]
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if not len(pts):
        return np.empty((0, 3), dtype=np.int32)
    keys64 = np.floor(pts / voxel_m).astype(np.int64)
    if np.any(keys64 < _INT32_MIN) or np.any(keys64 > _INT32_MAX):
        raise ValueError("voxel key exceeds int32 transport range")
    return np.unique(keys64, axis=0).astype(np.int32, copy=False)


def flatten_voxel_keys(keys):
    """Flatten an ``(N, 3)`` voxel array for a ROS ``int32[]`` field."""
    arr = np.asarray(keys)
    if arr.size == 0:
        return []
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("voxel keys must have shape (N, 3)")
    arr64 = arr.astype(np.int64, copy=False)
    if np.any(arr64 < _INT32_MIN) or np.any(arr64 > _INT32_MAX):
        raise ValueError("voxel key exceeds int32 transport range")
    return arr64.ravel().tolist()


def fusion_payload_from_points(points, label, voxel_m=VOXEL_SIZE_M):
    """Build one detection's transport payload; safe to call in a worker."""
    if points is None or not fusion_eligible_label(label):
        return []
    return flatten_voxel_keys(voxel_keys_from_points(points, voxel_m))


def decode_voxel_keys(flat_keys):
    """Decode and deduplicate a flat XYZ key sequence."""
    values = np.asarray(list(flat_keys), dtype=np.int64)
    if values.size == 0:
        return set()
    if values.size % 3:
        raise ValueError("fusion_voxel_keys length must be a multiple of 3")
    if np.any(values < _INT32_MIN) or np.any(values > _INT32_MAX):
        raise ValueError("voxel key exceeds int32 transport range")
    return {tuple(int(value) for value in row) for row in values.reshape(-1, 3)}


def _new_state(voxel_m):
    return {"voxel_m": float(voxel_m), "views": set(), "voxels": {}}


def _copy_state(state):
    if state is None:
        return None
    return {
        "voxel_m": float(state["voxel_m"]),
        "views": set(state["views"]),
        "voxels": {tuple(key): set(view_ids)
                   for key, view_ids in state["voxels"].items()},
    }


def _assert_voxel_size(state, voxel_m):
    if not np.isclose(float(state["voxel_m"]), float(voxel_m), rtol=0.0, atol=1e-9):
        raise ValueError(
            f"bbox fusion voxel size changed from {state['voxel_m']} to {voxel_m} m"
        )


def required_agreement(view_count):
    """Return the number of distinct views required for the current object."""
    return MIN_AGREEING_VIEWS if int(view_count) >= AGREEMENT_START_VIEWS else 1


def fused_bbox_from_state(state):
    """Fit the measured untrimmed AABB from retained voxel centres."""
    if state is None or not state["voxels"]:
        return None
    need = required_agreement(len(state["views"]))
    selected = [key for key, view_ids in state["voxels"].items()
                if len(view_ids) >= need]
    used_fallback = False
    if not selected:
        selected = list(state["voxels"])
        used_fallback = True
    keys = np.asarray(selected, dtype=np.float64)
    centres = (keys + 0.5) * float(state["voxel_m"])
    low = centres.min(axis=0)
    high = centres.max(axis=0)
    return {
        "x_min": float(low[0]), "x_max": float(high[0]),
        "y_min": float(low[1]), "y_max": float(high[1]),
        "z_min": float(low[2]), "z_max": float(high[2]),
        "source": "multi_observation_voxel_agreement",
        "voxel_size_m": float(state["voxel_m"]),
        "view_count": len(state["views"]),
        "required_views": need,
        "voxel_count": len(selected),
        "agreement_fallback": used_fallback,
    }


def fusion_summary(obj):
    """Return JSON-safe evidence counts without dumping the full voxel map."""
    state = getattr(obj, "_bbox_fusion_state", None)
    bbox = getattr(obj, "fused_bbox", None)
    if state is None or bbox is None:
        return None
    return {
        "voxel_size_m": float(state["voxel_m"]),
        "view_count": len(state["views"]),
        "observed_voxel_count": len(state["voxels"]),
        "retained_voxel_count": int(bbox["voxel_count"]),
        "required_views": int(bbox["required_views"]),
        "agreement_fallback": bool(bbox["agreement_fallback"]),
    }


def add_fusion_view(obj, label, view_id, voxel_m, flat_keys):
    """Add one detection view to ``obj`` and rebuild its separate fused box.

    Return ``True`` only when new evidence was added.  A repeated ``view_id`` is
    an idempotent no-op.
    """
    if not fusion_eligible_label(label):
        return False
    view_id = str(view_id or "").strip()
    if not view_id:
        raise ValueError("bbox fusion evidence requires a non-empty view_id")
    keys = decode_voxel_keys(flat_keys)
    if not keys:
        return False
    voxel_m = float(voxel_m)
    if voxel_m <= 0.0:
        raise ValueError("bbox fusion voxel size must be positive")

    state = getattr(obj, "_bbox_fusion_state", None)
    if state is None:
        state = _new_state(voxel_m)
        obj._bbox_fusion_state = state
    else:
        _assert_voxel_size(state, voxel_m)
    if view_id in state["views"]:
        return False

    state["views"].add(view_id)
    for key in keys:
        seen_by = state["voxels"].setdefault(key, set())
        # The policy asks only whether one or two views saw a voxel.  Retaining
        # more IDs would grow with the tour but would not change any decision.
        if len(seen_by) < MIN_AGREEING_VIEWS:
            seen_by.add(view_id)
    obj.fused_bbox = fused_bbox_from_state(state)
    return True


def reset_bbox_fusion(obj):
    """Drop geometry measured at an object's former physical position."""
    obj._bbox_fusion_state = None
    obj.fused_bbox = None


def merge_bbox_fusion(keeper, discard):
    """Idempotently merge two duplicate tracks' distinct-view voxel evidence."""
    left = getattr(keeper, "_bbox_fusion_state", None)
    right = getattr(discard, "_bbox_fusion_state", None)
    if right is None:
        return False
    if left is None:
        keeper._bbox_fusion_state = _copy_state(right)
        keeper.fused_bbox = fused_bbox_from_state(keeper._bbox_fusion_state)
        return True
    _assert_voxel_size(left, right["voxel_m"])

    changed = False
    before_views = len(left["views"])
    left["views"].update(right["views"])
    changed = len(left["views"]) != before_views
    for key, right_views in right["voxels"].items():
        left_views = left["voxels"].setdefault(tuple(key), set())
        before = set(left_views)
        if len(left_views) < MIN_AGREEING_VIEWS:
            for view_id in sorted(right_views):
                left_views.add(view_id)
                if len(left_views) >= MIN_AGREEING_VIEWS:
                    break
        changed = changed or left_views != before
    if changed:
        keeper.fused_bbox = fused_bbox_from_state(left)
    return changed


def request_fusion_fields(request):
    """Decode the optional fusion fields shared by AddObject and UpdateObject."""
    if not bool(getattr(request, "has_fusion_voxels", False)):
        return None
    return {
        "label": getattr(request, "label", ""),
        "view_id": getattr(request, "fusion_view_id", ""),
        "voxel_m": float(getattr(request, "fusion_voxel_size_m", 0.0)),
        "flat_keys": list(getattr(request, "fusion_voxel_keys", [])),
    }


def apply_request_fusion(obj, request, label=None):
    """Apply fusion evidence from a ROS service request when it is present."""
    fields = request_fusion_fields(request)
    if fields is None:
        return False
    return add_fusion_view(
        obj,
        label if label is not None else fields["label"],
        fields["view_id"],
        fields["voxel_m"],
        fields["flat_keys"],
    )


def fusion_http_fields(fusion):
    """Return the additive JSON fields used by the Graph API bridge."""
    if not fusion:
        return {}
    flat_keys = [int(value) for value in fusion.get("flat_keys", [])]
    if not flat_keys:
        return {}
    if len(flat_keys) % 3:
        raise ValueError("fusion_voxel_keys length must be a multiple of 3")
    view_id = str(fusion.get("view_id") or "").strip()
    voxel_m = float(fusion.get("voxel_m", 0.0))
    if not view_id or voxel_m <= 0.0:
        raise ValueError("fusion evidence requires view_id and positive voxel size")
    return {
        "fusion_view_id": view_id,
        "fusion_voxel_size_m": voxel_m,
        "fusion_voxel_keys": flat_keys,
    }
