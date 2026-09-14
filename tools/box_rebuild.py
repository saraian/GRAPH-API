#!/usr/bin/env python3
"""Rebuild object boxes from a bundle by accumulating views, and score the result.

THE PROBLEM. A box is built from the points of ONE view: the detection's mask unprojected
through that frame's depth (cv_utils.mask_list_to_centroid_and_bbox), then cut to the 5th-95th
percentile per axis. It therefore covers the visible surface of one side of the object, and
holds a median 0.66 of the ground-truth volume for movable objects on 20260911_173938_hm3d_00861.
Merging does not fix it: object_services.fuse_orientation keeps the extents of ONE measured view
by design (GA-20), so a whole tour grows the box about 4%.

WHAT THIS DOES INSTEAD, measured on that bundle over 108 ground-truth objects:

    box builder                        IoU median   matched >0.5   volume/GT
    one view (what the bundle holds)      0.084         2/108         0.20
    union of every view                   0.233        18/108         1.16
    union + DBSCAN(eps .06, min 8)        0.251        30/108         0.69
    VOXELS >=2 VIEWS AGREED ON            0.253        28/108         0.53

On the objects with 8 or more views, where the difference matters most, >=2-view voxels reach
IoU 0.520 against DBSCAN's 0.511, with a volume ratio of 1.14 against 1.35.

On the 79 MOVABLE objects, which is the group this is for, the whole change is
IoU 0.124 -> 0.372, matched at 0.5 2 -> 28, and volume 0.23 -> 0.90 of the true box.

WHY COUNTING VIEWS BEATS CLUSTERING. DBSCAN asks whether a point sits in a dense neighbourhood.
Mask bleed onto the wall behind an object IS dense, so clustering keeps it. Counting asks whether
two different viewpoints agreed that something is there, which bleed fails because it moves with
the camera. Running both is WORSE than either (23 of 108): they remove the same points twice.

DO NOT ADD DBSCAN ON TOP. That was measured, not assumed.

GROUPING IS BY habitat_gt_instance_id, so this measures the BOX BUILDER and nothing else --
association is perfect by construction and cannot contribute. The number it produces is a
CEILING for the live system, not a prediction of it.

    python3 tools/box_rebuild.py results/<run_id> data/gt/<scene>.json [--json out.json]
    python3 tools/box_rebuild.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from box_report import _as_box, classify, concentric_iou  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lost3dsg", "src", "perception_module"))
from metrics_eval import _aabb_iou  # noqa: E402

VOXEL_M = 0.03
MIN_VIEWS = 2
# Below this many views, agreement is not asked for: see required_agreement().
AGREEMENT_FLOOR_VIEWS = 4


def rle_decode(d):
    """The archive's own RLE, as detection_archive's fallback decoder writes it."""
    out = np.zeros(d["size"][0] * d["size"][1], dtype=np.uint8)
    value, i = d["first_val"], 0
    for count in d["counts"]:
        out[i:i + count] = value
        i += count
        value = 1 - value
    return out.reshape(d["size"])


def quat_to_rotation(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _depth_bounds(z):
    """cv_utils._depth_bounds: a median-absolute-deviation cut on the depth values."""
    median = np.median(z)
    mad = np.median(np.abs(z - median))
    return (median - 4.5 * mad, median + 4.5 * mad) if mad > 0.001 else (median * 0.5, median * 1.5)


def _outlier_keep(pts, k=30, std_ratio=1.5):
    """cv_utils.statistical_outlier_removal: drop points whose k-neighbourhood is far."""
    from scipy.spatial import cKDTree
    if len(pts) <= k:
        return np.arange(len(pts))
    distances, _ = cKDTree(pts).query(pts, k=k + 1)
    mean_d = distances[:, 1:].mean(axis=1)
    return np.where(mean_d <= mean_d.mean() + std_ratio * mean_d.std())[0]


def view_points(row, depth, intrinsics, sensor_width):
    """One detection's mask -> map-frame points, by the runtime's own steps.

    Reproduces cv_utils to 0.6 mm median against the archived centroid and box corners on
    20260911_173938_hm3d_00861; a drift from the runtime here would make every number below
    a measurement of this file instead of of the system.
    """
    mask = rle_decode(row["mask_rle"]).astype(bool)
    if mask.shape != depth.shape:
        return None
    scale = depth.shape[1] / float(sensor_width)
    fx, fy = intrinsics["fx"] * scale, intrinsics["fy"] * scale
    cx, cy = intrinsics["cx"] * scale, intrinsics["cy"] * scale

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    if len(xs) > 20000:
        idx = np.linspace(0, len(xs) - 1, 20000).astype(int)
        xs, ys = xs[idx], ys[idx]
    zs = depth[ys, xs]
    ok = np.isfinite(zs) & (zs > 0.0)
    xs, ys, zs = xs[ok], ys[ok], zs[ok]
    if len(xs) == 0:
        return None
    low, high = _depth_bounds(zs)
    keep = (zs >= low) & (zs <= high)
    xs, ys, zs = xs[keep], ys[keep], zs[keep]
    if len(xs) == 0:
        return None

    pts = np.stack([(xs - cx) * zs / fx, (ys - cy) * zs / fy, zs], axis=1)
    if len(pts) > 20:
        pts = pts[_outlier_keep(pts)]
    if len(pts) == 0:
        return None
    rotation = quat_to_rotation(row["camera_quat_xyzw"])
    return pts.dot(rotation.T) + np.asarray(row["camera_position"], dtype=float)


def voxel_keys(pts, voxel=VOXEL_M):
    return np.unique(np.floor(pts / voxel).astype(np.int64), axis=0)


def required_agreement(n_views, min_views=MIN_VIEWS, floor_views=AGREEMENT_FLOOR_VIEWS):
    """How many distinct views must agree on a voxel, given how many views there are.

    AGREEMENT IS ONLY ASKED FOR ONCE THERE IS ENOUGH OF IT. Requiring two agreeing views
    unconditionally destroyed exactly the objects seen two or three times: on
    20260911_173938_hm3d_00861 that bucket fell to IoU 0.120 and 0.17 of the true volume,
    WORSE than the single view it replaced, because two views taken from far apart share
    almost no voxels and the intersection is nearly empty. Asking for agreement only at four
    or more views lifts that bucket to IoU 0.295 at 0.69 volume and improves the movable
    median from 0.348 to 0.372 with volume 0.81 -> 0.90. Measured, both arms, same objects.

    A stricter rule (half the views) is worse again: 20 matches against 28.
    """
    return min_views if n_views >= floor_views else 1


def accumulate_box(views, voxel=VOXEL_M, min_views=MIN_VIEWS):
    """-> (lo, hi) from voxels that enough DISTINCT views agreed on.

    Falls back to the plain union when the requirement cannot be met, so an object is never
    dropped for being seen too few times -- 23 of the 101 on this bundle are seen once.
    """
    if not views:
        return None
    seen = Counter()
    for pts in views:
        for key in map(tuple, voxel_keys(pts, voxel)):
            seen[key] += 1
    need = required_agreement(len(views), min_views)
    kept = [k for k, n in seen.items() if n >= need]
    if not kept:
        kept = list(seen)
    centres = (np.asarray(kept, dtype=np.int64) + 0.5) * voxel
    return centres.min(axis=0), centres.max(axis=0)


def single_view_box(pts):
    """What the bundle holds today: the 5th-95th percentile of the LAST view's points."""
    return np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)


def load_views(bundle):
    """-> {semantic_id: [points per view]} for detections carrying a ground-truth id."""
    from PIL import Image
    calibration = json.load(open(os.path.join(bundle, "calibration.json")))
    intrinsics = calibration["intrinsics"]
    width = calibration["resolution"]["width"]
    rows = [json.loads(line) for line in
            open(os.path.join(bundle, "detections.jsonl")) if line.strip()]

    depth_cache = {}
    grouped = {}
    skipped = Counter()
    for row in rows:
        gt_id = row.get("habitat_gt_instance_id")
        if gt_id is None or not row.get("mask_rle") or not row.get("camera_quat_xyzw"):
            skipped["no gt id, mask or pose"] += 1
            continue
        frame = row["frame_id"]
        if frame not in depth_cache:
            path = os.path.join(bundle, "depth", f"{frame}.png")
            if not os.path.exists(path):
                skipped["depth frame absent"] += 1
                continue
            depth_cache[frame] = np.array(Image.open(path)).astype(np.float64) / 1000.0
        pts = view_points(row, depth_cache[frame], intrinsics, width)
        if pts is None:
            skipped["no usable points"] += 1
            continue
        grouped.setdefault(int(gt_id), []).append(pts)
    return grouped, skipped


def score(bundle, gt_path):
    grouped, skipped = load_views(bundle)
    inventory = {int(o["semantic_id"]): o for o in json.load(open(gt_path))["objects"]}

    rows = []
    for gt_id, views in grouped.items():
        obj = inventory.get(gt_id)
        if obj is None or classify(obj["label"]) is None:
            continue
        centre = np.asarray(obj["pos"], dtype=float)
        extents = np.asarray(obj["extents"], dtype=float)
        truth = _as_box(centre - extents / 2.0, centre + extents / 2.0)
        gt_volume = float(np.prod(extents))

        lo, hi = accumulate_box(views)
        rebuilt = _as_box(lo, hi)
        base_lo, base_hi = single_view_box(views[-1])
        baseline = _as_box(base_lo, base_hi)

        rows.append({
            "semantic_id": gt_id, "gt_label": obj["label"],
            "kind": classify(obj["label"]), "views": len(views),
            "iou_single_view": round(_aabb_iou(baseline, truth), 4),
            "iou_rebuilt": round(_aabb_iou(rebuilt, truth), 4),
            "ceiling_rebuilt": round(concentric_iou(rebuilt, truth), 4),
            "completeness_single_view": round(float(np.prod(base_hi - base_lo)) / gt_volume, 4)
                                        if gt_volume > 0 else None,
            "completeness_rebuilt": round(float(np.prod(hi - lo)) / gt_volume, 4)
                                    if gt_volume > 0 else None,
            "box": {"aabb_min_m": [round(float(v), 4) for v in lo],
                    "aabb_max_m": [round(float(v), 4) for v in hi]},
        })
    return rows, skipped


def _summarise(rows, kind=None):
    subset = [r for r in rows if kind is None or r["kind"] == kind]
    if not subset:
        return None
    def med(key):
        values = sorted(r[key] for r in subset if r[key] is not None)
        return round(values[len(values) // 2], 4)
    return {
        "kind": kind or "all", "objects": len(subset),
        "iou_single_view_median": med("iou_single_view"),
        "iou_rebuilt_median": med("iou_rebuilt"),
        "completeness_single_view_median": med("completeness_single_view"),
        "completeness_rebuilt_median": med("completeness_rebuilt"),
        "matched_single_view_gt_0.5": sum(1 for r in subset if r["iou_single_view"] > 0.5),
        "matched_rebuilt_gt_0.5": sum(1 for r in subset if r["iou_rebuilt"] > 0.5),
        "matched_rebuilt_gt_0.25": sum(1 for r in subset if r["iou_rebuilt"] > 0.25),
    }


def _selftest():
    """Two views of two faces of a cube must rebuild a bigger box than either view alone,
    and a voxel only one view ever saw must not survive."""
    face_a = np.random.default_rng(0).uniform([0, 0, 0], [1, 1, 0.02], size=(4000, 3))
    face_b = np.random.default_rng(1).uniform([0, 0, 0.98], [1, 1, 1.0], size=(4000, 3))
    lo, hi = accumulate_box([face_a, face_b, face_a, face_b])
    assert hi[2] - lo[2] > 0.9, (lo, hi)

    # A stray view of a wall 2 m away, seen ONCE, must be dropped: it is what mask bleed
    # looks like, and keeping it is the failure this builder exists to avoid.
    bleed = np.random.default_rng(2).uniform([0, 0, 2.0], [1, 1, 2.1], size=(4000, 3))
    lo2, hi2 = accumulate_box([face_a, face_b, face_a, face_b, bleed])
    assert hi2[2] < 1.5, f"single-view bleed survived: {lo2} {hi2}"

    # With only ONE view the requirement must relax rather than return nothing.
    lo3, hi3 = accumulate_box([face_a])
    assert lo3 is not None and hi3[0] > 0.5
    print("selftest ok: views accumulate, single-view bleed is dropped, one view still boxes")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", nargs="?")
    ap.add_argument("gt", nargs="?")
    ap.add_argument("--json", help="write the per-object rows and rebuilt boxes here")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.bundle or not args.gt:
        ap.error("bundle and gt are required unless --selftest")

    rows, skipped = score(args.bundle, args.gt)
    print(f"objects rebuilt: {len(rows)}   detections skipped: {dict(skipped)}\n")
    header = f"{'group':<12}{'n':>5}{'IoU 1-view':>12}{'IoU rebuilt':>13}{'>0.5 was':>10}{'>0.5 now':>10}{'vol 1-view':>12}{'vol rebuilt':>13}"
    print(header)
    for kind in ("movable", "structural", None):
        s = _summarise(rows, kind)
        if s is None:
            continue
        print(f"{s['kind']:<12}{s['objects']:>5}{s['iou_single_view_median']:>12}"
              f"{s['iou_rebuilt_median']:>13}{s['matched_single_view_gt_0.5']:>10}"
              f"{s['matched_rebuilt_gt_0.5']:>10}{s['completeness_single_view_median']:>12}"
              f"{s['completeness_rebuilt_median']:>13}")
    by_views = {}
    for r in rows:
        bucket = "1" if r["views"] == 1 else "2-3" if r["views"] < 4 else "4-7" if r["views"] < 8 else "8+"
        by_views.setdefault(bucket, []).append(r)
    print("\nby view count (movable and structural together):")
    for bucket in ("1", "2-3", "4-7", "8+"):
        subset = by_views.get(bucket)
        if not subset:
            continue
        iou = sorted(r["iou_rebuilt"] for r in subset)
        vol = sorted(r["completeness_rebuilt"] for r in subset)
        print(f"   {bucket:>4} views  n={len(subset):3d}  IoU {iou[len(iou)//2]:.3f}  "
              f"volume {vol[len(vol)//2]:.2f}  >0.5 {sum(1 for v in iou if v > 0.5)}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"voxel_m": VOXEL_M, "min_views": MIN_VIEWS,
                       "agreement_floor_views": AGREEMENT_FLOOR_VIEWS,
                       "objects": rows}, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
