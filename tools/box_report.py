#!/usr/bin/env python3
"""Split the object box score into PLACEMENT and COMPLETENESS, because one number hides which failed.

WHY THIS EXISTS. table_iv_objects in a bundle's eval/metrics.json reports 5 matched objects of
184 at IoU > 0.5 for 20260911_173938_hm3d_00861, which reads as total failure. It is not. The
merged boxes sit a median 0.21 m from the nearest ground-truth centre -- the placement is good --
and hold a median 0.30 of its volume, because a box is built from the VISIBLE surface points of
one view (cv_utils.mask_list_to_centroid_and_bbox) and the ground-truth box covers the whole
object. 86% of the pairs cannot reach IoU 0.5 even if centred perfectly. A single IoU conflates
"in the wrong place" with "we only ever saw the front of it", and at a 0.5 threshold it reports
both as failure.

So this reports three things per matched pair:
  placement    centre distance in metres -- did we put it in the right place
  completeness our volume / GT volume    -- how much of the object the box covers
  ceiling      the IoU those two extents could reach if the centres coincided exactly

`ceiling` is the honest bar. An IoU below its own ceiling is a placement loss; a LOW CEILING is
not a perception error at all, it is the view geometry, and no amount of better detection moves
it. Read them together or not at all.

THE IoU AND THE MATCHING ARE metrics_eval's OWN, imported rather than rewritten: a second
implementation would drift from the one the tables are built on and the comparison would stop
meaning anything. Only the REPORTING is new here.

    python3 tools/box_report.py results/<run_id> data/gt/<scene>.json [--json out.json]
    python3 tools/box_report.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lost3dsg", "src", "perception_module"))
from metrics_eval import _aabb_iou, assignment  # noqa: E402


def _as_box(lo, hi):
    return {"aabb_min_m": [float(v) for v in lo], "aabb_max_m": [float(v) for v in hi]}


def predicted_boxes(bundle):
    """-> [(object_id, label, box)] from the bundle's merged objects."""
    path = os.path.join(bundle, "persistent_perception.json")
    raw = json.load(open(path))
    objects = raw if isinstance(raw, list) else raw.get("objects", raw)
    out = []
    for obj in objects:
        bbox = obj.get("bbox") or {}
        try:
            lo = [bbox["x_min"], bbox["y_min"], bbox["z_min"]]
            hi = [bbox["x_max"], bbox["y_max"], bbox["z_max"]]
        except (KeyError, TypeError):
            # A merged object with no usable box is not scored, and is counted below so the
            # denominator says how many there were rather than quietly shrinking.
            continue
        out.append((obj.get("object_id"), obj.get("label"), _as_box(lo, hi)))
    return out, len(objects)


# OWNER RULING 2026-09-12. Structure is scored SEPARATELY from movable objects, and
# `unknown` leaves the inventory entirely.
#
# WHY `unknown` GOES RATHER THAN FORMING A THIRD GROUP: 70 of the 870 objects on hm3d_00861
# carry the label "unknown", which names no object. Scored as a miss it invents 70 failures;
# scored as a hit it invents 70 successes; either way it moves recall by 8% on a fact about
# the annotator rather than about perception. A denominator must be things we claim to map.
#
# WHY STRUCTURE STAYS, in its own group: a wall IS detected and IS mapped, but its box is a
# 3.4 m slab and ours is the patch of it that was visible, so a completeness of 0.00 against
# a wall says nothing about the object pipeline. Reported apart, both numbers are readable;
# pooled, the structural boxes dominate and hide the object result.
STRUCTURAL_LABELS = frozenset({
    "wall", "shower wall", "floor", "ceiling", "door", "doorway", "door frame",
    "window", "window frame", "stairs", "staircase", "railing", "beam", "column",
    "pillar", "ledge", "roof",
})
EXCLUDED_LABELS = frozenset({"unknown"})


def classify(label):
    """-> 'structural', 'movable' or None (excluded from every denominator)."""
    name = (label or "").strip().lower()
    if name in EXCLUDED_LABELS or not name:
        return None
    return "structural" if name in STRUCTURAL_LABELS else "movable"


def ground_truth_boxes(gt_path, group=None):
    """-> [(semantic_id, label, box)] from tools/extract_gt.py's inventory, ROS convention.

    `group` filters to 'structural' or 'movable'; None keeps both but still drops the
    excluded labels, so no caller can accidentally score against `unknown`.
    """
    out = []
    for obj in json.load(open(gt_path))["objects"]:
        kind = classify(obj["label"])
        if kind is None or (group is not None and kind != group):
            continue
        centre = np.asarray(obj["pos"], dtype=float)
        extents = np.asarray(obj["extents"], dtype=float)
        out.append((int(obj["semantic_id"]), obj["label"],
                    _as_box(centre - extents / 2.0, centre + extents / 2.0)))
    return out


def _extents(box):
    return (np.asarray(box["aabb_max_m"], dtype=float)
            - np.asarray(box["aabb_min_m"], dtype=float))


def _centre(box):
    return (np.asarray(box["aabb_max_m"], dtype=float)
            + np.asarray(box["aabb_min_m"], dtype=float)) / 2.0


def concentric_iou(a, b):
    """The IoU of two boxes' EXTENTS with their centres made to coincide.

    The best score the pair could reach if localisation were perfect, so a caller can tell a
    placement loss from a shape the view never contained. Axis-aligned, like the boxes.
    """
    ea, eb = np.maximum(_extents(a), 0.0), np.maximum(_extents(b), 0.0)
    inter = float(np.prod(np.minimum(ea, eb)))
    va, vb = float(np.prod(ea)), float(np.prod(eb))
    union = va + vb - inter
    return inter / union if union > 0 else 0.0


def report(bundle, gt_path, match_threshold=0.0, group=None):
    pred, n_pred_total = predicted_boxes(bundle)
    gt = ground_truth_boxes(gt_path, group)
    pairs = assignment([p[2] for p in pred], [g[2] for g in gt], match_threshold)

    rows = []
    for i, j, score in pairs:
        pbox, gbox = pred[i][2], gt[j][2]
        pv, gv = float(np.prod(_extents(pbox))), float(np.prod(_extents(gbox)))
        rows.append({
            "object_id": pred[i][0], "predicted_label": pred[i][1],
            "semantic_id": gt[j][0], "gt_label": gt[j][1],
            "iou": round(float(score), 4),
            "ceiling": round(concentric_iou(pbox, gbox), 4),
            "placement_m": round(float(np.linalg.norm(_centre(pbox) - _centre(gbox))), 4),
            "completeness": round(pv / gv, 4) if gv > 0 else None,
        })

    def med(key):
        vals = sorted(r[key] for r in rows if r[key] is not None)
        return round(vals[len(vals) // 2], 4) if vals else None

    ceilings = [r["ceiling"] for r in rows]
    summary = {
        "group": group or "all",
        "predicted_objects": n_pred_total,
        "predicted_objects_with_a_box": len(pred),
        "ground_truth_objects": len(gt),
        "pairs": len(rows),
        "placement_m_median": med("placement_m"),
        "completeness_median": med("completeness"),
        "iou_median": med("iou"),
        "ceiling_median": med("ceiling"),
        # The count that explains the headline number: pairs the view geometry alone keeps
        # below the 0.5 threshold, whatever perception did.
        "pairs_whose_ceiling_is_below_0.5": sum(1 for c in ceilings if c < 0.5),
        "matched_iou_gt_0.5": sum(1 for r in rows if r["iou"] > 0.5),
        "matched_iou_gt_0.25": sum(1 for r in rows if r["iou"] > 0.25),
        "matched_iou_gt_0.1": sum(1 for r in rows if r["iou"] > 0.1),
    }
    return summary, rows


def _print(summary, rows):
    print(f"=== {summary['group'].upper()} ===")
    print(f"predicted objects        {summary['predicted_objects']} "
          f"({summary['predicted_objects_with_a_box']} with a usable box)")
    print(f"ground-truth objects     {summary['ground_truth_objects']}")
    print(f"pairs scored             {summary['pairs']}")
    print()
    print(f"PLACEMENT    centre distance, median   {summary['placement_m_median']} m")
    print(f"COMPLETENESS our volume / GT, median   {summary['completeness_median']}")
    print(f"CEILING      IoU if perfectly centred  {summary['ceiling_median']}")
    print(f"IoU          actual, median            {summary['iou_median']}")
    print()
    print(f"matched at IoU > 0.5     {summary['matched_iou_gt_0.5']}")
    print(f"matched at IoU > 0.25    {summary['matched_iou_gt_0.25']}")
    print(f"matched at IoU > 0.1     {summary['matched_iou_gt_0.1']}")
    print(f"pairs the geometry alone holds below 0.5: "
          f"{summary['pairs_whose_ceiling_is_below_0.5']} of {summary['pairs']}")
    if rows:
        print("\nworst completeness (seen from too few sides):")
        for r in sorted(rows, key=lambda r: r["completeness"] or 0)[:5]:
            print(f"   {r['gt_label']:<16} completeness {r['completeness']:.2f} "
                  f"placement {r['placement_m']:.2f} m  iou {r['iou']:.3f} "
                  f"(ceiling {r['ceiling']:.3f})")


def _selftest():
    """The decomposition must separate a placement loss from a completeness loss."""
    unit = _as_box([0, 0, 0], [1, 1, 1])
    # Same shape, shifted: the ceiling is perfect, the IoU is not. A PLACEMENT loss.
    shifted = _as_box([0.5, 0, 0], [1.5, 1, 1])
    assert abs(concentric_iou(unit, shifted) - 1.0) < 1e-9, concentric_iou(unit, shifted)
    assert _aabb_iou(unit, shifted) < 0.5

    # Concentric, half the size on one axis: the ceiling IS the IoU. A COMPLETENESS loss,
    # and no better localisation can move it.
    half = _as_box([0, 0, 0.25], [1, 1, 0.75])
    assert abs(concentric_iou(unit, half) - _aabb_iou(unit, half)) < 1e-9
    assert concentric_iou(unit, half) < 0.51

    # A box that covers a third of the volume cannot reach 0.5 however it is placed -- the
    # claim the report rests on.
    third = _as_box([0, 0, 0], [1, 1, 1 / 3])
    assert concentric_iou(unit, third) < 0.5
    assert _aabb_iou(unit, third) <= concentric_iou(unit, third) + 1e-9
    print("selftest ok: placement and completeness losses separate, and the ceiling bounds the IoU")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("bundle", nargs="?", help="a run directory under results/")
    ap.add_argument("gt", nargs="?", help="the scene inventory from tools/extract_gt.py")
    ap.add_argument("--json", help="also write the per-pair rows here")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="minimum IoU for a pair to be reported (default 0.0: report every "
                         "assigned pair, so the failures are visible rather than filtered out)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.bundle or not args.gt:
        ap.error("bundle and gt are required unless --selftest")

    raw = json.load(open(args.gt))["objects"]
    counts = {"movable": 0, "structural": 0, "excluded": 0}
    for obj in raw:
        counts[classify(obj["label"]) or "excluded"] += 1
    print(f"ground-truth inventory: {len(raw)} objects -> {counts['movable']} movable, "
          f"{counts['structural']} structural, {counts['excluded']} excluded "
          f"({'/'.join(sorted(EXCLUDED_LABELS))})\n")

    out = {}
    for group in ("movable", "structural"):
        summary, rows = report(args.bundle, args.gt, args.threshold, group)
        _print(summary, rows)
        print()
        out[group] = {"summary": summary, "pairs": rows}
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"inventory": counts, "groups": out}, fh, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
