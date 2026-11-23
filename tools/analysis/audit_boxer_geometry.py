#!/usr/bin/env python3
"""Compare geometry on fixed archived identities; never mutate the run's objects.

This is a retrospective geometry comparison, not an online tracking replay. It
joins accepted mutation receipts to original detections, follows recorded merges,
and skips tracks with recorded relocations. It cannot predict new merge decisions.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lost3dsg/src/perception_module"))
import boxer_geometry
import metrics_eval


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def audit(bundle):
    detections = {(d.get("observation") or {}).get("observation_id"): d
                  for d in rows(bundle / "detections.jsonl")}
    parents = {}
    for event in rows(bundle / "hook_decisions.jsonl"):
        if event.get("kind") == "merge":
            parents[event["merged_from"]] = event["object"]

    def survivor(oid):
        seen = set()
        while oid in parents:
            if oid in seen:
                raise ValueError("Cycle in archived merge lineage")
            seen.add(oid)
            oid = parents[oid]
        return oid

    observations, moved = defaultdict(dict), set()
    for receipt in rows(bundle / "mutation_receipts.jsonl"):
        if not str(receipt.get("mutation_state", "")).startswith("applied"):
            continue
        oid = survivor(receipt["object_id"])
        if receipt.get("geometry_epoch", 1) > 1:
            moved.add(oid)
        ref = (receipt.get("observation") or {}).get("observation_id")
        detection = detections.get(ref)
        if detection and detection.get("bbox_3d"):
            observations[oid][ref] = detection["bbox_3d"]

    manifest = json.loads((bundle / "eval/manifest_eval.json").read_text())
    modified = deepcopy(manifest)
    changes = []
    for obj in modified["predicted_objects"]:
        oid = obj.get("object_id")
        if oid in moved or not observations[oid]:
            continue
        samples = list(observations[oid].values())
        box, _ = boxer_geometry.fuse_samples(samples)
        old_lo, old_hi = np.array(obj["aabb_min_m"]), np.array(obj["aabb_max_m"])
        lo = [box[f"{axis}_min"] for axis in "xyz"]
        hi = [box[f"{axis}_max"] for axis in "xyz"]
        before = float(np.prod(old_hi - old_lo))
        after = float(np.prod(np.subtract(hi, lo)))
        obj["aabb_min_m"], obj["aabb_max_m"] = lo, hi
        changes.append(dict(object_id=oid, label=obj.get("label"), samples=len(samples),
                            old_volume_m3=before, new_volume_m3=after,
                            old_to_new_volume_ratio=before / after, bbox=box))
    fields = ("precision_pct", "recall_pct", "matched_object_iou_mean",
              "matched_objects_iou_gt_0.5", "matched_objects", "predicted_objects")
    before = metrics_eval.objects([manifest])
    after = metrics_eval.objects([modified])
    return {
        "scope": "Geometry only, fixed archived identities/labels; not an online replay",
        "policy": boxer_geometry.POLICY,
        "reconstructed_objects": len(changes), "skipped_moved_tracks": len(moved),
        "before": {k: before.get(k) for k in fields},
        "after": {k: after.get(k) for k in fields}, "objects": changes,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "objects"}, indent=2))
