"""Regression coverage for the archived object-layer split."""
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "perception_module"
sys.path.insert(0, str(MODULE_DIR))

import metrics_eval  # noqa: E402


def _box(label, centre):
    x, y, z = centre
    return {"label": label, "category_name": label,
            "aabb_min_m": [x - .1, y - .1, z - .1],
            "aabb_max_m": [x + .1, y + .1, z + .1]}


def test_partitions_use_exact_bbox_fusion_labels_and_rematch_within_layer():
    scene = {
        "predicted_objects": [_box("chair", (0, 0, 0)),
                              _box("window", (2, 0, 0)),
                              _box("doormat", (4, 0, 0))],
        "ground_truth_objects": [_box("chair", (.1, 0, 0)),
                                  _box("window", (2.1, 0, 0)),
                                  _box("beam", (6, 0, 0)),
                                  _box("doormat", (4.1, 0, 0))],
    }
    report = metrics_eval.object_layer_partitions([scene], .5)
    assert report["partition_source"] == "bbox_fusion.STRUCTURAL_LABELS exact labels"
    movable, structural = report["movable"], report["structural"]
    assert (movable["predicted_objects"], movable["ground_truth_objects"],
            movable["matched_objects"], movable["semantic_exact_matches"]) == (2, 2, 2, 2)
    assert (structural["predicted_objects"], structural["ground_truth_objects"],
            structural["matched_objects"], structural["semantic_exact_matches"]) == (1, 2, 1, 1)


if __name__ == "__main__":
    test_partitions_use_exact_bbox_fusion_labels_and_rematch_within_layer()
    print("all checks passed")
