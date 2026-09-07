"""GA-19: a zero-area box must not be suppressed by a box it does not overlap.

`apply_nms` divided intersection by union with no epsilon; two degenerate boxes of one
class gave 0/0 = nan, and `nan <= threshold` is False, so the weaker was dropped. The
cloud twin (cloud/modal_perception.py) always had the epsilon.

Run: uv run --no-project --with numpy --with pyyaml --with scipy python3 test_nms_degenerate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rosstub  # noqa: E402

rosstub.install()

import numpy as np  # noqa: E402
from utils import apply_nms  # noqa: E402


def test_two_degenerate_boxes_both_survive():
    boxes = [[10, 10, 10, 10], [50, 50, 50, 50]]
    kept, labels, scores = apply_nms(boxes, ["chair", "chair"], [0.9, 0.8], iou_threshold=0.5)
    assert len(kept) == 2, (kept, labels, scores)


def test_a_real_overlap_is_still_suppressed():
    boxes = [[0, 0, 100, 100], [5, 5, 100, 100]]
    kept, _, _ = apply_nms(boxes, ["chair", "chair"], [0.9, 0.8], iou_threshold=0.5)
    assert len(kept) == 1, kept
    assert np.allclose(kept[0], [0, 0, 100, 100])


if __name__ == "__main__":
    test_two_degenerate_boxes_both_survive()
    test_a_real_overlap_is_still_suppressed()
    print("test_nms_degenerate: 2 passed")
