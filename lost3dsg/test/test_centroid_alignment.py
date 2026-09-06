"""GA-327: a degenerate mask must leave centroids_3d, bboxes_3d and points_out the same length (red-first, 2026-09-06)."""
import os, sys, types
sys.path.insert(0, "/DATA/FOUND/vendor/graph-api/lost3dsg/src/perception_module")
import rosstub; rosstub.install()
import numpy as np
import cv_utils
# rosstub stubs scipy, so the KDTree outlier pass cannot run on the host; it is not what this checks.
cv_utils.statistical_outlier_removal = lambda pts, k=20, std_ratio=2.0: np.ones(len(pts), dtype=bool)

class Log:
    def warn(self, m): pass
    def info(self, m): pass
class Node:
    def get_logger(self): return Log()
class CamInfo:
    k = [500.0, 0, 320.0, 0, 500.0, 240.0, 0, 0, 1]
    class header: stamp = 0
class TF:
    class transform:
        class translation: x = 0.0; y = 0.0; z = 0.0
        class rotation: x = 0.0; y = 0.0; z = 0.0; w = 1.0

H, W = 480, 640
depth = np.full((H, W), 2.0, dtype=np.float32)
good = np.zeros((H, W), np.uint8); good[100:200, 100:220] = 1                # a real blob
flat = np.zeros((H, W), np.uint8); flat[300, 100:400] = 1                    # one pixel row: y-span 0 -> degenerate
masks = [good, flat, good]
c, b = cv_utils.mask_list_to_centroid_and_bbox(masks, ["a", "b", "c"], depth, CamInfo(), Node(), transform=TF())
pts = []
c2, b2 = cv_utils.mask_list_to_centroid_and_bbox(masks, ["a", "b", "c"], depth, CamInfo(), Node(), transform=TF(), points_out=pts)
print("len centroids", len(c), "len bboxes", len(b), "len points", len(pts), "| middle bbox:", b[1])
assert len(c) == len(b) == len(pts) == len(masks), "centroids_3d drifted out of alignment"
assert b[1] is None and pts[1] is None and c[1] is None
assert c[0] == c[2] and b[0] == b[2]
print("OK: aligned")
