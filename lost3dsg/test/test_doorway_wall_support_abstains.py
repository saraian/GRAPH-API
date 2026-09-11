"""A doorway must be judged, not deleted, when no wall detector has confirmed anything.

WHAT THIS CATCHES. `_segment_regions_gvd` filters its doorway candidates by how well confirmed
depth walls support the proposed cut. The guard used to read

    if doorway_require_wall_support and self._active_detected_wall_support is not None:

and `_detected_wall_support` returns an ALL-ZERO raster -- not None -- whenever the feature is
enabled and no wall has been confirmed. A zeros array is not None, so the filter always ran,
every score was 0.0, and `0.0 < 0.16` deleted every doorway. MEASURED on the bundle
20260911_173938_hm3d_00861: 63 skeleton branches, 12 bottlenecks accepted by geometry, all 12
dropped, door_cuts=0, and one 58.8 m2 "living room" standing for a whole floor.

The distinction is ABSTAIN versus REJECT, and it died at the point of use.
`_door_wall_support_score` draws it correctly -- it returns 0.0 when it has no raster to
consult -- and the caller read that same 0.0 as a failing score.

The segmentation itself is OpenCV throughout, and `rosstub` stubs cv2, so the end-to-end split
runs only where cv2 is real (inside the stack image). What runs everywhere is the pair that
actually encodes the defect: the scorer's abstain, and the two guards that must consult it.
"""
# ruff: noqa: E402, I001  -- rosstub must be installed before the module under test is imported
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src", "perception_module"))
import rosstub; rosstub.install()  # noqa: E702
import numpy as np

from room_manager import RoomManager

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   os.pardir, "src", "perception_module", "room_manager.py")

rm = RoomManager(node=None)

# 1. THE SCORER ABSTAINS. No raster at all, and an all-zero raster, are the same answer: it
#    cannot say. Both must be reachable, or the guard below has nothing to protect against.
rm._active_detected_wall_support = None
assert rm._door_wall_support_score(10, 10, 0.0, 4, 0.05) == 0.0
rm._active_detected_wall_support = np.zeros((40, 40), dtype=np.float32)
assert rm._door_wall_support_score(10, 10, 0.0, 4, 0.05) == 0.0

# 2. AND IT SCORES when there IS evidence -- exercised where the answer is present, or step 1
#    proves only that the function returns 0.0 for everything.
support = np.zeros((40, 40), dtype=np.float32)
support[6:15, 10] = 1.0            # a wall the cut's two endpoints can terminate against
support[6:15, 18] = 1.0
rm._active_detected_wall_support = support
scored = rm._door_wall_support_score(10, 14, np.pi / 2, 4, 0.05)
assert scored > 0.0, f"a cut between two confirmed walls scored {scored}; the scorer is inert"

# 3. BOTH GUARDS CONSULT IT. `is not None` cannot tell "no wall detector" from "wall detector
#    running, no wall here", because the raster exists either way. Read from the source: the
#    condition lives inside a 200-line method that needs OpenCV to reach.
src = open(SRC).read()
guards = re.findall(
    r"if \(?\s*self\._params\.get\('doorway_require_wall_support'[^:]*?:", src, re.S)
assert len(guards) == 2, f"expected 2 wall-support guards, found {len(guards)}"
for i, g in enumerate(guards):
    assert "np.any(" in g, (
        f"guard {i + 1} admits an all-zero support raster as evidence:\n{g.strip()}")

# 4. END TO END where cv2 is real. Two 4.0 x 3.0 m rooms, one 0.9 m doorway, nothing else.
import cv2  # noqa: E402
# `isinstance(..., str)`, not `== "<stub>"`. rosstub's module answers EVERY attribute with an
# Any object, so `cv2.__version__` is not missing and is not the sentinel either -- comparing it
# to a string is False, the skip never fires, and the test dies inside cv2 instead of skipping.
if not isinstance(getattr(cv2, "__version__", None), str):
    print("OK: scorer abstains and scores, both guards consult it "
          "(end-to-end split skipped: cv2 is a stub on this host)")
    raise SystemExit(0)

RES, FREE, OCC = 0.05, 0, 100


class _Grid:
    def __init__(self, arr, res=RES):
        h, w = arr.shape
        origin = type("O", (), {"position": type("P", (), {"x": 0.0, "y": 0.0, "z": 0.0})(),
                                "orientation": type("Q", (), {"x": 0.0, "y": 0.0,
                                                              "z": 0.0, "w": 1.0})()})()
        self.info = type("I", (), {"height": h, "width": w,
                                   "resolution": res, "origin": origin})()
        self.data = arr.reshape(-1).tolist()


def two_rooms(door_m=0.9):
    rw, rh, wall = int(4.0 / RES), int(3.0 / RES), 3
    g = np.full((rh + 2 * wall, rw * 2 + 3 * wall), OCC, dtype=np.int16)
    g[wall:wall + rh, wall:wall + rw] = FREE
    x2 = wall + rw + wall
    g[wall:wall + rh, x2:x2 + rw] = FREE
    door = max(1, int(round(door_m / RES)))
    y0 = wall + rh // 2 - door // 2
    g[y0:y0 + door, wall + rw:x2] = FREE
    return _Grid(g)


live = RoomManager(node=None)
live._params["gvd_method"] = "ridge"
live._params["doorway_require_wall_support"] = True
n = len(live._segment_regions_gvd(two_rooms()))
assert live._active_detected_wall_support is not None and not np.any(live._active_detected_wall_support), \
    "this scene no longer produces the empty-but-not-None raster the guard is about"
assert n == 2, (f"two rooms joined by one doorway segmented as {n} region(s) with no wall "
                f"detector running -- the doorway filter rejects where it should abstain")
print(f"OK: scorer abstains and scores, both guards consult it, "
      f"and a two-room scene splits into {n} with no wall detector")
