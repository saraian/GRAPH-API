"""GA-30: the room centroid is the area centroid, and every room threshold is reachable from CFG['rooms']."""
# ruff: noqa: E402, I001  -- rosstub must be installed before the module under test is imported
import sys
sys.path.insert(0, "/DATA/FOUND/vendor/graph-api/lost3dsg/src/perception_module")
import rosstub; rosstub.install()  # noqa: E702
import numpy as np

import config
config.CFG.setdefault("rooms", {}).update({"room_nearest_fallback_m": 0.9, "region_match_iou_min": 0.33, "not_a_param": 1})
from room_manager import RoomManager  # noqa: E402

# A rectangle with one side subdivided: the vertex mean drifts toward the busy side, the area centroid does not.
rect = [(0, 0), (4, 0), (4, 2), (0, 2)]
busy = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 2), (0, 2)]
assert np.allclose(RoomManager._centroid(rect), (2.0, 1.0))
assert np.allclose(RoomManager._centroid(busy), (2.0, 1.0)), RoomManager._centroid(busy)
assert not np.allclose((np.mean([p[0] for p in busy]), np.mean([p[1] for p in busy])), (2.0, 1.0))
assert np.allclose(RoomManager._centroid([(1, 1), (3, 1), (5, 1)]), (3.0, 1.0))   # degenerate: vertex mean

rm = RoomManager(node=None)
assert rm._params["room_nearest_fallback_m"] == 0.9 and rm._params["region_match_iou_min"] == 0.33
assert "not_a_param" not in rm._params and rm._params["gvd_method"] == config.CFG["rooms"]["gvd_method"]
print("OK: area centroid, rooms thresholds reach _params")
