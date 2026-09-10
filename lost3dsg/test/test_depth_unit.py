"""GA-42: the depth unit follows the encoding, not the frame's largest pixel."""
# ruff: noqa: E402, I001  -- rosstub must be installed before the module under test is imported
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, "src", "perception_module"))
import rosstub; rosstub.install()  # noqa: E702
import numpy as np

from utils import depth_to_metres  # noqa: E402

mm = np.array([[1500, 0], [65535, 2000]], dtype=np.uint16)
assert np.allclose(depth_to_metres(mm), [[1.5, 0.0], [65.535, 2.0]])
m = np.array([[1.5, np.inf], [30.0, 2.0]], dtype=np.float32)     # metres with one far/inf pixel
assert np.allclose(depth_to_metres(m)[0, 0], 1.5) and depth_to_metres(m)[1, 0] == 30.0   # NOT divided by 1000
print("OK: uint16 -> /1000, float untouched")
