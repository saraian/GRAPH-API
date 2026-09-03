"""Self-check: _best_frame() picks the freshest live source, never a stale overlay.

The bug this pins: /image_with_bb arrives once per perception cycle (and never while
the agent walks), so serving node.latest_jpeg unconditionally froze the dashboard on
one annotated frame forever. Run: python3 viewer/test_best_frame.py
"""
import os
import sys
import types

# Stub ROS/CV deps so the bridge module imports outside the container
for name in ["cv2", "numpy", "rclpy", "uvicorn", "rclpy.node", "sensor_msgs",
             "sensor_msgs.msg", "lost3dsg", "lost3dsg.srv"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["rclpy.node"].Node = object
sys.modules["sensor_msgs.msg"].Image = object
for cls in ["AddObject", "RemoveObject", "UpdateObject", "MergeObjects", "DeleteObjects", "QueryObjects"]:
    setattr(sys.modules["lost3dsg.srv"], cls, type(cls, (), {}))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graph_api_bridge as b  # noqa: E402


class FakeNode:
    def __init__(self, annotated=None, ann_age=0.0, raw=None, raw_age=0.0):
        now = b.time.time()
        self.latest_jpeg = annotated
        self.last_frame_time = now - ann_age
        self.raw_jpeg = raw
        self.last_raw_time = now - raw_age


def run(node, host_frame):
    """Drive _best_frame with a given node state and host reachability."""
    b.get_node = lambda: node
    if host_frame is None:
        def boom(*a, **k):
            raise OSError("host unreachable")
        b.urllib.request.urlopen = boom
    else:
        class Resp:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

            def read(self_):
                return host_frame
        b.urllib.request.urlopen = lambda *a, **k: Resp()
    return b._best_frame()


def demo():
    b._latest_fresh_composite = lambda: None

    # 1. a CURRENT overlay wins — that is the frame with the boxes on it
    assert run(FakeNode(annotated=b"ANN", ann_age=0.2), b"HOST") == b"ANN"

    # 2. the moment it goes stale (mid-cycle, or the agent starts walking) the live
    #    host feed takes over instead of the dashboard freezing
    assert run(FakeNode(annotated=b"ANN", ann_age=5.0), b"HOST") == b"HOST"

    # 3. host down -> fresh raw ROS camera
    assert run(FakeNode(annotated=b"ANN", ann_age=5.0, raw=b"RAW", raw_age=0.5), None) == b"RAW"

    # 4. everything live is gone -> a stale overlay beats a blank panel
    assert run(FakeNode(annotated=b"ANN", ann_age=99.0, raw=b"RAW", raw_age=99.0), None) == b"ANN"

    # 5. nothing at all
    assert run(FakeNode(), None) is None

    # thresholds are the contract, not incidental
    assert b.ANNOTATED_MAX_AGE_SEC < b.RAW_MAX_AGE_SEC
    assert b.FEED_HEARTBEAT_SEC < 2.0, "must beat the viewer's 2s STALLED badge"
    print("test_best_frame: OK")


if __name__ == "__main__":
    demo()
