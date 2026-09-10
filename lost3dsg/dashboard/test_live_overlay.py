"""Integration check for the bridge's live overlay, with REAL OpenCV and ROS stubbed out.

RUN IT WITH THE CONDA PYTHON, which is the one on this host that has OpenCV:

    /home/xps/miniconda3/envs/habitat_env/bin/python found/dashboard/test_live_overlay.py

The system python3 has fastapi but no cv2; the conda env has cv2 but no fastapi and no ROS. So
everything the overlay does not exercise is stubbed and cv2/numpy are left REAL -- the drawing
is the thing under test, and a stubbed cv2 would asserts nothing while passing.

WHAT THIS DOES NOT COVER, stated so the pass is not read as more than it is: `_pose_from_tf`
is the live pose source and there was no ROS stack up when the overlay was written, so that
path has never answered. The test asserts only that it returns None with no node. Everything
downstream of the pose -- projection, drawing, caching, the disable switch and the failure
path -- runs here against real pixels.

The geometry itself is checked separately and more strictly in live_overlay.py's own
self-check, which reprojects 414 recorded detections against their recorded 2D boxes.
"""
import os
import sys
import types

sys.path.insert(0, os.environ.get('GRAPH_API_ROOT', '/DATA/GRAPH-API') + '/lost3dsg/src/perception_module')
BUNDLE = os.environ.get("BUNDLE", "") or str(
    __import__("dash_env").runs_dir() / "20260903_230232_hm3d_00861")
os.environ["GRAPH_API_OUTPUT_DIR"] = BUNDLE


class _Stub:
    def __init__(self, *a, **k): pass
    def __getattr__(self, _): return lambda *a, **k: None


class _Router:
    routes = []


class _App:
    router = _Router()
    def __init__(self, *a, **k): pass
    def mount(self, *a, **k): pass
    def __getattr__(self, _):
        # get / post / delete / put / middleware / on_event -- every FastAPI decorator is the
        # same shape here: take the route args, return an identity decorator.
        return lambda *a, **k: (lambda f: f)


def mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


mod("rclpy", init=lambda *a, **k: None, spin=lambda *a, **k: None,
    shutdown=lambda *a, **k: None, ok=lambda: True)
mod("rclpy.node", Node=_Stub)
sys.modules["rclpy"].node = sys.modules["rclpy.node"]
mod("cv_bridge", CvBridge=_Stub)
for pkg, names in (("sensor_msgs", ("Image", "CameraInfo")), ("std_msgs", ("String",)),
                   ("geometry_msgs", ("Point", "Pose", "Quaternion", "Vector3", "TransformStamped")),
                   ("nav_msgs", ("OccupancyGrid",))):
    mod(pkg)
    mod(pkg + ".msg", **{n: _Stub for n in names})
    setattr(sys.modules[pkg], "msg", sys.modules[pkg + ".msg"])
mod("lost3dsg")
mod("lost3dsg.srv", **{n: _Stub for n in ("AddObject", "DeleteObjects", "MergeObjects",
                                          "QueryObjects", "RemoveObject", "UpdateObject")})
sys.modules["lost3dsg"].srv = sys.modules["lost3dsg.srv"]
mod("uvicorn", run=lambda *a, **k: None)
mod("fastapi", FastAPI=_App, HTTPException=Exception, Request=_Stub)
mod("fastapi.responses", FileResponse=_Stub, JSONResponse=_Stub, Response=_Stub,
    StreamingResponse=_Stub)
mod("fastapi.staticfiles", StaticFiles=_Stub)
sys.modules["fastapi"].responses = sys.modules["fastapi.responses"]
sys.modules["fastapi"].staticfiles = sys.modules["fastapi.staticfiles"]

import glob  # noqa: E402

import cv2  # noqa: E402
import graph_api_bridge as gb  # noqa: E402

assert hasattr(cv2, "imdecode"), "cv2 is stubbed -- this would test nothing"
print(f"  cv2 {cv2.__version__} is real")

objs = gb._overlay_objects()
assert objs, "no world model objects read"
print(f"  world model: {len(objs)} objects")

# THE STALE-POSE GUARD IS THE POINT, so assert it in both directions. This bundle is an
# archive: its detections are hours old, and drawing boxes from an hours-old pose onto a live
# frame is exactly the "an older frame with boxes" defect this overlay exists to remove.
assert gb._pose_from_detections() is None, "a stale recorded pose must be refused"
print("  a stale recorded pose is refused (wall-clock age guard)")
recorded = gb._pose_from_detections(max_age=1e12)
assert recorded, "the recorded-pose reader found nothing at all"
print(f"  recorded-pose reader works: {recorded[2]}, pos "
      f"{[round(v, 2) for v in recorded[0]]}")
assert gb._pose_from_tf() is None, "there is no ROS node here, so TF must not answer"
print("  TF returns None with no node (WRITTEN, NEVER RUN against a live stack)")

# Drive the drawing with the recorded pose, since no live pose exists on this host.
gb._overlay_pose = lambda: recorded

i1 = gb._overlay_intrinsics(1280, 960)
i2 = gb._overlay_intrinsics(640, 480)
assert abs(i2["fx"] - i1["fx"] / 2) < 1e-6 and abs(i2["cx"] - i1["cx"] / 2) < 1e-6, \
    "intrinsics must scale with the frame, fx and cx are in pixels"
print(f"  intrinsics scale with frame size: fx {i1['fx']:.0f} @1280 -> {i2['fx']:.0f} @640")

frames = sorted(glob.glob(BUNDLE + "/frames/*.jpg"))
assert frames, "no frames in the bundle"
raw = open(frames[len(frames) // 2], "rb").read()
out = gb._with_overlay(raw)
assert out != raw, "the overlay drew nothing"
print(f"  drew on a real frame: {len(raw):,} -> {len(out):,} bytes")

assert gb._with_overlay(raw) is out, "the second call should hit the cache"
print("  repeat call served from cache")

gb.OVERLAY_ON = False
assert gb._with_overlay(raw) is raw, "disabled overlay must pass the frame through"
gb.OVERLAY_ON = True
print("  BRIDGE_OVERLAY=0 passes the frame through untouched")

before = gb._OVERLAY_FAILURES["n"]
assert gb._with_overlay(b"not a jpeg") == b"not a jpeg", "bad input must return the original"
assert gb._OVERLAY_FAILURES["n"] == before, "a non-decodable frame is not an exception path"
print("  undecodable input returns the original frame")

out_path = "/tmp/claude-1000/-DATA-GRAPH-API/68f0878a-088e-4ac1-8cb4-d2ea654233aa/scratchpad/bridge_overlay.jpg"
open(out_path, "wb").write(out)
print(f"  wrote {out_path}")
print("  ALL CHECKS PASSED")
