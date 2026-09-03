#!/usr/bin/env python3
"""Host-side habitat feed: renders a scene and streams frames over TCP.

Runs in a plain habitat_sim environment (no ROS). The container-side
habitat_feed_node.py connects, converts, and publishes to ROS topics + TF.
Protocol: length-prefixed pickle dicts {rgb, depth, cam_pos, cam_quat,
base_pos, base_quat, t, w, h, hfov}.

Motion has two phases:
  MAPPING   (first FEED_MAPPING_SECONDS): continuous coverage tour over the
            navmesh — greedy nearest-unvisited waypoints with a full
            look-around at each — so the SLAM map and room segmentation are
            built before any detection is attempted.
  DETECTION walk/dwell bursts (FEED_WALK / FEED_DWELL frames): perception
            only runs while the robot is still.

FEED_SHOW=1 opens a window with the agent camera; FEED_OVERLAY=1 additionally
draws the belief's 3D boxes (PCA-oriented when available) projected into it,
polled from the Graph API bridge at FEED_BRIDGE (default http://127.0.0.1:8081).
Only boxes actually in view are drawn (depth-tested against the frame, the same
rule as the ROS /image_with_bb overlay — box_view.py).

Multi-storey scenes: with config `habitat.single_floor` the tour goals stay on the
start floor (`habitat.floor_tolerance_m`), because the 2D SLAM grid cannot tell one
storey from another. Config is config.yaml / GRAPH_API_CONFIG, as on the ROS side.
"""
import collections
import functools
import json
import math
import os
import pickle
import socket
import struct
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import habitat_sim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
from box_view import BOX_EDGES, box_corners_map, project_visible  # noqa: E402  (ROS-free)
from config import CFG, CFG_PATH  # noqa: E402

_print = functools.partial(print, flush=True)  # nohup/file logs must not buffer
_log_ring = collections.deque(maxlen=400)      # served by the control server's /logs


def print(*args, **kw):  # noqa: A001  — keep the module's existing print() call sites
    _log_ring.append(" ".join(str(a) for a in args))
    _print(*args, **kw)

hab_cfg = CFG.get("habitat", {}) if isinstance(CFG, dict) else {}
SCENE = os.environ.get("HABITAT_SCENE", "train_99248")
DATASET = os.environ.get("HABITAT_DATASET", "/DATA/habitat_hospital/holodeck_clinical.scene_dataset_config.json")
PORT = int(os.environ.get("FEED_PORT", "7799"))
FPS = float(os.environ.get("FEED_FPS", hab_cfg.get("fps", 3.0)))
SEED = int(os.environ.get("FEED_SEED", "7"))
SHOW = os.environ.get("FEED_SHOW", "0") == "1"
OVERLAY = os.environ.get("FEED_OVERLAY", "0") == "1"

# GA-265. WHAT THE HABITAT WINDOW DRAWS, toggleable from the window itself AND from the
# dashboard, with one shared state so the two can never disagree.
#
# The window is the only view of the simulator that exists while a run is walking, and until
# now it drew every belief box unconditionally -- which at 269 objects is the same unreadable
# pile the dashboard had. Keys toggle layers in the window; the control server exposes the
# same dict at /layers so the dashboard can read it and set it. Neither side owns it.
LAYERS = {
    "boxes": True,        # b  belief boxes projected into the camera
    "labels": True,       # l  the label beside each box
    "admitted": True,     # 1  grade filters -- an object carries its verdict
    "held": True,         # 2
    "declined": False,    # 3  off by default: a decline is not in the map
    "nogrounds": True,    # 4
    "walls": False,       # w  detected wall segments, per-frame from depth (opt-in node)
    "hud": True,          # h  the frame counter and phase text
}
LAYER_KEYS = {ord("b"): "boxes", ord("l"): "labels", ord("1"): "admitted",
              ord("2"): "held", ord("3"): "declined", ord("4"): "nogrounds",
              ord("w"): "walls", ord("h"): "hud"}


def _publish_layers():
    """Write the layer state where the dashboard can read it. Same channel as
    merge_pending.json, and atomic for the same reason (GA-257)."""
    d = os.environ.get("RUN_DIR") or os.environ.get("GRAPH_API_OUTPUT_DIR") or ""
    if not d:
        return
    try:
        path = os.path.join(d, "feed_layers.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"t": time.time(), "layers": LAYERS,
                       "keys": {chr(k): v for k, v in LAYER_KEYS.items()}}, fh)
        os.replace(tmp, path)
    except OSError:
        pass
# GA-267. BRIDGE_PORT, honoured. This was a fixed 8081 while the bridge is moved to 8091
# whenever 8081 is taken -- which it is on this machine, by another project. The belief
# poller therefore reached nothing for a whole run, so the Habitat window drew no boxes
# while the dashboard (which the bridge itself serves) showed them. Two symptoms, one cause,
# and neither log said so because the poller swallows its errors by design.
# GA-270. From config, with the env override kept for a single run. See config.py
# "services": a hardcoded address here left the belief poller talking to a dead port for an
# entire run, and the window looked healthy the whole time.
_SVC = (CFG.get("services", {}) or {}) if isinstance(CFG, dict) else {}
BRIDGE = os.environ.get("FEED_BRIDGE") or (
    f"http://{_SVC.get('bridge_host', '127.0.0.1')}:"
    f"{os.environ.get('BRIDGE_PORT') or _SVC.get('bridge_port', 8081)}")
MAPPING_SECONDS = float(os.environ.get("FEED_MAPPING_SECONDS", hab_cfg.get("mapping_seconds", 0.0)))
SEND_TIMEOUT = float(os.environ.get("FEED_SEND_TIMEOUT", "10"))
# GA-120. WHICH FLOOR THIS RUN MAPS. Unset = whatever habitat drops the agent on, which is what
# every run before 2026-08-31 did — one uncontrolled random sample chose the storey of every map
# this project has ever made, and it landed on 1.35 each time. ROS z, which IS habitat y.
SPAWN_FLOOR = float(os.environ["FEED_SPAWN_FLOOR"]) if os.environ.get("FEED_SPAWN_FLOOR") else None
# GA-131. Ground-truth instance ids in the frame, for VALIDATION ONLY. Off by default.
GT_SEMANTIC = os.environ.get("FEED_GT_SEMANTIC", "0") == "1"
CTRL_PORT = int(os.environ.get("FEED_CTRL_PORT", "7790"))
# Where the per-frame stats and the BEV payload go. ONE directory, and it is an error for the
# run not to know which.
#
# GA-60 moved this off three hardcoded literals but left the fallback chain ["/tmp/graphapi_live",
# "/out", "/tmp"] with "first one that exists wins". /tmp ALWAYS exists, so the write could never
# fail and could never warn — it silently landed outside the bundle. Measured by the testing lane
# on 2026-08-31: run 19 (20260831_151134) has no feed_stats.json in its bundle and one at
# /tmp/graphapi_live/feed_stats.json with a matching mtime, holding walk 6, dwell 60, mapping 150,
# 351 steps, 2.4 m in 553.6 s. /tmp/feed_stats.json held a DIFFERENT run's stats from another day,
# so the third fallback was live too and runs were overwriting each other outside any bundle.
#
# That is how run 19's feed geometry became unrecoverable from its own bundle, which is what made
# the figure behind the dwell ruling an estimate rather than a measurement.
#
# So: OUT_DIR wins alone and is created if absent. Without it there is exactly one fallback and it
# ANNOUNCES ITSELF, because a stats file outside the bundle is invisible to the gate's a5 probe and
# to every reader afterwards.
if os.environ.get("OUT_DIR"):
    STATS_DIR = Path(os.environ["OUT_DIR"])
    STATS_DIR.mkdir(parents=True, exist_ok=True)
else:
    STATS_DIR = Path("/tmp/graphapi_live")
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[feed] WARNING: OUT_DIR is not set. feed_stats.json and bev_data.json go to "
          f"{STATS_DIR} and will NOT be in any run bundle. The next run overwrites them.",
          flush=True)

SINGLE_FLOOR = bool(hab_cfg.get("single_floor", True))
FLOOR_TOL = float(hab_cfg.get("floor_tolerance_m", 0.5))

# GA-52: say WHICH config was loaded, and the two values that come only from it.
# Every other habitat value the feed host reads has an environment override that the launcher
# always sets, so it is visible in the process line. These two have none — they are decided by
# the config file alone, and config.py returns its defaults SILENTLY when that file is absent.
# Nothing else in the tree reads them, so a two-storey arm that failed to take effect leaves a
# bundle identical to the single-storey one. CFG_PATH is None when the defaults are in force.
print(f"[config] loaded={CFG_PATH or '<defaults, no file found>'} "
      f"single_floor={SINGLE_FLOOR} floor_tolerance_m={FLOOR_TOL}", flush=True)
if CFG_PATH is None:
    print("[config] WARNING: running on config.py defaults — no file was read. "
          "Any value set in a yaml is NOT in force.", flush=True)
# OWNER RULING 25, 2026-09-01: raise the sensor to 1280x960 for the next run.
#
# WHY IT IS THE ONLY LEVER THAT ADDS INFORMATION: 640x480 is the ceiling on every small-object
# problem here. The air-conditioner sliver is 28x14 px at source; at 1280x960 it is 56x28, and no
# crop construction recovers detail the sensor never captured. Depth scales with it too, so the
# back-projection that produces the 3D box gets the same increase as the mask.
#
# MEASURED COST, render plus pickle, 20 frames, median, on the real scene:
#     640x480    28.7 ms   34.9 fps max   2.15 MB/frame
#     960x720    51.8 ms   19.3 fps       4.84 MB
#    1280x960    89.1 ms   11.2 fps       8.60 MB   <- 4x the pixels, 3.7x headroom at 3 fps
#   1920x1440   254.2 ms    3.9 fps      19.35 MB   <- 30% margin; a hiccup drops frames
#
# NOT MEASURED, and it must not be read as measured: this is render and serialisation only. It
# says nothing about the detector, segmentation or describer, where the ~42 s per-cycle overhead
# lives — larger frames will not help that and may worsen it.
#
# ATTRIBUTION WARNING, recorded because the run cannot recover it afterwards: resolution moves the
# detector, the segmentation, the depth and the describer AT ONCE. A gain measured at 1280x960 is
# a gain of the system, not of any subsystem, and cannot be assigned to one without a second arm.
W = int(os.environ.get("FEED_WIDTH", hab_cfg.get("width", 1280)))
H = int(os.environ.get("FEED_HEIGHT", hab_cfg.get("height", 960)))
HFOV = float(os.environ.get("FEED_HFOV", hab_cfg.get("hfov", 90.0)))
SENSOR_HEIGHT = 1.5


def make_sim():
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = SCENE
    cfg.scene_dataset_config_file = DATASET
    cfg.random_seed = SEED
    specs = []
    # GA-131. GROUND-TRUTH INSTANCE IDS, off unless asked for.
    #
    # The semantic sensor renders habitat's own instance id per pixel, so a detection's mask can be
    # intersected with it to yield the dominant TRUE object behind that detection. That is what
    # labels a pair as a genuine match or a genuine non-match. Without it, association validation
    # falls back to co-visible hard negatives, which are free and certain but NEGATIVES ONLY: they
    # bound false merges and say nothing about missed ones — and a missed merge is the
    # air-conditioner problem.
    #
    # MEASURED, not assumed: 12 of 12 frames on hm3d_00861 carry non-zero ids, 206 distinct,
    # uint32 480x640. Note this is a DIFFERENT channel from semantic_scene.regions/objects, whose
    # AABBs on this scene come back with [0,0,0] centres and -inf extents — the annotation
    # GEOMETRY is empty while the sensor is fine. One does not imply the other.
    #
    # OFF BY DEFAULT because it costs a third render and a third array in every frame, and no
    # runtime consumer may read it. It is validation data written to the bundle.
    _sensors = [("color_sensor", habitat_sim.SensorType.COLOR),
                ("depth_sensor", habitat_sim.SensorType.DEPTH)]
    if GT_SEMANTIC:
        _sensors.append(("semantic_sensor", habitat_sim.SensorType.SEMANTIC))
    for uuid, stype in _sensors:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [H, W]
        s.position = [0.0, SENSOR_HEIGHT, 0.0]
        s.hfov = HFOV
        specs.append(s)
    agent_cfg = habitat_sim.agent.AgentConfiguration(
        sensor_specifications=specs,
        action_space={
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.15)),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)),
            "move_backward": habitat_sim.agent.ActionSpec(
                "move_backward", habitat_sim.agent.ActuationSpec(amount=0.15)),
        },
    )
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))


def ensure_navmesh(sim):
    if sim.pathfinder.is_loaded:
        return True
    if os.environ.get("FEED_NAVMESH", "0") != "1":
        print("[feed] navmesh recompute disabled (FEED_NAVMESH!=1), blind wander")
        return False
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    settings.agent_height = 1.5
    settings.agent_radius = 0.2
    print("[feed] recomputing navmesh...")
    ok = sim.recompute_navmesh(sim.pathfinder, settings)
    print(f"[feed] navmesh recomputed: {ok}")
    return ok


def habitat_pose_to_ros(position, quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
    hx, hy, hz = float(position[0]), float(position[1]), float(position[2])
    ros_position = np.array([-hz, -hx, hy], dtype=np.float64)

    R_change = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    qx, qy, qz, qw = (float(v) for v in quat_xyzw)
    R_habitat = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ])
    R_ros = R_change @ R_habitat @ R_change.T

    trace = np.trace(R_ros)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R_ros[2, 1] - R_ros[1, 2]) * s
        y = (R_ros[0, 2] - R_ros[2, 0]) * s
        z = (R_ros[1, 0] - R_ros[0, 1]) * s
    else:
        if R_ros[0, 0] > R_ros[1, 1] and R_ros[0, 0] > R_ros[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R_ros[0, 0] - R_ros[1, 1] - R_ros[2, 2])
            w = (R_ros[2, 1] - R_ros[1, 2]) / s
            x = 0.25 * s
            y = (R_ros[0, 1] + R_ros[1, 0]) / s
            z = (R_ros[0, 2] + R_ros[2, 0]) / s
        elif R_ros[1, 1] > R_ros[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R_ros[1, 1] - R_ros[0, 0] - R_ros[2, 2])
            w = (R_ros[0, 2] - R_ros[2, 0]) / s
            x = (R_ros[0, 1] + R_ros[1, 0]) / s
            y = 0.25 * s
            z = (R_ros[1, 2] + R_ros[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R_ros[2, 2] - R_ros[0, 0] - R_ros[1, 1])
            w = (R_ros[1, 0] - R_ros[0, 1]) / s
            x = (R_ros[0, 2] + R_ros[2, 0]) / s
            y = (R_ros[1, 2] + R_ros[2, 1]) / s
            z = 0.25 * s
    return ros_position, np.array([x, y, z, w], dtype=np.float64)

# --- belief overlay: ROS map frame (z-up) -> habitat (y-up) -> camera pixels ---
def ros_to_habitat(p):
    # inverse of habitat_pose_to_ros: ros = (-hz, -hx, hy)
    return np.array([-p[1], p[2], -p[0]], dtype=np.float64)


def _quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def visible_points(corners, depth, Rt, cam, fx):
    """8 corners (+ centre) in the ROS map frame -> habitat camera -> OPTICAL frame,
    then the shared rule (box_view.project_visible): (corner pixels, n_visible)."""
    hab = np.array([ros_to_habitat(c) for c in corners + [np.mean(corners, axis=0)]])
    pc = (hab - cam) @ Rt.T              # Rt @ (p - cam) per row
    cam_xyz = np.column_stack([pc[:, 0], -pc[:, 1], -pc[:, 2]])   # habitat camera looks down -z, y up
    return project_visible(cam_xyz, depth, fx, fx, W / 2.0, H / 2.0, W, H)


def draw_walls(bgr, walls, cam_pos, cam_quat):
    """Draw detected wall segments as vertical quads. GA-271.

    A segment from wall_detector is (p0, p1, z_min, z_max, n_inliers, rms) in the MAP frame:
    a top-down line with the height band it was observed over. So each one is a quad -- the
    line swept through its own height -- which is what distinguishes it from room_manager's
    polygon edges, where an edge only means "observed free space stopped here" and carries no
    height at all.
    
    NO MAP IS REQUIRED. These come from the current depth frame, so they are available on a
    run with no accumulated geometry, which is the case this has to work in.

    The same projection as draw_belief -- map (z-up) -> habitat (y-up) -> pixels -- reused
    rather than reimplemented: a second copy would drift from the first, and a wall drawn
    with a slightly different transform looks like a detection error rather than a bug.
    """
    import cv2
    fx = (W / 2.0) / math.tan(math.radians(HFOV) / 2.0)
    Rt = _quat_to_rot(cam_quat).T
    cam = np.asarray(cam_pos, dtype=np.float64)
    for seg in walls:
        try:
            p0, p1 = seg[0], seg[1]
            z0, z1 = float(seg[2]), float(seg[3])
        except (IndexError, TypeError, ValueError):
            continue
        corners = [[float(p0[0]), float(p0[1]), z0], [float(p1[0]), float(p1[1]), z0],
                   [float(p1[0]), float(p1[1]), z1], [float(p0[0]), float(p0[1]), z1]]
        pts, n_vis = visible_points(corners, None, Rt, cam, fx)
        # visible_points APPENDS the centroid, so 4 corners come back as 5 points. Taking
        # them all would put the centre of the quad in its outline and draw a bow-tie.
        quad = pts[:4] if pts is not None else None
        if not n_vis or quad is None or len(quad) < 4 or any(q is None for q in quad):
            continue
        poly = np.array([[int(q[0]), int(q[1])] for q in quad], dtype=np.int32)
        # Outline plus a wash, so a wall reads as a surface without hiding what is in front
        # of it. Walls are context; the objects are the subject.
        overlay = bgr.copy()
        cv2.fillPoly(overlay, [poly], (90, 70, 40))
        cv2.addWeighted(overlay, 0.28, bgr, 0.72, 0, bgr)
        cv2.polylines(bgr, [poly], True, (200, 160, 90), 1, cv2.LINE_AA)


def draw_belief(bgr, belief, cam_pos, cam_quat, depth=None):
    """Draw the belief boxes that are in view: skipped when no test point is visible
    (occluded or outside the frame), thin when fewer than 5 of 9 are, full otherwise."""
    import cv2
    fx = (W / 2.0) / math.tan(math.radians(HFOV) / 2.0)
    R = _quat_to_rot(cam_quat)          # camera -> world (habitat)
    Rt = R.T
    cam = np.asarray(cam_pos, dtype=np.float64)
    for obj in belief:
        corners, oriented = box_corners_map(obj.get("bbox"))
        if not corners:
            continue
        pts, n_vis = visible_points(corners, depth, Rt, cam, fx)
        if not n_vis:
            continue
        color = (0, 200, 255) if oriented else (255, 160, 0)
        thick = 2 if n_vis >= 5 else 1
        for i, j in BOX_EDGES:
            cv2.line(bgr, pts[i], pts[j], color, thick, cv2.LINE_AA)
        top = min(pts, key=lambda p: p[1])
        cv2.putText(bgr, str(obj.get("label", "?")), (top[0], max(12, top[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thick, cv2.LINE_AA)


class BeliefPoller(threading.Thread):
    """Polls the bridge for the persistent belief once a second."""
    def __init__(self):
        super().__init__(daemon=True)
        self.objects = []
        self.walls, self.walls_available = [], False

    def run(self):
        while True:
            try:
                with urllib.request.urlopen(f"{BRIDGE}/persistent_perception", timeout=2) as r:
                    data = json.loads(r.read().decode())
                self.objects = data if isinstance(data, list) else data.get("data", [])
            except Exception:
                pass
            try:
                # GA-271. Walls come from the same hub as the belief, so the window and the
                # dashboard draw the same segments rather than two independent fits.
                with urllib.request.urlopen(f"{BRIDGE}/walls", timeout=2) as r:
                    w = json.loads(r.read().decode())
                self.walls = w.get("walls") or []
                self.walls_available = bool(w.get("available"))
            except Exception:
                pass
            time.sleep(1.0)


# --- HTTP control server (:CTRL_PORT) — the bridge proxies the viewer's teleop here ---
_HAVE_CV2 = [True]


class Ctrl:
    """State shared between the control server threads and the sim loop.
    ponytail: plain attributes, GIL-atomic reads/writes only — no lock needed."""
    def __init__(self):
        self.auto_mode = True
        self.actions = collections.deque()   # (act, {x,y,z,amount}) drained by the sim loop
        self.config = {}                     # perceive_while_moving/perm/temp/seg/det, echoed in bev_data
        self.latest_jpeg = None
        self.bev = {}


CTRL = Ctrl()


class CtrlHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):       # keep the feed log clean
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        try:
            self._route(url.path, q)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _route(self, path, q):
        if path == "/frame.jpg":
            jpeg = CTRL.latest_jpeg
            if not jpeg:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
        elif path in ("/feed", "/feed.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                jpeg = CTRL.latest_jpeg
                if jpeg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(max(1.0 / FPS, 0.1))
        elif path == "/bev_data":
            self._json(CTRL.bev)
        elif path == "/logs":
            self._json({"logs": list(_log_ring)})
        elif path == "/auto_mode":
            CTRL.auto_mode = q.get("enabled", "true").lower() in ("1", "true", "yes", "on")
            self._json({"success": True, "auto_mode": CTRL.auto_mode})
        elif path == "/action":
            act = q.get("act") or q.get("action") or ""
            params = {k: float(q[k]) for k in ("x", "y", "z", "amount") if q.get(k)}
            CTRL.actions.append((act, params))
            self._json({"success": True, "queued": act})
        elif path == "/set_config":
            CTRL.config.update(q)
            self._json({"success": True, "config": CTRL.config})
        elif path == "/layers":
            # GET returns the state; ?set=name:0|1 (repeatable) changes it. The dashboard and
            # the window write the SAME dict, so a toggle in either place is visible in both.
            changed = []
            # COMMA-SEPARATED, not a repeated parameter: `_route` receives only the flattened
            # `q`, which keeps the first value of each key, so `?set=a:0&set=b:0` would
            # silently apply just the first. Widening the signature to pass the raw query
            # would touch every route for one caller's convenience.
            #   /layers?set=boxes:0,labels:0
            for item in (q.get("set") or "").split(","):
                item = item.strip()
                if not item:
                    continue
                name, _, val = item.partition(":")
                if name in LAYERS:
                    LAYERS[name] = val.lower() not in ("0", "false", "off", "")
                    changed.append(name)
            if changed:
                _publish_layers()
                print(f"[feed] layers set from the dashboard: "
                      f"{ {k: LAYERS[k] for k in changed} }", flush=True)
            self._json({"layers": LAYERS, "changed": changed,
                        "keys": {chr(k): v for k, v in LAYER_KEYS.items()}})
        elif path == "/get_config":
            self._json({"success": True, "config": CTRL.config})
        else:
            self._json({"success": False, "error": f"unknown path {path}"}, code=404)


def start_ctrl_server():
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", CTRL_PORT), CtrlHandler)
    except OSError as exc:
        print(f"[feed] control server disabled ({exc})")
        return
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[feed] control server on :{CTRL_PORT} "
          "(frame.jpg feed.mjpg bev_data logs auto_mode action set_config)")



TEST_MODE = os.environ.get("FEED_TEST_MODE", "0").lower() in ("1", "true", "yes", "on")
TEST_RADIUS_M = float(os.environ.get("FEED_TEST_RADIUS", "4.0"))

# GA-213. AN EXPLICIT TEST SPAWN, in HABITAT coordinates, "x,y,z".
#
# Set it and the density search is skipped entirely. Three reasons that matters, and the
# third is why it exists at all:
#   1. REPRODUCIBLE. The search samples random navigable points, so two runs of the same
#      config could stand in different rooms and their object counts would not be comparable.
#   2. FAST. The search cost 400 navigable samples before the first frame, which delayed the
#      feed host past probe a6's 20 s window and SKIPPED it -- a skipped probe fails the gate.
#   3. It is the owner's decision which room a test runs in, and a decision belongs in a
#      config value rather than in the outcome of a random draw.
#
# The value for hm3d_00861's ground floor is recorded in the run config; the search PRINTS
# the point it chose in this same format, so a good spot found once can be pinned.
TEST_SPAWN = os.environ.get("FEED_TEST_SPAWN", "").strip()

# GA-219. A BOUNDED LOCAL WALK, in metres from the spawn. 0 = turn in place.
#
# Turning in place and walking freely BOTH fail to produce a publishable map, for opposite
# reasons, and both were measured today:
#   free walk   -> the agent climbed the stairs; node z spread 3.071 m over 368 nodes, and
#                  the map was REJECTED for straddling three storeys (owner ruling 25).
#   turn only   -> node z spread 0.000 m, single_floor True -- and 37 nodes at ONE distinct
#                  pose, footprint [0.0, 0.0] m, REJECTED against a 30-distinct-pose floor.
# A map needs DISTINCT POSITIONS for a footprint and a SINGLE STOREY to be usable, and those
# two demands pull opposite ways the moment the confinement is only a tolerance.
#
# This makes the confinement GEOMETRIC instead: goals are sampled inside a radius of the
# spawn, so the stairs are not merely discouraged but out of reach. It does not fix the
# floor_confinement defect -- teleport mode failed to hold on the free walk and that is still
# open -- it sidesteps it for a mapping run whose only job is one room.
TEST_WALK_RADIUS = float(os.environ.get("FEED_TEST_WALK_RADIUS", "0"))

# GA-256. A ROOM TOUR: visit N well-separated places on the traversed storey.
#
# WHY. Run 20260901_174810_hm3d_00861 recorded total_distance_m = 0.0 over 1,566 steps. Test
# mode turns in place (radius 0) or wanders a disc around the spawn (radius > 0), and neither
# leaves the room it started in. Three things the paper needs die on that: coverage, room
# segmentation (the ridge pass finds no critical points because there is only one room to
# segment), and the held-pool resolution rate -- a proposal set aside for a better view can
# only be resolved BY a better view, and 53 holds resolved none because the agent never moved.
#
# The waypoints are chosen by FARTHEST-POINT SAMPLING over navigable points on the storey, so
# they spread across the floor plan instead of clustering wherever the sampler happened to
# land. Seeded, so two runs of the same scene tour the same places and are comparable.
# Config first, environment second: the yaml states the intended setting and travels with
# the bundle, while the env var flips ONE run without editing a file everyone shares. Same
# precedence the rest of this host already uses.
TEST_TOUR = int(os.environ.get("FEED_TEST_TOUR", hab_cfg.get("tour_waypoints", 0)))
# Frames spent turning on arrival. A waypoint the agent walks through teaches the detector
# almost nothing: the objects that matter are the ones it stops and looks at.
TEST_TOUR_SCAN = int(os.environ.get("FEED_TEST_TOUR_SCAN", hab_cfg.get("tour_scan_frames", 12)))

# GA-258. DYNAMIC DWELL: stay while merges are still waiting to be confirmed.
#
# A merge commits only after `merge_min_consecutive` consecutive sweeps over the evidence
# threshold. MEASURED on 20260901_174810_hm3d_00861: of 265,944 held pairs the fused log-odds
# reached p90 7.77 against a threshold of 3.0 -- more than a tenth had ENOUGH evidence and
# were held anyway, because the agent turned away before a second consecutive sweep could see
# the pair. That is why 269 world-model entries carry only 117 labels.
#
# A fixed dwell is wrong in both directions: it wastes frames when nothing is pending, and
# leaves too early when something is. So the object manager publishes what is pending and the
# agent stays while that number is above zero -- bounded, because a pair that never resolves
# must not hold the run forever.
TOUR_DWELL_DYNAMIC = os.environ.get(
    "FEED_TEST_DWELL_DYNAMIC",
    "1" if hab_cfg.get("dwell_dynamic", True) else "0").lower() in ("1", "true", "yes", "on")
TOUR_DWELL_MIN = int(os.environ.get("FEED_TEST_DWELL_MIN", hab_cfg.get("dwell_min_frames", 8)))
TOUR_DWELL_MAX = int(os.environ.get("FEED_TEST_DWELL_MAX", hab_cfg.get("dwell_max_frames", 90)))
# RUN_DIR first: that is the bundle, and the container's /ws/output is bind-mounted onto it.
# GRAPH_API_OUTPUT_DIR is exported INSIDE the container only, so on the host it is empty and
# os.path.join("", "merge_pending.json") yields a bare relative name that never opens --
# measured on 20260902_125130, where every dwell reported "sweep None" and capped out.
_MERGE_PENDING_DIR = (os.environ.get("RUN_DIR")
                      or os.environ.get("GRAPH_API_OUTPUT_DIR")
                      or "")
_MERGE_PENDING_PATH = os.path.join(_MERGE_PENDING_DIR, "merge_pending.json")
if not _MERGE_PENDING_DIR:
    print("[feed] WARNING: neither RUN_DIR nor GRAPH_API_OUTPUT_DIR is set; the dynamic dwell "
          "cannot read pending merges and will run to FEED_TEST_DWELL_MAX at every waypoint",
          flush=True)
else:
    print(f"[feed] dynamic dwell reads {_MERGE_PENDING_PATH}", flush=True)


def _pending_merges():
    """-> (pending, sweep) from the object manager's sidecar, or (None, None).

    None means UNKNOWN, and the caller must not read it as zero: "nothing is pending" and
    "the file is not there yet" are different states, and treating the second as the first
    would cut every dwell short in exactly the runs where the manager is slow to start.
    """
    try:
        with open(_MERGE_PENDING_PATH) as fh:
            d = json.load(fh)
        return int(d.get("pending", 0)), int(d.get("sweep", -1))
    except (OSError, ValueError, TypeError):
        return None, None


def _dense_spawn_point(sim, spawn_floor, radius_m=4.0, candidates=400):
    """A navigable point with the MOST annotated objects around it. GA-212.

    TEST MODE exists to measure the pipeline, not the navigation. A tour spends most of its
    frames in corridors and doorways, so a short run sees few objects and the association
    layer -- which needs PAIRS -- gets almost nothing to decide. Standing still in the
    busiest room and turning gives the densest object stream the scene can offer, from one
    pose, with no path-following to fail.

    Density is counted from the scene's OWN semantic annotations, not from a previous run's
    output: the annotations are in the simulator's frame and need no map, no localisation
    and no assumption about what a past run happened to detect. When a scene carries no
    annotations this falls back to the ordinary spawn and SAYS SO -- an unannotated scene is
    not a reason to invent a density.
    """
    try:
        scene = sim.semantic_scene
        objs = [o for o in (scene.objects or []) if o is not None and o.aabb is not None]
    except Exception as exc:
        print(f"[feed] test mode: no semantic annotations ({type(exc).__name__}); "
              f"falling back to the ordinary spawn")
        return _spawn_point(sim, spawn_floor), None
    if not objs:
        print("[feed] test mode: scene carries no annotated objects; ordinary spawn")
        return _spawn_point(sim, spawn_floor), None

    centres = np.array([[o.aabb.center[0], o.aabb.center[1], o.aabb.center[2]] for o in objs])

    # THE SAME-STOREY FILTER IS GONE, AND IT HAS TO BE. Measured on hm3d_00861: EVERY
    # annotated object and region reports `aabb.center[1] == 0.00`, so the height channel
    # carries no information at all. A filter on it compared 0.0 against the navmesh height
    # and passed for BOTH floors -- it looked like a storey guard and was a no-op, which is
    # worse than no guard because it reads as one.
    #
    # THE NAVMESH IS THE ONE THAT WORKS. `spawn_floor` (FEED_SPAWN_FLOOR) constrains the
    # candidate points to a real storey, measured from navmesh geometry rather than from an
    # annotation that turns out to be empty. Density is then scored in x/z ONLY, where the
    # annotation is sound. Objects on the floor above still count toward a point below them,
    # and that is a KNOWN limitation of this scene's annotations, not an oversight.
    flat_ok = float(np.abs(centres[:, 1]).max()) < 1e-6
    if flat_ok:
        print("[feed] test mode: annotation heights are all zero in this scene; relying on "
              "FEED_SPAWN_FLOOR for the storey and scoring density in x/z only")
    best_n, best_pt = -1, None
    on_floor = 0
    for _ in range(candidates):
        p = np.array(sim.pathfinder.get_random_navigable_point())
        if spawn_floor is not None and abs(float(p[1]) - spawn_floor) > FLOOR_TOL:
            continue
        on_floor += 1
        d = np.linalg.norm(centres[:, [0, 2]] - p[[0, 2]], axis=1)
        n = int((d < radius_m).sum())
        if n > best_n:
            best_n, best_pt = n, p
    if spawn_floor is not None:
        print(f"[feed] test mode: {on_floor}/{candidates} sampled points were on floor "
              f"{spawn_floor:+.2f} (tolerance {FLOOR_TOL:.2f} m)")
    if best_pt is None:
        print("[feed] test mode: no navigable point matched the floor; ordinary spawn")
        return _spawn_point(sim, spawn_floor), None
    print(f"[feed] TEST MODE: spawning at the densest point — {best_n} annotated objects "
          f"within {radius_m:.1f} m; the agent will TURN IN PLACE and never navigate")
    # Printed in the exact form FEED_TEST_SPAWN accepts, so this spot can be pinned and the
    # search never repeated. THE COUNT IS AN OVERCOUNT where a scene's annotation heights are
    # degenerate: with every aabb.center[1] == 0, objects on the floor ABOVE still fall inside
    # an x/z radius. It ranks candidate points honestly against each other; it is not a claim
    # about what the camera will see.
    print(f"[feed] TEST MODE: pin this spot with "
          f"FEED_TEST_SPAWN={best_pt[0]:.4f},{best_pt[1]:.4f},{best_pt[2]:.4f}")
    return best_pt, best_n


def _spawn_point(sim, spawn_floor, tries=4000):
    """A navigable point, on the REQUESTED floor when one is asked for.

    WHY THIS MATTERS MORE THAN IT LOOKS. Everything downstream takes its floor from wherever the
    agent lands: Tour.floor_y reads the agent's position, SINGLE_FLOOR then confines every goal to
    that storey, and the topdown map is rendered for it. So this one sample decides which floor the
    whole run maps. hm3d_00861 has four floors and every run so far spawned randomly and landed on
    1.35 — the other three ([-1.59, 0.43, 2.21]) have never been mapped by any run.

    FRAMES, because getting this wrong is silent and the bug would look like a working run.
    habitat_pose_to_ros is `ros = (-hz, -hx, hy)`, so ROS z IS habitat y — equal, not negated. The
    floor heights from the navmesh clustering are ROS z, and `p[1]` here is habitat y, so they
    compare directly with no conversion. A negation here would spawn on a mirrored floor that does
    not exist, find nothing, and refuse — which is at least loud. Using habitat's `p[2]` instead
    would compare a height against a horizontal coordinate and silently pick a wrong floor, which
    is not.

    CEILING, stated rather than guarded: floor matching uses FLOOR_TOL (0.5 m), so two storeys
    closer together than 2 x FLOOR_TOL could both match a request and the first sample wins.
    hm3d_00861's floors are 0.86 m apart at their closest, so this is safe there. A scene with
    half-levels or a mezzanine needs the tolerance narrowed or the nearest-point rule instead of
    first-within-tolerance. ponytail: not built, because no scene in this project has one.

    REFUSES RATHER THAN FALLING BACK. If no navigable point is found on the requested floor, this
    raises. A fallback to a random point would map a different storey and the floor stamp would
    then honestly record that storey — so the bundle would be internally consistent and answer the
    wrong question. Rule 14's shape: fail where the cause is, not where the symptom appears.
    """
    if spawn_floor is None:
        return sim.pathfinder.get_random_navigable_point()
    best = None
    for _ in range(tries):
        p = sim.pathfinder.get_random_navigable_point()
        d = abs(float(p[1]) - spawn_floor)
        if d < FLOOR_TOL:
            print(f"[feed] spawn floor {spawn_floor:+.2f} (ROS z = habitat y): "
                  f"landed at {float(p[1]):+.3f}, {d:.3f} m from target", flush=True)
            return p
        if best is None or d < best[0]:
            best = (d, float(p[1]))
    raise SystemExit(
        f"[feed] FEED_SPAWN_FLOOR={spawn_floor} — no navigable point within "
        f"{FLOOR_TOL} m of that height in {tries} samples. Closest was {best[1]:+.3f} "
        f"({best[0]:.2f} m away). Refusing to spawn on a different floor: the run would map "
        f"the wrong storey and every artefact would agree with itself about it."
    )


def derive_floor_tolerance(scene_floors, configured):
    """Half the smallest gap between adjacent storeys, never more than the configured value.

    Owner ruling 19, 2026-09-01. The fixed 0.5 m default is LARGER THAN HALF the smallest gap on
    hm3d_00861: its floors are [-1.59, 0.43, 1.35, 2.21], so the tightest pair is 0.86 m apart and
    half of that is 0.43. At 0.5 m a point 0.5 m above floor 1.35 is inside the band of 1.35 AND
    within 0.36 m of 2.21 — the bands overlap, and which floor a point belongs to depends on which
    comparison ran first rather than on where it is.

    Derived from the scene rather than configured per scene, because a hand-set tolerance is a
    number someone has to remember to change when the scene changes, and nothing would tell them.
    The config value remains an upper bound so a scene can ask for tighter, never looser.
    """
    if len(scene_floors) < 2:
        return configured
    gaps = [b - a for a, b in zip(sorted(scene_floors), sorted(scene_floors)[1:])]
    return min(configured, min(gaps) / 2.0)


class FloorGuard:
    """Keeps a tour on its storey, by teleport when it drifts off.

    Owner ruling 20, 2026-09-01, in their words: "DEFINITELY FIX THIS, even if it requires a
    teleport (this should be a tunable config parameter)."

    WHY IT IS NEEDED DESPITE SINGLE_FLOOR. Tour goals are already filtered to the starting floor,
    but the WALK BETWEEN THEM is not: `agent.act("move_forward")` follows the navmesh, and HM3D
    navmeshes join storeys through the stairs into one island. So the agent walks upstairs on its
    way to a goal that is on its own floor. Measured on the -1.59 run: 251 nodes on -1.59, 99 on
    +1.35 and 18 on +0.43 — 32% of the map on storeys it was never meant to visit, and a 2D
    occupancy grid cannot represent them separately, so the map is wrong rather than merely mixed.

    TUNABLE, as ruled. `habitat.floor_confinement`:
        teleport  snap back onto the floor when drift exceeds the tolerance (default)
        warn      report the drift and let it continue — for diagnosing, not for mapping
        off       today's behaviour, kept so the defect can be reproduced deliberately

    The teleport target is the last on-floor position, snapped to the navmesh. Returning to where
    it was is safer than jumping to a fresh sample: a fresh sample teleports across the building
    and the pose jump is large enough to matter to a mapper that trusts odometry.
    """

    def __init__(self, floor_y, tol, mode="teleport"):
        self.floor_y, self.tol, self.mode = float(floor_y), float(tol), mode
        self.last_on_floor = None
        self.corrections = 0
        self.max_drift = 0.0

    def check(self, agent, pathfinder=None):
        """Call once per frame, after the motion. Returns True if it corrected."""
        if self.mode == "off":
            return False
        st = agent.get_state()
        y = float(st.position[1])
        drift = abs(y - self.floor_y)
        if drift <= self.tol:
            self.last_on_floor = np.array(st.position, dtype=np.float32)
            return False
        self.max_drift = max(self.max_drift, drift)
        if self.mode == "warn":
            print(f"[feed] FLOOR DRIFT {drift:.2f} m off {self.floor_y:+.2f} (warn only)", flush=True)
            return False
        if self.last_on_floor is None:
            # Nothing to go back to: drifted before ever being on the floor. Say so rather than
            # teleporting to a guess.
            print(f"[feed] FLOOR DRIFT {drift:.2f} m and no recorded on-floor position — "
                  "cannot correct", flush=True)
            return False
        target = self.last_on_floor
        if pathfinder is not None and pathfinder.is_loaded:
            snapped = np.asarray(pathfinder.snap_point(target), dtype=np.float32)
            if np.all(np.isfinite(snapped)):
                target = snapped
        st.position = np.asarray(target, dtype=np.float32)
        agent.set_state(st)
        self.corrections += 1
        print(f"[feed] floor guard: drifted {drift:.2f} m off {self.floor_y:+.2f}, "
              f"teleported back (correction {self.corrections})", flush=True)
        return True

    def report(self):
        return {"floor_y": round(self.floor_y, 3), "tolerance_m": round(self.tol, 3),
                "mode": self.mode, "corrections": self.corrections,
                "max_drift_m": round(self.max_drift, 3)}

def topdown_map_payload(sim, floor_y, mpp=0.05):
    """Static top-down navigability render for the viewer minimap background:
    {image: dataURI, bounds_min, bounds_max} in ROS ground coords (x at [0], y at [2],
    image rows increasing with ROS y) — the shape viewer.html already renders."""
    try:
        import base64

        import cv2
        if not sim.pathfinder.is_loaded:
            return None
        nav = sim.pathfinder.get_topdown_view(mpp, floor_y)   # [row=hab z, col=hab x]
        img = np.flip(nav.T, axis=(0, 1))                     # [row=ros y asc, col=ros x asc]
        bgra = np.zeros((*img.shape, 4), dtype=np.uint8)
        bgra[img] = (184, 163, 148, 110)                      # translucent slate
        ok, png = cv2.imencode(".png", bgra)
        if not ok:
            return None
        bmin, bmax = sim.pathfinder.get_bounds()
        return {
            "image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
            "bounds_min": [-float(bmax[2]), float(floor_y), -float(bmax[0])],
            "bounds_max": [-float(bmin[2]), float(floor_y), -float(bmin[0])],
        }
    except Exception as exc:
        print(f"[feed] topdown map skipped: {exc}")
        return None


# --- motion ---
class Tour:
    """Greedy nearest-unvisited coverage over navmesh samples, 360° scan at each.

    With config habitat.single_floor, goals are confined to the floor the agent starts
    on (|y - y0| < habitat.floor_tolerance_m). HM3D navmeshes join storeys through the
    stairs into one island, so an unfiltered sample sends the robot upstairs and the 2D
    grid flattens both floors into one plan.
    """
    def __init__(self, sim, rng, n_points=40):
        self.sim = sim
        self.follower = sim.make_greedy_follower(0, goal_radius=0.4)
        self.floor_y = float(sim.get_agent(0).get_state().position[1])
        self.todo = self._sample(n_points)
        self.goal = None
        self.scan_left = 0
        self.origin = None      # GA-219: set on the first step, the centre of the walk disc

    def _sample(self, n, max_tries=50):
        pts = []
        for _ in range(n * max_tries):
            p = np.array(self.sim.pathfinder.get_random_navigable_point())
            if not SINGLE_FLOOR or abs(p[1] - self.floor_y) < FLOOR_TOL:
                pts.append(p)
                if len(pts) == n:
                    break
        return pts

    def _tour_waypoints(self, n):
        """-> n navigable points spread over the storey, farthest-point sampled.

        Deterministic given the pathfinder seed. Starts from the agent's own position so the
        first leg is a real journey rather than a step to somewhere it already stands.
        """
        pool = []
        for _ in range(3000):
            p = np.array(self.sim.pathfinder.get_random_navigable_point())
            if SINGLE_FLOOR and abs(p[1] - self.floor_y) >= FLOOR_TOL:
                continue
            pool.append(p)
            if len(pool) >= 600:
                break
        if not pool:
            return []
        chosen = []
        # Farthest-point sampling: each new waypoint is the pool point furthest from every
        # point already chosen. Uniform random sampling clusters, and a tour of five points
        # that all sit in one corner is the failure this exists to avoid.
        ref = [np.array(self.origin if self.origin is not None else pool[0])]
        while len(chosen) < n and pool:
            best, bestd = None, -1.0
            for q in pool:
                d = min(float(np.linalg.norm(q[[0, 2]] - r[[0, 2]])) for r in ref)
                if d > bestd:
                    bestd, best = d, q
            if best is None:
                break
            chosen.append(best)
            ref.append(best)
            pool = [q for q in pool if not np.allclose(q, best)]
        return chosen

    def step(self, agent):
        if TEST_MODE and TEST_TOUR > 0:
            # GA-256. Tour mode takes precedence over the radius disc: a bounded walk and a
            # tour are different intentions, and silently blending them would produce a run
            # that is neither.
            if self.origin is None:
                self.origin = np.array(agent.get_state().position)
            if not hasattr(self, "_tour") or self._tour is None:
                self._tour = self._tour_waypoints(TEST_TOUR)
                self._tour_i = 0
                self._tour_scan = 0
                self._dwelling = False
                self._dwell_frames = 0
                print(f"[feed] TEST TOUR: {len(self._tour)} waypoints, "
                      f"scan {TEST_TOUR_SCAN} frames on arrival", flush=True)
                for k, w in enumerate(self._tour):
                    print(f"[feed]   waypoint {k}: "
                          f"({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})", flush=True)
            if self._tour_scan > 0:
                self._tour_scan -= 1
                agent.act("turn_left")
                return
            if self._dwelling:
                # GA-258. Past the minimum, keep turning while merges are pending.
                self._dwell_frames += 1
                pend, sweep = _pending_merges()
                if self._dwell_frames >= TOUR_DWELL_MAX:
                    print(f"[feed] waypoint {self._tour_i}: dwell capped at "
                          f"{self._dwell_frames} frames with {pend} still pending", flush=True)
                elif pend is None or pend > 0:
                    # Unknown counts as "keep looking": the cap bounds it either way, and
                    # leaving early on a missing file is the failure that costs the merge.
                    if self._dwell_frames % 10 == 0:
                        print(f"[feed] waypoint {self._tour_i}: dwelling, "
                              f"{pend if pend is not None else '?'} merges pending "
                              f"(sweep {sweep}, frame {self._dwell_frames})", flush=True)
                    agent.act("turn_left")
                    return
                else:
                    print(f"[feed] waypoint {self._tour_i}: nothing pending after "
                          f"{self._dwell_frames} frames — moving on", flush=True)
                self._dwelling = False
                self._dwell_frames = 0
                self._tour_i += 1
                return
            if self._tour_i >= len(self._tour):
                # Tour complete. Keep turning rather than stopping: a still camera is
                # indistinguishable from a crashed feed downstream.
                agent.act("turn_left")
                return
            goal = self._tour[self._tour_i]
            try:
                action = self.follower.next_action_along(goal)
            except Exception:
                action = None
            if action is None:
                print(f"[feed] TEST TOUR: reached waypoint {self._tour_i}", flush=True)
                self._tour_scan = max(TEST_TOUR_SCAN, TOUR_DWELL_MIN) if TOUR_DWELL_DYNAMIC \
                    else TEST_TOUR_SCAN
                # The minimum runs first as a plain countdown; the dynamic part takes over
                # after it, so there is always at least one full look before asking whether
                # anything is pending -- the answer is meaningless before the sweep that
                # follows the first look.
                self._dwelling = bool(TOUR_DWELL_DYNAMIC)
                self._dwell_frames = 0
                return
            self.sim.step(action)
            return
        if TEST_MODE and TEST_WALK_RADIUS <= 0:
            # GA-212: turn, and only turn. No goal, no follower, no path that can fail --
            # the point of test mode is that a stalled run cannot be blamed on navigation.
            agent.act("turn_left")
            return
        if TEST_MODE:
            # GA-219: walk, but never further than TEST_WALK_RADIUS from where we started.
            # Goals are re-sampled until one lands inside the disc, so the agent explores a
            # room without the tour's freedom to find a staircase.
            if self.origin is None:
                self.origin = np.array(agent.get_state().position)
            if self.goal is None:
                for _ in range(200):
                    p = np.array(self.sim.pathfinder.get_random_navigable_point())
                    if SINGLE_FLOOR and abs(p[1] - self.floor_y) >= FLOOR_TOL:
                        continue
                    if np.linalg.norm(p[[0, 2]] - self.origin[[0, 2]]) <= TEST_WALK_RADIUS:
                        self.goal = p
                        break
                else:
                    # No reachable goal inside the disc: turn rather than widen it. Widening
                    # silently would defeat the guarantee this mode exists to give.
                    agent.act("turn_left")
                    return
            try:
                action = self.follower.next_action_along(self.goal)
            except Exception:
                action = None
            if action is None:
                self.goal = None
                agent.act("turn_left")   # look around from the new spot, then pick another
                return
            self.sim.step(action)
            return
        if self.scan_left > 0:
            agent.act("turn_left")
            self.scan_left -= 1
            return
        if self.goal is None:
            if not self.todo:
                self.todo = self._sample(20)
            here = np.array(agent.get_state().position)
            i = int(np.argmin([np.linalg.norm(p - here) for p in self.todo]))
            self.goal = self.todo.pop(i)
        try:
            action = self.follower.next_action_along(self.goal)
        except Exception:
            action = None
        if action is None:          # arrived (or unreachable): look around, next goal
            self.goal = None
            self.scan_left = 36     # 36 x 10° = full turn
            return
        self.sim.step(action)


def main():
    global SHOW
    sim = make_sim()
    rng = np.random.default_rng(SEED)
    agent = sim.initialize_agent(0)

    have_nav = ensure_navmesh(sim)
    state = habitat_sim.AgentState()
    if have_nav:
        sim.pathfinder.seed(SEED)
        if TEST_MODE and TEST_SPAWN:
            try:
                _p = np.array([float(v) for v in TEST_SPAWN.split(",")], dtype=np.float32)
                if _p.shape != (3,):
                    raise ValueError("need exactly three comma-separated values")
                # Snapped to the navmesh: an explicit point a few centimetres off it is not
                # navigable and the agent would be unable to turn. Snapping is stated rather
                # than silent, because a large correction means the pinned point is stale --
                # a different scene, or a navmesh rebuilt since it was recorded.
                _snap = np.array(sim.pathfinder.snap_point(_p), dtype=np.float32)
                _d = float(np.linalg.norm(_snap - _p))
                state.position = _snap
                print(f"[feed] TEST MODE: PINNED spawn {TEST_SPAWN}"
                      + (f" (snapped {_d:.3f} m to the navmesh)" if _d > 1e-3 else "")
                      + "; the agent will TURN IN PLACE and never navigate")
                if _d > 0.5:
                    print(f"[feed] !! the pinned point moved {_d:.2f} m when snapped — it may "
                          f"belong to a different scene or an older navmesh")
            except (ValueError, TypeError) as exc:
                print(f"[feed] !! FEED_TEST_SPAWN={TEST_SPAWN!r} is not 'x,y,z' ({exc}); "
                      f"falling back to the density search")
                state.position, _dense_n = _dense_spawn_point(sim, SPAWN_FLOOR, TEST_RADIUS_M)
        elif TEST_MODE:
            state.position, _dense_n = _dense_spawn_point(sim, SPAWN_FLOOR, TEST_RADIUS_M)
        else:
            state.position = _spawn_point(sim, SPAWN_FLOOR)
    else:
        bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
        c = (np.array(bb.min) + np.array(bb.max)) / 2
        state.position = np.array([c[0], float(bb.min[1]) + 0.1, c[2]], dtype=np.float32)
    agent.set_state(state)

    cached_navmesh_pts = []
    try:
        if sim.pathfinder.is_loaded:
            for _ in range(300):
                p = sim.pathfinder.get_random_navigable_point()
                rp, _ = habitat_pose_to_ros(p, [0, 0, 0, 1])
                cached_navmesh_pts.append([float(rp[0]), float(rp[1]), float(rp[2])])
    except Exception as exc:
        print(f"[feed] navmesh sampling skipped: {exc}")

    tour = Tour(sim, rng) if have_nav else None
    poller = None
    if SHOW and OVERLAY:
        poller = BeliefPoller()
        poller.start()

    floor_ref = tour.floor_y if tour else float(agent.get_state().position[1])

    # GA-93. One map for a payload that advertised TWO floors, and the two numbers it advertised
    # were `[-2.5, 0.5]` TYPED INTO THE SOURCE — shipped for every scene whatever the scene held.
    #
    # Worse than invented, and the dashboard lane measured it: in run A the agent reported ROS
    # z=1.20 while the advertised floors were [-2.5, 0.5]. Nearest-to-1.20 is always 0.5, so an
    # auto-follow could never select anything else, and a user picking "Ground Floor (-2.5m)" was
    # choosing a height 3.7 m below the robot. The literals were not in the same coordinate frame
    # as the position they were compared against.
    #
    # Derived instead, from the navmesh this file already samples. pt[2] IS THE HEIGHT: these came
    # through habitat_pose_to_ros, so they are ROS coordinates, and the same index 2 is what
    # ros_agent_pos supplies below. Mixing the Habitat Y convention in here would compare a height
    # against a horizontal coordinate and pick the wrong floor in silence.
    #
    # Ported from the remedy tree (Phase 0: /DATA/GRAPH-API is a read-only remedy source), floor
    # derivation and maps dict only. Its on-disk cache under /tmp/graphapi_maps is DELIBERATELY
    # NOT taken: it writes outside the bundle, which is the defect just removed from STATS_DIR,
    # and its read and write paths are both `except Exception: pass`.
    # A CLUSTER OF HEIGHTS IS NOT A STOREY. GA-93 as first written called every height cluster a
    # floor, and on hm3d_00861 that invented two: measured over 20000 navigable samples,
    #     -1.59  46.8%  13.2 x 10.2 m  fully connected   <- a storey
    #     +0.43   0.7%   0.59 x 1.37 m                   <- A STAIR LANDING
    #     +1.35  45.9%  13.5 x  9.3 m  fully connected   <- a storey
    #     +2.21   3.3%  11.5 x  6.9 m  169/400 connected <- a fragmented gallery
    # Spawn control then faithfully sent runs to map the landing and the gallery, the publish gate
    # accepted the results because they were single-storey and opened, and floor_+0.43 — 1.04 x
    # 0.42 m of map — was reported as the end-to-end proof. It was: the tour mapped the whole
    # landing. There was nothing else there.
    #
    # THE THRESHOLD IS UNMEASURED. The two storeys sit at 46.8% and 45.9% and the two non-floors
    # at 0.7% and 3.3%, so anything between 4% and 45% separates them ON THIS SCENE — which is
    # exactly the gap that looks decisive on one example and is not a rule. 10% is a starting
    # point, and the measured shares are published beside it so the next scene either confirms it
    # or shows where it breaks. REPLACE FROM A SECOND MULTI-STOREY SCENE.
    #
    # Rejected clusters are reported as LEVELS, not dropped. A stair landing existing is true and
    # worth knowing; calling it a floor is what was wrong.
    min_share = float(hab_cfg.get("min_floor_share", 0.10))
    _heights = sorted(pt[2] for pt in cached_navmesh_pts)
    _clusters = []
    for sh in _heights:
        if _clusters and abs(sh - _clusters[-1][0]) < 0.8:
            _clusters[-1][1].append(sh)
        else:
            _clusters.append((round(sh, 2), [sh]))
    scene_floors, scene_levels = [], []
    for _first, members in _clusters:
        # THE MEDIAN, not the first member. _heights is sorted ascending, so the first member of a
        # cluster is its LOWEST point and using it biases every floor height down by roughly the
        # cluster's own spread. Real navmesh samples sit tight on the floor so it has not shown in
        # practice, but this number targets the spawn and names the published directory, and a
        # systematically low floor height is the kind of error that stays invisible until it puts
        # a run on the wrong storey. Found by running the clustering against a synthetic
        # distribution built from the measured hm3d shares, where it read -1.77 for -1.59.
        centre = round(members[len(members) // 2], 2)
        share = len(members) / max(len(_heights), 1)
        (scene_floors if share >= min_share else scene_levels).append(
            {"z": centre, "share": round(share, 4), "samples": len(members)})
    if not scene_floors:
        # Every cluster below the threshold. Do not silently produce no floors: take the largest
        # and say that the scene did not meet the bar, so the next reader knows which happened.
        if scene_levels:
            best = max(scene_levels, key=lambda c: c["share"])
            print(f"[feed] NO cluster reached min_floor_share {min_share}; using the largest "
                  f"({best['z']:+.2f}, {100*best['share']:.1f}%) and reporting the rest as levels",
                  flush=True)
            scene_floors = [best]
            scene_levels = [c for c in scene_levels if c is not best]
        else:
            scene_floors = [{"z": round(floor_ref, 2), "share": 1.0, "samples": 0}]
    if scene_levels:
        print("[feed] NOT floors (below min_floor_share "
              f"{min_share}): " + ", ".join(f"{c['z']:+.2f} at {100*c['share']:.1f}%"
                                            for c in scene_levels), flush=True)
    scene_floor_detail = {"floors": scene_floors, "levels": scene_levels,
                          "min_floor_share": min_share,
                          "min_floor_share_note": "UNMEASURED. See the comment above."}
    scene_floors = [c["z"] for c in scene_floors]

    # OWNER RULING 19: the tolerance comes from the scene, capped by the config. The fixed 0.5 m
    # is larger than half hm3d_00861's smallest gap (0.86 m), so two floor bands overlapped and a
    # point's floor depended on comparison order rather than on where it is.
    floor_tol = derive_floor_tolerance(scene_floors, FLOOR_TOL)

    # The tour's goals were filtered with the COARSE tolerance, because Tour is built before the
    # floors are known: the navmesh samples that derive them are collected after the seeded spawn,
    # and moving that would consume RNG draws and change every run's spawn point. Re-filter rather
    # than reorder — same goals, tighter test, reproducibility untouched.
    if tour is not None and SINGLE_FLOOR:
        _before = len(tour.todo)
        tour.todo = [g for g in tour.todo if abs(float(g[1]) - tour.floor_y) < floor_tol]
        if len(tour.todo) != _before:
            print(f"[feed] tour goals re-filtered at the derived tolerance {floor_tol:.2f} m: "
                  f"{_before} -> {len(tour.todo)}", flush=True)

    # OWNER RULING 20: keep the tour on its storey. Anchored to the nearest DERIVED floor rather
    # than to wherever the spawn landed, so it holds the storey and not the spawn point.
    _anchor = min(scene_floors, key=lambda f: abs(f - floor_ref)) if scene_floors else floor_ref
    floor_guard = FloorGuard(_anchor, floor_tol, str(hab_cfg.get("floor_confinement", "teleport")))
    print(f"[feed] floor guard: anchor {_anchor:+.2f}, tolerance {floor_tol:.2f} m, "
          f"mode {floor_guard.mode}", flush=True)

    topdown_maps = {}
    if have_nav:
        for f in scene_floors:
            m = topdown_map_payload(sim, f)
            if m:
                topdown_maps[str(f)] = m
    print(f"[feed] floors derived from navmesh: {scene_floors} ({len(topdown_maps)} maps rendered)",
          flush=True)
    start_ctrl_server()
    nav_target = None
    manual_scan = 0

    def exec_manual(act, p):
        nonlocal nav_target, manual_scan
        amount = float(p.get("amount", 0) or 0)

        def act_once(name, default):
            # exact amounts (viewer step sizes) without re-registering the action space
            spec = agent.agent_config.action_space[name].actuation
            old = spec.amount
            spec.amount = amount if amount > 0 else default
            try:
                agent.act(name)
            finally:
                spec.amount = old

        if act == "forward":
            act_once("move_forward", 0.15)
        elif act == "backward":
            act_once("move_backward", 0.15)
        elif act == "left":
            act_once("turn_left", 10.0)
        elif act == "right":
            act_once("turn_right", 10.0)
        elif act == "scan":
            manual_scan = 36            # one 10° turn per frame, like the tour scan
        elif act in ("teleport", "nav_goal"):
            target = ros_to_habitat((p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)))
            if sim.pathfinder.is_loaded:
                target = np.array(sim.pathfinder.snap_point(target))
            if act == "teleport":
                st = agent.get_state()
                st.position = np.asarray(target, dtype=np.float32)
                agent.set_state(st)
                nav_target = None
            else:
                nav_target = np.asarray(target)
        else:
            print(f"[feed] unknown action: {act}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(1)
    print(f"[feed] scene={SCENE} listening on :{PORT}, waiting for the ROS side...")
    conn, addr = srv.accept()
    conn.settimeout(SEND_TIMEOUT)  # a hard-killed container must not hang sendall forever
    print(f"[feed] connected: {addr}")

    period = 1.0 / FPS
    walk_frames = int(os.environ.get("FEED_WALK", hab_cfg.get("walk_frames", 6)))
    dwell_frames = int(os.environ.get("FEED_DWELL", hab_cfg.get("dwell_frames", 0)))
    # dwell_frames 0 is the DEFAULT as of 2026-08-31: the agent never stops. `phase % (walk +
    # dwell)` then becomes `% walk_frames`, `moving` is always true, and frames publish on every
    # tick. That is intended.
    #
    # What is NOT survivable is walk_frames + dwell_frames == 0 — a modulo by zero that raises
    # 150+ seconds into the run, after the scene has loaded and the ROS side has connected.
    # dwell used to be 60 and hid this; with dwell at 0 a single FEED_WALK=0 reaches it. Fail
    # here, before the socket, and say which name to change.
    if walk_frames + dwell_frames <= 0:
        raise SystemExit(
            f"[feed] FEED_WALK={walk_frames} and FEED_DWELL={dwell_frames} sum to "
            f"{walk_frames + dwell_frames}; the phase cycle divides by that sum. "
            "Set FEED_WALK to a positive number of frames."
        )

    # RESOLVED values, printed AFTER the environment has beaten the config file. Asked for by the
    # experiment lane, 2026-08-31, and the reason is measured: run 19 (20260831_151134) finished
    # cleanly and its dwell is in NONE of the four places it could be — no feed block in
    # run_metadata.json, no feed_stats.json, and config.yaml records the intent rather than the
    # effect (bundle 20260831_033330 has config.yaml mapping_seconds 0.0 against a feed_stats.json
    # and a log that both say 150). live_run.sh now stamps these into the bundle, but live_run.sh
    # is not the only way this file is started, and a run started any other way was exactly how
    # run 19 became unrecoverable. This line costs nothing and fails closed.
    print(f"[feed] resolved: walk={walk_frames} dwell={dwell_frames} fps={FPS} "
          f"mapping_seconds={MAPPING_SECONDS} seed={SEED} scene={SCENE}", flush=True)
    phase = 0
    t_start = time.time()
    mapping_announced = False

    ag_state = agent.get_state()
    t_start_sim = t_start
    total_steps = 0
    frame_seq = 0          # GA-121: ordinal of each published frame; see the frame dict
    total_distance_m = 0.0
    last_pos = np.asarray(ag_state.position, dtype=np.float64)

    while True:
        t0 = time.time()
        mapping = MAPPING_SECONDS > 0 and (t0 - t_start_sim) < MAPPING_SECONDS
        if mapping and not mapping_announced:
            print(f"[feed] MAPPING phase for {MAPPING_SECONDS:.0f}s (continuous coverage tour)")
            mapping_announced = True
        if not mapping and mapping_announced:
            print("[feed] DETECTION phase (walk/dwell)")
            mapping_announced = False

        while CTRL.actions:
            queued_act, queued_params = CTRL.actions.popleft()
            exec_manual(queued_act, queued_params)

        if not CTRL.auto_mode:
            label = "MANUAL"
            if manual_scan > 0:
                agent.act("turn_left")
                manual_scan -= 1
            elif nav_target is not None and tour is not None:
                try:
                    action = tour.follower.next_action_along(nav_target)
                except Exception:
                    action = None
                if action is None:
                    print("[feed] nav_goal reached")
                    nav_target = None
                else:
                    sim.step(action)
        elif mapping:
            label = "MAPPING"
            if tour is not None:
                tour.step(agent)
            else:
                agent.act("move_forward")
        else:
            phase = (phase + 1) % (walk_frames + dwell_frames)
            moving = phase < walk_frames
            label = "WALK" if moving else "DWELL"
            if moving:
                if tour is not None:
                    tour.step(agent)
                else:
                    collided = agent.act("move_forward")
                    if collided:
                        turn = "turn_left" if rng.random() < 0.5 else "turn_right"
                        for _ in range(int(rng.integers(9, 18))):
                            agent.act(turn)

        # OWNER RULING 20. Checked AFTER the motion and BEFORE the observation, so a frame is
        # never rendered from a pose the guard is about to reject — a corrected teleport would
        # otherwise put one off-floor frame into the map for every drift, which is the defect in
        # miniature.
        if floor_guard.check(agent, sim.pathfinder if have_nav else None):
            ag_state = agent.get_state()

        obs = sim.get_sensor_observations()
        ag_state = agent.get_state()
        curr_pos = np.asarray(ag_state.position, dtype=np.float64)
        step_dist = float(np.linalg.norm(curr_pos - last_pos))
        if step_dist > 0.001:
            total_distance_m += step_dist
        total_steps += 1
        last_pos = curr_pos

        # Export navigation stats & BEV data payload for dashboard consumption
        try:
            ros_agent_pos, ros_agent_quat = habitat_pose_to_ros(ag_state.position, [ag_state.rotation.x, ag_state.rotation.y, ag_state.rotation.z, ag_state.rotation.w])
            yaw = math.atan2(2.0 * (ros_agent_quat[3]*ros_agent_quat[2] + ros_agent_quat[0]*ros_agent_quat[1]), 1.0 - 2.0 * (ros_agent_quat[1]**2 + ros_agent_quat[2]**2))

            phase_rem = (walk_frames - (phase % (walk_frames + dwell_frames))) if label == "WALK" else (dwell_frames - (phase % (walk_frames + dwell_frames) - walk_frames)) if label == "DWELL" else 0
            feed_stats = {
                "elapsed_sec": round(time.time() - t_start_sim, 1),
                "total_steps": total_steps,
                "total_distance_m": round(total_distance_m, 2),
                "phase": label,
                "phase_walk_frames": walk_frames,
                "phase_dwell_frames": dwell_frames,
                "phase_remaining_frames": max(0, phase_rem),
                "mapping_seconds": MAPPING_SECONDS,
                # What the guard did. A run whose map is one storey because nothing drifted and
                # a run whose map is one storey because it was teleported back forty times are
                # different runs, and the map alone cannot tell them apart.
                "floor_guard": floor_guard.report(),
                "last_updated": time.time()
            }

            # Nearest floor to the agent's CURRENT height, recomputed per frame so the map
            # follows the robot up a staircase.
            #
            # ponytail: a legacy cache in the remedy tree stores the literal key "auto", and
            # float("auto") raised HERE every frame — taking the whole stats/BEV export down with
            # it, so feed_stats.json and bev_data.json were never written at all. This tree does
            # not read that cache, but the keys are strings and the guard costs one branch.
            current_z = float(ros_agent_pos[2])

            def _floor_dist(k):
                try:
                    return abs(float(k) - current_z)
                except (TypeError, ValueError):
                    return float("inf")

            active_map = topdown_maps.get(min(topdown_maps, key=_floor_dist)) if topdown_maps else None

            bev_payload = {
                "agent": {
                    "x": float(ros_agent_pos[0]),
                    "y": float(ros_agent_pos[1]),
                    "z": float(ros_agent_pos[2]),
                    "yaw": float(yaw)
                },
                "floors": scene_floors,
                # The clusters that did NOT qualify as storeys, with their measured shares. In the
                # artefact rather than only in a log, because "this scene has two floors" is a
                # claim a reader needs to be able to check — and because the two that were
                # wrongly called floors on hm3d_00861 produced two published maps before anyone
                # noticed. A rejected level is a fact about the scene, not a silence.
                "floor_detail": scene_floor_detail,
                "maps": topdown_maps,
                "navmesh": cached_navmesh_pts,
                # "map" keeps its meaning and its readers: the single map for the floor the
                # agent is on. "maps" is ADDED beside it, never in place of it.
                "map": active_map,
                "stats": feed_stats,
                "auto_mode": CTRL.auto_mode,
                "config": CTRL.config,
            }
            CTRL.bev = bev_payload

            # GA-60. These two were written to THREE hardcoded literals with no break, so a
            # run's per-run measurements landed in every one that existed — including /tmp,
            # outside any bundle — and OUT_DIR was ignored entirely. The launcher gives each
            # run its own scratch precisely so two runs cannot overwrite each other, and the
            # gate's a5 probe checks the scratch the launcher names: artefacts written anywhere
            # else are invisible to it. One directory, chosen by the launcher, and stop.
            with open(STATS_DIR / "feed_stats.json", "w") as f:
                json.dump(feed_stats, f)
            with open(STATS_DIR / "bev_data.json", "w") as f:
                json.dump(bev_payload, f)
        except Exception as exc:
            # GA-61, rule 14. This caught 45 lines -- the pose conversion, the stats and BEV
            # payloads, and the writes -- and turned any failure into one log line while the
            # run carried on. The per-frame viewpoint series and the traversal statistics are
            # produced here, so a muted failure yields a bundle that looks complete and is
            # missing exactly the inputs the detectable denominator needs.
            #
            # The owner's default remedy for rule 14 is to remove the handler and let it crash.
            # Re-raising achieves that -- the run stops -- and keeps the diagnostic, which bare
            # removal would lose. The behaviour is identical to removal; only the message
            # survives. It is annotation, not handling: nothing below this line continues.
            print(f"[feed] export error: {exc}")
            raise

        cam = ag_state.sensor_states["color_sensor"]
        cam_quat = np.array([cam.rotation.x, cam.rotation.y, cam.rotation.z, cam.rotation.w], dtype=np.float64)
        frame = {
            "rgb": np.ascontiguousarray(obs["color_sensor"][..., :3], dtype=np.uint8),
            "depth": np.ascontiguousarray(obs["depth_sensor"], dtype=np.float32),
            "cam_pos": np.asarray(cam.position, dtype=np.float64),
            "cam_quat": cam_quat,
            "base_pos": np.asarray(ag_state.position, dtype=np.float64),
            "base_quat": np.array([ag_state.rotation.x, ag_state.rotation.y, ag_state.rotation.z, ag_state.rotation.w], dtype=np.float64),
            "t": time.time(),
            # GA-121. THE IDENTITY OF THIS FRAME, so a detection can say which frame it came from.
            #
            # Nothing carried one before: "t" is a wall-clock float, which is a timestamp rather
            # than an identity — two consumers rounding it differently disagree about whether two
            # detections are co-visible, and a replayed bundle has no stable key at all.
            #
            # WHY IT IS WORTH A FIELD. Co-visibility is the only channel that separates "a
            # fragment of X" from "an object resting on X", and containment provably cannot:
            # perception's replay had the new design merging PILLOWS INTO THE BED they rest on at
            # +7.03 log-odds, containment 0.850. Two objects seen in the same frame at the same
            # time are two objects. That inference costs one integer per frame and nothing else.
            #
            # A DEDICATED COUNTER, not total_steps. They are equal today — one sim step per
            # published frame — but they answer different questions, and a counter reused for a
            # second meaning is one refactor away from being wrong about both. This one means
            # exactly "the ordinal of this published frame" and is incremented nowhere else.
            "frame_id": frame_seq,
            # GA-102. The viewer's layer toggles, delivered to the node that RASTERISES the
            # overlay. The viewer already sends /set_config?perm=..&temp=..&seg=..&det=.., the
            # bridge already forwards it, and this control server already stores it — but the
            # frames the viewer displays come from /image_with_bb, which perception_2 draws
            # BEFORE any flag is consulted. By the time a frame reaches the bridge the boxes are
            # burned in, so nothing downstream can filter them. The flags have to arrive with the
            # frame, which is the only channel that reaches the drawing code every cycle.
            #
            # Sent as-is, values included, so the consumer sees exactly what the viewer set. An
            # ABSENT key must not be read as "draw everything": that is indistinguishable from a
            # working toggle, and it is the defect class this whole review has been removing.
            "viz_config": dict(CTRL.config),
            # NAMED SO IT CANNOT BE MISTAKEN FOR A PERCEPTION OUTPUT. This is habitat's own
            # instance id per pixel — the answer, not an estimate of it. Nothing on the runtime
            # path may read it: a detector that can see the ground truth is not being measured,
            # it is being told. The key is absent entirely when the sensor is off, so a consumer
            # cannot read a zeros array as "no objects present".
            **({"gt_semantic_instance": np.ascontiguousarray(obs["semantic_sensor"], dtype=np.uint32)}
               if GT_SEMANTIC and "semantic_sensor" in obs else {}),
            "w": W, "h": H, "hfov": HFOV,
        }
        frame_seq += 1

        if _HAVE_CV2[0]:          # latest JPEG for the control server's frame.jpg / feed.mjpg
            try:
                import cv2
                ok, enc = cv2.imencode(".jpg", cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR),
                                       [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    CTRL.latest_jpeg = enc.tobytes()
            except Exception:
                _HAVE_CV2[0] = False

        if SHOW:
            try:
                import cv2
                bgr = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
                if LAYERS["walls"] and poller is not None and poller.walls:
                    # Under the boxes: a wall is context for the objects, not a peer of them.
                    draw_walls(bgr, poller.walls, frame["cam_pos"], cam_quat)
                if LAYERS["boxes"] and poller is not None and poller.objects:
                    draw_belief(bgr, poller.objects, frame["cam_pos"], cam_quat, frame["depth"])
                if LAYERS["hud"]:
                    cv2.putText(bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    # GA-265. The key map ON the window. A keyboard interface nobody can see
                    # is a keyboard interface nobody uses, and "press ? for help" still
                    # requires knowing that ? does anything. Each line shows the key, the
                    # layer, and whether it is currently ON -- so the HUD doubles as the
                    # state readout and there is nothing else to consult.
                    y = 74
                    cv2.putText(bgr, "KEYS", (10, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                                0.42, (150, 150, 150), 1)
                    # GA-267. The COUNT beside each layer, so the HUD says what the toggle
                    # would show. A layer reading ON with nothing behind it is exactly the
                    # state this run was in for an hour, and the HUD looked healthy.
                    _objs = (poller.objects if poller is not None else []) or []
                    _n_total = len(_objs)
                    for key, name in sorted(LAYER_KEYS.items(), key=lambda kv: kv[1]):
                        on = LAYERS[name]
                        col = (120, 255, 140) if on else (110, 110, 130)
                        if name in ("boxes", "labels"):
                            cnt = str(_n_total)
                        elif name == "walls":
                            cnt = (str(len(poller.walls)) if poller is not None
                                   and getattr(poller, "walls_available", False) else "off")
                        elif name == "hud":
                            cnt = ""
                        else:
                            # The belief the bridge serves carries no per-object verdict, so
                            # the grade filters have nothing to count. "n/a" rather than 0:
                            # zero would claim there are none of that grade, which is a
                            # different statement from having no data.
                            cnt = "n/a"
                        cv2.putText(bgr,
                                    f"{chr(key)}  {name:<10s} {'ON' if on else 'off':<3s} {cnt}",
                                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1,
                                    cv2.LINE_AA)
                        y += 18
                    if _n_total == 0:
                        cv2.putText(bgr, f"belief empty - polling {BRIDGE}", (12, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (110, 180, 255), 1,
                                    cv2.LINE_AA)
                        y += 18
                    cv2.putText(bgr, "h  hide this HUD", (12, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110, 110, 130), 1, cv2.LINE_AA)
                    cv2.putText(bgr, "q  close window", (12, y + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (110, 110, 130), 1, cv2.LINE_AA)
                    cv2.putText(bgr, "synced with the dashboard", (12, y + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 255), 1, cv2.LINE_AA)
                cv2.imshow("habitat feed (agent camera)", bgr)
                # GA-265. Keys toggle the SAME dict the dashboard writes, so the window and
                # the browser cannot drift apart. Every toggle republishes, which is how the
                # dashboard learns about a keypress it did not make.
                _k = cv2.waitKey(1) & 0xFF
                if _k == ord("q"):
                    cv2.destroyAllWindows()
                    SHOW = False
                elif _k in LAYER_KEYS:
                    name = LAYER_KEYS[_k]
                    LAYERS[name] = not LAYERS[name]
                    _publish_layers()
                    print(f"[feed] layer {name} -> {LAYERS[name]} (from the window)", flush=True)
                elif _k == ord("?"):
                    print("[feed] layer keys: " + ", ".join(
                        f"{chr(k)}={v}" for k, v in sorted(LAYER_KEYS.items())) +
                        ", q=close window", flush=True)
            except Exception as exc:
                print(f"[feed] cv2 display error (disabling GUI window): {exc}\n{traceback.format_exc()}")
                SHOW = False

        blob = pickle.dumps(frame, protocol=4)
        try:
            conn.sendall(struct.pack("!I", len(blob)) + blob)
        except (BrokenPipeError, ConnectionResetError, socket.error, OSError) as exc:
            print(f"[feed] client disconnected ({exc}), waiting for reconnect...")
            try:
                conn.close()
            except Exception:
                pass
            while True:
                try:
                    conn, addr = srv.accept()
                    conn.settimeout(SEND_TIMEOUT)
                    print(f"[feed] client reconnected: {addr}")
                    break
                except Exception:
                    time.sleep(0.5)

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()
