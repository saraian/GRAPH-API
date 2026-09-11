#!/usr/bin/env python3
"""Host-side habitat feed: renders a scene and streams frames over TCP.

Runs in a plain habitat_sim environment (no ROS). The container-side
habitat_feed_node.py connects, converts, and publishes to ROS topics + TF.
Protocol: length-prefixed pickle dicts {rgb, depth, cam_pos, cam_quat,
base_pos, base_quat, t, w, h, hfov}.

The HTTP control port also accepts runtime rigid-object commands from
habitat_feed_node.py, so run_habitat_script.py works with this headless feed
just as it does with habitat_camera_objects_node.py.

Motion comes from ONE policy: the precomputed exploration schedule named by
FEED_SCHEDULE. The agent drives the storey's Voronoi roadmap, turns a full
circle at each stop, and repeats the same lap FEED_LAPS times. There is no
fallback. A run with no schedule has no motion at all, so live_run.sh builds or
finds the schedule before it starts anything and refuses the run if it cannot.

The sampling policy this replaced — a mapping phase of greedy nearest-unvisited
navmesh samples, then walk/dwell bursts with an adaptive hold — is REMOVED
(owner, 2026-09-11). It moved on 1.0-1.8% of frames and its coverage is not
comparable with a schedule's. Bundles from before that date record
motion_policy "sampled" and must be read as a different experiment.

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
import base64
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
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import habitat_sim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
from box_view import BOX_EDGES, box_corners_map, project_visible  # noqa: E402  (ROS-free)
from config import CFG, CFG_PATH  # noqa: E402
import gt_codec as _gt_codec  # noqa: E402

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
# GA-473 (owner 2026-09-10). CONFIG FIRST, ENVIRONMENT SECOND, like every other feed setting.
# These four were the only ones with no config key, so a run that set everything else in
# config.yaml still needed them on the command line. The precedence is the one the rest of this
# file already uses: the yaml states the intended setting and travels with the bundle; the
# environment variable flips ONE run without editing a file everyone shares.
SHOW = os.environ.get("FEED_SHOW", "1" if hab_cfg.get("show", False) else "0") == "1"
OVERLAY = os.environ.get("FEED_OVERLAY", "1" if hab_cfg.get("overlay", False) else "0") == "1"

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
# FEED_MAPPING_SECONDS / habitat.mapping_seconds is RETIRED with the sampling policy it selected.
# It was never a duration: it chose between the coverage tour and the walk/dwell bursts, and both
# are gone. It REFUSES rather than being ignored -- a retired knob that is silently accepted is a
# knob somebody sets, and then a run does something other than what its config says. Zero and unset
# are the inert values every current config already carries, so only a real request fails here.
_mapping_seconds = os.environ.get("FEED_MAPPING_SECONDS", hab_cfg.get("mapping_seconds", 0.0))
if float(_mapping_seconds or 0.0) > 0:
    raise SystemExit(
        f"[feed] mapping_seconds={_mapping_seconds} but the mapping phase is removed with the "
        "sampling policy (owner 2026-09-11). The schedule drives the whole run. Remove the key "
        "from the config, or clear FEED_MAPPING_SECONDS.")
SEND_TIMEOUT = float(os.environ.get("FEED_SEND_TIMEOUT", "10"))
# GA-120. WHICH FLOOR THIS RUN MAPS. Unset = whatever habitat drops the agent on, which is what
# every run before 2026-08-31 did — one uncontrolled random sample chose the storey of every map
# this project has ever made, and it landed on 1.35 each time. ROS z, which IS habitat y.
SPAWN_FLOOR = float(os.environ["FEED_SPAWN_FLOOR"]) if os.environ.get("FEED_SPAWN_FLOOR") else None
# GA-131. Ground-truth instance ids in the frame, for VALIDATION ONLY. Off by default.
GT_SEMANTIC = os.environ.get("FEED_GT_SEMANTIC", "1" if hab_cfg.get("gt_semantic", False) else "0") == "1"
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
CAMERA_PITCH_DEG = float(os.environ.get("FEED_CAMERA_PITCH_DEG", hab_cfg.get("camera_pitch_deg", 0.0)))
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
        # CameraSensorSpec.orientation uses Habitat's XYZ Euler angles in radians;
        # rotate around X so the same pitch is applied to RGB/depth/semantic views.
        s.orientation = [math.radians(CAMERA_PITCH_DEG), 0.0, 0.0]
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


# GA-102. The bridge's /persistent_perception stamps each object with its admission grade;
# this maps the four-valued grade onto the LAYERS keys the window and the dashboard toggle.
GRADE_LAYER = {"admit": "admitted", "hold": "held", "decline": "declined", "reject": "declined",
               "no_grounds": "nogrounds", "abstain": "nogrounds"}


def grade_layer(obj):
    """The LAYERS key an object's grade falls under, or None when it carries no grade (an
    older bridge, or an object no admission row links to)."""
    return GRADE_LAYER.get(str(obj.get("grade") or "").lower())


def visible_belief(belief, layers=None):
    """The objects the grade toggles leave on. An ungraded object is never filtered: hiding
    it would claim a grade nobody recorded."""
    layers = LAYERS if layers is None else layers
    return [o for o in belief if not (grade_layer(o) and not layers[grade_layer(o)])]


def grade_counts(belief):
    """Objects per grade layer for the HUD, or None when no object carries a grade at all --
    "n/a" on the HUD rather than 0, which would claim there are none of that grade."""
    if not any("grade" in o for o in belief):
        return None
    counts = {k: 0 for k in ("admitted", "held", "declined", "nogrounds")}
    for o in belief:
        if grade_layer(o):
            counts[grade_layer(o)] += 1
    return counts


def draw_belief(bgr, belief, cam_pos, cam_quat, depth=None, labels=True):
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
        if not labels:
            continue
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
        # HTTP threads enqueue requests, but only the simulator thread is
        # allowed to mutate Habitat's scene graph.
        self.object_commands = collections.deque()
        self.object_catalog = {"templates": []}
        # Published by the sim thread each frame, read by /revisit_status. Swapped whole.
        self.revisit = None


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

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path != "/object_command":
            self._json({"success": False, "error": f"unknown path {url.path}"}, code=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65536:
                raise ValueError("invalid object command body size")
            command = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(command, dict):
                raise ValueError("object command must be a JSON object")
            done = threading.Event()
            item = {"command": command, "done": done, "result": None}
            CTRL.object_commands.append(item)
            if not done.wait(timeout=10.0):
                item["cancelled"] = True
                self._json({
                    "success": False,
                    "action": command.get("action"),
                    "request_id": command.get("request_id"),
                    "message": "timeout waiting for the Habitat simulator thread",
                }, code=504)
                return
            self._json(item["result"])
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._json({"success": False, "message": str(exc)}, code=400)

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
        elif path == "/object_catalog":
            self._json(CTRL.object_catalog)
        elif path == "/auto_mode":
            CTRL.auto_mode = q.get("enabled", "true").lower() in ("1", "true", "yes", "on")
            self._json({"success": True, "auto_mode": CTRL.auto_mode})
        elif path == "/action":
            act = q.get("act") or q.get("action") or ""
            params = {k: float(q[k]) for k in ("x", "y", "z", "amount") if q.get(k)}
            # `resume` stays a string: it is a flag, and float("0") would make "false" raise.
            if q.get("resume") is not None:
                params["resume"] = q["resume"]
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
        elif path == "/revisit_status":
            # /action is fire-and-forget, so without this a caller cannot tell a reach from a
            # refusal. Single attribute read: the sim thread swaps CTRL.revisit whole.
            self._json({"success": True, **(CTRL.revisit or {"active": None, "last": None})})
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
          "(frame.jpg feed.mjpg bev_data logs auto_mode action set_config object_command)")


class DynamicObjectController:
    """Rigid-object operations used by run_habitat_script through the feed node.

    All methods run in the main simulation loop.  The HTTP handler only queues
    JSON commands, avoiding concurrent access to Habitat-Sim's scene graph.
    """

    def __init__(self, sim, agent):
        self.sim = sim
        self.agent = agent
        self.templates = sim.get_object_template_manager()
        self.objects = sim.get_rigid_object_manager()
        self.spawned = set()
        # Keep the ManagedRigidObject wrappers alive.  Some Habitat-Sim builds
        # release the underlying scene object when the last Python wrapper is
        # collected, even though its id was returned by the manager.
        self.spawned_objects = {}
        self.capture_sensor = None
        self._load_templates()
        CTRL.object_catalog = {"templates": self._catalog_names()}
        print(f"[feed] dynamic objects ready: {len(CTRL.object_catalog['templates'])} templates")

    def _load_templates(self):
        # This process runs on the host, not in the ROS container.  The
        # repository-local habitat directory is therefore the useful default;
        # HABITAT_EXAMPLE_OBJECTS_DIR still overrides it for other layouts.
        default_root = (hab_cfg.get("dataset_root") or
                        str(Path(__file__).resolve().parent.parent / "habitat"))
        requested = os.environ.get(
            "HABITAT_EXAMPLE_OBJECTS_DIR",
            os.path.join(default_root, "habitat_objects", "configs"),
        )
        if not os.path.isdir(requested):
            print(f"[feed] object template directory not found: {requested}")
            return
        directories = []
        for root, _dirs, files in os.walk(requested):
            if any(name.endswith(".object_config.json") for name in files):
                directories.append(root)
        for directory in directories:
            try:
                self.templates.load_configs(directory)
            except Exception as exc:
                print(f"[feed] failed loading object templates from {directory}: {exc}")

    def _handles(self):
        return list(self.templates.get_template_handles())

    def _catalog_names(self):
        names = set()
        for handle in self._handles():
            name = os.path.basename(str(handle)).lower()
            for suffix in (".object_config.json", ".json"):
                if name.endswith(suffix):
                    name = name[:-len(suffix)]
                    break
            if name:
                names.add(name)
        return sorted(names)

    def _resolve_template(self, requested):
        handles = self._handles()
        if requested is None or str(requested).strip().lower() in ("", "random"):
            return handles[0] if handles else None
        requested = str(requested).strip()
        if requested in handles:
            return requested
        wanted = os.path.basename(requested).lower()
        for suffix in (".object_config.json", ".json"):
            if wanted.endswith(suffix):
                wanted = wanted[:-len(suffix)]
                break
        for handle in handles:
            stem = os.path.basename(str(handle)).lower()
            for suffix in (".object_config.json", ".json"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break
            if stem == wanted:
                return handle
        return None

    @staticmethod
    def _position(value):
        p = np.asarray(value, dtype=np.float32)
        if p.shape != (3,) or not np.all(np.isfinite(p)) or np.any(np.abs(p) > 100.0):
            raise ValueError("position must contain three finite coordinates within +/-100 m")
        return p

    def _object(self, object_id):
        object_id = int(object_id)
        obj = self.spawned_objects.get(object_id)
        if obj is None:
            obj = self.objects.get_object_by_id(object_id)
        if obj is None:
            raise ValueError(f"object id={object_id} not found")
        return obj

    def _camera_state(self):
        state = self.agent.get_state()
        sensor = state.sensor_states.get("color_sensor")
        return ((np.asarray(sensor.position, dtype=np.float64), sensor.rotation)
                if sensor is not None else
                (np.asarray(state.position, dtype=np.float64), state.rotation))

    @staticmethod
    def _rotmat(q):
        return _quat_to_rot([q.x, q.y, q.z, q.w])

    @staticmethod
    def _bottom_offset(obj):
        try:
            return -float(obj.root_scene_node.cumulative_bb.min.y)
        except Exception:
            return -float(obj.aabb.min.y)

    def _pixel_hit(self, pixel):
        import magnum as mn
        if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
            raise ValueError("pixel must be [u, v]")
        u, v = int(pixel[0]), int(pixel[1])
        if not (0 <= u < W and 0 <= v < H):
            raise ValueError("pixel is outside the camera image")
        fx = (W / 2.0) / math.tan(math.radians(HFOV) / 2.0)
        direction = np.array([(u - W / 2.0) / fx, -(v - H / 2.0) / fx, -1.0])
        direction /= np.linalg.norm(direction)
        camera, rotation = self._camera_state()
        direction = self._rotmat(rotation) @ direction
        hits = self.sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(camera), mn.Vector3(direction)))
        if not hits.has_hits():
            raise ValueError("pixel does not intersect the scene")
        return hits.hits[0]

    def spawn(self, command):
        import magnum as mn
        handle = self._resolve_template(command.get("template"))
        if handle is None:
            raise ValueError(f"template {command.get('template')!r} not found")
        obj = self.objects.add_object_by_template_handle(handle)
        if obj is None:
            raise RuntimeError(f"Habitat could not instantiate template {handle!r}")
        try:
            scale = float(command.get("object_scale", 1.0))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("object_scale must be a positive number")
            if abs(scale - 1.0) > 1e-6:
                obj.root_scene_node.scale(mn.Vector3(scale))
            if "position" in command:
                position = self._position(command["position"])
            else:
                camera, rotation = self._camera_state()
                position = camera + self._rotmat(rotation) @ np.array([0.0, 0.0, -1.5])
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            obj.translation = position
            obj.rotation = mn.Quaternion.rotation(mn.Deg(0.0), mn.Vector3(0.0, 1.0, 0.0))
            obj.awake = True
            object_id = int(obj.object_id)
            self.spawned.add(object_id)
            self.spawned_objects[object_id] = obj
        except Exception:
            self.objects.remove_object_by_id(int(obj.object_id))
            raise
        print(f"[feed] scripted object spawned: id={object_id} handle={handle} pos={position.tolist()}")
        return {"success": True, "action": "spawn", "object_id": object_id,
                "handle": str(handle), "position": [float(v) for v in position],
                "target_category": command.get("target_category"),
                "target_surface_point": command.get("target_surface_point")}

    def move(self, command):
        obj = self._object(command["object_id"])
        mode = "position"
        extra = {}
        if "position" in command:
            position = self._position(command["position"])
        elif "pixel" in command:
            hit = self._pixel_hit(command["pixel"])
            normal = np.asarray([float(hit.normal.x), float(hit.normal.y),
                                 float(hit.normal.z)], dtype=np.float32)
            normal /= max(float(np.linalg.norm(normal)), 1e-6)
            offset = self._bottom_offset(obj) + 0.01 if abs(float(normal[1])) > 0.8 else 0.01
            position = np.asarray(hit.point, dtype=np.float32) + normal * offset
            mode = "pixel"
            extra = {"pixel": [int(v) for v in command["pixel"]],
                     "hit_object_id": int(hit.object_id)}
        else:
            raise ValueError("move requires position or pixel")
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        obj.translation = position
        obj.awake = True
        return {"success": True, "action": "move", "object_id": int(obj.object_id),
                "mode": mode, "position": [float(v) for v in position], **extra}

    def remove(self, command):
        if command.get("all") is True:
            for object_id in list(self.spawned):
                self.objects.remove_object_by_id(object_id)
            count = len(self.spawned)
            self.spawned.clear()
            self.spawned_objects.clear()
            return {"success": True, "action": "remove", "object_id": None,
                    "removed_count": count, "position": None}
        object_id = int(command["object_id"])
        if object_id not in self.spawned:
            raise ValueError(f"object id={object_id} was not spawned by this feed")
        obj = self._object(object_id)
        position = [float(obj.translation.x), float(obj.translation.y), float(obj.translation.z)]
        self.objects.remove_object_by_id(object_id)
        self.spawned.remove(object_id)
        self.spawned_objects.pop(object_id, None)
        return {"success": True, "action": "remove", "object_id": object_id,
                "position": position}

    def _capture_sensor(self):
        if self.capture_sensor is not None:
            return self.capture_sensor
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = "object_capture_sensor"
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.resolution = [H, W]
        spec.position = [0.0, 0.0, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        spec.hfov = 55.0
        self.sim.add_sensor(spec)
        registry = getattr(self.sim, "sensors", None)
        self.capture_sensor = registry[spec.uuid] if registry is not None else None
        if self.capture_sensor is None or not hasattr(self.capture_sensor, "sensor_object"):
            raise RuntimeError("this Habitat build does not expose the auxiliary VisualSensor")
        return self.capture_sensor

    def capture(self, command):
        import magnum as mn
        object_id = int(command["object_id"]) if "object_id" in command else None
        obj = self._object(object_id) if object_id is not None else None
        target = (obj.transformation.transform_point(obj.aabb.center()) if obj is not None
                  else mn.Vector3(self._position(command["position"])))
        if command.get("capture_eye") is not None:
            eye = mn.Vector3(self._position(command["capture_eye"]))
        else:
            current, _rotation = self._camera_state()
            eye = mn.Vector3(current)
        sensor = self._capture_sensor()
        camera = sensor.sensor_object
        node = camera.object() if callable(getattr(camera, "object", None)) else camera.object
        view = mn.Matrix4.look_at(eye, target, mn.Vector3(0.0, 1.0, 0.0))
        try:
            agent_world = self.agent.scene_node.absolute_transformation()
            node.transformation = agent_world.inverted() @ view
            rgb = self.sim.get_sensor_observations().get("object_capture_sensor")
            if rgb is None:
                raise RuntimeError("no observation from object capture sensor")
            rgb = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
        finally:
            camera.set_transformation_from_spec()
        return {"success": True, "action": "capture", "object_id": object_id,
                "rgb_zlib": base64.b64encode(
                    zlib.compress(rgb.tobytes(), level=1)).decode("ascii"),
                "rgb_width": int(rgb.shape[1]), "rgb_height": int(rgb.shape[0])}

    def execute(self, command):
        action = str(command.get("action", "")).lower()
        try:
            handler = {"spawn": self.spawn, "move": self.move,
                       "remove": self.remove, "capture": self.capture}.get(action)
            if handler is None:
                raise ValueError(f"unsupported object action {action!r}")
            result = handler(command)
        except Exception as exc:
            result = {"success": False, "action": action, "message": str(exc)}
            print(f"[feed] object command {action} failed: {exc}")
        if command.get("request_id") is not None:
            result["request_id"] = str(command["request_id"])
        return result



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
# THE RETIRED SAMPLING-POLICY SETTINGS, named so a stale config can be refused by name.
#
# FEED_TEST_WALK_RADIUS confined a free walk to a disc around the spawn; FEED_TEST_TOUR and
# FEED_TEST_TOUR_SCAN chose and scanned farthest-point waypoints; the dwell_* family drove the
# walk/dwell burst cycle. All of them belonged to the coverage tour, which is removed (owner
# 2026-09-11). A schedule states its own stops and its own scan angle per stop, so none of these
# has anything left to control. They are listed rather than deleted so that a config still
# carrying them fails with the key name instead of running with it silently ignored.
_TOUR_RETIRED = ("tour_waypoints", "tour_scan_frames", "dwell_dynamic",
                 "dwell_min_frames", "dwell_max_frames", "walk_frames", "dwell_frames",
                 "dwell_mode", "mapping_seconds")

# GA-434 / RULE 73. THE BASE RUN TOURS THE WHOLE HOUSE: every storey, teleporting to the next one
# when a storey is finished. Owner, 2026-09-10: "no caps this time, we need to perform a full house
# tour, all storeys (if we finish a storey, just teleport to the next storey). This is our base run
# policy from now on."
#
# OFF BY DEFAULT, AND THAT IS NOT A RETREAT FROM THE POLICY. The owner ruled on 2026-09-10 that
# each storey gets its OWN mapping session: ruling 25 stands, so no map may straddle storeys. Under
# that ruling the house is toured by RELAUNCHING the stack once per storey, each launch spawning on
# its storey through FEED_SPAWN_FLOOR, so a mid-run teleport is not how a base run moves between
# floors and a default of ON would be a setting every base run has to remember to switch off.
#
# THE MACHINERY IS GONE WITH THE SAMPLING TOUR (2026-09-11). It lived in Tour, which chose the next
# storey and teleported to it; a schedule is one storey by construction, and a multi-storey schedule
# would need its own stair-crossing legs that no roadmap in schedules/ contains. So the switch now
# REFUSES instead of doing nothing: the base-run shape is unchanged (one launch per storey through
# run_house.sh), and a config asking for the other shape must be told the shape no longer exists.
TOUR_ALL_FLOORS = os.environ.get(
    "FEED_TOUR_ALL_FLOORS",
    "1" if hab_cfg.get("tour_all_floors", False) else "0").lower() in ("1", "true", "yes", "on")
if TOUR_ALL_FLOORS:
    raise SystemExit(
        "[feed] FEED_TOUR_ALL_FLOORS / habitat.tour_all_floors asks for one continuous session "
        "across every storey. That shape belonged to the sampling tour, which is removed "
        "(owner 2026-09-11); a schedule drives one storey. Tour the house with run_house.sh, "
        "which relaunches the stack once per storey and is the base-run policy (rule 73).")

# GA-434 / RULE 73. A NO-CAP RUN NEEDS ITS OWN ENDING, and until now it had none.
#
# The tour turned in place forever when it ran out of waypoints, and a cap script stopped the run
# from outside. Rule 73 removes the cap ("no caps this time"), so with nothing else changed a base
# run would tour the house and then spin until somebody noticed. The feed ends itself instead: it
# keeps feeding for a settle period after the last storey, then writes feed_ended.json into the
# bundle, which live_stack_container.sh watches for and shuts the stack down on.
#
# 90 s IS A CHOSEN NUMBER, NOT A MEASURED ONE. It has to cover the object manager's last merge
# sweeps -- a pair over the evidence threshold still needs merge_min_consecutive sweeps to commit,
# and the dwell logic exists because turning away early is what strands them. Raise it if a run
# ends with pending merges; the count is in the dwell lines.
TOUR_END_SETTLE_S = float(os.environ.get("FEED_TOUR_END_SETTLE_S",
                                         hab_cfg.get("tour_end_settle_s", 90.0)))

# GA-441. THE PRECOMPUTED EXPLORATION SCHEDULE. A schedule is one storey's roadmap and the order to
# walk it, built offline from the navmesh by lost3dsg/test/voronoi_roadmap.py: waypoints on the
# generalized Voronoi diagram -- the line equidistant from two or more walls, which runs down the
# middle of corridors -- visited depth-first from the busiest junction, with a 360 degree scan at
# each first arrival.
#
# WHY PRECOMPUTED AND NOT SAMPLED PER RUN. The old policy drew random navigable points and walked to
# them, and MEASURED on 20260909_004443 it covered 4.8 m in 18.7 minutes: 6 of every 96 frames were
# allowed to move, and 84% of those went into a 36-frame look-around on arrival. A schedule also
# makes two runs of one scene visit the SAME places in the SAME order, which is what makes them
# comparable at all.
#
# IT IS MANDATORY since 2026-09-11: the walk/dwell path it used to be optional against is removed,
# so main() refuses a run that has no schedule rather than publishing a stationary robot.
SCHEDULE_PATH = os.environ.get("FEED_SCHEDULE", hab_cfg.get("schedule", "") or "").strip()
EXPLORATION_LAPS = int(os.environ.get("FEED_EXPLORATION_LAPS",
                                      hab_cfg.get("exploration_laps", 3)))
# GA-466 (owner 2026-09-10). HOW THE AGENT GETS FROM ONE STOP TO THE NEXT, chosen by
# `habitat.navigation_mode`:
#   navigate  drive it, with the goto skill's own arrival test (the follower, an arrival tolerance
#             and a frame cap). This is what a robot does, and it is what produces the frames
#             between two stops -- a corridor is where half the objects are seen.
#   teleport  set the pose directly, snapped to the navmesh. No travel frames, so a lap costs only
#             its scans; use it when the question is what the scans see, not how the agent got there.
# `move_function` is the old name for the same setting and still works.
NAVIGATION_MODE = os.environ.get(
    "FEED_NAVIGATION_MODE",
    hab_cfg.get("navigation_mode", hab_cfg.get("move_function", "navigate")) or "navigate").strip()
MOVE_FN = os.environ.get("FEED_MOVE_FN", NAVIGATION_MODE).strip()
# module:function called after every completed 360 degree scan. The dynamic dataset update belongs
# here: the scan is the moment the world model has just been shown a place, so it is the moment a
# change to that place is worth making.
POST_SCAN_HOOK = os.environ.get("FEED_POST_SCAN_HOOK",
                                hab_cfg.get("post_scan_hook", "") or "").strip()
# REVISIT (`/action?act=goto`). Drive the agent to one point, look around, rejoin the tour.
# The scene is dynamic, so answering "what changed over there" needs a way to go and look;
# this file supplies the mechanism only -- which point, and when, belongs to the caller.
#
# 36 frames is a full turn at the 10 degrees per `turn_left` the manual scan already assumes.
REVISIT_SCAN_FRAMES = int(os.environ.get("FEED_REVISIT_SCAN", hab_cfg.get("revisit_scan_frames", 36)))
# Arrival tolerance, against the follower's own goal_radius of 0.4 m plus a margin: the follower
# stops "close enough", and a target it never approached must not be read as a reach.
REVISIT_ARRIVAL_TOL_M = float(os.environ.get("FEED_REVISIT_ARRIVAL_TOL",
                                             hab_cfg.get("revisit_arrival_tol_m", 1.0)))
# A walk that never finishes has to end by itself. The floor guard teleports the agent back the
# moment it drifts off-storey, so a path crossing a staircase can otherwise loop forever:
# walk, get pulled back, re-plan the same path. Bounded here rather than diagnosed at 3 a.m.
REVISIT_MAX_FRAMES = int(os.environ.get("FEED_REVISIT_MAX_FRAMES",
                                        hab_cfg.get("revisit_max_frames", 600)))

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
# RUN_DIR first: that is the bundle, and the container's /ws/output is bind-mounted onto it.
# GRAPH_API_OUTPUT_DIR is exported INSIDE the container only, so on the host it is empty and
# os.path.join("", "merge_pending.json") yields a bare relative name that never opens --
# measured on 20260902_125130, where every dwell reported "sweep None" and capped out.
_MERGE_PENDING_DIR = (os.environ.get("RUN_DIR")
                      or os.environ.get("GRAPH_API_OUTPUT_DIR")
                      or "")
_MERGE_PENDING_PATH = os.path.join(_MERGE_PENDING_DIR, "merge_pending.json")
if not _MERGE_PENDING_DIR:
    print("[feed] WARNING: neither RUN_DIR nor GRAPH_API_OUTPUT_DIR is set; the end-of-run settle "
          "cannot read pending merges and will report them as unknown", flush=True)
else:
    print(f"[feed] pending merges read from {_MERGE_PENDING_PATH}", flush=True)


def _read_signal():
    """The object manager's merge_pending.json as a dict, or None when absent or unreadable.

    None means UNKNOWN, and the caller must not read it as zero: "nothing is pending" and
    "the file is not there yet" are different states, and treating the second as the first
    would call a settle finished in exactly the runs where the manager is slow to start.
    """
    try:
        with open(_MERGE_PENDING_PATH) as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _pending_merges():
    """-> (pending, sweep) for the end-of-run settle (GA-258), or (None, None)."""
    d = _read_signal()
    try:
        return (int(d.get("pending", 0)), int(d.get("sweep", -1))) if d else (None, None)
    except (ValueError, TypeError):
        return None, None


# EVERY KNOB OF THE SAMPLING POLICY REFUSES, and it refuses HERE, before the scene loads and
# before the socket opens. The adaptive hold (GA-339), the walk/dwell burst cycle and the greedy
# coverage tour are removed (owner 2026-09-11); a config still carrying their keys was written for
# a run this file can no longer perform, and accepting it silently would produce a bundle whose
# config.yaml describes motion that never happened. Named one by one, with the file that holds
# them, so the message says what to delete rather than that something is wrong.
_stale = [k for k in _TOUR_RETIRED if hab_cfg.get(k) is not None]
_stale_env = [e for e in ("FEED_WALK", "FEED_DWELL", "FEED_DWELL_MODE", "FEED_DWELL_MIN",
                          "FEED_DWELL_MAX", "FEED_DWELL_SIGNAL_MAX_AGE_S", "FEED_TEST_TOUR",
                          "FEED_TEST_TOUR_SCAN", "FEED_TEST_DWELL_DYNAMIC", "FEED_TEST_DWELL_MIN",
                          "FEED_TEST_DWELL_MAX", "FEED_TEST_WALK_RADIUS")
              if os.environ.get(e)]
if _stale or _stale_env:
    raise SystemExit(
        "[feed] the sampling policy is removed (owner 2026-09-11); the schedule drives every run.\n"
        + (f"       habitat keys in {CFG_PATH}: {', '.join(_stale)}\n" if _stale else "")
        + (f"       environment: {', '.join(_stale_env)}\n" if _stale_env else "")
        + "       Delete them. FEED_SCHEDULE and FEED_LAPS are what set the motion now.")


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
        self.reanchors = 0

    def reanchor(self, floor_y):
        """Move the guard to a new storey after a DELIBERATE teleport (rule 73).

        LAST_ON_FLOOR IS CLEARED, and that is the whole point of the method. The guard corrects
        drift by teleporting to the last position it saw on its own storey. Left in place across a
        storey change, the first drift on the new storey would send the agent back DOWNSTAIRS, and
        the second storey would never be toured -- a full-house tour that silently tours one floor
        twice. Clearing it costs one uncorrected drift at most: check() says so and does nothing
        until a position on the new storey has been seen.
        """
        self.floor_y = float(floor_y)
        self.last_on_floor = None
        self.reanchors += 1

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
        bmin, bmax = sim.pathfinder.get_bounds()
        ok, png = cv2.imencode(".png", bgra)
        if not ok:
            return None
        return {
            "image": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
            "bounds_min": [-float(bmax[2]), float(floor_y), -float(bmax[0])],
            "bounds_max": [-float(bmin[2]), float(floor_y), -float(bmin[0])],
        }
    except Exception as exc:
        print(f"[feed] topdown map skipped: {exc}")
        return None


# --- movers: how the agent gets from one stop to the next ---
#
# EACH RETURNS True WHEN THE STOP IS REACHED. Keep the contract that narrow: a mover decides how to
# travel, never when to scan or where to go next, so a new one cannot quietly change the schedule.
def _move_navigate(sim, agent, goal, follower, state):
    """Drive to the stop. -> "arrived", "unreachable", "timeout", or None while travelling.

    THE ARRIVAL IS MEASURED, NOT INFERRED. `next_action_along` returns None both for "arrived" and
    for "no path exists", and the old tour treated both as a reach -- that is how a run kept
    "arriving" at goals it never approached and covered 4.8 m in 18.7 minutes. This is the goto
    skill's test: when the follower stops, compare the horizontal distance against
    REVISIT_ARRIVAL_TOL_M and call it what it is. Horizontal only, because the target sits on the
    navmesh and the agent's origin is its base.

    THE FRAME CAP IS NOT DECORATION. The floor guard teleports the agent back the moment it drifts
    off-storey, so a leg that crosses a staircase can loop for ever: walk, get pulled back, re-plan
    the same path.
    """
    if state["frames"] >= REVISIT_MAX_FRAMES:
        return "timeout"
    try:
        action = follower.next_action_along(np.asarray(goal, dtype=np.float32))
    except Exception as exc:
        state["why"] = f"follower raised {type(exc).__name__}: {exc}"
        return "unreachable"
    if action is not None:
        sim.step(action)
        state["frames"] += 1
        return None
    here = np.asarray(agent.get_state().position, dtype=np.float64)
    dist = float(np.hypot(here[0] - float(goal[0]), here[2] - float(goal[2])))
    state["distance"] = dist
    if dist > REVISIT_ARRIVAL_TOL_M:
        state["why"] = f"stopped {dist:.2f} m short (tolerance {REVISIT_ARRIVAL_TOL_M:.2f} m)"
        return "unreachable"
    return "arrived"


def _move_teleport(sim, agent, goal, follower, state):
    """Put the agent on the stop. -> "arrived", or "unreachable" when the navmesh refuses it.

    SNAPPED TO THE NAVMESH FIRST. A schedule point comes from a rasterised roadmap at 5 cm, so it
    can sit a few centimetres off the walkable surface; placing the agent there would leave it
    standing in a wall and every frame from that stop would be wrong. If the snap moves it further
    than the arrival tolerance, the stop is refused rather than silently relocated.
    """
    target = np.asarray(goal, dtype=np.float32)
    pf = getattr(sim, "pathfinder", None)
    if pf is not None and getattr(pf, "is_loaded", False):
        snapped = np.asarray(pf.snap_point(target), dtype=np.float32)
        if not np.all(np.isfinite(snapped)):
            state["why"] = "snap_point returned no navigable point"
            return "unreachable"
        moved = float(np.hypot(snapped[0] - target[0], snapped[2] - target[2]))
        if moved > REVISIT_ARRIVAL_TOL_M:
            state["why"] = f"nearest navigable point is {moved:.2f} m away"
            return "unreachable"
        state["distance"] = moved
        target = snapped
    st = agent.get_state()
    st.position = target
    agent.set_state(st)
    return "arrived"


# `follower` and `straight` are the previous names, kept so an existing command line still runs.
MOVERS = {"navigate": _move_navigate, "teleport": _move_teleport,
          "follower": _move_navigate, "straight": _move_navigate}


def _fire_post_scan(ctx):
    """Called once per completed 360 degree scan. -> what the hook returned, or None.

    THE EVENT IS RECORDED WHETHER OR NOT A HOOK IS CONFIGURED. A trigger nobody can see afterwards
    is not a trigger; scan_events.jsonl in the bundle is the record that a stop was scanned, when,
    and what the hook did about it.

    THE HOOK IS NAMED, NOT WIRED IN. `module:function` in FEED_POST_SCAN_HOOK or habitat.post_scan_hook.
    The dynamic dataset update goes here: the scan is the moment the world model has just been shown
    a place, so it is the moment to change that place and let the next lap find the difference.
    A hook that raises STOPS THE RUN rather than being swallowed -- a dataset update that silently
    failed would leave a bundle whose laps claim a change that never happened.
    """
    out = None
    if POST_SCAN_HOOK:
        mod_name, _, fn_name = POST_SCAN_HOOK.partition(":")
        if not fn_name:
            raise SystemExit(f"FEED_POST_SCAN_HOOK={POST_SCAN_HOOK!r} is not module:function")
        import importlib
        out = getattr(importlib.import_module(mod_name), fn_name)(ctx)
    try:
        with open(STATS_DIR / "scan_events.jsonl", "a") as fh:
            fh.write(json.dumps({**ctx, "hook": POST_SCAN_HOOK or None,
                                 "hook_result": out, "t": time.time()}) + "\n")
    except (OSError, TypeError) as exc:
        print(f"[feed] scan event not recorded: {exc}", flush=True)
    return out


class ScheduledTour:
    """Drives a precomputed schedule. Same interface as Tour: step(agent) and house_done.

    ONE LAP IS THE FILE; the run repeats it EXPLORATION_LAPS times. Laps are identical by design
    (owner, 2026-09-10): the same trajectory driven again, so a difference between two laps is a
    difference in the WORLD, not in the route.
    """

    def __init__(self, sim, schedule, laps, move_fn):
        self.sim = sim
        self.follower = sim.make_greedy_follower(0, goal_radius=0.4)
        self.points = list(schedule["trajectory"])
        self.laps = max(1, int(laps))
        self.move = MOVERS[move_fn]
        self.move_name = move_fn
        self.i = 0
        self.lap = 0
        self.scan_left = 0
        self.scans_done = 0
        self.travel_frames = 0
        self.skipped = []
        self.leg = {"frames": 0}      # per-leg state the mover keeps: frames, distance, why
        # THE SAME SURFACE AS Tour, because main() and the goto skill hold whichever one exists and
        # read these off it without asking which. Missing floor_y crashed the first scheduled run at
        # habitat_feed_host.py:2027 after the schedule had already loaded -- the storey the guard
        # anchors to, the revisit counters the bundle reports, and the goto entry point all live
        # here. Substitutability is the contract; a partial one fails only at runtime.
        self.floor_y = float(schedule.get("height", 0.0))
        self.todo = []
        self.revisit = None
        self.last_revisit = None
        self.revisits_requested = 0
        self.revisits_reached = 0
        self.revisits_failed = 0

        self.house_done = False
        self.floor_order = [round(float(schedule.get("height", 0.0)), 2)]
        self._tour_reached = 0
        self._tour_planned_total = len(self.points)
        print(f"[feed] SCHEDULE: {len(self.points)} points, "
              f"{sum(1 for p in self.points if p['scan_deg'])} stops, {self.laps} lap(s), "
              f"navigation_mode {move_fn}", flush=True)

    def bind_floors(self, scene_floors, tol, guard):
        """A schedule is one storey by construction, so there is nothing to plan. Anchor the guard."""
        self.floor_guard = guard
        if guard is not None:
            guard.reanchor(self.floor_y)
        print(f"[feed] SCHEDULE: single storey {self.floor_y:+.2f}; the floor guard is anchored "
              f"there and no storey change is planned", flush=True)

    def start_revisit(self, *a, **k):
        """goto is refused while a schedule drives: the two would fight over the same agent.

        A revisit walks somewhere, scans and rejoins the tour by restoring a saved tour index. A
        schedule has no such index to restore, and silently dropping the agent back mid-trajectory
        would leave the lap claiming stops it never reached.
        """
        print("[feed] goto refused: a schedule is driving this run; stop it or run without "
              "FEED_SCHEDULE", flush=True)
        return None

    def _scan_frames(self, deg):
        return max(1, int(round(deg / 10.0)))     # the turn action is 10 degrees

    def step(self, agent):
        if self.house_done:
            agent.act("turn_left")
            return
        if self.scan_left > 0:
            self.scan_left -= 1
            agent.act("turn_left")
            if self.scan_left == 0:
                self.scans_done += 1
                pt = self.points[self.i]
                _fire_post_scan({"event": "scan_complete", "lap": self.lap,
                                 "stop": pt.get("stop"), "point_index": self.i,
                                 "xyz": pt["xyz"], "scan_deg": pt["scan_deg"],
                                 "scans_done": self.scans_done,
                                 "stops_total": sum(1 for p in self.points if p["scan_deg"]),
                                 "laps_total": self.laps})
                self._advance()
            return

        pt = self.points[self.i]
        verdict = self.move(self.sim, agent, pt["xyz"], self.follower, self.leg)
        if verdict is None:
            return                       # still travelling
        if verdict != "arrived":
            # A LEG THAT CANNOT BE DRIVEN MUST NOT HOLD THE RUN, and it must not be counted as a
            # visit either. The stop is skipped, the reason is named, and the count goes into the
            # bundle -- a lap that skipped nine stops is not the same lap as one that skipped none.
            self.skipped.append({"point_index": self.i, "stop": pt.get("stop"), "lap": self.lap,
                                 "verdict": verdict, "why": self.leg.get("why", ""),
                                 "frames": self.leg["frames"], "xyz": pt["xyz"]})
            print(f"[feed] SCHEDULE: point {self.i} {verdict}"
                  f"{' — ' + self.leg['why'] if self.leg.get('why') else ''}, skipping", flush=True)
            self._advance()
            return
        self.travel_frames += self.leg["frames"]
        if pt["scan_deg"]:
            self._tour_reached += 1
            self.scan_left = self._scan_frames(pt["scan_deg"])
        else:
            self._advance()

    def _advance(self):
        self.leg = {"frames": 0}
        self.i += 1
        if self.i < len(self.points):
            return
        self.i = 0
        self.lap += 1
        if self.lap >= self.laps:
            self.house_done = True
            print(f"[feed] SCHEDULE COMPLETE: {self.laps} lap(s), {self.scans_done} scans",
                  flush=True)
        else:
            print(f"[feed] SCHEDULE: lap {self.lap + 1} of {self.laps}", flush=True)

    def report(self):
        return {"schedule_points": len(self.points), "laps": self.laps, "lap_reached": self.lap,
                "scans_done": self.scans_done, "navigation_mode": self.move_name,
                "travel_frames": self.travel_frames,
                "stops_skipped": len(self.skipped), "skipped": self.skipped[:40]}


def schedule_overlay(schedule, laps):
    """-> the schedule in ROS ground coords for the viewer, or None.

    Three lists, because they are drawn differently: `path` is the polyline the agent follows,
    `stops` are the 360 scan points in visit order, `root` is where the search starts. `y` rides
    along on every point so the 3D scene can place them at the storey's height instead of guessing.
    """
    if not schedule:
        return None

    def ros(p):
        return [round(-float(p[2]), 3), round(-float(p[0]), 3), round(float(p[1]), 3)]

    traj = schedule.get("trajectory") or []
    return {
        "storey_y": schedule.get("height"),
        "laps": laps,
        "path": [ros(t["xyz"]) for t in traj],
        "stops": [{"order": t.get("stop"), "xyz": ros(t["xyz"]), "scan_deg": t["scan_deg"]}
                  for t in traj if t.get("scan_deg")],
        "root": ros(schedule["root"]) if schedule.get("root") else None,
        "note": "ROS ground coords, the same frame as agent and map.bounds_*; y is the storey height",
    }


def load_schedule(path, floor_y, tol=0.75):
    """-> the storey's schedule from a scene schedule file, or None.

    The file holds every storey of the scene; the run wants the one it is standing on. Matched on
    height rather than on order, because the order in the file is the histogram's, not the run's.
    """
    with open(path) as fh:
        doc = json.load(fh)
    entries = [e for e in doc.get("schedule", []) if "skipped" not in e]
    if not entries:
        raise SystemExit(f"[feed] {path} holds no usable storey schedule")
    best = min(entries, key=lambda e: abs(float(e["height"]) - float(floor_y)))
    if abs(float(best["height"]) - float(floor_y)) > tol:
        raise SystemExit(
            f"[feed] FEED_SCHEDULE={path} has no storey within {tol} m of the spawn height "
            f"{floor_y:+.2f}; its storeys are "
            + ", ".join(f"{e['height']:+.2f}" for e in entries)
            + ". Refusing to tour a different storey than the one the agent stands on.")
    print(f"[feed] schedule storey {best['height']:+.2f} for spawn {floor_y:+.2f} "
          f"({len(entries)} storey(s) in {os.path.basename(path)})", flush=True)
    return best


# --- motion ---
class RevisitState:
    """One in-flight `goto`: where, which phase, and how to rejoin the tour.

    REPLACED WHOLE, NEVER MUTATED FIELD BY FIELD. Ctrl carries no lock ("GIL-atomic reads and
    writes only"), and the sim thread reads this while an HTTP thread may be installing the
    next one. A single attribute swap is atomic; three independent attributes are not, and the
    torn read is a target from one request with the phase of another.
    """

    TRAVEL = "travel"
    SCAN = "scan"

    def __init__(self, target_hab, target_ros, scan_frames, resume, saved_tour_i=None,
                 snap_distance_m=0.0):
        self.target_hab = np.asarray(target_hab, dtype=np.float64)
        self.target_ros = tuple(float(v) for v in target_ros)
        self.scan_frames = int(scan_frames)
        self.resume = bool(resume)
        self.saved_tour_i = saved_tour_i
        self.snap_distance_m = float(snap_distance_m)
        self.phase = self.TRAVEL
        self.frames_travelled = 0
        self.scan_left = int(scan_frames)

    def status(self):
        return {"phase": self.phase, "target_ros": list(self.target_ros),
                "frames_travelled": self.frames_travelled, "scan_left": self.scan_left,
                "resume": self.resume, "snap_distance_m": round(self.snap_distance_m, 3)}


# class Tour lived here: greedy nearest-unvisited coverage over navmesh samples, a 360 scan at
# each waypoint, the multi-storey house tour and the dynamic dwell. REMOVED 2026-09-11 on the
# owner's instruction, with adaptive_hold.py and the walk/dwell burst cycle. It moved on 1.0-1.8%
# of frames with mapping_seconds 0 and its coverage is not comparable with a schedule's, so a
# bundle recording motion_policy "sampled" answers a different question from one recording
# "schedule". ScheduledTour above is the only motion policy now. Recover it from git history if
# a random-sampling baseline is ever wanted: it is at habitat_feed_host.py in commit eae203e.


def main():
    global SHOW
    sim = make_sim()
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
    object_controller = DynamicObjectController(sim, agent)

    # No handler here (owner ruling 2026-09-08 12:15, rule 14): a sampling failure stops the launch.
    # The old `except Exception` continued with NO points, so the storey clustering below ran
    # against an empty list and the spawn-floor filter had nothing to filter on.
    cached_navmesh_pts = []
    if sim.pathfinder.is_loaded:
        for _ in range(300):
            p = sim.pathfinder.get_random_navigable_point()
            rp, _ = habitat_pose_to_ros(p, [0, 0, 0, 1])
            cached_navmesh_pts.append([float(rp[0]), float(rp[1]), float(rp[2])])

    # THE SCHEDULE IS THE ONLY MOTION POLICY (owner 2026-09-11). The sampling tour that used to
    # stand here as the fallback is removed, so there is nothing to fall back TO: a run without a
    # schedule would publish frames from a robot that never moves, which is worse than no run.
    # live_run.sh builds or finds the schedule and refuses the launch before this file starts, and
    # this refusal is the same statement for anyone starting the feed host on its own.
    if not SCHEDULE_PATH:
        raise SystemExit(
            "[feed] FEED_SCHEDULE is not set and the sampling policy is removed, so this run "
            "would have no motion at all. Build the scene's schedule with schedule_batch.py, or "
            "start the run through live_run.sh, which does it for you.")
    if not have_nav:
        raise SystemExit(
            f"[feed] scene {SCENE} has no loaded navmesh, so a schedule cannot be driven. "
            "The sampling policy that used to run without one is removed.")
    if MOVE_FN not in MOVERS:
        raise SystemExit(f"[feed] FEED_MOVE_FN={MOVE_FN!r} is not one of {sorted(MOVERS)}")
    _floor_now = float(agent.get_state().position[1])
    _sched_doc = load_schedule(SCHEDULE_PATH, _floor_now)
    schedule_payload = schedule_overlay(_sched_doc, EXPLORATION_LAPS)
    tour = ScheduledTour(sim, _sched_doc, EXPLORATION_LAPS, MOVE_FN)
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

    # RULE 73. The itinerary is planned HERE and not in Tour.__init__ because the storeys are not
    # known until the navmesh samples above have been clustered, and Tour is built before that.
    if tour is not None:
        tour.bind_floors(scene_floors, floor_tol, floor_guard)

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
        elif act == "goto":
            _start_goto(p)
        else:
            print(f"[feed] unknown action: {act}")

    def _start_goto(p):
        """Validate a revisit target and hand it to the tour. Refuses rather than half-fails.

        Every rejection below is a case that would otherwise fail SILENTLY somewhere later:
        off the navmesh gives a nan the follower never reports, and off-storey livelocks
        against the floor guard instead of raising.
        """
        if tour is None:
            print("[feed] goto refused: no navmesh, so there is no follower to drive", flush=True)
            return
        ros = (float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0)))
        requested = ros_to_habitat(ros)
        target = requested
        snap_distance = 0.0
        if sim.pathfinder.is_loaded:
            snapped = np.asarray(sim.pathfinder.snap_point(requested), dtype=np.float64)
            if not np.all(np.isfinite(snapped)):
                print(f"[feed] goto refused: ROS ({ros[0]:.2f}, {ros[1]:.2f}, {ros[2]:.2f}) "
                      "does not snap to the navmesh", flush=True)
                tour.revisits_requested += 1
                tour.revisits_failed += 1
                tour.last_revisit = {"outcome": "refused", "detail": "off navmesh",
                                     "target_ros": list(ros), "frames_travelled": 0,
                                     "resumed": False}
                return
            snap_distance = float(np.linalg.norm(snapped - requested))
            target = snapped
        # ROS z IS habitat y (ros = -hz, -hx, hy), so the two heights compare directly.
        drift = abs(float(target[1]) - floor_guard.floor_y)
        if drift > floor_guard.tol:
            print(f"[feed] goto refused: target sits {float(target[1]):+.2f} against storey "
                  f"{floor_guard.floor_y:+.2f}, {drift:.2f} m off (tolerance "
                  f"{floor_guard.tol:.2f} m). Cross-storey revisit is not this command.",
                  flush=True)
            tour.revisits_requested += 1
            tour.revisits_failed += 1
            tour.last_revisit = {"outcome": "refused",
                                 "detail": f"off storey by {drift:.2f} m",
                                 "target_ros": list(ros), "frames_travelled": 0,
                                 "resumed": False}
            return
        if snap_distance > 1.0:
            # Served a neighbour, not what was asked for. Say so: silence here reads downstream
            # as "the robot looked at the thing", when it looked at the nearest floor to it.
            print(f"[feed] goto: target snapped {snap_distance:.2f} m onto the navmesh", flush=True)
        amount = float(p.get("amount", 0) or 0)
        scan = int(amount) if amount > 0 else REVISIT_SCAN_FRAMES
        resume = str(p.get("resume", "1")).lower() not in ("0", "false", "no", "off")
        tour.start_revisit(target, ros, scan, resume, snap_distance)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(1)
    print(f"[feed] scene={SCENE} listening on :{PORT}, waiting for the ROS side...")
    conn, addr = srv.accept()
    conn.settimeout(SEND_TIMEOUT)  # a hard-killed container must not hang sendall forever
    print(f"[feed] connected: {addr}")

    period = 1.0 / FPS

    # RESOLVED values, printed AFTER the environment has beaten the config file. Asked for by the
    # experiment lane, 2026-08-31, and the reason is measured: run 19 (20260831_151134) finished
    # cleanly and its dwell is in NONE of the four places it could be — no feed block in
    # run_metadata.json, no feed_stats.json, and config.yaml records the intent rather than the
    # effect (bundle 20260831_033330 has config.yaml mapping_seconds 0.0 against a feed_stats.json
    # and a log that both say 150). live_run.sh now stamps these into the bundle, but live_run.sh
    # is not the only way this file is started, and a run started any other way was exactly how
    # run 19 became unrecoverable. This line costs nothing and fails closed.
    print(f"[feed] resolved: fps={FPS} laps={EXPLORATION_LAPS} seed={SEED} scene={SCENE} "
          f"camera_pitch_deg={CAMERA_PITCH_DEG} navigation_mode={MOVE_FN}", flush=True)
    print(f"[feed] schedule={SCHEDULE_PATH} storey={tour.floor_y:+.2f} "
          f"points={len(tour.points)} signal={_MERGE_PENDING_PATH}", flush=True)
    t_start = time.time()

    ag_state = agent.get_state()
    t_start_sim = t_start
    total_steps = 0
    frame_seq = 0          # GA-121: ordinal of each published frame; see the frame dict
    # GA-37. Two counts with two names. total_steps is RENDERED frames; frames_sent_ok is
    # frames sendall() accepted for the ROS side (the socket is reliable, so a frame the
    # peer never read is one that was in flight when it dropped the connection); the
    # peer's own count is in feed_node_stats.json. A coverage denominator built from
    # rendered frames counts viewpoints perception never saw (21-34% of frames in the
    # two old bundles that carried a viewpoint file).
    frames_sent_ok = 0
    frames_send_failed = 0
    frame_poses_path = STATS_DIR / "frame_poses.jsonl"
    total_distance_m = 0.0
    last_pos = np.asarray(ag_state.position, dtype=np.float64)

    house_done_at = None
    end_reason = None
    settle_pending_start = settle_pending_end = None
    while True:
        t0 = time.time()
        # RULE 73. THE RUN ENDS ITSELF. See TOUR_END_SETTLE_S: with no cap, nothing else would.
        # The break is at the TOP of the loop, after a full frame has been sent and feed_stats.json
        # rewritten, so the bundle already holds a complete set of statistics when it fires.
        if tour is not None and getattr(tour, "house_done", False):
            if house_done_at is None:
                house_done_at = t0
                # GA-440. MEASURE THE SETTLE INSTEAD OF DEFENDING THE NUMBER. TOUR_END_SETTLE_S is a
                # CHOSEN 90 s: it has to cover the object manager's last merge sweeps, and a pair
                # over the evidence threshold still needs merge_min_consecutive sweeps to commit.
                # Sampling pending merges at both ends turns "is 90 s enough" into a bundle field:
                # settle_pending_end above zero says the feed left while merges were still resolving,
                # and the run says so about itself rather than waiting for somebody to notice.
                settle_pending_start, _sw = _pending_merges()
                print(f"[feed] settling for {TOUR_END_SETTLE_S:.0f}s before ending the feed "
                      f"({settle_pending_start if settle_pending_start is not None else '?'} "
                      f"merges pending)", flush=True)
            elif (t0 - house_done_at) >= TOUR_END_SETTLE_S:
                settle_pending_end, _sw = _pending_merges()
                end_reason = "house_tour_complete"
                break
        while CTRL.actions:
            queued_act, queued_params = CTRL.actions.popleft()
            exec_manual(queued_act, queued_params)

        # Scene-graph mutations are queued by the HTTP threads and executed
        # here, on the same thread that steps and renders Habitat-Sim.
        while CTRL.object_commands:
            item = CTRL.object_commands.popleft()
            if item.get("cancelled"):
                continue
            item["result"] = object_controller.execute(item["command"])
            item["done"].set()

        if tour is not None and tour.revisit is not None:
            # AHEAD OF THE MANUAL BRANCH TOO. A goto is an explicit instruction; having it
            # silently ignored because the viewer happened to leave auto_mode off is the
            # failure this command exists to remove.
            label = "REVISIT"
            tour.step(agent)
        elif not CTRL.auto_mode:
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
        else:
            # THE SCHEDULE OWNS THE MOTION, and it is the only thing that does. Its own stops and
            # scans say when the agent stands still, so there is no separate dwell to run beside it.
            label = "SCHEDULE"
            tour.step(agent)

        if tour is not None:
            CTRL.revisit = {"active": tour.revisit.status() if tour.revisit else None,
                            "last": tour.last_revisit,
                            "requested": tour.revisits_requested,
                            "reached": tour.revisits_reached,
                            "failed": tour.revisits_failed}

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

            # Frames left in the current leg or scan. -1 is UNKNOWN, never 0 (rule 5): a leg's
            # length depends on the path the mover finds, which is not known until it arrives.
            phase_rem = tour.scan_left if tour.scan_left > 0 else -1
            feed_stats = {
                "elapsed_sec": round(time.time() - t_start_sim, 1),
                "total_steps": total_steps,
                "frames_rendered": total_steps,
                "frames_sent_ok": frames_sent_ok,
                "frames_send_failed": frames_send_failed,
                "total_distance_m": round(total_distance_m, 2),
                "phase": label,
                "phase_remaining_frames": phase_rem if phase_rem < 0 else max(0, phase_rem),
                # GA-431. The tour existed only as log text, so no bundle could say whether its own
                # tour FINISHED. Four keys, and the first two are settings while the last two are
                # outcomes -- the distinction that let a configured tour be read as a completed one
                # for six days (rule 68). `requested` is what the recipe asked for; `planned` is what
                # farthest-point sampling could actually place on the storey, which can be fewer;
                # `reached` is the count of arrivals; `ended_on_index` is where the run stopped.
                # A tour finished when reached == planned and planned > 0.
                "tour_waypoints_planned": tour._tour_planned_total,
                "tour_waypoints_reached": tour._tour_reached,
                # RULE 73. planned counts EVERY storey's waypoints; the per-storey key above is
                # the last storey only, and a full-house run must not be read as a short one.
                "tour_waypoints_planned_all_floors": tour._tour_planned_total,
                "tour_all_floors": TOUR_ALL_FLOORS,
                # WHICH POLICY DROVE THIS RUN. Constant now that the sampling policy is removed,
                # and KEPT for exactly that reason: bundles made before 2026-09-11 record "sampled"
                # and measure a different experiment, so a reader needs the key to tell them apart.
                "motion_policy": "schedule",
                "schedule_path": SCHEDULE_PATH,
                "schedule": tour.report(),
                "post_scan_hook": POST_SCAN_HOOK or None,
                # A schedule is one storey by construction (rule 73 is satisfied by one launch per
                # storey, not by one feed touring them all), so these are 1 and the storey it drove.
                "tour_floors_planned": 1,
                "tour_floors_toured": len(tour.floor_order),
                "tour_floor_order": list(tour.floor_order),
                "tour_ended_on_index": tour.i,
                "test_mode": TEST_MODE,
                # What the guard did. A run whose map is one storey because nothing drifted and
                # a run whose map is one storey because it was teleported back forty times are
                # different runs, and the map alone cannot tell them apart.
                "floor_guard": floor_guard.report(),
                # Revisits are recorded as three counts, not one: a run where every goto was
                # refused off-storey and one where none was ever sent both have zero reaches.
                "revisits_requested": getattr(tour, "revisits_requested", 0) if tour else 0,
                "revisits_reached": getattr(tour, "revisits_reached", 0) if tour else 0,
                "revisits_failed": getattr(tour, "revisits_failed", 0) if tour else 0,
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
                # GA-470 (owner 2026-09-10). THE SCHEDULE AS DATA, for the minimap and the 3D scene
                # to draw. Published here rather than painted into the minimap PNG, because the
                # image is the background and the drawing is the dashboard's: a vector overlay can
                # be toggled, hit-tested and drawn in the mesh view, and a baked one cannot.
                #
                # ROS GROUND COORDS, the same frame as "agent" and as map.bounds_min/bounds_max, so
                # a client places a point with (p[0] - bounds_min[0]) / scale and nothing else. The
                # schedule itself is habitat coords; converting here means one conversion in one
                # place instead of one per consumer. ROS = (-hab_z, -hab_x, hab_y).
                "schedule": schedule_payload,
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
            # Sent run-length encoded (gt_codec, exact, ~3% of the raw 4.9 MB, ~15 ms): the raw
            # uint32 array cut the feed to 0.10 frames/s and dropped the socket twice in run
            # 20260906_234050; a PNG was exact but cost 245 ms a frame (run 20260907_001120).
            **({"gt_semantic_rle": _gt_codec.encode(obs["semantic_sensor"])}
               if GT_SEMANTIC and "semantic_sensor" in obs else {}),
            # GA-479. The exploration schedule, so the ROS side can draw it in rviz. The SAME
            # payload the dashboard reads out of bev_data.json -- one conversion to ROS ground
            # coords in one place, so the minimap, the mesh view and rviz cannot disagree about
            # where a stop is. About 180 points on hm3d_00861, next to a 921 kB RGB frame, so it
            # rides on every frame rather than on the first one: a reconnect then needs no state.
            "schedule": schedule_payload,
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
                    draw_belief(bgr, visible_belief(poller.objects), frame["cam_pos"], cam_quat,
                                frame["depth"], labels=LAYERS["labels"])
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
                    _counts = grade_counts(_objs)
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
                            # None when the bridge sent no grade at all (GA-102): "n/a"
                            # rather than 0, which is a different statement from no data.
                            cnt = "n/a" if _counts is None else str(_counts[name])
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
            frames_sent_ok += 1
            # GA-37. The viewpoint line is appended AFTER the send succeeds, so the series
            # holds only frames the ROS side was handed; it carries frame_id so a reader can
            # join it to feed_node_stats.json. CAMERA pose (1.5 m above the base), which is
            # what rendered the frame; base_z for reference. Consumed by tools/frustum_gt.py.
            _cp, _cq = habitat_pose_to_ros(frame["cam_pos"], frame["cam_quat"])
            _yaw = math.atan2(2.0 * (_cq[3] * _cq[2] + _cq[0] * _cq[1]), 1.0 - 2.0 * (_cq[1] ** 2 + _cq[2] ** 2))
            with open(frame_poses_path, "a") as fp:
                fp.write(json.dumps({
                    "frame_id": frame["frame_id"], "stamp": frame["t"],
                    "x": float(_cp[0]), "y": float(_cp[1]), "z": float(_cp[2]), "yaw": float(_yaw),
                    "qx": float(_cq[0]), "qy": float(_cq[1]), "qz": float(_cq[2]), "qw": float(_cq[3]),
                    "base_z": float(ros_agent_pos[2]), "phase": label,
                }) + "\n")
        except (BrokenPipeError, ConnectionResetError, socket.error, OSError) as exc:
            frames_send_failed += 1
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

    # RULE 73. The stack has no other way to learn that the tour is over: the feed host runs on the
    # HOST and rtabmap runs in the container, and the only thing they share is this directory
    # (STATS_DIR is RUN_DIR, bind-mounted onto the container's /ws/output).
    # NOT a bare touch: a marker with no reason in it cannot distinguish a finished tour from a
    # crash that happened to leave a file behind.
    # GA-440. The two readings that say whether the settle was long enough. UNKNOWN IS NOT ZERO
    # (rule 5): _pending_merges returns None when the signal file is absent, and a null here means
    # the feed could not read the count -- not that nothing was pending.
    _settled = (settle_pending_end == 0) if settle_pending_end is not None else None
    feed_stats["ended_reason"] = end_reason
    feed_stats["house_tour_complete"] = bool(tour is not None and getattr(tour, "house_done", False))
    feed_stats["settle_pending_start"] = settle_pending_start
    feed_stats["settle_pending_end"] = settle_pending_end
    feed_stats["settle_was_enough"] = _settled
    with open(STATS_DIR / "feed_stats.json", "w") as f:
        json.dump(feed_stats, f)
    marker = {"reason": end_reason, "t": time.time(),
              "floors_toured": list(getattr(tour, "floor_order", [])) if tour else [],
              "settle_s": TOUR_END_SETTLE_S,
              "settle_pending_start": settle_pending_start,
              "settle_pending_end": settle_pending_end,
              "settle_was_enough": _settled,
              "settle_note": "TOUR_END_SETTLE_S is CHOSEN, not measured. settle_pending_end above "
                             "zero means the feed ended while merges were still resolving, so the "
                             "settle was too short for this run; null means the pending count could "
                             "not be read, which is not the same as zero.",
              "total_steps": total_steps, "frames_sent_ok": frames_sent_ok}
    with open(STATS_DIR / "feed_ended.json", "w") as f:
        json.dump(marker, f)
    print(f"[feed] FEED ENDED ({end_reason}); wrote {STATS_DIR / 'feed_ended.json'}", flush=True)


if __name__ == "__main__":
    main()
