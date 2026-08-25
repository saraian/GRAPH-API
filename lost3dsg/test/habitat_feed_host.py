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
from config import CFG  # noqa: E402

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
BRIDGE = os.environ.get("FEED_BRIDGE", "http://127.0.0.1:8081")
MAPPING_SECONDS = float(os.environ.get("FEED_MAPPING_SECONDS", hab_cfg.get("mapping_seconds", 0.0)))
SEND_TIMEOUT = float(os.environ.get("FEED_SEND_TIMEOUT", "10"))
CTRL_PORT = int(os.environ.get("FEED_CTRL_PORT", "7790"))
SINGLE_FLOOR = bool(hab_cfg.get("single_floor", True))
FLOOR_TOL = float(hab_cfg.get("floor_tolerance_m", 0.5))
W, H, HFOV = 640, 480, 90.0
SENSOR_HEIGHT = 1.5


def make_sim():
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = SCENE
    cfg.scene_dataset_config_file = DATASET
    cfg.random_seed = SEED
    specs = []
    for uuid, stype in (("color_sensor", habitat_sim.SensorType.COLOR),
                        ("depth_sensor", habitat_sim.SensorType.DEPTH)):
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

    def run(self):
        while True:
            try:
                with urllib.request.urlopen(f"{BRIDGE}/persistent_perception", timeout=2) as r:
                    data = json.loads(r.read().decode())
                self.objects = data if isinstance(data, list) else data.get("data", [])
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

    def _sample(self, n, max_tries=50):
        pts = []
        for _ in range(n * max_tries):
            p = np.array(self.sim.pathfinder.get_random_navigable_point())
            if not SINGLE_FLOOR or abs(p[1] - self.floor_y) < FLOOR_TOL:
                pts.append(p)
                if len(pts) == n:
                    break
        return pts

    def step(self, agent):
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
        state.position = sim.pathfinder.get_random_navigable_point()
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
    topdown_map = topdown_map_payload(sim, floor_ref) if have_nav else None
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
    dwell_frames = int(os.environ.get("FEED_DWELL", hab_cfg.get("dwell_frames", 60)))
    phase = 0
    t_start = time.time()
    mapping_announced = False

    ag_state = agent.get_state()
    t_start_sim = t_start
    total_steps = 0
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
                "last_updated": time.time()
            }

            bev_payload = {
                "agent": {
                    "x": float(ros_agent_pos[0]),
                    "y": float(ros_agent_pos[1]),
                    "z": float(ros_agent_pos[2]),
                    "yaw": float(yaw)
                },
                "floors": [-2.5, 0.5],
                "navmesh": cached_navmesh_pts,
                "map": topdown_map,
                "stats": feed_stats,
                "auto_mode": CTRL.auto_mode,
                "config": CTRL.config,
            }
            CTRL.bev = bev_payload

            for stats_dir in [Path("/tmp/graphapi_live"), Path("/out"), Path("/tmp")]:
                if stats_dir.exists():
                    with open(stats_dir / "feed_stats.json", "w") as f:
                        json.dump(feed_stats, f)
                    with open(stats_dir / "bev_data.json", "w") as f:
                        json.dump(bev_payload, f)
        except Exception as exc:
            print(f"[feed] export error: {exc}")

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
            "w": W, "h": H, "hfov": HFOV,
        }

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
                if poller is not None and poller.objects:
                    draw_belief(bgr, poller.objects, frame["cam_pos"], cam_quat, frame["depth"])
                cv2.putText(bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("habitat feed (agent camera)", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    SHOW = False
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
