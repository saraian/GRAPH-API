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
polled from the Graph API bridge at FEED_BRIDGE (default http://127.0.0.1:8080).
Only boxes actually in view are drawn (depth-tested against the frame, the same
rule as the ROS /image_with_bb overlay — box_view.py).

Multi-storey scenes: with config `habitat.single_floor` the tour goals stay on the
start floor (`habitat.floor_tolerance_m`), because the 2D SLAM grid cannot tell one
storey from another. Config is config.yaml / GRAPH_API_CONFIG, as on the ROS side.
"""
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
import urllib.request

import numpy as np
import habitat_sim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
from box_view import BOX_EDGES, box_corners_map, project_visible  # noqa: E402  (ROS-free)
from config import CFG  # noqa: E402

print = functools.partial(print, flush=True)  # nohup/file logs must not buffer

SCENE = os.environ.get("HABITAT_SCENE", "train_99248")
DATASET = os.environ.get("HABITAT_DATASET", "/DATA/habitat_hospital/holodeck_clinical.scene_dataset_config.json")
PORT = int(os.environ.get("FEED_PORT", "7799"))
FPS = float(os.environ.get("FEED_FPS", "3"))
SEED = int(os.environ.get("FEED_SEED", "7"))
SHOW = os.environ.get("FEED_SHOW", "0") == "1"
OVERLAY = os.environ.get("FEED_OVERLAY", "0") == "1"
BRIDGE = os.environ.get("FEED_BRIDGE", "http://127.0.0.1:8080")
MAPPING_SECONDS = float(os.environ.get("FEED_MAPPING_SECONDS", "0"))
SINGLE_FLOOR = bool(CFG["habitat"]["single_floor"])
FLOOR_TOL = float(CFG["habitat"]["floor_tolerance_m"])
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

    tour = Tour(sim, rng) if have_nav else None
    poller = None
    if SHOW and OVERLAY:
        poller = BeliefPoller()
        poller.start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(1)
    print(f"[feed] scene={SCENE} listening on :{PORT}, waiting for the ROS side...")
    conn, addr = srv.accept()
    print(f"[feed] connected: {addr}")

    period = 1.0 / FPS
    walk_frames = int(os.environ.get("FEED_WALK", "8"))
    dwell_frames = int(os.environ.get("FEED_DWELL", "18"))
    phase = 0
    t_start = time.time()
    mapping_announced = False
    while True:
        t0 = time.time()
        mapping = MAPPING_SECONDS > 0 and (t0 - t_start) < MAPPING_SECONDS
        if mapping and not mapping_announced:
            print(f"[feed] MAPPING phase for {MAPPING_SECONDS:.0f}s (continuous coverage tour)")
            mapping_announced = True
        if not mapping and mapping_announced:
            print("[feed] DETECTION phase (walk/dwell)")
            mapping_announced = False

        if mapping:
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

        if SHOW:
            import cv2
            bgr = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
            if poller is not None and poller.objects:
                draw_belief(bgr, poller.objects, frame["cam_pos"], cam_quat, frame["depth"])
            cv2.putText(bgr, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("habitat feed (agent camera)", bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                SHOW = False

        blob = pickle.dumps(frame, protocol=4)
        try:
            conn.sendall(struct.pack("!I", len(blob)) + blob)
        except (BrokenPipeError, ConnectionResetError):
            print("[feed] client disconnected, waiting for reconnect...")
            conn, addr = srv.accept()
            print(f"[feed] reconnected: {addr}")

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


if __name__ == "__main__":
    main()
