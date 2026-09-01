#!/usr/bin/env python3

import os

os.environ["DISPLAY"] = ":1"

import ctypes
import math
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from nav_msgs.msg import Odometry

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

# Aggiungi il path del viewer al sys.path
sys.path.insert(0, "/root/exchange/habitat-sim/examples")

# These imports MUST follow the sys.path.insert above: they resolve against the
# habitat-sim examples directory added there, so moving them to the top of the file
# breaks them. Marked rather than moved -- an autofix that reorders imports breaks an
# import that had to come first for its side effect.
import habitat_sim  # noqa: E402
import magnum as mn  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from habitat_sim.utils.settings import default_sim_settings  # noqa: E402
from magnum.platform.glfw import Application  # noqa: E402
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster  # noqa: E402
# `Timer` was imported and never used. Removed rather than noqa'd: the module still loads
# for HabitatSimInteractiveViewer, so no import side effect changes.
from viewer import HabitatSimInteractiveViewer  # noqa: E402

DEFAULT_REALISTIC_PATTERNS = [
    "cup",
    "mug",
    "glass",
    "bottle",
    "bowl",
    "plate",
    "can",
    "jar",
    "vase",
    "tumbler",
    "goblet",
    "teapot",
    "wine",
]

DEFAULT_EXAMPLE_OBJECTS_DIR = os.path.expanduser(
    "~/exchange/lost3dsg/data/objects/example_objects"
)


def numpy_to_image_msg_rgb(np_img, stamp, frame_id: str = "habitat_camera_optical"):
    """
    np_img: H x W x 3, uint8 RGB
    """
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = np_img.shape[0]
    msg.width = np_img.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = msg.width * 3
    msg.data = np_img.astype(np.uint8).tobytes()
    return msg


def numpy_to_image_msg_depth(np_img, stamp, frame_id: str = "habitat_camera_optical"):
    """
    np_img: H x W, float32 (metri)
    """
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = np_img.shape[0]
    msg.width = np_img.shape[1]
    msg.encoding = "32FC1"
    msg.is_bigendian = False
    msg.step = msg.width * 4
    msg.data = np_img.astype(np.float32).tobytes()
    return msg


def habitat_pose_to_ros(position, rotation_quat):
    """
    Converte posizione e quaternione dal sistema di coordinate Habitat
    (Y-up, right-handed: X destra, Y su, Z verso l'osservatore/indietro)
    al sistema ROS (Z-up, right-handed: X avanti, Y sinistra, Z su).
    """
    hx, hy, hz = float(position[0]), float(position[1]), float(position[2])
    ros_position = np.array([-hz, -hx, hy], dtype=np.float64)

    R_change = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    qx, qy, qz, qw = (
        float(rotation_quat.x),
        float(rotation_quat.y),
        float(rotation_quat.z),
        float(rotation_quat.w),
    )

    R_habitat = np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )

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


def _quat_to_rotmat(qx, qy, qz, qw):
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def _rotmat_to_quat(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float64)


def _relative_ros_transform(parent_pos, parent_quat, child_pos, child_quat):
    parent_R = _quat_to_rotmat(*parent_quat)
    child_R = _quat_to_rotmat(*child_quat)
    rel_pos = parent_R.T @ (child_pos - parent_pos)
    rel_R = parent_R.T @ child_R
    rel_quat = _rotmat_to_quat(rel_R)
    return rel_pos, rel_quat


class HabitatRosViewerWithObjects(HabitatSimInteractiveViewer):
    def __init__(self, sim_settings: Dict[str, Any]):
        rclpy.init()
        self._ros_node = rclpy.create_node("habitat_sim_node")

        super().__init__(sim_settings)

        qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Publisher ROS
        self._rgb_pub = self._ros_node.create_publisher(Image, "/camera/rgb", qos_sensor)
        self._depth_pub = self._ros_node.create_publisher(Image, "/camera/depth", qos_sensor)
        self._odom_pub = self._ros_node.create_publisher(Odometry, "/odom", qos_sensor)
        self._camera_info_pub = self._ros_node.create_publisher(
            CameraInfo, "/camera/camera_info", qos_sensor
        )

        self._tf_broadcaster = TransformBroadcaster(self._ros_node)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self._ros_node)
        self._map_frame_id = "map"
        self._odom_frame_id = "odom"
        self._base_frame_id = "base_link"
        self._camera_frame_id = "habitat_camera"
        self._camera_optical_frame_id = "habitat_camera_optical"

        self._camera_info_msg = self._make_camera_info_msg(frame_id=self._camera_optical_frame_id)
        self._publish_static_tfs()
        self._publish_timer = self._ros_node.create_timer(0.1, self._publish_observations)

        self._object_template_mgr = self.sim.get_object_template_manager()
        self._rigid_object_mgr = self.sim.get_rigid_object_manager()
        self._spawned_object_ids: List[int] = []
        self._blacklisted_handles = set()
        self._example_template_handles = []
        self._refresh_template_cache()
        self._load_optional_object_templates()
        self._refresh_template_cache()

        # Inserisce subito un RigidObject davanti alla camera, così la modalità
        # GRAB è verificabile senza dover premere prima "o".
        if not self._spawn_object(
            prefer_file=False,
            prefer_realistic=True,
            force_realistic=False,
        ):
            self._ros_node.get_logger().warn(
                "Nessun RigidObject creato automaticamente: "
                "controlla HABITAT_EXAMPLE_OBJECTS_DIR e i template disponibili."
            )

        self._ros_node.get_logger().info(
            "Habitat ROS viewer ready. Premi 'm' per GRAB, 'o' per aggiungere "
            "un altro oggetto e 'u' per rimuovere l'ultimo."
        )

    def _make_camera_info_msg(self, frame_id: str = "habitat_camera_optical") -> CameraInfo:
        width = int(self.sim_settings["width"])
        height = int(self.sim_settings["height"])

        sensor_spec = self.cfg.agents[self.agent_id].sensor_specifications
        color_spec = next(s for s in sensor_spec if s.uuid == "color_sensor")
        hfov_deg = float(color_spec.hfov)
        hfov_rad = math.radians(hfov_deg)

        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width = width
        msg.height = height
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]
        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return msg

    def _load_optional_object_templates(self):
        """
        Carica i template del dataset example_objects.
        """
        example_dir = os.environ.get(
            "HABITAT_EXAMPLE_OBJECTS_DIR", DEFAULT_EXAMPLE_OBJECTS_DIR
        ).strip()
        if not os.path.isdir(example_dir):
            self._ros_node.get_logger().warn(
                f"Directory example_objects non trovata: {example_dir}"
            )
            return

        try:
            before_handles = set(self._object_template_mgr.get_file_template_handles())
            loaded = self._object_template_mgr.load_configs(example_dir)
            after_handles = set(self._object_template_mgr.get_file_template_handles())
            self._example_template_handles = sorted(after_handles - before_handles)
            self._ros_node.get_logger().info(
                f"Caricati {len(loaded)} template oggetto da {example_dir}"
            )
        except Exception as exc:
            self._ros_node.get_logger().warn(
                f"Impossibile caricare template da {example_dir}: {exc}"
            )

    def _refresh_template_cache(self):
        self._file_handles = list(self._object_template_mgr.get_file_template_handles())
        self._synth_handles = list(self._object_template_mgr.get_synth_template_handles())
        self._all_handles = list(self._object_template_mgr.get_template_handles())
        if not self._example_template_handles:
            self._example_template_handles = list(self._file_handles)

        if self._example_template_handles:
            self._ros_node.get_logger().info(
                "Template example_objects disponibili:\n"
                + "\n".join(sorted(self._example_template_handles))
            )
        else:
            self._ros_node.get_logger().warn(
                "Nessun template oggetto disponibile in example_objects."
            )

    def _pick_template_handle(
        self,
        prefer_file: bool = False,
        prefer_realistic: bool = True,
        force_realistic: bool = False,
    ) -> Optional[str]:
        if self._example_template_handles:
            return random.choice(self._example_template_handles)
        if prefer_file and self._file_handles:
            return random.choice(self._file_handles)
        if self._file_handles:
            return random.choice(self._file_handles)
        return None

    def _candidate_handles(
        self,
        prefer_file: bool = False,
        prefer_realistic: bool = True,
        force_realistic: bool = False,
    ) -> List[str]:
        pools: List[List[str]] = []

        if self._example_template_handles:
            pools.append(list(self._example_template_handles))
        elif self._file_handles:
            pools.append(list(self._file_handles))

        merged: List[str] = []
        seen = set()
        for pool in pools:
            random.shuffle(pool)
            for handle in pool:
                if handle in seen or handle in self._blacklisted_handles:
                    continue
                seen.add(handle)
                merged.append(handle)
        return merged

    def _get_camera_state(self):
        """
        Ritorna la posa del sensore RGB in coordinate Habitat.
        Se il sensore non e' disponibile, ricade sulla posa dell'agente.
        """
        agent_state = self.default_agent.get_state()
        sensor_state = agent_state.sensor_states.get("color_sensor")
        if sensor_state is not None:
            return np.array(sensor_state.position, dtype=np.float64), sensor_state.rotation
        return np.array(agent_state.position, dtype=np.float64), agent_state.rotation

    def _spawn_object(
        self,
        prefer_file: bool = False,
        prefer_realistic: bool = True,
        force_realistic: bool = False,
    ) -> bool:
        handles = self._candidate_handles(
            prefer_file=prefer_file,
            prefer_realistic=prefer_realistic,
            force_realistic=force_realistic,
        )
        if not handles:
            if force_realistic:
                self._ros_node.get_logger().warn(
                    "Nessun template realistico disponibile per lo spawn."
                )
            else:
                self._ros_node.get_logger().warn(
                    "Nessun template oggetto disponibile per lo spawn."
                )
            return False

        obj = None
        handle = None
        last_error = None
        for candidate in handles:
            try:
                obj = self._rigid_object_mgr.add_object_by_template_handle(candidate)
                handle = candidate
                if obj is not None:
                    break
            except Exception as exc:
                last_error = exc
                self._blacklisted_handles.add(candidate)
                continue

        if obj is None:
            self._ros_node.get_logger().warn(
                "Nessun template spawnabile ha funzionato"
                + (f" (ultimo errore: {last_error})" if last_error else "")
            )
            return False

        cam_pos, cam_rot = self._get_camera_state()
        cam_R = _quat_to_rotmat(cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w)
        forward = cam_R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        right = cam_R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        spawn_point = cam_pos + forward * 1.5 + right * random.uniform(-0.15, 0.15)
        spawn_point += np.array([0.0, 0.15, 0.0], dtype=np.float64)

        # L'oggetto resta separato dalla camera, ma deve essere DYNAMIC:
        # il viewer implementa il grab con un RigidConstraint e un oggetto
        # KINEMATIC può causare il fallimento della creazione/aggiornamento
        # del vincolo.
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        obj.translation = np.array(spawn_point, dtype=np.float32)
        obj.rotation = mn.Quaternion.rotation(
            mn.Deg(random.uniform(0.0, 360.0)), mn.Vector3(0.0, 1.0, 0.0)
        )
        obj.awake = True

        self._spawned_object_ids.append(int(obj.object_id))
        self._ros_node.get_logger().info(
            f"Oggetto aggiunto: id={obj.object_id}, handle='{handle}', "
            f"pos=({spawn_point[0]:.2f}, {spawn_point[1]:.2f}, {spawn_point[2]:.2f}), "
            f"tot={len(self._spawned_object_ids)}"
        )
        return True

    def _remove_last_object(self) -> bool:
        if not self._spawned_object_ids:
            self._ros_node.get_logger().info("No spawned object to remove.")
            return False

        object_id = self._spawned_object_ids[-1]
        try:
            # Un vincolo che punta all'oggetto rimosso diventa invalido e può
            # generare errori nel viewer al frame successivo.
            if (
                self.mouse_grabber is not None
                and self.mouse_grabber.settings.object_id_a == object_id
            ):
                del self.mouse_grabber
                self.mouse_grabber = None
            self._rigid_object_mgr.remove_object_by_id(object_id)
            self._spawned_object_ids.pop()
            self._ros_node.get_logger().info(
                f"Oggetto rimosso: id={object_id}, rimasti={len(self._spawned_object_ids)}"
            )
            return True
        except Exception as exc:
            self._ros_node.get_logger().warn(f"Could not remove object id={object_id}: {exc}")
            return False

    def _remove_all_objects(self) -> bool:
        removed_any = False
        while self._spawned_object_ids:
            removed_any = self._remove_last_object() or removed_any
        return removed_any

    def _snap_grabbed_object_to_surface(self) -> None:
        """Place a grabbed object just above the first surface below it.

        Thin shelves and imperfect scene collision meshes can leave a dynamic
        object slightly intersecting a surface when the mouse constraint is
        released.  A short vertical ray gives Bullet a clean starting pose.
        """
        grabber = self.mouse_grabber
        if grabber is None:
            return

        object_id = grabber.settings.object_id_a
        obj = self._rigid_object_mgr.get_object_by_id(object_id)
        if obj is None:
            return

        try:
            half_height = 0.5 * float(obj.aabb.size_y())
            origin = mn.Vector3(obj.translation) + mn.Vector3(
                0.0, -half_height - 0.01, 0.0
            )
            ray = habitat_sim.geo.Ray(origin, mn.Vector3(0.0, -1.0, 0.0))
            hits = self.sim.cast_ray(ray=ray)
            if not hits.has_hits():
                return

            hit = next(
                (candidate for candidate in hits.hits
                 if candidate.object_id != object_id),
                None,
            )
            if hit is None:
                return

            obj.translation = mn.Vector3(
                obj.translation.x,
                float(hit.point.y) + half_height + 0.01,
                obj.translation.z,
            )
            obj.awake = True
        except Exception as exc:
            self._ros_node.get_logger().debug(
                f"Snap superficie non applicato all'oggetto {object_id}: {exc}"
            )

    def _publish_static_tfs(self):
        stamp = self._ros_node.get_clock().now().to_msg()

        # map -> odom
        t_map_odom = TransformStamped()
        t_map_odom.header.stamp = stamp
        t_map_odom.header.frame_id = self._map_frame_id
        t_map_odom.child_frame_id = self._odom_frame_id
        t_map_odom.transform.translation.x = 0.0
        t_map_odom.transform.translation.y = 0.0
        t_map_odom.transform.translation.z = 0.0
        t_map_odom.transform.rotation.x = 0.0
        t_map_odom.transform.rotation.y = 0.0
        t_map_odom.transform.rotation.z = 0.0
        t_map_odom.transform.rotation.w = 1.0
        self._static_tf_broadcaster.sendTransform(t_map_odom)

        # habitat_camera -> habitat_camera_optical
        t_cam_opt = TransformStamped()
        t_cam_opt.header.stamp = stamp
        t_cam_opt.header.frame_id = self._camera_frame_id
        t_cam_opt.child_frame_id = self._camera_optical_frame_id
        t_cam_opt.transform.translation.x = 0.0
        t_cam_opt.transform.translation.y = 0.0
        t_cam_opt.transform.translation.z = 0.0
        t_cam_opt.transform.rotation.x = -0.5
        t_cam_opt.transform.rotation.y = 0.5
        t_cam_opt.transform.rotation.z = -0.5
        t_cam_opt.transform.rotation.w = 0.5
        self._static_tf_broadcaster.sendTransform(t_cam_opt)

    def draw_event(
        self,
        simulation_call: Optional[Callable] = None,
        global_call: Optional[Callable] = None,
        active_agent_id_and_sensor_name: Tuple[int, str] = (0, "color_sensor"),
    ) -> None:
        super().draw_event(simulation_call, global_call, active_agent_id_and_sensor_name)
        rclpy.spin_once(self._ros_node, timeout_sec=0)

    def key_press_event(self, event):
        key = Application.Key
        mod = Application.Modifier
        shift_pressed = bool(event.modifiers & mod.SHIFT)

        if event.key == key.O:
            if self._spawn_object(
                prefer_file=False,
                prefer_realistic=True,
                force_realistic=shift_pressed,
            ):
                event.accepted = True
                return
        elif event.key == key.U:
            if shift_pressed:
                self._remove_all_objects()
            else:
                self._remove_last_object()
            event.accepted = True
            return

        super().key_press_event(event)

    def pointer_release_event(self, event):
        # Esegui lo snap mentre il vincolo esiste ancora; il metodo base lo
        # elimina subito dopo il rilascio del mouse.
        self._snap_grabbed_object_to_surface()
        super().pointer_release_event(event)

    def print_help_text(self) -> None:
        super().print_help_text()
        self._ros_node.get_logger().info(
            """
Object editing:
'o': Spawn a realistic object in front of the camera if available.
'+SHIFT o': Force a realistic object, without falling back to primitives.
'u': Remove the most recently spawned object.
'+SHIFT u': Remove all spawned objects.
"""
        )

    def _publish_observations(self):
        observations = self.sim.get_sensor_observations()
        agent_state = self.default_agent.get_state()
        now = self._ros_node.get_clock().now().to_msg()

        if "color_sensor" in observations:
            rgb = observations["color_sensor"]
            if rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            self._rgb_pub.publish(
                numpy_to_image_msg_rgb(rgb, now, frame_id=self._camera_optical_frame_id)
            )

        if "depth_sensor" in observations:
            depth = observations["depth_sensor"].squeeze()
            self._depth_pub.publish(
                numpy_to_image_msg_depth(depth, now, frame_id=self._camera_optical_frame_id)
            )

        self._camera_info_msg.header.stamp = now
        self._camera_info_pub.publish(self._camera_info_msg)
        self._publish_camera_tf(now, agent_state)

    def _publish_camera_tf(self, stamp, agent_state):
        sensor_state = agent_state.sensor_states.get("color_sensor")
        base_position, base_quat = habitat_pose_to_ros(
            agent_state.position, agent_state.rotation
        )
        if sensor_state is not None:
            cam_position, cam_quat = habitat_pose_to_ros(
                sensor_state.position, sensor_state.rotation
            )
        else:
            cam_position, cam_quat = base_position, base_quat

        rel_position, rel_quat = _relative_ros_transform(
            base_position, base_quat, cam_position, cam_quat
        )

        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = stamp
        t_odom_base.header.frame_id = self._odom_frame_id
        t_odom_base.child_frame_id = self._base_frame_id
        t_odom_base.transform.translation.x = float(base_position[0])
        t_odom_base.transform.translation.y = float(base_position[1])
        t_odom_base.transform.translation.z = float(base_position[2])
        t_odom_base.transform.rotation.x = float(base_quat[0])
        t_odom_base.transform.rotation.y = float(base_quat[1])
        t_odom_base.transform.rotation.z = float(base_quat[2])
        t_odom_base.transform.rotation.w = float(base_quat[3])
        self._tf_broadcaster.sendTransform(t_odom_base)

        # Pubblica anche il messaggio /odom con la stessa posa
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = self._odom_frame_id
        odom_msg.child_frame_id = self._base_frame_id
        odom_msg.pose.pose.position.x = float(base_position[0])
        odom_msg.pose.pose.position.y = float(base_position[1])
        odom_msg.pose.pose.position.z = float(base_position[2])
        odom_msg.pose.pose.orientation.x = float(base_quat[0])
        odom_msg.pose.pose.orientation.y = float(base_quat[1])
        odom_msg.pose.pose.orientation.z = float(base_quat[2])
        odom_msg.pose.pose.orientation.w = float(base_quat[3])
        self._odom_pub.publish(odom_msg)

        t_base_cam = TransformStamped()
        t_base_cam.header.stamp = stamp
        t_base_cam.header.frame_id = self._base_frame_id
        t_base_cam.child_frame_id = self._camera_frame_id
        t_base_cam.transform.translation.x = float(rel_position[0])
        t_base_cam.transform.translation.y = float(rel_position[1])
        t_base_cam.transform.translation.z = float(rel_position[2])
        t_base_cam.transform.rotation.x = float(rel_quat[0])
        t_base_cam.transform.rotation.y = float(rel_quat[1])
        t_base_cam.transform.rotation.z = float(rel_quat[2])
        t_base_cam.transform.rotation.w = float(rel_quat[3])
        self._tf_broadcaster.sendTransform(t_base_cam)

    def exit_event(self, event):
        try:
            self._remove_all_objects()
        except Exception:
            pass
        try:
            self._ros_node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        super().exit_event(event)


def main():
    print(">>> habitat_camera_objects_node main starting")

    sim_settings: Dict[str, Any] = default_sim_settings.copy()

    # Scena e dataset
    sim_settings["scene"] = (
        "/root/exchange/lost3dsg/habitat/hm3d-val-habitat-v0.2/"
        "00801-HaxA7YrQdEC/HaxA7YrQdEC.basis.glb"
    )
    sim_settings["scene_dataset"] = (
        "/root/exchange/lost3dsg/habitat/hm3d-val-habitat-v0.2/"
        "hm3d_annotated_basis.scene_dataset_config.json"
    )

    # Risoluzione camera
    sim_settings["width"] = 640
    sim_settings["height"] = 480

    # Sensori da abilitare
    sim_settings["color_sensor"] = True
    sim_settings["depth_sensor"] = True
    sim_settings["semantic_sensor"] = False

    # Agente di default
    sim_settings["default_agent"] = 0

    # Parametri richiesti dal viewer interattivo
    sim_settings["window_width"] = 640
    sim_settings["window_height"] = 480
    sim_settings["enable_batch_renderer"] = False
    sim_settings["num_environments"] = 1
    sim_settings["composite_files"] = None
    sim_settings["use_default_lighting"] = False
    sim_settings["enable_hbao"] = False
    sim_settings["default_agent_navmesh"] = False
    sim_settings["enable_physics"] = True

    app = HabitatRosViewerWithObjects(sim_settings)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
