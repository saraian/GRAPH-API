#!/usr/bin/env python3

import os
os.environ.setdefault("DISPLAY", ":1")

import ctypes
import json
import math
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from nav_msgs.msg import Odometry

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

# Aggiungi il path del viewer al sys.path
VIEWER_EXAMPLES = os.environ.get(
    "HABITAT_VIEWER_EXAMPLES", "/root/exchange/habitat-sim/examples"
)
if VIEWER_EXAMPLES not in sys.path:
    sys.path.insert(0, VIEWER_EXAMPLES)

import magnum as mn
import numpy as np
import rclpy
import habitat_sim
from geometry_msgs.msg import TransformStamped
from habitat_sim.utils.settings import default_sim_settings
from magnum.platform.glfw import Application
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from viewer import HabitatSimInteractiveViewer, Timer

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
    "~/exchange/lost3dsg/habitat/habitat_objects/configs"
)


def global_object_scale(override=None) -> float:
    """Legge il fattore di scala comune agli oggetti spawnati."""
    try:
        factor = float(
            override if override is not None
            else os.environ.get("HABITAT_OBJECT_SCALE", "1.0")
        )
    except (TypeError, ValueError):
        raise ValueError("HABITAT_OBJECT_SCALE deve essere un numero positivo")
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("HABITAT_OBJECT_SCALE deve essere un numero positivo")
    return factor


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

        navmesh = os.environ.get("HABITAT_NAVMESH", "")
        if not navmesh:
            scene_path = os.path.abspath(str(sim_settings.get("scene", "")))
            if scene_path.endswith(".basis.glb"):
                navmesh = scene_path[:-4] + ".navmesh"
        if navmesh and not self.sim.pathfinder.is_loaded:
            if os.path.isfile(navmesh) and not self.sim.pathfinder.load_nav_mesh(navmesh):
                self._ros_node.get_logger().warn(f"Navmesh non caricata: {navmesh}")

        qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Publisher ROS
        self._rgb_pub = self._ros_node.create_publisher(Image, "/camera/rgb", qos_sensor)
        self._object_capture_pub = self._ros_node.create_publisher(
            Image, "/habitat/object_capture/rgb", qos_sensor
        )
        self._depth_pub = self._ros_node.create_publisher(Image, "/camera/depth", qos_sensor)
        self._odom_pub = self._ros_node.create_publisher(Odometry, "/odom", qos_sensor)
        self._camera_info_pub = self._ros_node.create_publisher(
            CameraInfo, "/camera/camera_info", qos_sensor
        )
        self._set_object_position_sub = self._ros_node.create_subscription(
            String,
            "/habitat/set_object_position",
            self._handle_set_object_position,
            qos_sensor,
        )
        self._spawn_object_sub = self._ros_node.create_subscription(
            String,
            "/habitat/spawn_object",
            self._handle_spawn_object,
            qos_sensor,
        )
        self._remove_object_sub = self._ros_node.create_subscription(
            String,
            "/habitat/remove_object",
            self._handle_remove_object,
            qos_sensor,
        )
        self._capture_object_view_sub = self._ros_node.create_subscription(
            String,
            "/habitat/capture_object_view",
            self._handle_capture_object_view,
            qos_sensor,
        )
        self._object_command_result_pub = self._ros_node.create_publisher(
            String,
            "/habitat/object_command_result",
            qos_sensor,
        )
        catalog_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._object_catalog_pub = self._ros_node.create_publisher(
            String, "/habitat/object_catalog", catalog_qos
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
        self._object_capture_sensor = None
        self._pending_object_captures: List[Dict[str, Any]] = []
        self._active_request_id = None
        self._spawned_object_ids: List[int] = []
        # Gli oggetti creati da uno script devono restare nella posa approvata:
        # alcune collision mesh HM3D hanno piccoli buchi e farebbero cadere un
        # rigid body dinamico fuori dalla scena prima della foto.
        self._scripted_object_ids = set()
        self._pending_move_verifications: List[Dict[str, Any]] = []
        self._blacklisted_handles = set()
        self._example_template_handles = []
        self._refresh_template_cache()
        self._load_optional_object_templates()
        self._refresh_template_cache()
        self._publish_object_catalog()

        # Lo spawn dimostrativo è opt-in: un oggetto dinamico creato sempre
        # all'avvio cade sul pavimento e può essere confuso con quello richiesto.
        if os.environ.get("HABITAT_SPAWN_DEMO_OBJECT", "0") == "1":
            if not self._spawn_object(
                prefer_file=False,
                prefer_realistic=True,
                force_realistic=False,
            ):
                self._ros_node.get_logger().warn(
                    "Nessun RigidObject dimostrativo creato: controlla "
                    "HABITAT_EXAMPLE_OBJECTS_DIR e i template disponibili."
                )

        self._ros_node.get_logger().info(
            "Habitat ROS viewer ready. Premi 'm' per GRAB, 'o' per aggiungere "
            "un altro oggetto e 'u' per rimuovere l'ultimo. "
            "Comandi ROS: spawn, move e remove via topic JSON."
        )

    def _set_object_position(self, object_id: int, position) -> bool:
        """Imposta la posa di un oggetto usando coordinate globali Habitat.

        ``position`` e' una sequenza [x, y, z] in metri nel frame globale
        Habitat. L'origine dell'oggetto e' normalmente il suo centro di
        massa/origine del template, non il punto di contatto con il pavimento.
        """
        obj = self._rigid_object_mgr.get_object_by_id(int(object_id))
        if obj is None:
            self._ros_node.get_logger().warn(
                f"Posizione rifiutata: oggetto id={object_id} non trovato."
            )
            return False

        try:
            target = np.asarray(position, dtype=np.float32)
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError("position deve contenere tre numeri finiti")

            # Limite prudenziale per evitare comandi evidentemente errati.
            if np.any(np.abs(target) > 100.0):
                raise ValueError("coordinate fuori dal limite +/-100 m")

            require_elevated = int(object_id) in self._scripted_object_ids
            if not self._validate_position_support(
                obj, target, require_elevated=require_elevated
            ):
                self._ros_node.get_logger().warn(
                    "Move applicato con verifica supporto non concorde; "
                    "la decisione della superficie appartiene al planner."
                )

            # Gli oggetti creati dagli script restano cinematici; quelli
            # manuali restano dinamici per continuare a supportare il grab.
            obj.motion_type = (
                habitat_sim.physics.MotionType.KINEMATIC
                if int(object_id) in self._scripted_object_ids
                else habitat_sim.physics.MotionType.DYNAMIC
            )
            obj.translation = target
            try:
                obj.linear_velocity = mn.Vector3(0.0)
                obj.angular_velocity = mn.Vector3(0.0)
            except Exception:
                # Alcune versioni/template possono non esporre queste
                # proprieta'; la posa resta comunque valida.
                pass
            obj.awake = True
            self._ros_node.get_logger().info(
                f"Oggetto spostato: id={object_id}, "
                f"pos=({target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f})"
            )
            return True
        except Exception as exc:
            self._ros_node.get_logger().warn(
                f"Impossibile spostare oggetto id={object_id}: {exc}"
            )
            return False

    def _handle_set_object_position(self, msg: String) -> None:
        """Riceve una posizione 3D oppure un pixel da proiettare nella scena.

        Formati supportati:
          {"object_id": 1, "position": [x, y, z]}
          {"object_id": 1, "pixel": [u, v]}
        """
        try:
            command = json.loads(msg.data)
            if not isinstance(command, dict):
                raise ValueError("il comando deve essere un oggetto JSON")

            object_id = int(command["object_id"])
            self._active_request_id = command.get("request_id")
            if "position" in command:
                success = self._set_object_position(object_id, command["position"])
                result = {
                    "success": bool(success),
                    "action": "move",
                    "object_id": object_id,
                    "mode": "position",
                    "target_category": command.get("target_category"),
                    "target_surface_point": command.get("target_surface_point"),
                }
                # Il comando e' gia' stato applicato da _set_object_position.
                # Alcune build non rieseguono con affidabilita' il timer di
                # verifica dopo un render di cattura; confermiamo subito e il
                # runner attende prima della foto.
                self._publish_object_command_result(result)
                return

            if "pixel" in command:
                pixel = command["pixel"]
                if not isinstance(pixel, (list, tuple)) or len(pixel) != 2:
                    raise ValueError("pixel deve essere [u, v]")

                point, hit_object_id, hit_normal = self._pixel_to_world(
                    int(pixel[0]),
                    int(pixel[1]),
                )
                if point is None:
                    self._ros_node.get_logger().warn(
                        f"Pixel {pixel} non interseca la scena."
                    )
                    self._publish_object_command_result({
                        "success": False,
                        "action": "move",
                        "object_id": object_id,
                        "mode": "pixel",
                        "pixel": [int(pixel[0]), int(pixel[1])],
                        "message": "il pixel non interseca la scena",
                    })
                    return

                target_position = self._object_position_on_hit(
                    object_id,
                    point,
                    hit_normal,
                )
                self._ros_node.get_logger().info(
                    f"Pixel {pixel} convertito in posizione "
                    f"({target_position[0]:.3f}, {target_position[1]:.3f}, "
                    f"{target_position[2]:.3f}); hit_object_id={hit_object_id}"
                )
                success = self._set_object_position(object_id, target_position)
                result = {
                    "success": bool(success),
                    "action": "move",
                    "object_id": object_id,
                    "mode": "pixel",
                    "pixel": [int(pixel[0]), int(pixel[1])],
                    "hit_object_id": int(hit_object_id),
                    "position": target_position,
                }
                self._publish_object_command_result(result)
                return

            raise ValueError("specificare 'position' oppure 'pixel'")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._ros_node.get_logger().warn(
                "Comando posizione non valido. Formato richiesto "
                '{"object_id": 1, "position": [x, y, z]} oppure '
                '{"object_id": 1, "pixel": [u, v]}: '
                f"{exc}"
            )
            self._publish_object_command_result({
                "success": False,
                "action": "move",
                "message": str(exc),
            })

    def _publish_object_command_result(self, result: Dict[str, Any], request_id=None) -> None:
        request_id = request_id if request_id is not None else self._active_request_id
        if request_id is not None:
            result.setdefault("request_id", str(request_id))
        result_msg = String()
        result_msg.data = json.dumps(result)
        self._object_command_result_pub.publish(result_msg)

    def _schedule_move_verification(
        self,
        object_id: int,
        expected_position,
        result: Dict[str, Any],
    ) -> None:
        """Ritarda l'esito del move finche' la fisica non si e' stabilizzata."""
        if not result.get("success"):
            self._publish_object_command_result(result)
            return

        self._pending_move_verifications.append({
            "object_id": int(object_id),
            "expected_position": np.asarray(expected_position, dtype=np.float64),
            "result": result,
            "wait_frames": 5,
            "stable_frames": 0,
            "max_frames": 20,
        })

    def _schedule_spawn_verification(self, object_id: int, result: Dict[str, Any]) -> None:
        """Attende che un oggetto appena creato termini l'assestamento fisico."""
        self._pending_move_verifications.append({
            "object_id": int(object_id),
            "expected_position": None,
            "result": result,
            "wait_frames": 8,
            "stable_frames": 0,
            "max_frames": 40,
        })

    def _is_object_supported(self, obj) -> bool:
        try:
            bottom_y = float(obj.translation.y) + float(obj.aabb.min.y)
            # Parte poco sopra il fondo reale dell'AABB: il precedente raggio
            # partiva gia' 2 cm sopra e accettava oggetti sospesi fino a 8 cm.
            origin = mn.Vector3(obj.translation.x, bottom_y + 0.05, obj.translation.z)
            ray = habitat_sim.geo.Ray(origin, mn.Vector3(0.0, -1.0, 0.0))
            hits = self.sim.cast_ray(ray=ray)
            if not hits.has_hits():
                return False

            for hit in hits.hits:
                if hit.object_id == obj.object_id:
                    continue
                gap = bottom_y - float(hit.point.y)
                return -0.01 <= gap <= 0.025
            return False
        except Exception:
            return False

    def _process_pending_move_verifications(self) -> None:
        if not self._pending_move_verifications:
            return

        remaining = []
        for pending in self._pending_move_verifications:
            if pending["wait_frames"] > 0:
                pending["wait_frames"] -= 1
                remaining.append(pending)
                continue

            object_id = pending["object_id"]
            obj = self._rigid_object_mgr.get_object_by_id(object_id)
            if obj is None:
                result = dict(pending["result"])
                result.update({
                    "success": False,
                    "verified": False,
                    "message": "oggetto non piu' presente dopo il posizionamento",
                })
                self._publish_object_command_result(result)
                continue

            actual = np.array(
                [float(obj.translation.x), float(obj.translation.y), float(obj.translation.z)],
                dtype=np.float64,
            )
            expected_position = pending["expected_position"]
            position_error = (
                float(np.linalg.norm(actual - expected_position))
                if expected_position is not None else 0.0
            )
            supported = self._is_object_supported(obj)
            try:
                speed = float(np.linalg.norm(obj.linear_velocity))
            except Exception:
                speed = 0.0

            stable = position_error <= 0.08 and supported and speed <= 0.15
            if stable:
                pending["stable_frames"] += 1
            else:
                pending["stable_frames"] = 0

            if pending["stable_frames"] >= 2:
                result = dict(pending["result"])
                result.update({
                    "success": True,
                    "verified": True,
                    "settled": expected_position is None,
                    "actual_position": [
                        float(actual[0]),
                        float(actual[1]),
                        float(actual[2]),
                    ],
                })
                self._publish_object_command_result(result)
                continue

            pending["max_frames"] -= 1
            if pending["max_frames"] <= 0:
                result = dict(pending["result"])
                result.update({
                    "success": False,
                    "verified": False,
                    "message": "oggetto instabile o caduto dopo il posizionamento",
                    "actual_position": [
                        float(actual[0]),
                        float(actual[1]),
                        float(actual[2]),
                    ],
                })
                self._publish_object_command_result(result)
            else:
                remaining.append(pending)

        self._pending_move_verifications = remaining

    def _handle_spawn_object(self, msg: String) -> None:
        """Spawn JSON: {"template": "...", "position": [x, y, z]}."""
        try:
            command = json.loads(msg.data) if msg.data.strip() else {}
            if not isinstance(command, dict):
                raise ValueError("il comando deve essere un oggetto JSON")

            requested_template = command.get("template")
            self._active_request_id = command.get("request_id")
            visual_surface_validated = command.get("visual_surface_validated") is True
            object_scale = command.get("object_scale")
            template = self._resolve_template_handle(requested_template)
            if (
                requested_template
                and str(requested_template).lower() != "random"
                and template is None
            ):
                available = [
                    os.path.basename(handle)
                    for handle in self._example_template_handles[:30]
                ]
                message = (
                    f"template '{requested_template}' non trovato. "
                    f"Esempi disponibili: {available}"
                )
                self._ros_node.get_logger().warn(message)
                self._publish_object_command_result({
                    "success": False,
                    "action": "spawn",
                    "message": message,
                })
                return

            position = command.get("position")
            if position is not None:
                position = np.asarray(position, dtype=np.float32)
                if position.shape != (3,) or not np.all(np.isfinite(position)):
                    raise ValueError("position deve contenere tre numeri finiti")

            spawned = self._spawn_object(
                template_handle=template,
                spawn_position=position,
                scripted=True,
                visual_surface_validated=visual_surface_validated,
                object_scale=object_scale,
            )
            if not spawned:
                self._publish_object_command_result({
                    "success": False,
                    "action": "spawn",
                    "message": "nessun oggetto creato",
                })
                return

            object_id = int(self._spawned_object_ids[-1])
            obj = self._rigid_object_mgr.get_object_by_id(object_id)
            spawn_result = {
                "success": True,
                "action": "spawn",
                "object_id": object_id,
                "handle": str(obj.handle) if obj is not None else None,
                "target_category": command.get("target_category"),
                "target_surface_point": command.get("target_surface_point"),
                "position": [
                    float(obj.translation.x),
                    float(obj.translation.y),
                    float(obj.translation.z),
                ] if obj is not None else None,
            }
            # Lo spawn deve rispondere subito: il client deve poter associare
            # l'id all'oggetto anche se la fisica lo sta ancora assestando.
            # L'attesa avviene nel runner *prima della foto*, cosi' non puo'
            # piu' far scadere l'azione spawn.
            self._publish_object_command_result(spawn_result)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._ros_node.get_logger().warn(f"Comando spawn non valido: {exc}")
            self._publish_object_command_result({
                "success": False,
                "action": "spawn",
                "message": str(exc),
            })

    def _resolve_template_handle(self, requested_template) -> Optional[str]:
        """Risolve un nome breve, ad esempio ``banana``, in un template."""
        if requested_template is None:
            return None

        requested = str(requested_template).strip()
        if not requested or requested.lower() == "random":
            return None
        if requested in self._all_handles:
            return requested

        requested_name = os.path.basename(requested).lower()
        if requested_name.endswith(".object_config.json"):
            requested_name = requested_name[:-len(".object_config.json")]
        elif requested_name.endswith(".json"):
            requested_name = requested_name[:-len(".json")]

        handles = self._example_template_handles + self._file_handles
        for handle in handles:
            filename = os.path.basename(handle).lower()
            stem = filename
            if stem.endswith(".object_config.json"):
                stem = stem[:-len(".object_config.json")]
            elif stem.endswith(".json"):
                stem = stem[:-len(".json")]
            if stem == requested_name:
                return handle

        return None

    def _handle_remove_object(self, msg: String) -> None:
        """Remove JSON: {"object_id": 1} oppure {"all": true}."""
        try:
            command = json.loads(msg.data) if msg.data.strip() else {}
            if not isinstance(command, dict):
                raise ValueError("il comando deve essere un oggetto JSON")
            self._active_request_id = command.get("request_id")

            if bool(command.get("all", False)):
                success = self._remove_all_objects()
                removed_id = None
                removed_position = None
            elif "object_id" in command:
                object_id = int(command["object_id"])
                obj = self._rigid_object_mgr.get_object_by_id(object_id)
                removed_position = (
                    [float(obj.translation.x), float(obj.translation.y), float(obj.translation.z)]
                    if obj is not None else None
                )
                success = self._remove_object_by_id(object_id)
                removed_id = object_id if success else None
            else:
                raise ValueError("specificare 'object_id' oppure 'all': true")

            self._publish_object_command_result({
                "success": bool(success),
                "action": "remove",
                "object_id": removed_id,
                "position": removed_position if success else None,
            })
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._ros_node.get_logger().warn(f"Comando remove non valido: {exc}")
            self._publish_object_command_result({
                "success": False,
                "action": "remove",
                "message": str(exc),
            })

    def _get_object_capture_sensor(self):
        """Restituisce una camera RGB dedicata, senza spostare l'agente."""
        if self._object_capture_sensor is not None:
            return self._object_capture_sensor
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = "object_capture_sensor"
        spec.sensor_type = habitat_sim.SensorType.COLOR
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.resolution = [int(self.sim_settings["height"]), int(self.sim_settings["width"])]
        spec.position = [0.0, 0.0, 0.0]
        spec.orientation = [0.0, 0.0, 0.0]
        spec.hfov = 55.0
        self.sim.add_sensor(spec)
        # ``node_sensors`` espone il CameraSensor C++ nudo, che in alcune
        # build non possiede ``sensor_object``. Il registro del Simulator
        # conserva invece il wrapper VisualSensor con il SceneNode necessario
        # per posizionare una camera senza cambiare AgentState.
        sensor_registry = getattr(self.sim, "sensors", None)
        try:
            sensor = sensor_registry[spec.uuid] if sensor_registry is not None else None
        except (KeyError, TypeError):
            sensor = None
        if sensor is None or not hasattr(sensor, "sensor_object"):
            raise RuntimeError(
                "questa build Habitat non espone il VisualSensor dedicato "
                "attraverso sim.sensors"
            )
        self._object_capture_sensor = sensor
        return self._object_capture_sensor

    def _handle_capture_object_view(self, msg: String) -> None:
        """Accoda la cattura: il render avviene nel timer del viewer."""
        try:
            command = json.loads(msg.data)
            preferred_eye = command.get("capture_eye")
            if preferred_eye is not None:
                preferred_eye = np.asarray(preferred_eye, dtype=np.float32)
                if preferred_eye.shape != (3,) or not np.all(np.isfinite(preferred_eye)):
                    raise ValueError("capture_eye deve contenere tre numeri finiti")
                preferred_eye = preferred_eye.tolist()
            capture_attempt = max(1, int(command.get("capture_attempt", 1)))
            if "object_id" in command:
                object_id = int(command["object_id"])
                if self._rigid_object_mgr.get_object_by_id(object_id) is None:
                    raise ValueError(f"oggetto id={object_id} non trovato")
                self._pending_object_captures.append({
                    "object_id": object_id,
                    "request_id": command.get("request_id"),
                    "preferred_eye": preferred_eye,
                    "capture_attempt": capture_attempt,
                })
            elif "position" in command:
                position = np.asarray(command["position"], dtype=np.float32)
                if position.shape != (3,) or not np.all(np.isfinite(position)):
                    raise ValueError("position deve contenere tre numeri finiti")
                self._pending_object_captures.append({
                    "position": position.tolist(),
                    "request_id": command.get("request_id"),
                    "preferred_eye": preferred_eye,
                    "capture_attempt": capture_attempt,
                })
            else:
                raise ValueError("specificare object_id oppure position")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._publish_object_command_result({
                "success": False,
                "action": "capture",
                "message": str(exc),
            })

    def _capture_object_view(
        self, object_id: Optional[int] = None, position=None, request_id=None,
        preferred_eye=None, capture_attempt=1,
    ) -> None:
        """Renderizza una vista ravvicinata al sicuro, tra due frame del viewer."""
        try:
            obj = self._rigid_object_mgr.get_object_by_id(object_id) if object_id is not None else None
            if object_id is not None and obj is None:
                raise ValueError(f"oggetto id={object_id} non trovato")

            # Il sensore dedicato e' figlio dell'agente ma la sua trasformazione
            # locale viene ricavata dalla posa mondo desiderata. In questo modo
            # non cambiamo AgentState, la camera RGB pubblicata o i TF del robot.
            sensor = self._get_object_capture_sensor()
            capture_sensor = sensor.sensor_object
            sensor_node = getattr(capture_sensor, "object", None)
            sensor_node = sensor_node() if callable(sensor_node) else sensor_node
            if sensor_node is None or not hasattr(sensor_node, "transformation"):
                raise RuntimeError("SceneNode non disponibile per la camera di cattura")
            # L'AABB esposta da questo rigid object e' nel frame locale del
            # template. Il suo centro va quindi trasformato con la posa
            # dell'oggetto: usare il centro nudo inquadrava il pavimento,
            # mentre usare solo ``translation`` poteva lasciare la banana
            # fuori dal riquadro.
            target = (
                obj.transformation.transform_point(obj.aabb.center())
                if obj is not None else mn.Vector3(position)
            )
            size = obj.aabb.size() if obj is not None else mn.Vector3(0.5)
            radius = max(float(size.x), float(size.y), float(size.z)) * 0.5
            eye = self._find_object_capture_eye(
                obj, target, radius, preferred_eye=preferred_eye,
                search_offset=max(0, int(capture_attempt) - 1),
            )
            self._ros_node.get_logger().info(
                "Object-capture v5: "
                f"id={object_id}, target=({target.x:.3f}, {target.y:.3f}, {target.z:.3f}), "
                f"eye=({eye.x:.3f}, {eye.y:.3f}, {eye.z:.3f})"
            )
            camera_view = mn.Matrix4.look_at(
                eye, target, mn.Vector3(0.0, 1.0, 0.0)
            )
            try:
                agent_world = self.default_agent.scene_node.absolute_transformation()
                sensor_node.transformation = (
                    # Questa build Habitat memorizza nel nodo del sensore la
                    # view transform. L'inversa produceva una camera ruotata
                    # verso il soffitto e frame quasi interamente neri.
                    agent_world.inverted() @ camera_view
                )
                # La verifica definitiva avviene *dopo* aver spostato la
                # camera ausiliaria. Non ci fidiamo soltanto del candidato
                # geometrico calcolato da _find_object_capture_eye().
                actual_eye = getattr(sensor_node, "absolute_translation", eye)
                actual_eye = actual_eye() if callable(actual_eye) else actual_eye
                actual_eye = mn.Vector3(actual_eye)
                if not self._same_semantic_room(actual_eye, target):
                    raise RuntimeError(
                        "camera ausiliaria e target non risultano nella stessa stanza"
                    )
                sight = target - actual_eye
                if sight.length() <= 1e-6:
                    raise RuntimeError("camera ausiliaria coincidente con il target")
                sight_hits = self.sim.cast_ray(
                    habitat_sim.geo.Ray(actual_eye, sight.normalized())
                )
                target_distance = sight.length()
                visible = (
                    sight_hits.has_hits()
                    and (
                        (obj is not None and sight_hits.hits[0].object_id == obj.object_id)
                        or (
                            obj is None
                            and float(sight_hits.hits[0].ray_distance) >= target_distance
                        )
                    )
                )
                if not visible:
                    first_id = (
                        sight_hits.hits[0].object_id if sight_hits.has_hits() else None
                    )
                    raise RuntimeError(
                        "target non visibile dalla posa effettiva della camera "
                        f"ausiliaria (first_hit={first_id})"
                    )
                self._ros_node.get_logger().info(
                    "Object-capture visibility after camera move: "
                    f"wanted={object_id}, camera_distance={target_distance:.3f}, "
                    f"first_hit={sight_hits.hits[0].object_id}"
                )
                observations = self.sim.get_sensor_observations()
                rgb = observations.get("object_capture_sensor")
                if rgb is None:
                    raise RuntimeError("nessuna osservazione dalla camera oggetto")
                if rgb.shape[-1] == 4:
                    rgb = rgb[:, :, :3]
                self._object_capture_pub.publish(
                    numpy_to_image_msg_rgb(
                        rgb,
                        self._ros_node.get_clock().now().to_msg(),
                        frame_id=f"object_capture_{object_id if object_id is not None else 'removed'}",
                    )
                )
                self._publish_object_command_result({
                    "success": True,
                    "action": "capture",
                    "object_id": object_id,
                }, request_id=request_id)
            finally:
                # Riporta il sensore nella trasformazione prevista dal suo spec.
                # L'agente non e' mai stato alterato.
                capture_sensor.set_transformation_from_spec()
        except Exception as exc:
            # La camera ausiliaria dipende dalla versione di Habitat-Sim; un
            # errore di rendering non deve interrompere il viewer o lo script.
            self._ros_node.get_logger().warn(f"Cattura oggetto non riuscita: {exc}")
            self._publish_object_command_result({
                "success": False,
                "action": "capture",
                "message": str(exc),
            }, request_id=request_id)

    def _semantic_regions_at(self, point) -> set:
        """Restituisce gli indici delle stanze semantiche contenenti un punto."""
        semantic_scene = getattr(self.sim, "semantic_scene", None)
        getter = getattr(semantic_scene, "get_regions_for_point", None)
        if getter is None:
            return set()
        try:
            return {int(region) for region in getter(mn.Vector3(point))}
        except (TypeError, ValueError, RuntimeError):
            return set()

    def _same_semantic_room(self, first, second) -> bool:
        first_regions = self._semantic_regions_at(first)
        second_regions = self._semantic_regions_at(second)
        # Alcune scene non espongono regioni: in quel caso la linea di vista
        # resta il controllo disponibile, senza inventare una stanza.
        return not first_regions or not second_regions or bool(first_regions & second_regions)

    def _capture_eye_has_line_of_sight(self, obj, eye, target) -> bool:
        direction = target - eye
        if direction.length() < 0.05:
            return False
        hits = self.sim.cast_ray(
            habitat_sim.geo.Ray(eye, direction.normalized())
        )
        if not hits.has_hits():
            return False
        if obj is not None:
            return hits.hits[0].object_id == obj.object_id
        return float(hits.hits[0].ray_distance) >= direction.length()

    def _find_object_capture_eye(
        self, obj, target, radius, preferred_eye=None, search_offset=0
    ):
        """Trova una posa vicina e nella stessa stanza per la camera ausiliaria."""
        if preferred_eye is not None:
            compiled_eye = mn.Vector3(preferred_eye)
            if (
                self._same_semantic_room(compiled_eye, target)
                and self._capture_eye_has_line_of_sight(obj, compiled_eye, target)
            ):
                return compiled_eye
        current_camera, _ = self._get_camera_state()
        current_eye = mn.Vector3(current_camera)
        current_distance = (target - current_eye).length()
        # La cattura non deve fallire solo perché l'oggetto è oltre 2 m: una
        # vista distante ma realmente libera è comunque una cattura valida.
        max_current_distance = max(8.0, radius * 16.0)
        if (
            current_distance <= max_current_distance
            and self._same_semantic_room(current_eye, target)
            and self._capture_eye_has_line_of_sight(obj, current_eye, target)
        ):
            return current_eye

        # I punti sono proiettati sulla navmesh: una posa generata solo con un
        # offset XYZ puo' finire oltre un muro, producendo uno sfondo nero.
        directions = (
            (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
            (0.707, 0.707), (0.707, -0.707), (-0.707, 0.707), (-0.707, -0.707),
        )
        offset = int(search_offset) % len(directions)
        directions = directions[offset:] + directions[:offset]
        eye_lifts = (
            max(0.15, radius * 0.5),
            max(0.35, radius),
            max(0.80, radius * 1.5),
        )
        for distance in (
            max(0.8, radius * 4.0), max(1.25, radius * 6.0), 2.0, 2.75
        ):
            for dx, dz in directions:
                candidate = target + mn.Vector3(distance * dx, 0.0, distance * dz)
                floor = self.sim.pathfinder.snap_point(candidate)
                floor_values = (float(floor.x), float(floor.y), float(floor.z))
                if not all(math.isfinite(value) for value in floor_values):
                    continue
                # snap_point puo' restituire il punto navigabile piu' vicino
                # ma in un'altra stanza, dall'altro lato di una parete. In
                # quel caso una foto dal punto snap-pato e' parzialmente nera
                # o inquadra il lato sbagliato dell'ambiente.
                snap_offset = math.hypot(
                    float(floor.x - candidate.x), float(floor.z - candidate.z)
                )
                # La navmesh può essere leggermente arretrata rispetto alla
                # proiezione X/Z del piano d'arredo; la linea di vista resta
                # il controllo definitivo della posa della camera.
                if snap_offset > 0.75:
                    continue
                # Il punto navmesh stabilisce soltanto X/Z e la stanza. Per
                # oggetti su tavoli o mensole il suo Y e' il pavimento, spesso
                # oltre un metro sotto il target: la camera deve invece restare
                # circa alla quota dell'oggetto per non essere occlusa dal tavolo.
                for lift in eye_lifts:
                    eye = mn.Vector3(floor.x, target.y + lift, floor.z)
                    if not self._same_semantic_room(eye, target):
                        continue
                    ray_direction = target - eye
                    if ray_direction.length() < 0.05:
                        continue
                    # Evita una posa navmesh valida ma con una parete tra camera
                    # e oggetto. Il primo hit deve essere l'oggetto richiesto.
                    if self._capture_eye_has_line_of_sight(obj, eye, target):
                        return eye

        raise RuntimeError(
            "nessuna posa navigabile nella stessa stanza con linea di vista sul target"
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

        HabitatSim ``load_configs`` legge i file object_config nella directory
        indicata, ma non attraversa automaticamente le sottocartelle. Il
        dataset HM3D invece separa normalmente ``configs/`` da ``meshes/``;
        se l'utente indica la radice del dataset, selezioniamo quindi
        automaticamente la cartella delle configurazioni.
        """
        requested_dir = os.environ.get(
            "HABITAT_EXAMPLE_OBJECTS_DIR", DEFAULT_EXAMPLE_OBJECTS_DIR
        ).strip()
        example_dir = requested_dir
        if os.path.isdir(example_dir) and not any(
            name.endswith(".object_config.json")
            for name in os.listdir(example_dir)
        ):
            configs_dir = os.path.join(example_dir, "configs")
            if os.path.isdir(configs_dir):
                example_dir = configs_dir

        if not os.path.isdir(example_dir):
            self._ros_node.get_logger().warn(
                f"Directory template non trovata: {requested_dir}"
            )
            return

        if not any(
            name.endswith(".object_config.json")
            for name in os.listdir(example_dir)
        ):
            self._ros_node.get_logger().warn(
                f"Nessun file .object_config.json in {example_dir} "
                f"(directory richiesta: {requested_dir})"
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

    def _publish_object_catalog(self):
        """Pubblica i nomi brevi dei template realmente caricati da Habitat."""
        names = set()
        for handle in self._example_template_handles:
            name = os.path.basename(str(handle)).lower()
            if name.endswith(".object_config.json"):
                name = name[:-len(".object_config.json")]
            elif name.endswith(".json"):
                name = name[:-len(".json")]
            if name:
                names.add(name)
        message = String()
        message.data = json.dumps({"templates": sorted(names)})
        self._object_catalog_pub.publish(message)
        self._ros_node.get_logger().info(
            f"Catalogo oggetti pubblicato: {len(names)} template"
        )

    def _pick_template_handle(
        self,
        prefer_file: bool = False,
        prefer_realistic: bool = True,
        force_realistic: bool = False,
    ) -> Optional[str]:
        candidates = self._candidate_handles(
            prefer_file=prefer_file,
            prefer_realistic=prefer_realistic,
            force_realistic=force_realistic,
        )
        if candidates:
            return random.choice(candidates)
        return None

    def _candidate_handles(
        self,
        prefer_file: bool = False,
        prefer_realistic: bool = True,
        force_realistic: bool = False,
    ) -> List[str]:
        pools: List[List[str]] = []

        if self._example_template_handles:
            pool = list(self._example_template_handles)
            if prefer_realistic or force_realistic:
                realistic = [
                    handle for handle in pool
                    if any(
                        token in os.path.basename(str(handle)).lower()
                        for token in DEFAULT_REALISTIC_PATTERNS
                    )
                ]
                pool = realistic if realistic or force_realistic else pool
            if pool:
                pools.append(pool)
        elif self._file_handles:
            pool = list(self._file_handles)
            if prefer_realistic or force_realistic:
                realistic = [
                    handle for handle in pool
                    if any(
                        token in os.path.basename(str(handle)).lower()
                        for token in DEFAULT_REALISTIC_PATTERNS
                    )
                ]
                pool = realistic if realistic or force_realistic else pool
            if pool:
                pools.append(pool)

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

    @staticmethod
    def _visual_support_offset(obj) -> float:
        """Distanza tra origine e fondo della mesh renderizzata locale."""
        try:
            return -float(obj.root_scene_node.cumulative_bb.min.y)
        except (AttributeError, TypeError, ValueError):
            return -float(obj.aabb.min.y)

    def _spawn_object(
        self,
        prefer_file: bool = False,
        prefer_realistic: bool = True,
        force_realistic: bool = False,
        template_handle: Optional[str] = None,
        spawn_position=None,
        scripted: bool = False,
        visual_surface_validated: bool = False,
        object_scale=None,
    ) -> bool:
        if template_handle:
            handles = [str(template_handle)]
        else:
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

        factor = global_object_scale(object_scale)
        if abs(factor - 1.0) > 1e-6:
            # In questa build ManagedRigidObject.scale e' un Vector3 read-only;
            # la trasformazione va applicata al SceneNode radice.
            obj.root_scene_node.scale(mn.Vector3(factor))

        if spawn_position is None:
            cam_pos, cam_rot = self._get_camera_state()
            cam_R = _quat_to_rotmat(cam_rot.x, cam_rot.y, cam_rot.z, cam_rot.w)
            forward = cam_R @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
            right = cam_R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            spawn_point = cam_pos + forward * 1.5 + right * random.uniform(-0.15, 0.15)
            spawn_point += np.array([0.0, 0.15, 0.0], dtype=np.float64)
        else:
            spawn_point = np.asarray(spawn_position, dtype=np.float64)

        # Gli oggetti manuali restano dinamici per il grab del viewer. Quelli
        # da script sono cinematici: devono documentare la posa richiesta e
        # non dipendere da collisioni incomplete della scena.
        obj.motion_type = (
            habitat_sim.physics.MotionType.KINEMATIC
            if scripted else habitat_sim.physics.MotionType.DYNAMIC
        )
        obj.translation = np.array(spawn_point, dtype=np.float32)
        # Controllo diagnostico: la decisione sulla superficie target
        # appartiene al planner/LLM, non a questo executor.
        if not self._validate_position_support(
            obj, spawn_point, require_elevated=scripted
        ):
            self._ros_node.get_logger().warn(
                "Spawn applicato con verifica supporto non concorde; "
                "la decisione della superficie appartiene al planner."
            )
        obj.rotation = (
            mn.Quaternion.rotation(mn.Deg(0.0), mn.Vector3(0.0, 1.0, 0.0))
            if scripted else mn.Quaternion.rotation(
                mn.Deg(random.uniform(0.0, 360.0)), mn.Vector3(0.0, 1.0, 0.0)
            )
        )
        obj.awake = True
        self._spawned_object_ids.append(int(obj.object_id))
        if scripted:
            self._scripted_object_ids.add(int(obj.object_id))
        self._ros_node.get_logger().info(
            f"Oggetto aggiunto: id={obj.object_id}, handle='{handle}', "
            f"pos=({obj.translation.x:.2f}, {obj.translation.y:.2f}, {obj.translation.z:.2f}), "
            f"tot={len(self._spawned_object_ids)}"
        )
        return True

    def _validate_position_support(self, obj, requested_position, require_elevated=False) -> bool:
        """Valida che la posa richiesta abbia un supporto, senza modificarla."""
        try:
            requested = np.asarray(requested_position, dtype=np.float64)
            support_offset = self._visual_support_offset(obj)
            expected_support_y = float(requested[1]) - support_offset
            origin = mn.Vector3(
                float(requested[0]), float(requested[1] + 2.0), float(requested[2])
            )
            hits = self.sim.cast_ray(habitat_sim.geo.Ray(origin, mn.Vector3(0.0, -1.0, 0.0)))
            if not hits.has_hits():
                self._ros_node.get_logger().warn("Spawn support check: ray verticale senza hit")
                return False
            stage_id = getattr(habitat_sim, "stage_id", None)
            supports = [
                hit for hit in hits.hits
                if hit.object_id != obj.object_id
                and (stage_id is None or int(hit.object_id) == int(stage_id))
            ]
            # Il primo hit non è sempre il piano corretto: in HM3D possono
            # esserci piani sovrapposti (soffitto, ripiano interno, arredi).
            # Scegliamo quello più vicino alla quota prevista dal template.
            support = min(
                supports,
                key=lambda hit: abs(float(hit.point.y) - expected_support_y),
                default=None,
            )
            if support is None:
                self._ros_node.get_logger().warn("Spawn support check: nessun hit diverso dall'oggetto")
                return False
            if abs(float(support.point.y) - expected_support_y) > 0.12:
                self._ros_node.get_logger().warn(
                    "Spawn support check: quota supporto non compatibile "
                    f"(attesa={expected_support_y:.3f}, trovata={float(support.point.y):.3f})"
                )
                return False

            normal = getattr(support, "normal", None)
            if normal is not None:
                normal_length = math.sqrt(sum(float(normal[index]) ** 2 for index in range(3)))
                if normal_length <= 1e-8 or float(normal.y) / normal_length < 0.75:
                    self._ros_node.get_logger().warn(
                        "Spawn support check: hit non orizzontale o rivolto verso il basso"
                    )
                    return False

            # Se la navmesh è disponibile, usiamola come controllo di stanza;
            # lo spawn resta comunque possibile con una scena senza navmesh.
            if self.sim.pathfinder.is_loaded:
                floor = self.sim.pathfinder.snap_point(mn.Vector3(
                    float(requested[0]), float(support.point.y), float(requested[2])
                ))
                if not all(math.isfinite(float(value)) for value in (floor.x, floor.y, floor.z)):
                    self._ros_node.get_logger().warn("Spawn support check: navmesh non valida")
                    return False
                navmesh_distance = math.hypot(
                    float(floor.x) - requested[0], float(floor.z) - requested[2]
                )
                if navmesh_distance > 0.35:
                    self._ros_node.get_logger().warn(
                        "Spawn support check: supporto troppo lontano dalla navmesh "
                        f"({navmesh_distance:.3f} m)"
                    )
                    return False
                if require_elevated and float(support.point.y) - float(floor.y) < 0.30:
                    self._ros_node.get_logger().warn(
                        "Spawn support check: supporto non abbastanza elevato "
                        f"(delta_y={float(support.point.y) - float(floor.y):.3f} m)"
                    )
                    return False
            return True
        except Exception as exc:
            self._ros_node.get_logger().debug(f"Snap spawn non riuscito: {exc}")
            return False

    def _remove_object_by_id(self, object_id: int) -> bool:
        object_id = int(object_id)
        if object_id not in self._spawned_object_ids:
            self._ros_node.get_logger().info(
                f"Nessun oggetto spawnato con id={object_id}."
            )
            return False

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
            self._spawned_object_ids.remove(object_id)
            self._scripted_object_ids.discard(object_id)
            self._ros_node.get_logger().info(
                f"Oggetto rimosso: id={object_id}, rimasti={len(self._spawned_object_ids)}"
            )
            return True
        except Exception as exc:
            self._ros_node.get_logger().warn(f"Impossibile rimuovere oggetto id={object_id}: {exc}")
            return False

    def _remove_last_object(self) -> bool:
        if not self._spawned_object_ids:
            self._ros_node.get_logger().info("Nessun oggetto spawnato da rimuovere.")
            return False

        return self._remove_object_by_id(self._spawned_object_ids[-1])

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
            support_offset = self._visual_support_offset(obj)
            origin = mn.Vector3(obj.translation) + mn.Vector3(
                0.0, -support_offset - 0.01, 0.0
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
                float(hit.point.y) + support_offset + 0.01,
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

    def _pixel_to_world(self, pixel_x: int, pixel_y: int):
        """Converte un pixel della camera nel primo hit della scena.

        Ritorna (punto_habitat, object_id_colpito, normale), oppure
        (None, None, None).
        """
        try:
            pixel_x = int(pixel_x)
            pixel_y = int(pixel_y)

            width = int(self.sim_settings["width"])
            height = int(self.sim_settings["height"])
            if not (0 <= pixel_x < width and 0 <= pixel_y < height):
                return None, None, None

            sensor_spec = self.cfg.agents[self.agent_id].sensor_specifications
            color_spec = next(
                sensor for sensor in sensor_spec if sensor.uuid == "color_sensor"
            )
            hfov_rad = math.radians(float(color_spec.hfov))
            fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
            fy = fx
            cx = width / 2.0
            cy = height / 2.0

            # Camera frame Habitat: X destra, Y alto, -Z in avanti.
            ray_direction_camera = np.array(
                [
                    (pixel_x - cx) / fx,
                    -(pixel_y - cy) / fy,
                    -1.0,
                ],
                dtype=np.float64,
            )
            ray_direction_camera /= np.linalg.norm(ray_direction_camera)

            camera_position, camera_rotation = self._get_camera_state()
            camera_R = _quat_to_rotmat(
                camera_rotation.x,
                camera_rotation.y,
                camera_rotation.z,
                camera_rotation.w,
            )
            ray_direction_world = camera_R @ ray_direction_camera

            ray = habitat_sim.geo.Ray(
                mn.Vector3(camera_position),
                mn.Vector3(ray_direction_world),
            )
            hits = self.sim.cast_ray(ray=ray)

            if not hits.has_hits():
                return None, None, None

            hit = hits.hits[0]
            hit_normal = getattr(hit, "normal", mn.Vector3(0.0, 1.0, 0.0))
            return hit.point, hit.object_id, hit_normal
        except Exception as exc:
            self._ros_node.get_logger().debug(
                f"Impossibile convertire il pixel in posizione Habitat: {exc}"
            )
            return None, None, None

    def _object_position_on_hit(self, object_id: int, point, normal) -> list:
        """Restituisce una posa appena sopra una superficie colpita."""
        obj = self._rigid_object_mgr.get_object_by_id(int(object_id))
        if obj is None:
            raise ValueError(f"oggetto id={object_id} non trovato")

        normal_np = np.array(
            [float(normal.x), float(normal.y), float(normal.z)],
            dtype=np.float32,
        )
        normal_length = float(np.linalg.norm(normal_np))
        if normal_length < 1e-6:
            normal_np = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            normal_np /= normal_length

        # Per le superfici orizzontali usiamo il fondo reale dell'AABB, non
        # meta' altezza: l'origine del template puo' essere decentrata.
        if abs(float(normal_np[1])) > 0.8:
            offset = self._visual_support_offset(obj) + 0.01
        else:
            offset = 0.5 * float(max(
                obj.aabb.size_x(),
                obj.aabb.size_y(),
                obj.aabb.size_z(),
            )) + 0.01

        target = np.array(
            [float(point.x), float(point.y), float(point.z)],
            dtype=np.float32,
        ) + normal_np * offset
        return [float(target[0]), float(target[1]), float(target[2])]

    def _log_clicked_world_position(self, event) -> None:
        """Stampa il primo punto della scena colpito dal pixel cliccato."""
        pixel_x = int(event.position.x)
        pixel_y = int(event.position.y)
        point, hit_object_id, _ = self._pixel_to_world(pixel_x, pixel_y)

        if point is None:
            self._ros_node.get_logger().info(
                f"Click pixel=({pixel_x}, {pixel_y}): nessun impatto"
            )
            return

        try:
            self._ros_node.get_logger().info(
                f"Click pixel=({pixel_x}, {pixel_y}) -> "
                f"Habitat position=({float(point.x):.3f}, "
                f"{float(point.y):.3f}, {float(point.z):.3f}), "
                f"object_id={hit_object_id}"
            )
        except Exception as exc:
            self._ros_node.get_logger().debug(
                f"Impossibile calcolare la posizione del click: {exc}"
            )

    def pointer_press_event(self, event):
        self._log_clicked_world_position(event)
        super().pointer_press_event(event)

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
        # Le verifiche fisiche non devono dipendere dal rendering: una camera
        # temporanea non supportata da una build Habitat non puo' bloccare il
        # risultato di un move e far scadere il runner.
        self._process_pending_move_verifications()
        try:
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
        except Exception as exc:
            self._ros_node.get_logger().warn(
                f"Pubblicazione camera saltata, fisica ancora attiva: {exc}"
            )
        if self._pending_object_captures:
            capture = self._pending_object_captures.pop(0)
            self._capture_object_view(
                object_id=capture.get("object_id"), position=capture.get("position"),
                request_id=capture.get("request_id"),
                preferred_eye=capture.get("preferred_eye"),
                capture_attempt=capture.get("capture_attempt", 1),
            )

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
    sim_settings["scene"] = os.environ.get(
        "HABITAT_SCENE",
        "/root/exchange/lost3dsg/habitat/hm3d-val-habitat-v0.2/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb",
    )
    sim_settings["scene_dataset"] = os.environ.get(
        "HABITAT_SCENE_DATASET",
        "/root/exchange/lost3dsg/habitat/hm3d-val-semantic-configs-v0.2/"
        "hm3d_annotated_basis.scene_dataset_config.json",
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
