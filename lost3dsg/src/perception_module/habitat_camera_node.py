#!/usr/bin/env python3

import os
os.environ["DISPLAY"] = ":1"

import ctypes
import math
import sys
from typing import Any, Callable, Dict, Optional, Tuple

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

# Aggiungi il path del viewer al sys.path
sys.path.insert(0, "/root/exchange/habitat-sim/examples")

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from viewer import HabitatSimInteractiveViewer, Timer
from habitat_sim.utils.settings import default_sim_settings
import magnum as mn


def numpy_to_image_msg_rgb(np_img, stamp, frame_id: str = "habitat_camera"):
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


def numpy_to_image_msg_depth(np_img, stamp, frame_id: str = "habitat_camera"):
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

    Mapping assi: ROS_x = -Habitat_z, ROS_y = -Habitat_x, ROS_z = Habitat_y

    Args:
        position: array-like [x, y, z] in coordinate Habitat
        rotation_quat: oggetto quaternion Habitat/magnum con attributi x, y, z, w
                       (scalar-last, stessa convenzione di ROS)

    Returns:
        (ros_position: np.ndarray[3], ros_quat_xyzw: np.ndarray[4])
    """
    hx, hy, hz = float(position[0]), float(position[1]), float(position[2])
    ros_position = np.array([-hz, -hx, hy], dtype=np.float64)

    # Matrice di cambio base (stessa permutazione/segno usata per la posizione)
    # R_change = [[0, 0, -1],
    #             [-1, 0, 0],
    #             [0, 1, 0]]
    R_change = np.array([
        [0.0,  0.0, -1.0],
        [-1.0, 0.0,  0.0],
        [0.0,  1.0,  0.0],
    ])

    qx, qy, qz, qw = float(rotation_quat.x), float(rotation_quat.y), float(rotation_quat.z), float(rotation_quat.w)

    # Quaternione -> matrice di rotazione (Habitat frame)
    R_habitat = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),       2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),         1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),       1 - 2*(qx**2 + qy**2)],
    ])

    # Cambia base: R_ros = R_change @ R_habitat @ R_change^T
    R_ros = R_change @ R_habitat @ R_change.T

    # Matrice di rotazione -> quaternione (ROS, scalar-last)
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


class HabitatRosViewer(HabitatSimInteractiveViewer):
    """
    Estende il viewer interattivo originale di habitat-sim aggiungendo
    la pubblicazione di RGB, depth e camera_info su topic ROS2 ad ogni
    frame renderizzato. La finestra GLFW rimane interattiva (tastiera/mouse)
    esattamente come nel viewer originale.
    """

    def __init__(self, sim_settings: Dict[str, Any]):
        # Init ROS prima del viewer
        rclpy.init()
        self._ros_node = rclpy.create_node("habitat_sim_node")

        # Init viewer (crea finestra GLFW + simulatore)
        super().__init__(sim_settings)

        # QoS compatibile con i subscriber sensoriali (BEST_EFFORT)
        qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        # Publisher ROS
        self._rgb_pub = self._ros_node.create_publisher(Image, "/camera/rgb", qos_sensor)
        self._depth_pub = self._ros_node.create_publisher(Image, "/camera/depth", qos_sensor)
        self._camera_info_pub = self._ros_node.create_publisher(
            CameraInfo, "/camera/camera_info", qos_sensor
        )

        # Broadcaster TF per la posa della camera (map -> habitat_camera)
        self._tf_broadcaster = TransformBroadcaster(self._ros_node)
        self._map_frame_id = "map"
        self._camera_frame_id = "habitat_camera"

        # Pre-calcola il messaggio CameraInfo (statico finché risoluzione/FOV non cambiano)
        self._camera_info_msg = self._make_camera_info_msg(frame_id=self._camera_frame_id)

        # Pubblica RGB/depth/camera_info/TF a frequenza fissa (10 Hz),
        # disaccoppiata dal framerate di rendering GLFW per evitare
        # movimento "a scatti" quando si naviga con tastiera/mouse.
        self._publish_timer = self._ros_node.create_timer(0.1, self._publish_observations)

        self._ros_node.get_logger().info("Habitat ROS viewer ready!")

    def _make_camera_info_msg(self, frame_id: str = "habitat_camera") -> CameraInfo:
        width = int(self.sim_settings["width"])
        height = int(self.sim_settings["height"])

        # Recupera l'hfov del color sensor dalla config dell'agente
        sensor_spec = self.cfg.agents[self.agent_id].sensor_specifications
        color_spec = next(s for s in sensor_spec if s.uuid == "color_sensor")
        hfov_deg = float(color_spec.hfov)  # gradi
        hfov_rad = math.radians(hfov_deg)

        # Focal length da hfov (camera pinhole)
        fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
        # Assumendo pixel quadrati, fy = fx
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

    def draw_event(
        self,
        simulation_call: Optional[Callable] = None,
        global_call: Optional[Callable] = None,
        active_agent_id_and_sensor_name: Tuple[int, str] = (0, "color_sensor"),
    ) -> None:
        # Chiama il draw_event originale del viewer (rendering + movimento agente)
        super().draw_event(simulation_call, global_call, active_agent_id_and_sensor_name)

        # Pompa solo gli eventi ROS pendenti (timer di pubblicazione a 10Hz).
        # La pubblicazione vera e propria NON avviene qui per non introdurre
        # variabilità nel frame time del rendering (causa di movimento a scatti).
        rclpy.spin_once(self._ros_node, timeout_sec=0)

    def _publish_observations(self):
        observations = self.sim.get_sensor_observations()
        now = self._ros_node.get_clock().now().to_msg()

        if "color_sensor" in observations:
            rgb = observations["color_sensor"]
            if rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            self._rgb_pub.publish(numpy_to_image_msg_rgb(rgb, now, frame_id=self._camera_frame_id))

        if "depth_sensor" in observations:
            depth = observations["depth_sensor"].squeeze()
            self._depth_pub.publish(numpy_to_image_msg_depth(depth, now, frame_id=self._camera_frame_id))

        # CameraInfo con timestamp aggiornato, stesso frame_id/stamp di RGB e depth
        self._camera_info_msg.header.stamp = now
        self._camera_info_pub.publish(self._camera_info_msg)

        # TF: posa della camera dell'agente nel frame "map"
        self._publish_camera_tf(now)

    def _publish_camera_tf(self, stamp):
        agent_state = self.default_agent.get_state()
        sensor_state = agent_state.sensor_states.get("color_sensor")

        # Usa la posa del sensore se disponibile, altrimenti quella dell'agente
        if sensor_state is not None:
            position = sensor_state.position
            rotation = sensor_state.rotation
        else:
            position = agent_state.position
            rotation = agent_state.rotation

        ros_position, ros_quat = habitat_pose_to_ros(position, rotation)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self._map_frame_id
        t.child_frame_id = self._camera_frame_id

        t.transform.translation.x = float(ros_position[0])
        t.transform.translation.y = float(ros_position[1])
        t.transform.translation.z = float(ros_position[2])

        t.transform.rotation.x = float(ros_quat[0])
        t.transform.rotation.y = float(ros_quat[1])
        t.transform.rotation.z = float(ros_quat[2])
        t.transform.rotation.w = float(ros_quat[3])

        self._tf_broadcaster.sendTransform(t)

    def exit_event(self, event):
        # Chiudi ROS in modo pulito prima di uscire
        try:
            self._ros_node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        super().exit_event(event)


def main():
    print(">>> habitat_camera_node main starting")

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

    app = HabitatRosViewer(sim_settings)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()