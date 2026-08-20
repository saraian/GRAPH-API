#!/usr/bin/env python3


import os


os.environ["DISPLAY"] = ":1"


import ctypes
import math
import sys
from typing import Any, Callable, Dict, Optional, Tuple
from nav_msgs.msg import Odometry


flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)


# Aggiungi il path del viewer al sys.path
sys.path.insert(0, "/root/exchange/habitat-sim/examples")


import magnum as mn
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from habitat_sim.utils.settings import default_sim_settings
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


from viewer import HabitatSimInteractiveViewer, Timer



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



class HabitatRosViewer(HabitatSimInteractiveViewer):
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


        self._ros_node.get_logger().info("Habitat ROS viewer ready!")


    def _make_camera_info_msg(self, frame_id: str = "habitat_camera_optical") -> CameraInfo:
        width = int(self.sim_settings["width"])
        height = int(self.sim_settings["height"])


        sensor_spec = self.cfg.agents[self.agent_id].sensor_specifications
        color_spec = next(s for s in sensor_spec if s.uuid == "color_sensor")
        hfov_deg = float(color_spec.hfov)  # gradi
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
        # Rotazione optical: roll=-90deg (X), yaw=-90deg (Z)
        # Equivalente a: (rx, ry, rz, rz_w) = (-0.5, 0.5, -0.5, 0.5)
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