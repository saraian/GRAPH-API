#!/usr/bin/env python3
"""ROS side of the habitat TCP feed (pairs with test/habitat_feed_host.py).

Connects to the host feed, publishes /camera/rgb, /camera/depth,
/camera/camera_info and the TF chain map -> habitat_camera ->
habitat_camera_optical, using the same conventions as habitat_camera_node.py
(z-up ROS map frame; optical rotation (-0.5, 0.5, -0.5, 0.5)). Every message
of a frame shares one stamp, so TF-at-image-stamp lookups resolve exactly.

  FEED_HOST=127.0.0.1 FEED_PORT=7799 ros2 run lost3dsg habitat_feed_node.py
"""
import math
import os
import pickle
import socket
import struct

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

FRAME_MAP = "map"
FRAME_ODOM = "odom"
FRAME_BASE = "base_link"
FRAME_CAMERA = "habitat_camera"
FRAME_OPTICAL = "habitat_camera_optical"


# --- habitat (y-up) -> ROS (z-up) pose conversion, copied verbatim from
# habitat_camera_node.habitat_pose_to_ros (that module also imports habitat_sim
# and the interactive viewer, so it cannot be imported here). ---
def habitat_pose_to_ros(position, quat_xyzw):
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


def _quat_to_rotmat(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
    ], dtype=np.float64)


def _rotmat_to_quat(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return np.array([(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])


def _relative_ros_transform(parent_pos, parent_quat, child_pos, child_quat):
    parent_R = _quat_to_rotmat(*parent_quat)
    child_R = _quat_to_rotmat(*child_quat)
    return parent_R.T @ (child_pos - parent_pos), _rotmat_to_quat(parent_R.T @ child_R)


def _tf(stamp, parent, child, pos, quat):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in pos)
    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (float(v) for v in quat)
    return t


class HabitatFeedNode(Node):
    def __init__(self):
        super().__init__("habitat_feed_node")
        # RELIABLE like habitat_camera_node: rtabmap subscribes reliable and
        # refuses best-effort publishers; the perception's best-effort
        # subscribers accept reliable publishers fine.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.pub_rgb = self.create_publisher(Image, "/camera/rgb", qos)
        self.pub_depth = self.create_publisher(Image, "/camera/depth", qos)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera_info", qos)
        self.pub_odom = self.create_publisher(Odometry, "/odom", qos)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)

        now = self.get_clock().now().to_msg()
        # same static links as habitat_camera_node: map->odom identity, camera->optical
        self.static_tf.sendTransform([
            _tf(now, FRAME_MAP, FRAME_ODOM, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            _tf(now, FRAME_CAMERA, FRAME_OPTICAL, (0.0, 0.0, 0.0), (-0.5, 0.5, -0.5, 0.5)),
        ])

        host = os.environ.get("FEED_HOST", "127.0.0.1")
        port = int(os.environ.get("FEED_PORT", "7799"))
        self.sock = socket.create_connection((host, port), timeout=30)
        self.sock.settimeout(30)
        self.get_logger().info(f"connected to habitat feed at {host}:{port}")
        self.buf = b""
        self.frames = 0
        self.create_timer(0.01, self.poll)

    def _recv_frame(self):
        while len(self.buf) < 4:
            self.buf += self.sock.recv(1 << 20)
        n = struct.unpack("!I", self.buf[:4])[0]
        while len(self.buf) < 4 + n:
            self.buf += self.sock.recv(1 << 20)
        frame = pickle.loads(self.buf[4:4 + n])
        self.buf = self.buf[4 + n:]
        return frame

    def poll(self):
        try:
            frame = self._recv_frame()
        except socket.timeout:
            self.get_logger().warn("feed timeout, retrying")
            return

        stamp = self.get_clock().now().to_msg()
        w, h = frame["w"], frame["h"]

        cam_pos, cam_quat = habitat_pose_to_ros(frame["cam_pos"], frame["cam_quat"])
        if "base_pos" in frame:
            base_pos, base_quat = habitat_pose_to_ros(frame["base_pos"], frame["base_quat"])
        else:  # older feed host without agent pose: body = camera
            base_pos, base_quat = cam_pos, cam_quat
        rel_pos, rel_quat = _relative_ros_transform(base_pos, base_quat, cam_pos, cam_quat)
        # dynamic chain: odom->base_link (agent body), base_link->habitat_camera
        self.tf.sendTransform([
            _tf(stamp, FRAME_ODOM, FRAME_BASE, base_pos, base_quat),
            _tf(stamp, FRAME_BASE, FRAME_CAMERA, rel_pos, rel_quat),
        ])
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = FRAME_ODOM
        odom.child_frame_id = FRAME_BASE
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = (float(v) for v in base_pos)
        (odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,
         odom.pose.pose.orientation.z, odom.pose.pose.orientation.w) = (float(v) for v in base_quat)
        self.pub_odom.publish(odom)

        rgb = Image(height=h, width=w, encoding="rgb8", is_bigendian=False, step=w * 3)
        rgb.header.stamp = stamp
        rgb.header.frame_id = FRAME_OPTICAL
        rgb.data = frame["rgb"].tobytes()
        self.pub_rgb.publish(rgb)

        depth = Image(height=h, width=w, encoding="32FC1", is_bigendian=False, step=w * 4)
        depth.header.stamp = stamp
        depth.header.frame_id = FRAME_OPTICAL
        depth.data = frame["depth"].tobytes()
        self.pub_depth.publish(depth)

        fx = (w / 2.0) / math.tan(math.radians(frame["hfov"]) / 2.0)
        info = CameraInfo(width=w, height=h, distortion_model="plumb_bob")
        info.header.stamp = stamp
        info.header.frame_id = FRAME_OPTICAL
        info.d = [0.0] * 5
        info.k = [fx, 0.0, w / 2.0, 0.0, fx, h / 2.0, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, w / 2.0, 0.0, 0.0, fx, h / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)

        self.frames += 1
        if self.frames % 30 == 1:
            self.get_logger().info(f"frames relayed: {self.frames}")


def main():
    rclpy.init()
    node = HabitatFeedNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
