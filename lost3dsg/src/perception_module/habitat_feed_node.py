#!/usr/bin/env python3
"""ROS side of the habitat TCP feed (pairs with test/habitat_feed_host.py).

Connects to the host feed, publishes /camera/rgb, /camera/depth,
/camera/camera_info and the TF chain map -> habitat_camera ->
habitat_camera_optical, using the same conventions as habitat_camera_node.py
(z-up ROS map frame; optical rotation (-0.5, 0.5, -0.5, 0.5)). Every message
of a frame shares one stamp, so TF-at-image-stamp lookups resolve exactly.

  FEED_HOST=127.0.0.1 FEED_PORT=7799 ros2 run lost3dsg habitat_feed_node.py
"""
import json
import array
import math
import os
import pickle
import socket
import struct
import time

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

FRAME_MAP = "map"
FRAME_ODOM = "odom"
FRAME_BASE = "base_link"
FRAME_CAMERA = "habitat_camera"
FRAME_OPTICAL = "habitat_camera_optical"


# --- habitat (y-up) -> ROS (z-up) pose conversion, copied verbatim from
# habitat_camera_node.habitat_pose_to_ros (that module also imports habitat_sim
# and the interactive viewer, so it cannot be imported here). ---
def _u8(buf):
    """bytes -> array('B') for a uint8[] message field. MEASURED in the run image
    (2026-09-07, run H profile): assigning bytes makes the generated setter validate every
    byte in Python -- 3.2 s for the 3.7 MB colour frame and 4.2 s for the 4.9 MB depth frame,
    every frame -- and that is where this node's 100% core went. An array.array('B') takes the
    setter's fast path: 16 ms and 5 ms. Same bytes on the wire."""
    return array.array("B", buf)


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
        # GT-ONLY CHANNEL. habitat's semantic sensor renders its own instance id per pixel,
        # which is the join a detection needs to be labelled with ground truth. The feed host
        # has been putting it in the payload under `gt_semantic_instance` whenever
        # FEED_GT_SEMANTIC=1, and NOTHING READ IT -- the channel existed at one end and was
        # never opened at the other.
        #
        # The topic name says what it is so no runtime consumer can pick it up by accident:
        # nothing in the perception or association path subscribes to it, and the only
        # consumer is the per-detection archive, which is validation output.
        # GA-330 follow-up: a COMPRESSED message. The raw 32SC1 Image (4.9 MB at 1280x960)
        # over reliable DDS stalled this node's receive loop and cut the feed to 0.10 frames/s
        # in run 20260906_234050. The payload is gt_codec's lossless run-length form (~3% of
        # raw, milliseconds each way); the perception node decodes it with the same module.
        self.pub_gt_semantic = self.create_publisher(CompressedImage, "/gt/semantic_instance", qos)
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)

        now = self.get_clock().now().to_msg()
        # GA-359. ONE authority for map -> odom. Under FEED_POSE_SOURCE=simulator (the A arm,
        # the default) this node publishes the static identity, so every box is placed with
        # Habitat's true pose. Under rtabmap it publishes NOTHING for that link: rtabmap's
        # localiser owns it (publish_tf_map, simulator lane's launch flag). Two publishers for
        # one transform was the state every run before this had (GA-362: publish_tf was
        # undeclared and rtabmap published beside the identity; the identity won by luck).
        # An unknown value raises: a pose source is not something to default silently.
        self.pose_source = os.environ.get("FEED_POSE_SOURCE", "simulator").strip().lower()
        if self.pose_source not in ("simulator", "rtabmap"):
            raise ValueError(f"FEED_POSE_SOURCE must be 'simulator' or 'rtabmap', got {self.pose_source!r}")
        static_links = [_tf(now, FRAME_CAMERA, FRAME_OPTICAL, (0.0, 0.0, 0.0), (-0.5, 0.5, -0.5, 0.5))]
        if self.pose_source == "simulator":
            static_links.insert(0, _tf(now, FRAME_MAP, FRAME_ODOM, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        self.static_tf.sendTransform(static_links)
        self.get_logger().info(f"[feed] pose_source={self.pose_source}: map->odom "
                               f"{'identity from this node' if self.pose_source == 'simulator' else 'left to rtabmap'}")

        self._host = os.environ.get("FEED_HOST", "127.0.0.1")
        self._port = int(os.environ.get("FEED_PORT", "7799"))
        self.sock = None
        self.buf = b""
        self.frames = 0
        self._last_frame_id = None
        self._reconnects = 0
        self._connect()
        self.create_timer(0.01, self.poll)
        # GA-200: a heartbeat carrying the FRAME COUNT. The run that produced this fix sat
        # for six minutes with one frame relayed and no further log line, and every liveness
        # signal said healthy -- the process was up, the port was open, the preflight was
        # green. A frozen counter has to be visible from the log, not only from `docker exec`.
        self.create_timer(10.0, self._heartbeat)

    def _connect(self):
        """Open the feed socket. Raises on failure; the caller decides whether to retry."""
        self.sock = socket.create_connection((self._host, self._port), timeout=30)
        self.sock.settimeout(30)
        self.buf = b""
        self.get_logger().info(
            f"connected to habitat feed at {self._host}:{self._port}"
            + (f" (reconnect #{self._reconnects})" if self._reconnects else ""))

    def _heartbeat(self):
        self.get_logger().info(
            f"feed heartbeat: frames={self.frames} reconnects={self._reconnects} "
            f"connected={self.sock is not None}")
        self._write_stats()

    def _write_stats(self):
        """GA-37. The receiving side's own count, in the bundle, beside the host's
        frames_sent_ok (feed_stats.json). frames_received is what this node relayed to
        ROS; last_frame_id is the host's ordinal of the newest one, so the two files join."""
        out = os.environ.get("LOST3DSG_OUTPUT_DIR", "/ws/output")
        if not os.path.isdir(out):
            return
        path = os.path.join(out, "feed_node_stats.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"frames_received": self.frames, "last_frame_id": self._last_frame_id,
                       "reconnects": self._reconnects, "last_updated": time.time()}, f)
        os.replace(tmp, path)

    def _reconnect(self, why):
        """Drop the dead socket and try again. NEVER spin silently.

        The failure this exists for: the peer closes, `recv()` returns b'' WITHOUT raising,
        and the framing loop below `while len(self.buf) < 4` appends nothing forever. The
        process stays RUNNABLE, burns a core, logs nothing and relays no frame -- which is
        indistinguishable from "busy" to every check we had.
        """
        self._reconnects += 1
        self.get_logger().warn(f"feed connection lost ({why}); reconnecting (#{self._reconnects})")
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass
        self.sock = None
        try:
            self._connect()
        except OSError as exc:
            # Stay disconnected and try again on the next poll rather than dying: the host
            # sits in accept() waiting, so the door is open and the next tick may succeed.
            self.get_logger().warn(f"reconnect failed: {type(exc).__name__}: {exc}")

    def _recv_frame(self):
        # GA-200: an EMPTY recv is END OF STREAM, not "no data yet". `recv` returns b'' when
        # the peer has closed, and it does so WITHOUT raising -- so both loops below used to
        # append nothing forever. Every read is checked; there is no path that appends b''.
        while len(self.buf) < 4:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise ConnectionResetError("feed closed while reading the length prefix")
            self.buf += chunk
        n = struct.unpack("!I", self.buf[:4])[0]
        while len(self.buf) < 4 + n:
            chunk = self.sock.recv(1 << 20)
            if not chunk:
                raise ConnectionResetError(
                    f"feed closed with {len(self.buf) - 4}/{n} bytes of the frame received")
            self.buf += chunk
        frame = pickle.loads(self.buf[4:4 + n])
        self.buf = self.buf[4 + n:]
        return frame

    def poll(self):
        if self.sock is None:
            try:
                self._connect()
            except OSError:
                return          # the 10 s heartbeat reports that we are still disconnected
        try:
            frame = self._recv_frame()
        except socket.timeout:
            self.get_logger().warn("feed timeout, retrying")
            return
        except (ConnectionError, OSError) as exc:
            self._reconnect(f"{type(exc).__name__}: {exc}")
            return
        except (pickle.UnpicklingError, struct.error, EOFError) as exc:
            # A malformed frame means the stream is out of sync; the buffer cannot be
            # trusted from here, so drop the connection rather than reinterpret it.
            self._reconnect(f"malformed frame: {type(exc).__name__}: {exc}")
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
        rgb.data = _u8(frame["rgb"].tobytes())
        self.pub_rgb.publish(rgb)

        depth = Image(height=h, width=w, encoding="32FC1", is_bigendian=False, step=w * 4)
        depth.header.stamp = stamp
        depth.header.frame_id = FRAME_OPTICAL
        depth.data = _u8(frame["depth"].tobytes())
        self.pub_depth.publish(depth)

        # Published with THE SAME STAMP as rgb and depth -- the join is by stamp and must be
        # exact, because a GT label taken from a neighbouring frame is worse than no label.
        # The host sends the frame already encoded (`gt_semantic_rle`, gt_codec's run-length
        # form); a host still sending the raw array (`gt_semantic_instance`) is encoded here,
        # so either side can be updated first.
        blob = frame.get("gt_semantic_rle")
        if blob is None and frame.get("gt_semantic_instance") is not None:
            import gt_codec
            blob = gt_codec.encode(frame["gt_semantic_instance"])
        if blob is not None:
            sem_msg = CompressedImage()
            sem_msg.header.stamp = stamp
            sem_msg.header.frame_id = FRAME_OPTICAL
            sem_msg.format = "gt_codec run-length uint32 instance ids (GTRL)"
            sem_msg.data = _u8(bytes(blob))
            self.pub_gt_semantic.publish(sem_msg)

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
        self._last_frame_id = frame.get("frame_id")
        if self.frames % 30 == 1:
            self.get_logger().info(f"frames relayed: {self.frames}")


def main():
    rclpy.init()
    node = HabitatFeedNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
