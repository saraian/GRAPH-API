"""Measured ROS occupancy-grid BEV; independent of Habitat and perception/VLM."""
import base64
import math
import threading
import time

import cv2
import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException


def yaw(q):
    return math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))


def grid_image(msg):
    """Rows increase with ROS +y, matching the viewer's down-screen +y axis."""
    w, h, res = msg.info.width, msg.info.height, msg.info.resolution
    if w <= 0 or h <= 0 or res <= 0 or len(msg.data) != w*h:
        raise ValueError('invalid occupancy grid dimensions/resolution/data')
    cells = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
    pixels = np.full((h, w), 32, dtype=np.uint8)
    known = cells >= 0
    pixels[known] = np.round(235 - 220*np.clip(cells[known], 0, 100)/100).astype(np.uint8)
    ok, png = cv2.imencode('.png', pixels)
    if not ok:
        raise ValueError('could not encode occupancy grid')
    p = msg.info.origin.position
    return {'image': 'data:image/png;base64,' + base64.b64encode(png).decode(),
            'origin': [p.x, p.y], 'yaw': yaw(msg.info.origin.orientation),
            'width_m': w*res, 'height_m': h*res, 'resolution': res,
            'bounds_min': [p.x, p.z, p.y],
            'bounds_max': [p.x+w*res, p.z, p.y+h*res]}


class RosBEV:
    def __init__(self, node, cfg):
        self.node = node
        self.frame = (cfg.get('tf') or {}).get('world_frame', 'map')
        opts = cfg.get('bev') or {}
        self.base = opts.get('base_frame', 'base_footprint')
        self.topic = opts.get('map_topic', '/rtabmap/map')
        self.lock = threading.Lock()
        self.map = None
        self.agent = None
        self.map_received = None
        self.pose_received = None
        self.last_pose_stamp = None
        self.map_error = f'waiting for {self.topic} in {self.frame}'
        self.pose_error = f'waiting for TF {self.frame} <- {self.base}'
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.subscription = node.create_subscription(OccupancyGrid, self.topic, self.on_map, qos)
        self.timer = node.create_timer(0.2, self.on_pose)

    def on_map(self, msg):
        try:
            if msg.header.frame_id != self.frame:
                raise ValueError(f'grid frame {msg.header.frame_id!r} != {self.frame!r}')
            entry = grid_image(msg)
        except ValueError as exc:
            with self.lock:
                self.map_error = str(exc)
            return
        with self.lock:
            self.map = entry
            self.map_received = time.monotonic()
            self.map_error = None

    def on_pose(self):
        try:
            tf = self.buffer.lookup_transform(self.frame, self.base, Time())
        except TransformException as exc:
            with self.lock:
                self.pose_error = str(exc)
            return
        p = tf.transform.translation
        stamp = (tf.header.stamp.sec, tf.header.stamp.nanosec)
        with self.lock:
            # Use progress/receive time, never compare host wall time to robot stamps.
            if stamp != self.last_pose_stamp:
                self.pose_received = time.monotonic()
                self.last_pose_stamp = stamp
            self.agent = {'x': p.x, 'y': p.y, 'z': p.z, 'yaw': yaw(tf.transform.rotation)}
            self.pose_error = None

    def payload(self):
        with self.lock:
            now = time.monotonic()
            age = None if self.map_received is None else now-self.map_received
            pose_age = None if self.pose_received is None else now-self.pose_received
            errors = [s for s in (self.map_error, self.pose_error) if s]
            if pose_age is not None and pose_age > 3:
                errors.append('robot TF has not advanced for more than 3 seconds')
            return {'source': 'ros_occupancy_grid', 'frame_id': self.frame,
                    'map_topic': self.topic, 'map': self.map, 'agent': self.agent,
                    'map_age_s': age, 'pose_age_s': pose_age,
                    'floors': [], 'navmesh': [], 'source_errors': errors,
                    'capabilities': {'ground_truth_navmesh': False, 'floor_selection': False}}
