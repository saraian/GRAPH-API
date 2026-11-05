"""Readiness preserves real ROS image messages until detection consumes them."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/perception_module'))
from utils import SyncedCameraData


def test_readiness_is_non_destructive_but_default_read_consumes_frame():
    bridge = CvBridge()
    rgb = bridge.cv2_to_imgmsg(np.zeros((3, 4, 3), dtype=np.uint8), encoding='bgr8')
    depth = bridge.cv2_to_imgmsg(np.ones((3, 4), dtype=np.float32), encoding='32FC1')
    info = CameraInfo(width=4, height=3)
    for message in (rgb, depth, info):
        message.header.frame_id = 'camera'
        message.header.stamp.sec = 10
    camera = SyncedCameraData.__new__(SyncedCameraData)
    camera.node = SimpleNamespace(get_logger=lambda: Mock())
    camera.bridge = bridge
    camera.sync_tolerance_sec = 2.0
    camera.default_camera_frame = 'camera'
    camera.cached_rgb, camera.cached_depth, camera.cached_camera_info = rgb, depth, info
    transform = object()
    camera.cached_transform = transform
    camera._rgb_received_mono = time.monotonic()

    assert camera.get_synced_data(consume=False)['transform'] is transform
    assert camera.cached_rgb is rgb
    assert camera.cached_depth is depth
    assert camera.cached_camera_info is info
    assert camera.get_synced_data()['transform'] is transform
    assert camera.cached_rgb is None
    assert camera.cached_depth is None
    assert camera.cached_camera_info is None
    assert camera.cached_transform is None
