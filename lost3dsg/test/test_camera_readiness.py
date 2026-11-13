"""Readiness preserves real ROS image messages until detection consumes them."""
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from threading import Event, RLock
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/perception_module'))
from utils import SyncedCameraData


def _camera():
    bridge = CvBridge()
    rgb = bridge.cv2_to_imgmsg(np.zeros((3, 4, 3), dtype=np.uint8), encoding='bgr8')
    depth = bridge.cv2_to_imgmsg(np.ones((3, 4), dtype=np.float32), encoding='32FC1')
    info = CameraInfo(width=4, height=3)
    for message in (rgb, depth, info):
        message.header.frame_id = 'camera'
        message.header.stamp.sec = 10
    camera = SyncedCameraData.__new__(SyncedCameraData)
    camera._cache_lock = RLock()
    camera.node = SimpleNamespace(get_logger=lambda: Mock())
    camera.bridge = bridge
    camera.sync_tolerance_sec = 2.0
    camera.default_camera_frame = 'camera'
    camera.cached_rgb, camera.cached_depth, camera.cached_camera_info = rgb, depth, info
    transform = object()
    camera.cached_transform = transform
    camera._rgb_received_mono = time.monotonic()
    return camera, rgb, depth, info, transform


def test_readiness_is_non_destructive_but_default_read_consumes_frame():
    camera, rgb, depth, info, transform = _camera()

    assert camera.get_synced_data(consume=False)['transform'] is transform
    assert camera.cached_rgb is rgb
    assert camera.cached_depth is depth
    assert camera.cached_camera_info is info
    assert camera.get_synced_data()['transform'] is transform
    assert camera.cached_rgb is None
    assert camera.cached_depth is None
    assert camera.cached_camera_info is None
    assert camera.cached_transform is None


def test_concurrent_consumers_cannot_clear_transform_during_conversion():
    camera, rgb, _, _, transform = _camera()
    converting, release, second_started, second_done = Event(), Event(), Event(), Event()
    convert = camera.bridge.imgmsg_to_cv2

    def paused_convert(message, *args, **kwargs):
        if message is rgb and not converting.is_set():
            converting.set()
            assert release.wait(3), 'conversion was not released'
        return convert(message, *args, **kwargs)

    camera.bridge.imgmsg_to_cv2 = paused_convert

    def second_read():
        second_started.set()
        try:
            return camera.get_synced_data()
        finally:
            second_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(camera.get_synced_data)
        try:
            assert converting.wait(3)
            second = pool.submit(second_read)
            assert second_started.wait(3)
            assert not second_done.wait(0.2), 'another reader consumed a frame mid-conversion'
        finally:
            release.set()
        assert first.result(timeout=3)['transform'] is transform
        assert second.result(timeout=3) is None
