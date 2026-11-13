"""Fresh sensor snapshots must not wait for or mutate an active segmentation."""
import sys
import threading
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.analysis.probe_room_segmentation import manager


class SensorMailbox(unittest.TestCase):
    def setUp(self):
        self.rm = manager()
        self.addCleanup(self.rm._room_vlm_executor.shutdown)

    def test_sensors_progress_while_segmentation_holds_lock(self):
        done = threading.Event()
        def receive():
            self.rm._camera_info_callback("new camera")
            self.rm._cache_cloud("new cloud", "map", "map", "stamp", 123.)
            done.set()
        with self.rm._lock:
            thread = threading.Thread(target=receive)
            thread.start()
            try:
                self.assertTrue(done.wait(1), "sensor intake waited for segmentation")
                self.assertIsNone(self.rm._latest_cloud_points)
                self.rm._consume_sensor_state()
                self.assertEqual(self.rm._latest_cloud_points, "new cloud")
                self.assertEqual(self.rm._latest_cloud_received_at, 123.)
                self.assertEqual(self.rm._camera_info, "new camera")
            finally:
                thread.join(timeout=1)

    def test_latest_samples_are_bounded_and_empty_cloud_clears_old_data(self):
        for i in range(1000):
            self.rm._cache_cloud(i, "map", "map", i, float(i))
        self.rm._cache_cloud("ground", "ground", "map", None, 800.)
        self.assertLessEqual(len(self.rm._pending_sensor_state), 7)
        with self.rm._lock:
            self.rm._consume_sensor_state()
        self.assertEqual(self.rm._latest_cloud_points, 999)
        self.assertEqual(self.rm._latest_cloud_stamp, 999)
        self.assertEqual(self.rm._latest_cloud_ground_points, "ground")
        self.rm._cache_cloud(None, "map", "map", 1000, 1000.)
        self.assertEqual(self.rm._latest_cloud_points, 999)
        with self.rm._lock:
            self.rm._consume_sensor_state()
        self.assertIsNone(self.rm._latest_cloud_points)
        self.assertEqual(self.rm._latest_cloud_received_at, 1000.)
        self.assertIsNone(self.rm._last_grid_signature)
