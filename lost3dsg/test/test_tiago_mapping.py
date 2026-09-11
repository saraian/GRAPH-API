"""ROS integration checks. Run in PAL container with an isolated ROS_DOMAIN_ID."""
import base64
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/perception_module'))
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import TransformBroadcaster
from ros_bev import RosBEV, grid_image


class MappingTests(unittest.TestCase):
    def grid(self):
        msg = OccupancyGrid()
        msg.header.frame_id = 'found_map'
        msg.info.width, msg.info.height, msg.info.resolution = 3, 2, 0.5
        msg.info.origin.position.x, msg.info.origin.position.y = 10., -2.
        msg.info.origin.orientation.z = math.sin(math.pi/4)
        msg.info.origin.orientation.w = math.cos(math.pi/4)
        msg.data = [-1, 0, 100, 100, 50, 0]
        return msg

    def test_grid_coordinates_and_unknown_space(self):
        entry = grid_image(self.grid())
        raw = base64.b64decode(entry['image'].split(',')[1])
        pixels = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(pixels, [[32,235,15],[15,125,235]])
        self.assertEqual(entry['origin'], [10.,-2.])
        self.assertEqual(entry['width_m'], 1.5)
        self.assertAlmostEqual(entry['yaw'], math.pi/2)
        msg = self.grid(); msg.data = [0]
        with self.assertRaises(ValueError): grid_image(msg)

    def test_mapping_tf_preflight_over_dds(self):
        from check_mapping_tf import MappingTFCheck
        from tf2_msgs.msg import TFMessage
        rclpy.init()
        node = MappingTFCheck()
        publisher = Node('tf_ownership_test_publisher')
        pub = publisher.create_publisher(TFMessage, '/tf', 10)
        try:
            self.assertEqual(node.result()[0], 2)  # silence is not a successful check
            def deliver(parent, child, expected):
                tf = TransformStamped()
                tf.header.frame_id, tf.child_frame_id = parent, child
                tf.transform.rotation.w = 1.
                deadline = time.monotonic()+5
                while time.monotonic()<deadline:
                    tf.header.stamp = publisher.get_clock().now().to_msg()
                    pub.publish(TFMessage(transforms=[tf]))
                    rclpy.spin_once(node, timeout_sec=.05)
                    if node.result()[0] == expected: return
                self.fail(node.result())
            deliver('odom', 'base_footprint', 0)
            deliver('map', 'odom', 1)
            self.assertIn('pal module stop localization', node.result()[1])
            self.assertEqual(node.parents, {'map'})
        finally:
            node.destroy_node(); publisher.destroy_node(); rclpy.shutdown()

    def test_fov_uses_configured_world_and_camera_at_image_stamp(self):
        from config import CFG
        from perception_utils import compute_fov_volume_from_depth
        from sensor_msgs.msg import CameraInfo
        from builtin_interfaces.msg import Time as Stamp
        from unittest.mock import Mock
        camera = CameraInfo()
        camera.header.frame_id = 'head_front_camera_color_optical_frame'
        camera.k = [100.,0.,1.,0.,100.,1.,0.,0.,1.]
        node = Mock()
        tf = TransformStamped()
        tf.transform.rotation.w = 1.
        tf.transform.translation.x = 2.
        node.tf_buffer.lookup_transform.return_value = tf
        with patch.dict(CFG['tf'], world_frame='found_map'):
            volume = compute_fov_volume_from_depth(np.full((4,4), 1000, np.uint16),
                        camera, node, stride=1, stamp=Stamp(sec=123))
        self.assertIsNotNone(volume)
        self.assertGreater(volume['x_min'], 1.9)
        args = node.tf_buffer.lookup_transform.call_args.args
        self.assertEqual(args[:2], ('found_map', camera.header.frame_id))
        self.assertEqual(args[2].nanoseconds, 123000000000)

    def test_real_dds_grid_tf_and_bridge_endpoints(self):
        import graph_api_bridge as bridge
        rclpy.init()
        publisher = Node('tiago_mapping_test_publisher')
        subscriber = Node('tiago_mapping_test_bev')
        bev = RosBEV(subscriber, {'tf': {'world_frame':'found_map'}})
        pub = publisher.create_publisher(OccupancyGrid, '/rtabmap/map',
              QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        broadcaster = TransformBroadcaster(publisher)
        try:
            deadline = time.monotonic()+10
            while time.monotonic() < deadline:
                pub.publish(self.grid())
                tf = TransformStamped()
                # Deliberately skewed robot clock: receive age must remain fresh.
                tf.header.stamp = publisher.get_clock().now().to_msg()
                tf.header.stamp.sec -= 181
                tf.header.frame_id, tf.child_frame_id = 'found_map', 'base_footprint'
                tf.transform.translation.x = 11.
                tf.transform.rotation.w = 1.
                broadcaster.sendTransform(tf)
                rclpy.spin_once(subscriber, timeout_sec=.05)
                if bev.payload()['map'] and bev.payload()['agent']: break
            payload = bev.payload()
            self.assertIsNotNone(payload['map'])
            self.assertEqual(payload['agent']['x'], 11.)
            self.assertLess(payload['pose_age_s'], 1.)
            bad = self.grid(); bad.header.frame_id = 'map'; bev.on_map(bad)
            self.assertIn('grid frame', bev.payload()['source_errors'][0])
            bev.on_map(self.grid())
            with tempfile.TemporaryDirectory() as output:
                Path(output, 'perception.log').write_text('camera received\n')
                Path(output, 'rtabmap.log').write_text('map updated\n')
                fake_node = type('Bridge', (), {'ros_bev': bev})()
                with patch.dict(os.environ, GRAPH_API_OUTPUT_DIR=output), \
                     patch.object(bridge, '_node', fake_node), \
                     patch.object(bridge, '_BRIDGE_CFG', {'bev': {'source': 'ros'}}), \
                     patch.object(bridge, '_SVC_B', {'feed_enabled':False}), \
                     patch.object(bridge.urllib.request, 'urlopen', side_effect=AssertionError('Habitat must not be contacted')):
                    data = json.loads(bridge.proxy_bev_data().body)
                    self.assertTrue(data['map']['url'].startswith('/bev_map/'))
                    mid = data['map']['url'].split('/')[-1]
                    self.assertEqual(bridge.get_bev_map(mid).media_type, 'image/png')
                    logs = bridge.get_logs()['logs']
                    self.assertIn('[perception] camera received', logs)
                    self.assertIn('[rtabmap] map updated', logs)
        finally:
            subscriber.destroy_node(); publisher.destroy_node(); rclpy.shutdown()

if __name__ == '__main__': unittest.main()
