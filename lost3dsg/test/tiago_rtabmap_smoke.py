"""Exercise installed RTAB-Map with synthetic RGB-D + TIAGo-shaped TF, no VLM.
Run in an isolated ROS domain. Synthetic input validates wiring, not map quality.
"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/perception_module'))
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from ros_bev import RosBEV

rclpy.init()
node = Node('tiago_rtabmap_smoke')
bev = RosBEV(node, {'tf': {'world_frame': 'found_map'}})
pubs = [node.create_publisher(t, topic, qos_profile_sensor_data) for t, topic in [
    (Image, '/head_front_camera/rgb/image_raw'),
    (Image, '/head_front_camera/depth/image_raw'),
    (CameraInfo, '/head_front_camera/rgb/camera_info'),
    (Odometry, '/mobile_base_controller/odom')]]
static = StaticTransformBroadcaster(node)
broadcaster = TransformBroadcaster(node)
tf = TransformStamped(); tf.header.frame_id = 'base_footprint'
tf.child_frame_id = 'head_front_camera_color_optical_frame'
tf.transform.translation.z = 1.
# Optical +z forward, +x right, +y down in the robot base frame.
tf.transform.rotation.x, tf.transform.rotation.y = -.5, .5
tf.transform.rotation.z, tf.transform.rotation.w = -.5, .5
static.sendTransform(tf)
rng = np.random.default_rng(8)
texture = rng.integers(0, 255, (240,320,3), dtype=np.uint8)
rgb = Image(); rgb.height, rgb.width, rgb.encoding, rgb.step = 240,320,'rgb8',960
rgb.data = texture.tobytes()
depth = Image(); depth.height, depth.width, depth.encoding, depth.step = 240,320,'32FC1',1280
depth.data = np.full((240,320), 2., dtype=np.float32).tobytes()
info = CameraInfo(); info.height, info.width = 240,320
info.k = [250.,0.,160.,0.,250.,120.,0.,0.,1.]
info.p = [250.,0.,160.,0.,0.,250.,120.,0.,0.,0.,1.,0.]
info.distortion_model = 'plumb_bob'; info.d = [0.]*5
odom = Odometry(); odom.header.frame_id = 'odom'; odom.child_frame_id = 'base_footprint'
odom.pose.pose.orientation.w = 1.
with tempfile.TemporaryDirectory(prefix='tiago_slam_smoke_') as tmp:
    log_path = Path(tmp, 'rtabmap.log')
    with log_path.open('w') as log:
        proc = subprocess.Popen(['ros2','launch','rtabmap_launch','rtabmap.launch.py',
          'visual_odometry:=false', 'frame_id:=base_footprint', 'map_frame_id:=found_map',
          'odom_topic:=/mobile_base_controller/odom', 'rgb_topic:=/head_front_camera/rgb/image_raw',
          'depth_topic:=/head_front_camera/depth/image_raw', 'camera_info_topic:=/head_front_camera/rgb/camera_info',
          'qos:=2','approx_sync:=true','rtabmap_viz:=false','rviz:=false','publish_tf_map:=true',
          'database_path:='+tmp+'/rtabmap.db', 'rtabmap_args:=--Grid/FromDepth true --RGBD/NeighborLinkRefining false'],
          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+35
            while time.monotonic()<deadline and proc.poll() is None:
                stamp = node.get_clock().now().to_msg()
                for msg in (rgb,depth,info):
                    msg.header.stamp = stamp
                    msg.header.frame_id = tf.child_frame_id
                odom.header.stamp = stamp
                moving = TransformStamped(); moving.header = odom.header
                moving.child_frame_id = 'base_footprint'; moving.transform.rotation.w = 1.
                broadcaster.sendTransform(moving)
                for pub, msg in zip(pubs,(rgb,depth,info,odom)): pub.publish(msg)
                for _ in range(5): rclpy.spin_once(node, timeout_sec=.02)
                if bev.payload()['map'] and bev.payload()['agent']: break
            payload = bev.payload()
            assert payload['map'] is not None, payload
            assert payload['agent'] is not None, payload
            print('PASS: installed RTAB-Map produced occupancy grid and found_map -> odom -> base_footprint TF')
            print({k:payload[k] for k in ('frame_id','map_topic','agent','source_errors')})
        except BaseException:
            print(log_path.read_text()[-18000:])
            raise
        finally:
            os.killpg(proc.pid, signal.SIGINT)
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=5)
            import sqlite3
            with sqlite3.connect(str(Path(tmp,'rtabmap.db'))) as db:
                print('Database integrity:', db.execute('pragma integrity_check').fetchone()[0])
                print('Stored SLAM nodes:', db.execute('select count(*) from Node').fetchone()[0])
node.destroy_node(); rclpy.shutdown()
