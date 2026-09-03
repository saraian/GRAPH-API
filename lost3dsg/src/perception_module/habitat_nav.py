#!/usr/bin/env python3

import os
import math
import numpy as np
import magnum as mn
import rclpy
from config import CFG

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import habitat_sim
from habitat_sim.utils import common as utils

TEST_SCENE = CFG["habitat"]["nav_scene"]
SCENE_DATASET = CFG["habitat"]["nav_scene_dataset"]
OPTIONAL_NAVMESH = CFG["habitat"]["nav_navmesh"]

IMG_WIDTH = 640
IMG_HEIGHT = 480
SENSOR_HEIGHT = 1.5
HFOV_DEG = 90.0
FRAME_ID_MAP = "map"
FRAME_ID_BASE = "base_link"
FRAME_ID_CAMERA = "habitat_camera"
CONTROL_FREQUENCY = 10.0
FRAME_SKIP = 6
DEFAULT_FORWARD_STEP = 0.25
DEFAULT_TURN_DEG = 30.0
EPS = 1e-5


def build_camera_info(width, height, hfov_deg, stamp, frame_id):
    hfov_rad = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = width
    msg.height = height
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def rgb_to_msg(rgb, stamp, frame_id):
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = rgb.shape[0]
    msg.width = rgb.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = rgb.shape[1] * 3
    msg.data = rgb.tobytes()
    return msg


def depth_to_msg(depth, stamp, frame_id):
    depth = np.ascontiguousarray(depth.astype(np.float32))
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = depth.shape[0]
    msg.width = depth.shape[1]
    msg.encoding = "32FC1"
    msg.is_bigendian = False
    msg.step = depth.shape[1] * 4
    msg.data = depth.tobytes()
    return msg


def make_cfg():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = TEST_SCENE
    sim_cfg.scene_dataset_config_file = SCENE_DATASET
    sim_cfg.enable_physics = True
    sim_cfg.allow_sliding = True

    sensor_specs = []

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [IMG_HEIGHT, IMG_WIDTH]
    rgb_spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    rgb_spec.hfov = mn.Deg(HFOV_DEG)
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(rgb_spec)

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [IMG_HEIGHT, IMG_WIDTH]
    depth_spec.position = [0.0, SENSOR_HEIGHT, 0.0]
    depth_spec.hfov = mn.Deg(HFOV_DEG)
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=DEFAULT_FORWARD_STEP)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=DEFAULT_TURN_DEG)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=DEFAULT_TURN_DEG)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


class HabitatNode(Node):
    def __init__(self):
        super().__init__("habitat_sim_node")

        qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.sim = habitat_sim.Simulator(make_cfg())
        self.agent = self.sim.initialize_agent(0)

        init_state = habitat_sim.AgentState()
        init_state.position = np.array([-0.6, 0.0, 0.0], dtype=np.float32)
        self.agent.set_state(init_state)

        if os.path.exists(OPTIONAL_NAVMESH):
            ok = self.sim.pathfinder.load_nav_mesh(OPTIONAL_NAVMESH)
            self.get_logger().info(f"Explicit navmesh load: {ok}")
        else:
            self.get_logger().warn("No precomputed navmesh found, skipping explicit load.")

        self.pub_rgb = self.create_publisher(Image, "/camera/rgb", qos_sensor)
        self.pub_depth = self.create_publisher(Image, "/camera/depth", qos_sensor)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera_info", qos_sensor)
        self.pub_pose = self.create_publisher(PoseStamped, "/habitat/agent_pose", qos_sensor)
        self.pub_collision = self.create_publisher(String, "/habitat/collision", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.sub_action = self.create_subscription(String, "/habitat/action", self.action_callback, 10)
        self.sub_cmd_vel = self.create_subscription(Twist, "/habitat/cmd_vel", self.cmd_vel_callback, 10)

        self.camera_info_msg = build_camera_info(
            IMG_WIDTH, IMG_HEIGHT, HFOV_DEG, self.get_clock().now().to_msg(), FRAME_ID_CAMERA
        )

        self.action_names = list(self.agent.agent_config.action_space.keys())
        self.time_step = 1.0 / (FRAME_SKIP * CONTROL_FREQUENCY)
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.ang_vel_is_local = True

        self.last_collided = False
        self.pending_discrete_action = None
        self.pending_continuous_cmd = None

        self.control_timer = self.create_timer(1.0 / CONTROL_FREQUENCY, self.control_loop)
        self.publish_static_tfs()
        self.publish_current_observation()
        self.get_logger().info("Habitat ROS node ready.")

    def publish_static_tfs(self):
        stamp = self.get_clock().now().to_msg()
        static_tf = TransformStamped()
        static_tf.header.stamp = stamp
        static_tf.header.frame_id = FRAME_ID_BASE
        static_tf.child_frame_id = FRAME_ID_CAMERA
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = SENSOR_HEIGHT
        static_tf.transform.translation.z = 0.0
        static_tf.transform.rotation.x = 0.5
        static_tf.transform.rotation.y = -0.5
        static_tf.transform.rotation.z = 0.5
        static_tf.transform.rotation.w = -0.5
        self.static_tf_broadcaster.sendTransform(static_tf)

    def publish_dynamic_tfs(self, stamp):
        state = self.agent.get_state()

        tf_map_base = TransformStamped()
        tf_map_base.header.stamp = stamp
        tf_map_base.header.frame_id = FRAME_ID_MAP
        tf_map_base.child_frame_id = FRAME_ID_BASE
        tf_map_base.transform.translation.x = float(state.position[0])
        tf_map_base.transform.translation.y = float(state.position[1])
        tf_map_base.transform.translation.z = float(state.position[2])
        q = state.rotation
        tf_map_base.transform.rotation.x = float(q.x)
        tf_map_base.transform.rotation.y = float(q.y)
        tf_map_base.transform.rotation.z = float(q.z)
        tf_map_base.transform.rotation.w = float(q.w)

        self.tf_broadcaster.sendTransform(tf_map_base)

    def action_callback(self, msg):
        action = msg.data.strip()
        if action not in self.action_names:
            self.get_logger().warn(f"Invalid action: {action}")
            return
        self.pending_discrete_action = action
        self.pending_continuous_cmd = None

    def cmd_vel_callback(self, msg):
        self.pending_continuous_cmd = {
            "forward_velocity": float(msg.linear.x),
            "rotation_velocity": float(msg.angular.z),
        }
        self.pending_discrete_action = None

    def control_loop(self):
        collided = False

        if self.pending_discrete_action is not None:
            collided = self.execute_discrete_action(self.pending_discrete_action)
            self.pending_discrete_action = None
        elif self.pending_continuous_cmd is not None:
            collided = self.execute_continuous_action(self.pending_continuous_cmd)

        self.last_collided = collided
        self.pub_collision.publish(String(data=str(collided).lower()))
        self.publish_current_observation()

    def execute_discrete_action(self, action):
        discrete_action = self.agent.agent_config.action_space[action]
        did_collide = False

        if self.agent.controls.is_body_action(discrete_action.name):
            did_collide = self.agent.controls.action(
                self.agent.scene_node,
                discrete_action.name,
                discrete_action.actuation,
                apply_filter=True,
            )
        else:
            for _, sensor in self.agent.sensors.items():
                habitat_sim.errors.assert_obj_valid(sensor)
                self.agent.controls.action(
                    sensor.object,
                    discrete_action.name,
                    discrete_action.actuation,
                    apply_filter=False,
                )

        for _frame in range(FRAME_SKIP):
            self.sim.step_physics(self.time_step)

        return did_collide

    def execute_continuous_action(self, cmd):
        self.vel_control.linear_velocity = np.array([0.0, 0.0, -cmd["forward_velocity"]], dtype=np.float32)
        self.vel_control.angular_velocity = np.array([0.0, cmd["rotation_velocity"], 0.0], dtype=np.float32)
        collided = False

        for _ in range(FRAME_SKIP):
            agent_state = self.agent.state
            previous_rigid_state = habitat_sim.RigidState(
                utils.quat_to_magnum(agent_state.rotation), agent_state.position
            )

            target_rigid_state = self.vel_control.integrate_transform(
                self.time_step, previous_rigid_state
            )

            end_pos = self.sim.step_filter(
                previous_rigid_state.translation, target_rigid_state.translation
            )

            agent_state.position = end_pos
            agent_state.rotation = utils.quat_from_magnum(target_rigid_state.rotation)
            self.agent.set_state(agent_state)

            dist_moved_before_filter = float(
                np.linalg.norm(target_rigid_state.translation - previous_rigid_state.translation)
            )
            dist_moved_after_filter = float(
                np.linalg.norm(end_pos - previous_rigid_state.translation)
            )

            if (dist_moved_after_filter + EPS) < dist_moved_before_filter:
                collided = True

            self.sim.step_physics(self.time_step)

        return collided

    def publish_current_observation(self):
        obs = self.sim.get_sensor_observations()
        stamp = self.get_clock().now().to_msg()

        if "color_sensor" in obs:
            self.pub_rgb.publish(rgb_to_msg(obs["color_sensor"], stamp, FRAME_ID_CAMERA))

        if "depth_sensor" in obs:
            self.pub_depth.publish(depth_to_msg(obs["depth_sensor"].squeeze(), stamp, FRAME_ID_CAMERA))

        self.camera_info_msg.header.stamp = stamp
        self.pub_info.publish(self.camera_info_msg)
        self.pub_pose.publish(self.build_pose_msg(stamp))
        self.publish_dynamic_tfs(stamp)

    def build_pose_msg(self, stamp):
        state = self.agent.get_state()
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = FRAME_ID_MAP
        pose_msg.pose.position.x = float(state.position[0])
        pose_msg.pose.position.y = float(state.position[1])
        pose_msg.pose.position.z = float(state.position[2])
        q = state.rotation
        pose_msg.pose.orientation.x = float(q.x)
        pose_msg.pose.orientation.y = float(q.y)
        pose_msg.pose.orientation.z = float(q.z)
        pose_msg.pose.orientation.w = float(q.w)
        return pose_msg

    def destroy_node(self):
        try:
            self.sim.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = HabitatNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()