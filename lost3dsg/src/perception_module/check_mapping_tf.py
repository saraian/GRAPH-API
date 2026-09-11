#!/usr/bin/env python3
"""Read-only preflight: RTAB-Map must be the sole publisher parenting odom."""
import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


class MappingTFCheck(Node):
    def __init__(self, odom_frame='odom'):
        super().__init__('found_mapping_tf_check')
        self.odom_frame = odom_frame.lstrip('/')
        self.parents = set()
        self.seen_odometry_tf = False
        self.create_subscription(TFMessage, '/tf', self.observe,
            QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(TFMessage, '/tf_static', self.observe,
            QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def observe(self, msg):
        for tf in msg.transforms:
            parent = tf.header.frame_id.lstrip('/')
            child = tf.child_frame_id.lstrip('/')
            if child == self.odom_frame:
                self.parents.add(parent)
            if parent == self.odom_frame:
                self.seen_odometry_tf = True

    def result(self):
        if self.parents:
            return 1, (f'RTAB-Map cannot own {self.odom_frame}: TF already has parent(s) '
                       f'{sorted(self.parents)}. For FOUND mapping, stop PAL localization '
                       'and slam modules on the robot (pal module stop localization; '
                       'pal module stop slam), or stop the other mapping publisher, '
                       'then retry. Keep robot odometry and camera drivers running.')
        if not self.seen_odometry_tf:
            return 2, (f'No TF from {self.odom_frame} was received; ownership check is '
                       'inconclusive. Check robot odometry and DDS before starting mapping.')
        return 0, f'Observed {self.odom_frame} odometry TF with no existing parent publisher.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--odom-frame', default='odom')
    parser.add_argument('--seconds', type=float, default=5.0)
    args = parser.parse_args()
    rclpy.init()
    node = MappingTFCheck(args.odom_frame)
    try:
        deadline = time.monotonic()+args.seconds
        while time.monotonic() < deadline and not node.parents:
            rclpy.spin_once(node, timeout_sec=min(.1, max(0., deadline-time.monotonic())))
        code, message = node.result()
        print(message, flush=True)
        return code
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
