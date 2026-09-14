#!/usr/bin/env python3
"""Replay a rosbag TF stream while removing conflicting global-map frames.

The TIAGo recordings contain both the normal robot/camera transforms and a
recorded global transform (``map -> odom``).  A fresh RTAB-Map mapping run must
receive the former, but it must be the only publisher of the latter.  The bag
launcher remaps the recorded TF topics to ``/bag/tf`` and ``/bag/tf_static``;
this node republishes the safe transforms on the standard ``/tf`` topics.
"""

from __future__ import annotations

import os
from typing import Iterable

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from tf2_msgs.msg import TFMessage


def _normalise_frame(frame: str) -> str:
    return str(frame or "").strip().lstrip("/")


def _frames_from_environment() -> set[str]:
    raw = os.environ.get("TIAGO_BAG_TF_DROP_FRAMES", "map")
    # Accept both comma-separated and whitespace-separated values so the
    # launcher can be used comfortably from a shell.
    values = raw.replace(",", " ").split()
    frames = {_normalise_frame(value) for value in values}
    frames.discard("")
    return frames or {"map"}


def _qos(*, reliable: bool, transient_local: bool) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=100,
        reliability=(
            ReliabilityPolicy.RELIABLE
            if reliable
            else ReliabilityPolicy.BEST_EFFORT
        ),
        durability=(
            DurabilityPolicy.TRANSIENT_LOCAL
            if transient_local
            else DurabilityPolicy.VOLATILE
        ),
    )


class TfFilterRelay(Node):
    """Forward robot/camera TF and discard transforms touching configured frames."""

    def __init__(self) -> None:
        super().__init__("tiago_bag_tf_filter")
        self._drop_frames = _frames_from_environment()
        self._dropped_dynamic = 0
        self._dropped_static = 0
        self._static_by_edge: dict[tuple[str, str], object] = {}

        dynamic_input_qos = _qos(reliable=False, transient_local=False)
        # RTAB-Map and the Graph API use reliable TF subscriptions.  A
        # reliable publisher is compatible with both reliable and best-effort
        # subscribers, whereas a best-effort publisher would be invisible to
        # those consumers.
        dynamic_output_qos = _qos(reliable=True, transient_local=False)
        static_qos = _qos(reliable=True, transient_local=True)

        self._dynamic_pub = self.create_publisher(
            TFMessage, "/tf", dynamic_output_qos
        )
        self._static_pub = self.create_publisher(
            TFMessage, "/tf_static", static_qos
        )
        self.create_subscription(
            TFMessage, "/bag/tf", self._on_dynamic, dynamic_input_qos
        )
        self.create_subscription(
            TFMessage, "/bag/tf_static", self._on_static, static_qos
        )
        self.get_logger().info(
            "Forwarding /bag/tf -> /tf and /bag/tf_static -> /tf_static; "
            f"dropping transforms touching frames: {sorted(self._drop_frames)}"
        )

    def _is_conflicting(self, transform: object) -> bool:
        parent = _normalise_frame(transform.header.frame_id)
        child = _normalise_frame(transform.child_frame_id)
        return parent in self._drop_frames or child in self._drop_frames

    @staticmethod
    def _message(transforms: Iterable[object]) -> TFMessage:
        message = TFMessage()
        message.transforms = list(transforms)
        return message

    def _on_dynamic(self, message: TFMessage) -> None:
        safe = []
        for transform in message.transforms:
            if self._is_conflicting(transform):
                self._dropped_dynamic += 1
            else:
                safe.append(transform)
        if safe:
            self._dynamic_pub.publish(self._message(safe))
        if self._dropped_dynamic and self._dropped_dynamic % 100 == 0:
            self.get_logger().info(
                f"Dropped {self._dropped_dynamic} conflicting dynamic transforms"
            )

    def _on_static(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if self._is_conflicting(transform):
                self._dropped_static += 1
                continue
            key = (
                _normalise_frame(transform.header.frame_id),
                _normalise_frame(transform.child_frame_id),
            )
            self._static_by_edge[key] = transform

        # Keep the full safe static set in the transient-local message.  This
        # makes the relay correct even if a consumer joins after the first bag
        # message, while still allowing the bag to contain multiple TF chunks.
        if self._static_by_edge:
            self._static_pub.publish(self._message(self._static_by_edge.values()))
        if self._dropped_static and self._dropped_static % 10 == 0:
            self.get_logger().info(
                f"Dropped {self._dropped_static} conflicting static transforms"
            )


def main() -> None:
    rclpy.init()
    node = TfFilterRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
