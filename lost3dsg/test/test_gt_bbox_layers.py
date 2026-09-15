#!/usr/bin/env python3
"""Contract checks for the split ground-truth RViz layers.

The host-side feed imports Habitat, while the ROS-side publisher imports rclpy, so importing either
module in a lightweight source checkout is not a useful unit test. These checks validate the shared
wire/display contract and exercise the same label classification rules against representative HM3D
labels without starting either runtime.
"""
import ast
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.join(HERE, "habitat_feed_host.py")
NODE = os.path.join(HERE, "..", "src", "perception_module", "habitat_feed_node.py")
CONFIG = os.path.join(HERE, "..", "src", "perception_module", "config.yaml")
RVIZ = os.path.join(HERE, "live.rviz")

OBJECTS_TOPIC = "/ground_truth_bbox_objects"
STRUCTURAL_TOPIC = "/ground_truth_bbox_structural"


def _source(path):
    with open(path, encoding="utf-8") as stream:
        return stream.read()


def _structural_classifier(source):
    """Extract the host classifier with its configured module-level set for a small probe.

    Executing the complete host module would require Habitat-Sim and would open its control path.
    The classifier is deliberately self-contained, so compiling just its function is enough to
    verify the actual source logic rather than copying that logic into this test.
    """
    tree = ast.parse(source)
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef)
              and node.name == "_is_gt_structural_category")
    namespace = {"_GT_STRUCTURE_CATEGORIES": {
        "wall", "floor", "ceiling", "door", "doorway", "door frame", "window",
        "window frame", "window shutter", "bar", "air conditioner", "lamp",
        "shower wall", "shower floor",
        "shower ceiling", "shower door frame", "stairs", "stairs railing",
        "compound wall", "recessed wall", "wall panel", "ceiling dome",
    }}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), HOST, "exec"), namespace)
    return namespace["_is_gt_structural_category"]


def test_config_contains_the_structural_hm3d_labels():
    config = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    labels = set(config["perception"]["excluded_labels"])
    expected = {
        "door", "door frame", "window", "window frame", "window shutter",
        "wall", "wall panel", "shower wall", "shower floor", "shower ceiling",
        "shower door frame", "stairs", "stairs railing", "compound wall",
        "recessed wall", "ceiling dome", "bar", "air conditioner", "lamp",
    }
    assert expected <= labels


def test_gt_classifier_separates_architecture_from_real_objects():
    is_structural = _structural_classifier(_source(HOST))
    for label in (
        "door", "window", "window frame", "window shutter", "shower wall",
        "shower door frame", "stairs railing", "recessed wall", "ceiling dome",
        "bar", "air conditioner", "lamp",
    ):
        assert is_structural(label), label
    for label in ("chair", "bed", "bathroom cabinet", "glass"):
        assert not is_structural(label), label


def test_host_and_node_use_the_two_layer_contract():
    host = _source(HOST)
    node = _source(NODE)
    assert '"is_structural": _is_gt_structural_category(category)' in host
    assert '"is_structural", False' in node
    assert OBJECTS_TOPIC in node and STRUCTURAL_TOPIC in node
    assert "pub_ground_truth_bbox_objects" in node
    assert "pub_ground_truth_bbox_structural" in node
    assert "GT_BBOX_INCLUDE_STRUCTURE" not in host


def test_rviz_has_independent_transient_local_displays():
    config = yaml.safe_load(open(RVIZ, encoding="utf-8"))
    displays = config["Visualization Manager"]["Displays"]
    hits = {
        (display.get("Topic") or {}).get("Value"): display
        for display in displays
        if (display.get("Topic") or {}).get("Value") in {
            OBJECTS_TOPIC, STRUCTURAL_TOPIC,
        }
    }
    assert set(hits) == {OBJECTS_TOPIC, STRUCTURAL_TOPIC}
    assert hits[OBJECTS_TOPIC]["Enabled"] is True
    assert hits[STRUCTURAL_TOPIC]["Enabled"] is False
    for topic, display in hits.items():
        assert display["Topic"]["Durability Policy"] == "Transient Local", topic
        assert display["Topic"]["Reliability Policy"] == "Reliable", topic


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
