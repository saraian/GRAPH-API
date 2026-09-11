#!/usr/bin/env python3
"""The rviz schedule overlay: the producer and the consumer must agree on every name.

GA-479. `habitat_feed_host.py::schedule_overlay` builds the schedule payload; the ROS node
`habitat_feed_node.py::_publish_schedule` reads it and draws the markers. They are two files in two
processes with no shared type, so a rename on either side yields an EMPTY rviz display and no error
anywhere -- the marker builder catches its own exceptions on purpose, so that a drawing failure
cannot cost the run its frames.

Nothing else can catch it: no test can import the ROS message types on the host, and the run itself
reports the empty display as a silence. So the check is structural, on the source.

Run: python3 test_ga479_schedule_markers.py   (or under pytest)
"""
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
HOST = os.path.join(HERE, "habitat_feed_host.py")
NODE = os.path.join(HERE, "..", "src", "perception_module", "habitat_feed_node.py")
RVIZ = os.path.join(HERE, "live.rviz")

TOPIC = "/schedule_markers"


def _fn(path, name):
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{os.path.basename(path)} has no {name}()")


def _overlay_keys():
    """-> the keys schedule_overlay() puts in its payload."""
    fn = _fn(HOST, "schedule_overlay")
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    raise AssertionError("schedule_overlay() returns no dict literal")


def _read_keys():
    """-> the payload keys _publish_schedule() reads, from `sched.get(...)` and `s["..."]`."""
    fn = _fn(NODE, "_publish_schedule")
    keys = set()
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sched" and node.args
                and isinstance(node.args[0], ast.Constant)):
            keys.add(node.args[0].value)
    return keys


def test_every_key_the_node_reads_is_produced():
    produced, read = _overlay_keys(), _read_keys()
    assert read, "the marker builder reads no key -- the parser found nothing to check"
    missing = read - produced
    assert not missing, (
        f"habitat_feed_node.py::_publish_schedule reads {sorted(missing)}, which "
        f"habitat_feed_host.py::schedule_overlay does not produce (it produces {sorted(produced)})")


def test_stop_fields_agree():
    """The per-stop dict is built in one file and indexed in the other."""
    fn = _fn(NODE, "_publish_schedule")
    src = ast.unparse(fn)
    for field in ("xyz", "order"):
        assert f'"{field}"' in src or f"'{field}'" in src, \
            f"the marker builder no longer reads the stop field {field!r}"
    host = open(HOST).read()
    for field in ("xyz", "order"):
        assert f'"{field}"' in host, f"schedule_overlay() no longer emits the stop field {field!r}"


def test_the_frame_carries_the_schedule():
    """The payload has to REACH the node. It rides on the TCP frame, not on bev_data.json."""
    fn = _fn(HOST, "main")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and node.targets
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "frame"
                and isinstance(node.value, ast.Dict)):
            keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
            assert "schedule" in keys, (
                "the frame sent to the ROS side carries no 'schedule' key, so "
                "_publish_schedule() is never called and the rviz display stays empty")
            return
    raise AssertionError("main() builds no frame dict")


def test_the_node_publishes_on_the_topic_rviz_subscribes_to():
    node_src = open(NODE).read()
    assert TOPIC in node_src, f"habitat_feed_node.py does not name {TOPIC}"
    rviz = open(RVIZ).read()
    assert TOPIC in rviz, f"live.rviz has no display on {TOPIC}"


def test_the_rviz_display_is_transient_local():
    """The schedule is published ONCE. A Volatile subscriber that starts later sees nothing."""
    import yaml
    doc = yaml.safe_load(open(RVIZ))
    displays = doc["Visualization Manager"]["Displays"]
    hits = [d for d in displays if (d.get("Topic") or {}).get("Value") == TOPIC]
    assert len(hits) == 1, f"live.rviz has {len(hits)} displays on {TOPIC}, expected 1"
    d = hits[0]
    assert d["Topic"]["Durability Policy"] == "Transient Local", (
        f"the {TOPIC} display is {d['Topic']['Durability Policy']}; the publisher sends the "
        "schedule once, before rviz finishes starting, so a Volatile display stays empty")
    assert d["Enabled"] is True, "the schedule display is off by default"


def test_the_launcher_hands_rviz_the_config_that_has_the_display():
    """habitat_launch.py started rviz2 with no -d, so live.rviz was never opened by a run."""
    launch = open(os.path.join(HERE, "..", "launch", "habitat_launch.py")).read()
    assert "RVIZ_CONFIG" in launch, "habitat_launch.py ignores RVIZ_CONFIG, so it opens no layout"
    assert "'-d'" in launch or '"-d"' in launch, "habitat_launch.py passes no -d to rviz2"
    stack = open(os.path.join(HERE, "live_stack_container.sh")).read()
    assert "RVIZ_CONFIG=" in stack and "live.rviz" in stack, \
        "live_stack_container.sh does not point RVIZ_CONFIG at live.rviz"


def test_the_marker_builder_cannot_kill_the_feed():
    fn = _fn(NODE, "_publish_schedule")
    assert any(isinstance(n, ast.Try) for n in fn.body), \
        "_publish_schedule() has no try/except; a bad payload would stop the frame handler"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    print(json.dumps({"failed": fails}))
    raise SystemExit(1 if fails else 0)
