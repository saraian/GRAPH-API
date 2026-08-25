#!/usr/bin/env python3
"""Self-check for the habitat_feed_host control server (no habitat needed).

Stubs habitat_sim/config/box_view, imports the module, starts the HTTP server
on a free port and exercises every endpoint the bridge proxies.
Run: python3 test_ctrl_server.py
"""
import json
import os
import sys
import types
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# stub the sim-only imports before loading the module under test
fake_hab = types.ModuleType("habitat_sim")
fake_hab.agent = types.SimpleNamespace(ActionSpec=object, ActuationSpec=object,
                                       AgentConfiguration=object)
fake_hab.SensorType = types.SimpleNamespace(COLOR=0, DEPTH=1)
sys.modules["habitat_sim"] = fake_hab
fake_box = types.ModuleType("box_view")
fake_box.BOX_EDGES, fake_box.box_corners_map, fake_box.project_visible = [], None, None
sys.modules["box_view"] = fake_box
fake_cfg = types.ModuleType("config")
fake_cfg.CFG = {"habitat": {"single_floor": True, "floor_tolerance_m": 0.5}}
sys.modules["config"] = fake_cfg

os.environ["FEED_CTRL_PORT"] = "0"  # kernel-assigned free port
import habitat_feed_host as h  # noqa: E402


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
        return json.loads(r.read().decode())


def demo():
    httpd = h.ThreadingHTTPServer(("127.0.0.1", 0), h.CtrlHandler)
    h.threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    assert get(port, "/auto_mode?enabled=0") == {"success": True, "auto_mode": False}
    assert h.CTRL.auto_mode is False
    assert get(port, "/auto_mode?enabled=true")["auto_mode"] is True

    get(port, "/action?act=forward&amount=0.5")
    get(port, "/action?act=nav_goal&x=1.5&y=-2&z=0.1")
    assert list(h.CTRL.actions) == [("forward", {"amount": 0.5}),
                                    ("nav_goal", {"x": 1.5, "y": -2.0, "z": 0.1})]

    cfg = get(port, "/set_config?perceive_while_moving=true&seg=1")["config"]
    assert cfg == {"perceive_while_moving": "true", "seg": "1"}
    assert get(port, "/get_config")["config"] == cfg

    h.print("[feed] test line")
    assert "[feed] test line" in get(port, "/logs")["logs"]

    h.CTRL.bev = {"agent": {"x": 0}}
    assert get(port, "/bev_data") == {"agent": {"x": 0}}

    # frame.jpg: 503 without a frame, 200 with one
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/frame.jpg", timeout=2)
        raise AssertionError("expected 503")
    except urllib.error.HTTPError as e:
        assert e.code == 503
    h.CTRL.latest_jpeg = b"\xff\xd8fake"
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/frame.jpg", timeout=2) as r:
        assert r.read() == b"\xff\xd8fake"

    try:
        get(port, "/nope")
        raise AssertionError("expected 404")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    httpd.shutdown()
    print("test_ctrl_server: OK")


if __name__ == "__main__":
    demo()
