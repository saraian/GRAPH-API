#!/usr/bin/env python3
"""Self-check for the habitat_feed_host control server (no habitat needed).

Stubs habitat_sim/config/box_view, imports the module, starts the HTTP server
on a free port and exercises every endpoint the bridge proxies.
Run: python3 test_ctrl_server.py
"""
import importlib.util
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
# GA-52: habitat_feed_host logs WHICH config it loaded, so it imports CFG_PATH beside CFG.
# None is the honest stub value — it is what config.py returns when no file was read, and it
# exercises the warning branch rather than the quiet one.
fake_cfg.CFG_PATH = None
sys.modules["config"] = fake_cfg

os.environ["FEED_CTRL_PORT"] = "0"  # kernel-assigned free port
import habitat_feed_host as h  # noqa: E402


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2) as r:
        return json.loads(r.read().decode())


def test_ctrl_server():
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

    # Placed AFTER the exact-equality check above: /set_config MERGES, so keys set by the
    # block below would still be in the dict that assertion compares against.
    # --- the visibility gate, resolved from what the HOST sends -----------------------------
    # GA-56 part 4. The control server hands /set_config values through as query STRINGS —
    # "true", "5" — never as the bool and int a yaml would give. visibility() is the only
    # thing that coerces them, and it is also where the strict toggle is applied, so an
    # uncoerced "false" is truthy and the switch silently does nothing.
    #
    # The real function, not the stub. `fake_cfg` above replaces the config module with a
    # two-key namespace so habitat_feed_host can import it; asserting against that would
    # test the stub and prove nothing about the run. This loads config.py from disk and
    # calls ITS visibility, which resolves against the real _DEFAULTS.
    _spec = importlib.util.spec_from_file_location(
        "real_config", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "src", "perception_module", "config.py"))
    real_config = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(real_config)

    cfg = get(port, "/set_config?strict_visibility=true&min_visible_points=5")["config"]
    assert cfg["strict_visibility"] == "true", cfg   # a string, not a bool
    assert cfg["min_visible_points"] == "5", cfg     # a string, not an int

    # The strict toggle alone must bite. min_visible_points is seeded at startup, so it is
    # never absent, and a "default when missing" made this switch a no-op on both overlays.
    assert real_config.visibility(cfg)[0] == 5, "strict must raise the floor to 5"
    off = get(port, "/set_config?strict_visibility=false&min_visible_points=1")["config"]
    assert real_config.visibility(off)[0] == 1, "not strict must leave the value alone"
    assert real_config.visibility(
        get(port, "/set_config?strict_visibility=true")["config"])[0] == 5
    assert real_config.visibility(
        get(port, "/set_config?strict_visibility=false")["config"])[0] == 1

    # Strict is a FLOOR, not a fixed value: it may raise the count and never lower it.
    high = get(port, "/set_config?strict_visibility=true&min_visible_points=8")["config"]
    assert real_config.visibility(high)[0] == 8, "strict must not lower an explicit 8"

    # Garbage must never raise. These arrive from a URL a person typed.
    # /set_config MERGES rather than replaces, so strict_visibility is still "true" from the
    # call above. It is set explicitly here: a test that depends on state left by an earlier
    # request reads as an assertion about junk handling and is really an assertion about
    # request order, and it breaks when someone inserts a line between the two.
    junk = get(port, "/set_config?min_visible_points=junk&depth_tol_abs=nonsense"
                     "&strict_visibility=false")["config"]
    assert real_config.visibility(junk)[0] == 1, "unparsable count falls back, does not raise"
    assert real_config.visibility(junk)[1] == 0.10, "unparsable tolerance falls back"
    # and the same junk under strict still resolves to the floor rather than raising
    strict_junk = dict(junk, strict_visibility="true")
    assert real_config.visibility(strict_junk)[0] == 5

    # No live state at all resolves entirely from the configuration.
    assert real_config.visibility()[1:] == (0.10, 0.05)
    # ----------------------------------------------------------------------------------------

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
    # A pytest-collectable name AND a script entry. Named `demo` before, which pytest does not
    # collect -- so `pytest lost3dsg/test/` reported success over an empty collection while this
    # file's assertions never ran. An empty collection reporting green is the defect, not the
    # layout: it is a gate that cannot fail.
    test_ctrl_server()
