#!/usr/bin/env python3
"""The precomputed exploration schedule: the runner, the movers, and the post-scan trigger.

GA-441. The feed can drive a schedule built offline from the navmesh instead of sampling random
navigable goals. Three things must hold and none of them is obvious from reading the code:

  1. the run tours the storey it is STANDING on, and refuses rather than touring another;
  2. every lap is the SAME trajectory, so a difference between laps is a difference in the world;
  3. the trigger fires ONCE per completed 360 degree scan, and a hook that raises stops the run.

The stubs are installed inside each test and removed after it, so they cannot poison a pytest
session that also imports the real habitat_sim.

Run: python3 test_ga441_scheduled_tour.py   (or under pytest)
"""
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


@contextlib.contextmanager
def feed_host(**env):
    saved_mods = {k: sys.modules.get(k) for k in ("habitat_sim", "box_view", "config")}
    saved_env = {k: os.environ.get(k)
                 for k in list(env) + ["RUN_DIR", "OUT_DIR", "FEED_CTRL_PORT"]}
    saved_path = list(sys.path)
    tmp = tempfile.mkdtemp()
    try:
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
        fake_cfg.CFG_PATH = None
        sys.modules["config"] = fake_cfg
        os.environ["FEED_CTRL_PORT"] = "0"
        # STATS_DIR comes from OUT_DIR (habitat_feed_host.py:152); RUN_DIR is what the dynamic
        # dwell reads. Set both, or scan_events.jsonl lands in /tmp/graphapi_live and the assertions
        # below read another run's file.
        os.environ["RUN_DIR"] = tmp
        os.environ["OUT_DIR"] = tmp
        for k, v in env.items():
            os.environ[k] = v
        sys.path.insert(0, HERE)
        spec = importlib.util.spec_from_file_location(
            "_feed_sched_under_test", os.path.join(HERE, "habitat_feed_host.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod, tmp
    finally:
        sys.path[:] = saved_path
        for k, v in saved_mods.items():
            sys.modules.pop(k, None) if v is None else sys.modules.__setitem__(k, v)
        for k, v in saved_env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        sys.modules.pop("_feed_sched_under_test", None)


def schedule_file(tmp, heights=(1.21,), stops=3):
    """A scene schedule: one entry per storey, each a short there-and-back trajectory."""
    doc = {"scene_id": "test", "navmesh": "none", "schedule": []}
    for h in heights:
        traj = []
        for i in range(stops):
            traj.append({"xyz": [float(i), h, 0.0], "scan_deg": 360, "stop": i, "leg": i, "lap": 0})
            traj.append({"xyz": [i + 0.5, h, 0.0], "scan_deg": 0, "stop": None, "leg": i, "lap": 0})
        doc["schedule"].append({"height": h, "trajectory": traj, "waypoints": stops})
    p = os.path.join(tmp, "scene.schedule.json")
    json.dump(doc, open(p, "w"))
    return p


class FakeAgent:
    def __init__(self, y=1.21):
        self.state = types.SimpleNamespace(
            position=np.array([0.0, y, 0.0], dtype=np.float32),
            rotation=types.SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0))
        self.acts = []

    def get_state(self):
        return self.state

    def set_state(self, st):
        self.state = st

    def act(self, name):
        self.acts.append(name)


def fake_sim(arrive_after=1):
    """A sim whose follower reports arrival after `arrive_after` calls, then resets."""
    box = {"n": 0}

    def next_action(goal):
        box["n"] += 1
        return None if box["n"] % arrive_after == 0 else "move_forward"


    sim = types.SimpleNamespace()
    sim.make_greedy_follower = lambda *a, **k: types.SimpleNamespace(next_action_along=next_action)
    sim.step = lambda action: None
    return sim


def drive(tour, agent, limit=20000):
    for i in range(limit):
        tour.step(agent)
        if tour.house_done:
            return i
    return None


def test_the_storey_is_matched_to_where_the_agent_stands():
    with feed_host(FEED_SCHEDULE="x") as (mod, tmp):
        p = schedule_file(tmp, heights=(-1.59, 1.21))
        assert mod.load_schedule(p, 1.19)["height"] == 1.21
        assert mod.load_schedule(p, -1.55)["height"] == -1.59


def test_a_storey_that_is_not_in_the_file_is_refused():
    """Touring a storey the agent is not on would map one place and label it another."""
    with feed_host(FEED_SCHEDULE="x") as (mod, tmp):
        p = schedule_file(tmp, heights=(1.21,))
        try:
            mod.load_schedule(p, -1.59)
        except SystemExit as exc:
            assert "no storey within" in str(exc), exc
        else:
            raise AssertionError("a spawn on an unscheduled storey must refuse, not pick the nearest")


def test_every_lap_is_the_same_trajectory_and_the_run_ends():
    with feed_host(FEED_SCHEDULE="x") as (mod, tmp):
        sch = json.load(open(schedule_file(tmp, stops=3)))["schedule"][0]
        tour = mod.ScheduledTour(fake_sim(), sch, laps=3, move_fn="teleport")
        agent = FakeAgent()
        assert drive(tour, agent) is not None, "the schedule never completed"
        assert tour.lap == 3 and tour.scans_done == 9, (tour.lap, tour.scans_done)
        events = [json.loads(x) for x in open(os.path.join(tmp, "scan_events.jsonl"))]
        assert [e["lap"] for e in events] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert [e["stop"] for e in events] == [0, 1, 2] * 3, "the laps must visit the same stops"


def test_the_trigger_fires_once_per_scan_and_carries_the_context():
    with feed_host(FEED_SCHEDULE="x") as (mod, tmp):
        sch = json.load(open(schedule_file(tmp, stops=2)))["schedule"][0]
        seen = []
        mod.POST_SCAN_HOOK = "_hook_mod:record"
        hook = types.ModuleType("_hook_mod")
        hook.record = lambda ctx: seen.append(ctx) or {"moved": 1}
        sys.modules["_hook_mod"] = hook
        try:
            tour = mod.ScheduledTour(fake_sim(), sch, laps=2, move_fn="teleport")
            drive(tour, FakeAgent())
        finally:
            sys.modules.pop("_hook_mod", None)
        assert len(seen) == 4, f"one call per completed scan, got {len(seen)}"
        assert seen[0]["event"] == "scan_complete" and seen[0]["scan_deg"] == 360
        assert seen[0]["stops_total"] == 2 and seen[0]["laps_total"] == 2
        rec = [json.loads(x) for x in open(os.path.join(tmp, "scan_events.jsonl"))]
        assert all(e["hook_result"] == {"moved": 1} for e in rec), "the hook's answer is recorded"


def test_a_hook_that_raises_stops_the_run():
    """A dataset update that failed silently would leave laps claiming a change that never happened."""
    with feed_host(FEED_SCHEDULE="x") as (mod, tmp):
        sch = json.load(open(schedule_file(tmp, stops=1)))["schedule"][0]
        mod.POST_SCAN_HOOK = "_boom_mod:boom"
        boom = types.ModuleType("_boom_mod")

        def _raise(ctx):
            raise RuntimeError("dataset update failed")
        boom.boom = _raise
        sys.modules["_boom_mod"] = boom
        try:
            tour = mod.ScheduledTour(fake_sim(), sch, laps=1, move_fn="teleport")
            try:
                drive(tour, FakeAgent())
            except RuntimeError as exc:
                assert "dataset update failed" in str(exc)
            else:
                raise AssertionError("a raising hook must stop the run, not be swallowed")
        finally:
            sys.modules.pop("_boom_mod", None)


def test_navigate_measures_arrival_instead_of_trusting_the_follower():
    """`next_action_along` returns None for "arrived" AND for "no path". Distance decides."""
    with feed_host(FEED_SCHEDULE="x") as (mod, tmp):
        far = types.SimpleNamespace(next_action_along=lambda g: None)   # stops immediately
        sim = fake_sim()
        agent = FakeAgent()                                             # stands at (0, y, 0)
        st = {"frames": 0}
        assert mod._move_navigate(sim, agent, [0.0, 1.21, 0.0], far, st) == "arrived"
        st = {"frames": 0}
        assert mod._move_navigate(sim, agent, [9.0, 1.21, 0.0], far, st) == "unreachable", \
            "a follower that stops 9 m short has NOT arrived"
        assert "short" in st["why"], st

        # and a leg that never finishes ends on the cap rather than running for ever
        never = types.SimpleNamespace(next_action_along=lambda g: "move_forward")
        st = {"frames": mod.REVISIT_MAX_FRAMES}
        assert mod._move_navigate(sim, agent, [9.0, 1.21, 0.0], never, st) == "timeout"


def test_the_movers_are_selectable_and_teleport_snaps_to_the_navmesh():
    with feed_host(FEED_SCHEDULE="x") as (mod, tmp):
        assert set(mod.MOVERS) >= {"navigate", "teleport"}

        # TELEPORT SNAPS TO THE NAVMESH. A schedule point comes from a 5 cm raster and can sit off
        # the walkable surface; placing the agent there leaves it inside a wall.
        agent = FakeAgent()
        sim = fake_sim()
        sim.pathfinder = types.SimpleNamespace(
            is_loaded=True, snap_point=lambda p: np.array([p[0] + 0.05, p[1], p[2]], np.float32))
        st = {"frames": 0}
        assert mod._move_teleport(sim, agent, [3.0, 1.21, 4.0], None, st) == "arrived"
        assert abs(float(agent.get_state().position[0]) - 3.05) < 1e-4, agent.get_state().position

        # A SNAP THAT MOVES THE AGENT FURTHER THAN THE ARRIVAL TOLERANCE IS A REFUSAL, not a
        # silent relocation to somewhere else entirely.
        sim.pathfinder = types.SimpleNamespace(
            is_loaded=True, snap_point=lambda p: np.array([p[0] + 9.0, p[1], p[2]], np.float32))
        st = {"frames": 0}
        assert mod._move_teleport(sim, agent, [3.0, 1.21, 4.0], None, st) == "unreachable"
        assert "9.0" in st["why"], st
        # a stop the agent already stands on is reached with no snap needed
        sim.pathfinder = types.SimpleNamespace(is_loaded=False)
        st = {"frames": 0}
        assert mod._move_teleport(sim, agent, [0.0, 1.21, 0.0], None, st) == "arrived"


def test_an_unreachable_point_is_skipped_rather_than_holding_the_run():
    """The follower answers None for "arrived" and for "no path"; a schedule point can miss the navmesh."""
    with feed_host(FEED_SCHEDULE="x") as (mod, tmp):
        sch = json.load(open(schedule_file(tmp, stops=2)))["schedule"][0]
        never = types.SimpleNamespace()
        never.make_greedy_follower = lambda *a, **k: types.SimpleNamespace(
            next_action_along=lambda goal: "move_forward")     # never arrives
        never.step = lambda action: None
        tour = mod.ScheduledTour(never, sch, laps=1, move_fn="navigate")
        assert drive(tour, FakeAgent()) is not None, "a stuck leg must not hold the run for ever"
        assert tour.scans_done == 0, "nothing was reached, so nothing was scanned"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
