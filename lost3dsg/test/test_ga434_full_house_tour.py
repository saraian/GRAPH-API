#!/usr/bin/env python3
"""Self-check for the full-house tour (rule 73). No habitat needed.

Rule 73, owner 2026-09-10: a base run tours EVERY storey and teleports to the next one when a
storey is finished. Three things can go wrong and all three are checked here:

  1. the floor guard drags the agent back downstairs after a deliberate teleport, so the second
     storey is never toured while every log line still says the tour is progressing;
  2. a storey with no navigable point ends the tour instead of being skipped;
  3. the itinerary silently includes the storey the agent already stands on, touring it twice.

THE STUBS ARE INSTALLED INSIDE THE TEST AND REMOVED AFTER IT. A module-level stub of habitat_sim
stays in sys.modules for the rest of the pytest session and breaks unrelated files that import the
real thing.

Run: python3 test_ga434_full_house_tour.py   (or under pytest)
"""
import contextlib
import importlib.util
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


@contextlib.contextmanager
def feed_host(**env):
    """Import habitat_feed_host under stubs, then undo everything."""
    saved_mods = {k: sys.modules.get(k) for k in ("habitat_sim", "box_view", "config")}
    saved_env = {k: os.environ.get(k) for k in env}
    saved_path = list(sys.path)
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
        for k, v in env.items():
            os.environ[k] = v
        sys.path.insert(0, HERE)
        spec = importlib.util.spec_from_file_location(
            "_feed_host_under_test", os.path.join(HERE, "habitat_feed_host.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path[:] = saved_path
        for k, v in saved_mods.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.modules.pop("_feed_host_under_test", None)


class FakeAgent:
    def __init__(self, pos):
        self.state = types.SimpleNamespace(position=np.array(pos, dtype=np.float32))
        self.acts = []

    def get_state(self):
        return self.state

    def set_state(self, st):
        self.state = st

    def act(self, name):
        self.acts.append(name)


def fake_sim(floor_y):
    """A sim whose navigable points all sit on `box.y`, which the test moves."""
    box = types.SimpleNamespace(y=floor_y, n=0)

    def point():
        box.n += 1
        return np.array([box.n % 7 * 1.5, box.y, box.n % 5 * 1.7], dtype=np.float32)

    sim = types.SimpleNamespace()
    sim.pathfinder = types.SimpleNamespace(get_random_navigable_point=point, is_loaded=False)
    sim.make_greedy_follower = lambda *a, **k: types.SimpleNamespace(
        next_action_along=lambda goal: None)
    sim.get_agent = lambda i: FakeAgent([0.0, box.y, 0.0])
    sim.step = lambda action: None
    sim._box = box
    return sim


def build(mod, start_y, floors, tol=0.43):
    sim = fake_sim(start_y)
    tour = mod.Tour(sim, np.random.default_rng(7), n_points=3)
    guard = mod.FloorGuard(start_y, tol, "teleport")
    tour.bind_floors(floors, tol, guard)
    return sim, tour, guard


def test_itinerary_excludes_the_start_storey_and_is_ordered():
    with feed_host(FEED_TEST_TOUR="3", FEED_TOUR_ALL_FLOORS="1") as mod:
        _, tour, _ = build(mod, 1.35, [-1.59, 0.43, 1.35, 2.21])
        assert tour.floors_todo == [-1.59, 0.43, 2.21], tour.floors_todo
        assert tour.floor_order == [1.35]


def test_off_switch_tours_one_storey():
    with feed_host(FEED_TEST_TOUR="3", FEED_TOUR_ALL_FLOORS="0") as mod:
        _, tour, _ = build(mod, 1.35, [-1.59, 0.43, 1.35, 2.21])
        assert tour.floors_todo == []


def test_advance_teleports_replans_and_reanchors_the_guard():
    with feed_host(FEED_TEST_TOUR="3", FEED_TOUR_ALL_FLOORS="1") as mod:
        sim, tour, guard = build(mod, 1.35, [-1.59, 0.43, 1.35, 2.21])
        mod._spawn_point = lambda s, z, tries=4000: (
            setattr(s._box, "y", z) or np.array([1.0, z, 2.0], dtype=np.float32))
        # a position on the OLD storey, as one full storey of touring would have left behind
        agent = FakeAgent([3.0, 1.35, 4.0])
        guard.check(agent)
        assert guard.last_on_floor is not None

        assert tour._advance_floor(agent) is True
        assert abs(float(agent.get_state().position[1]) - (-1.59)) < 1e-6
        assert abs(tour.floor_y - (-1.59)) < 1e-6
        assert abs(guard.floor_y - (-1.59)) < 1e-6
        # THE TRAP: a stale on-floor position would teleport the agent back to 1.35.
        assert guard.last_on_floor is None
        assert guard.reanchors == 1
        assert tour._tour_i == 0 and len(tour._tour) == 3
        assert tour._tour_planned_total >= 3
        assert tour.floor_order == [1.35, -1.59]
        # and the guard does not correct on the new storey
        assert guard.check(agent) is False
        assert guard.corrections == 0


def test_a_storey_with_no_navigable_point_is_skipped_not_fatal():
    with feed_host(FEED_TEST_TOUR="3", FEED_TOUR_ALL_FLOORS="1") as mod:
        sim, tour, guard = build(mod, 1.35, [-1.59, 0.43, 1.35, 2.21])

        def spawn(s, z, tries=4000):
            if z == -1.59:
                raise SystemExit("no navigable point")
            s._box.y = z
            return np.array([1.0, z, 2.0], dtype=np.float32)

        mod._spawn_point = spawn
        agent = FakeAgent([3.0, 1.35, 4.0])
        assert tour._advance_floor(agent) is True
        assert abs(tour.floor_y - 0.43) < 1e-6, tour.floor_y
        assert tour.floors_todo == [2.21]


def test_the_house_ends_after_the_last_storey():
    with feed_host(FEED_TEST_TOUR="3", FEED_TOUR_ALL_FLOORS="1") as mod:
        sim, tour, guard = build(mod, 1.35, [1.35, 2.21])
        mod._spawn_point = lambda s, z, tries=4000: (
            setattr(s._box, "y", z) or np.array([1.0, z, 2.0], dtype=np.float32))
        agent = FakeAgent([3.0, 1.35, 4.0])
        assert tour._advance_floor(agent) is True
        assert tour._advance_floor(agent) is False
        assert tour.floor_order == [1.35, 2.21]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
