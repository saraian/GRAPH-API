#!/usr/bin/env python3
"""Self-check for the revisit (`goto`) state machine (no habitat needed).

Drives Tour.step through a revisit with a stub follower, because the two failure modes this
feature exists to prevent are both SILENT in the simulator: an unreachable target reads as a
reach, and an off-storey target livelocks against the floor guard instead of raising.

Run: python3 test_revisit.py
"""
import datetime
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
import habitat_feed_host as h  # noqa: E402


class FakeAgent:
    """Records actions and reports a position the test controls."""

    def __init__(self, position=(0.0, 0.0, 0.0)):
        self.position = np.asarray(position, dtype=np.float64)
        self.actions = []

    def get_state(self):
        return types.SimpleNamespace(position=self.position)

    def act(self, name):
        self.actions.append(name)


def make_tour(agent, actions):
    """A Tour with its simulator and follower stubbed.

    `actions` is the sequence next_action_along returns; None means the follower has stopped,
    which is exactly the ambiguity under test.
    """
    tour = h.Tour.__new__(h.Tour)          # bypass __init__: it needs a real simulator
    steps = []
    tour.sim = types.SimpleNamespace(step=steps.append)
    tour.steps = steps
    seq = list(actions)
    tour.follower = types.SimpleNamespace(
        next_action_along=lambda goal: seq.pop(0) if seq else None)
    tour.revisit = None
    tour.last_revisit = None
    tour.revisits_requested = tour.revisits_reached = tour.revisits_failed = 0
    return tour


def test_reaches_and_scans():
    """Arriving at the target scans, then reports a reach."""
    agent = FakeAgent((1.0, 0.0, 2.0))
    tour = make_tour(agent, ["move_forward", "move_forward"])
    tour.start_revisit(np.array([1.0, 0.0, 2.0]), (1.0, 2.0, 0.0), scan_frames=3,
                       resume=False, snap_distance_m=0.0)

    tour.step(agent)                       # walks
    tour.step(agent)                       # walks
    assert tour.steps == ["move_forward", "move_forward"], tour.steps
    assert tour.revisit.frames_travelled == 2

    tour.step(agent)                       # follower returns None; agent IS at the target
    assert tour.revisit.phase == h.RevisitState.SCAN
    for _ in range(3):
        tour.step(agent)
    tour.step(agent)                       # scan exhausted -> closes
    assert agent.actions == ["turn_left"] * 3, agent.actions
    assert tour.revisit is None
    assert tour.last_revisit["outcome"] == "reached", tour.last_revisit
    assert tour.revisits_reached == 1 and tour.revisits_failed == 0


def test_unreachable_is_not_a_reach():
    """THE POINT OF THIS FILE. The follower says None for 'arrived' AND for 'no path'.

    The tour branch treats both as a reach, so a target behind a closed door would be reported
    as observed. Distance is what separates them.
    """
    agent = FakeAgent((0.0, 0.0, 0.0))     # never moves
    tour = make_tour(agent, [])            # follower says None immediately
    tour.start_revisit(np.array([9.0, 0.0, 9.0]), (9.0, 9.0, 0.0), scan_frames=3,
                       resume=False, snap_distance_m=0.0)

    tour.step(agent)
    assert tour.revisit is None, "an unreachable target must close the revisit"
    assert tour.last_revisit["outcome"] == "unreachable", tour.last_revisit
    assert agent.actions == [], "nothing was observed, so nothing may be scanned"
    assert tour.revisits_reached == 0 and tour.revisits_failed == 1


def test_timeout_bounds_a_livelock():
    """A walk that never ends must end itself: the floor guard can pull the agent back forever."""
    agent = FakeAgent((0.0, 0.0, 0.0))
    tour = make_tour(agent, ["move_forward"] * (h.REVISIT_MAX_FRAMES + 10))
    tour.start_revisit(np.array([5.0, 0.0, 5.0]), (5.0, 5.0, 0.0), scan_frames=1,
                       resume=False, snap_distance_m=0.0)

    for _ in range(h.REVISIT_MAX_FRAMES + 1):
        tour.step(agent)
    assert tour.revisit is None, "the revisit must not outlive REVISIT_MAX_FRAMES"
    assert tour.last_revisit["outcome"] == "timeout", tour.last_revisit


def test_resume_restores_the_tour_index():
    """Resuming must not consume a waypoint: the tour's own scan advances _tour_i, this must not."""
    agent = FakeAgent((1.0, 0.0, 1.0))
    tour = make_tour(agent, [])
    tour._tour = [np.zeros(3)] * 10
    tour._tour_i = 4
    tour._tour_scan = 0
    tour._dwelling = False
    tour._dwell_frames = 0

    tour.start_revisit(np.array([1.0, 0.0, 1.0]), (1.0, 1.0, 0.0), scan_frames=1,
                       resume=True, snap_distance_m=0.0)
    tour._tour_i = 7                       # the tour moved on while the revisit was queued
    tour.step(agent)                       # arrives -> scan
    tour.step(agent)                       # scans
    tour.step(agent)                       # closes

    assert tour._tour_i == 4, f"expected the saved index back, got {tour._tour_i}"
    assert tour.last_revisit["resumed"] is True


def test_frames_convert_round_trip():
    """A goto arrives in ROS. ROS z IS habitat y, and getting that backwards picks a wrong floor."""
    ros = (1.5, -2.0, 0.75)
    hab = h.ros_to_habitat(ros)
    assert abs(float(hab[1]) - ros[2]) < 1e-9, "ROS z must equal habitat y"
    back, _ = h.habitat_pose_to_ros(hab)
    assert np.allclose(back, ros), (back, ros)


if __name__ == "__main__":
    test_reaches_and_scans()
    test_unreachable_is_not_a_reach()
    test_timeout_bounds_a_livelock()
    test_resume_restores_the_tour_index()
    test_frames_convert_round_trip()
    print("test_revisit: OK |",
          datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
