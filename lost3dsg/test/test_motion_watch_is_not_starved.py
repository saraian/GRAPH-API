"""The motion watch must not share a callback group with the perception cycle.

WHAT THIS CATCHES. `joint_callback` is the ONLY publisher of /robot_movement_detected.
object_manager_6 latches `_moving_since` on its "moving" edge and opens the latch only on its
"stopped" edge; while the latch is closed, `_try_process` discards every matched
description/bbox pair as "observed during motion", so nothing reaches the object store.

The 1 Hz motion timer was created on `perception_cb_group`, a MutuallyExclusiveCallbackGroup
shared with the 0.5 s perception cycle -- and that cycle blocks for a whole detect plus VLM
round trip. MEASURED on 20260911_181716_hm3d_00861: the timer evaluated FOUR times in 18
minutes. It caught one moving edge at +29 s, never saw a stationary sample again, never
published the stop, and the run discarded 426 of 426 pairs and ended with 5 objects from 455
detections. 20260911_173938_hm3d_00861 is the control: same single latch, closed at +535 s,
184 objects.

Exercised by CALLING `_create_timers` against a recorder, so the assertion is about the wiring
the node really builds rather than about the text of the file.
"""
# ruff: noqa: E402, I001  -- rosstub must be installed before the module under test is imported
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src", "perception_module"))
import rosstub; rosstub.install()  # noqa: E702

import perception_2


class _Recorder:
    """Enough of the node for `_create_timers`, and nothing else."""

    def __init__(self, with_queue):
        self.sensor_cb_group = "sensor"
        self.perception_cb_group = "perception"
        self.motion_cb_group = "motion"
        self.frame_queue = object() if with_queue else None
        self.timers = []

    def __getattr__(self, name):
        # The timers are created from bound methods this recorder does not have. Named
        # placeholders, so a timer's identity in `timers` is still the method it would run.
        if name.startswith("_") or name.endswith("_callback"):
            fn = lambda *a, **k: None            # noqa: E731
            fn.__name__ = name
            return fn
        raise AttributeError(name)

    def create_timer(self, period, cb, callback_group=None):
        self.timers.append((period, getattr(cb, "__name__", str(cb)), callback_group))

    def create_publisher(self, *a, **k):
        return object()

    def get_logger(self):
        class _L:
            def info(self, *a, **k):
                pass
        return _L()


for with_queue in (False, True):
    r = _Recorder(with_queue)
    perception_2.DetectObjectsNode._create_timers(r)
    by_name = {name: (period, grp) for period, name, grp in r.timers}

    assert "joint_callback" in by_name, f"the motion watch timer is gone: {by_name}"
    assert "_perception_timer_callback" in by_name, by_name
    motion_period, motion_grp = by_name["joint_callback"]
    _, cycle_grp = by_name["_perception_timer_callback"]

    assert motion_grp != cycle_grp, (
        "the motion watch shares a callback group with the perception cycle; on a mutually "
        "exclusive group it is serialised behind the detect + VLM round trip and the 'stopped' "
        "edge is never published")
    # It must also not ride the SENSOR group: that one is reentrant, and two concurrent runs of
    # the watch would difference a joint position against one the other had just replaced.
    assert motion_grp != r.sensor_cb_group, (
        "the motion watch is on the reentrant sensor group; it mutates last_joint_positions "
        "and must not run concurrently with itself")
    assert motion_period <= 1.0, f"the motion watch period slipped to {motion_period}s"

# And the group the node hands it must be a real mutually-exclusive one, not a leftover string.
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                        "src", "perception_module", "perception_2.py")).read()
assert "self.motion_cb_group = MutuallyExclusiveCallbackGroup()" in src, \
    "motion_cb_group is not built as a MutuallyExclusiveCallbackGroup"

print("OK: the motion watch runs on its own mutually-exclusive group, "
      "off both the perception cycle and the reentrant sensor group")
