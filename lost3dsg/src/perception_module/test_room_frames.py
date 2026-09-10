#!/usr/bin/env python3
"""Runnable check for the room-frame capture gate (`python3 test_room_frames.py`).

The ontology side types each saved view of a room and takes a majority, so WHICH views
get saved decides the vote. One entering view is demonstrably not enough: in run
20260826_0808 the entering frame of room_0 typed as a corridor (doorway view) while the
room's contents were office furniture.

Guards the gate in `perception_utils.room_frame_due`, used by
`object_manager_6._room_frames_for`.
"""
from perception_utils import room_frame_due

STRIDE = 1.5
CAP = 5


def pose(x, y, stamp=None):
    return {"pose": {"x": x, "y": y}, "stamp": stamp}


def main():
    # entry: the first view of a room is always due, pose or not
    assert room_frame_due([], (0.0, 0.0), STRIDE, CAP) is True
    assert room_frame_due([], None, STRIDE, CAP) is True

    frames = [pose(0.0, 0.0)]

    # standing still / small shuffle -> not due (a burst from one spot would bias the vote)
    assert room_frame_due(frames, (0.0, 0.0), STRIDE, CAP) is False
    assert room_frame_due(frames, (1.0, 0.0), STRIDE, CAP) is False
    assert room_frame_due(frames, (0.0, 1.49), STRIDE, CAP) is False

    # a full stride of travel in any direction -> due
    assert room_frame_due(frames, (1.5, 0.0), STRIDE, CAP) is True
    assert room_frame_due(frames, (0.0, -1.5), STRIDE, CAP) is True
    assert room_frame_due(frames, (1.2, 1.2), STRIDE, CAP) is True   # diagonal, ~1.70 m

    # distance is measured from the LAST captured view, not the entry one
    two = [pose(0.0, 0.0), pose(3.0, 0.0)]
    assert room_frame_due(two, (3.4, 0.0), STRIDE, CAP) is False
    assert room_frame_due(two, (4.6, 0.0), STRIDE, CAP) is True

    # no current pose -> entry frame only, never a blind burst
    assert room_frame_due(frames, None, STRIDE, CAP) is False
    assert room_frame_due([{"pose": None}], None, STRIDE, CAP) is False

    # An unposed entry frame must not freeze the room at one view. perception_2 publishes
    # the agent pose at the END of a detection cycle, after the descriptions that trigger
    # the entry frame, so the first view is routinely saved before any pose arrives.
    # Anchoring travel on it left room_0 stuck at a single doorway view (run 20260826_0835).
    assert room_frame_due([{"pose": None}], (9.0, 9.0), STRIDE, CAP) is True

    # travel is measured from the last POSED view, not simply the last view
    mixed = [{"pose": None}, pose(0.0, 0.0), {"pose": None}]
    assert room_frame_due(mixed, (0.5, 0.0), STRIDE, CAP) is False
    assert room_frame_due(mixed, (2.0, 0.0), STRIDE, CAP) is True

    # the cap holds however far the robot travels
    full = [pose(float(i) * 3.0, 0.0) for i in range(CAP)]
    assert room_frame_due(full, (99.0, 0.0), STRIDE, CAP) is False
    assert room_frame_due(full[:-1], (99.0, 0.0), STRIDE, CAP) is True
    assert room_frame_due([], (0.0, 0.0), STRIDE, 0) is False

    # The same image is never a new view. Camera and pose streams tick at different
    # rates, so a travel-triggered capture can arrive while _latest_rgb is still the
    # frame already saved; frames are named by image stamp, so re-saving it overwrote
    # the earlier view (run 20260826_0849 logged 4 captures but left 3 files).
    seen = [pose(0.0, 0.0, stamp=100.0)]
    assert room_frame_due(seen, (9.0, 9.0), STRIDE, CAP, stamp=100.0) is False
    assert room_frame_due(seen, (9.0, 9.0), STRIDE, CAP, stamp=101.0) is True
    # a fresh image still does not licence a capture from the same spot
    assert room_frame_due(seen, (0.2, 0.0), STRIDE, CAP, stamp=101.0) is False
    # and the entry frame is still taken whatever the stamp
    assert room_frame_due([], (0.0, 0.0), STRIDE, CAP, stamp=100.0) is True

    print(f"room frame gate OK: entry + every {STRIDE} m, cap {CAP}, "
          "no-pose -> entry only, same image -> not a new view")


if __name__ == "__main__":
    main()
