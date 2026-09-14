"""A motion edge must not erase geometry already captured for the active cycle."""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import object_manager_6 as om6  # noqa: E402
from world_model import wm  # noqa: E402


def observation(index):
    return NS(
        schema_version=1, run_id="run", producer_id="producer",
        capture_id="producer:1:0", capture_identity_kind="stamp_within_producer",
        capture_stamp=NS(sec=1, nanosec=0), camera_frame_id="camera",
        cycle_id="cycle", detection_index=index,
        observation_id=f"producer:cycle:{index}",
    )


def box(index):
    return NS(
        label="chair", observation=observation(index),
        x_min=float(index), x_max=float(index) + 0.5,
        y_min=0.0, y_max=0.5, z_min=0.0, z_max=0.5,
        has_orientation=False, has_bbox_2d=False,
        has_clip_embedding=False, has_fusion_voxels=False,
    )


def description(index):
    return NS(
        label="chair", observation=observation(index), color="unknown",
        material="unknown", description=f"chair {index}", crop_path="", status="ok",
    )


node = object.__new__(om6.ObjectManagerService)
node.exploration_mode = True
node.robot_has_moved = False
node._moving_since = None
node.tracking_step_counter = node.exploration_frame_counter = 0
node.last_room_check_time = 1e12
node._vlm_status_counts = {}
node.uncertain_objects = []
node.latest_bboxes = {}
node.latest_fov_volume = None
node.ga493_replay_capture = None
node.object_services = NS(log_both=lambda *args, **kwargs: None)
node.get_clock = lambda: NS(now=lambda: NS(to_msg=lambda: NS(sec=2, nanosec=0)))
node.room_manager = NS(
    current_room_id="room_1",
    assign_room_by_geometry=lambda value: "room_1",
    update_all_rooms_semantics=lambda objects: None,
)
node.tracking_activated_pub = node.kb_add_pub = rosstub.Any()
rows = []
node.decision_log = NS(write=lambda *args, **kwargs: rows.append((args, kwargs)))
proposals = []
node.filter_hook = NS(
    name="test",
    judge=lambda proposal: proposals.append(proposal) or NS(
        admitted=False, outcome="refused", reason="test", annotation={}, provisional=False),
)
node._room_frames_for = lambda room_id: []
node.check_tracking_transition = lambda *args: (False, None, 0.0)
node.delete_uncertain_objects = lambda volume: False
node.publish_kb_facts = lambda objects: []
node.publish_kb_relation_facts = lambda: []
node.flush_scan_summary = lambda frame_id=None: None
node._record_sighting = lambda obj, timestamp: None
node.update_spatial_relations = lambda: None
wm.persistent_perceptions.clear()

request = NS(
    bboxes=NS(
        header=NS(stamp=NS(sec=1, nanosec=0)), cycle_id="cycle",
        fov_x_max=0.0, fov_y_max=0.0, fov_z_max=0.0,
        boxes=[box(0), box(1)],
    ),
    descriptions=NS(descriptions=[description(0), description(1)]),
)

original_embedding = om6.get_embedding
calls = 0


def embedding(model, text):
    global calls
    calls += 1
    if calls == 1:
        om6.ObjectManagerService.movement_callback(node, NS(data=True))
        assert node.latest_bboxes == {}, "the live-view cache must be invalidated by motion"
    return None


om6.get_embedding = embedding
om6.EXPLORATION_FRAME_LIMIT = 0
try:
    response = om6.ObjectManagerService.object_tracking_callback(node, request, NS())
finally:
    om6.get_embedding = original_embedding

assert response.status == "success"
assert [p["observation"]["observation_id"] for p in proposals] == [
    "producer:cycle:0", "producer:cycle:1"
]
assert not [row for row in rows if row[0] and row[0][0] == "unpaired_description"]
print("OK: movement clears the live cache but both ObservationRef-keyed cycle entries survive")
