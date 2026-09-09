"""GA-10: on the exploration->tracking transition branch, a REFUSED update must reach the
admission seam (filter_hook.judge) instead of ending the detection with a bare `continue`."""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import object_manager_6 as om6  # noqa: E402
from world_model import wm  # noqa: E402

BOX = dict(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, z_min=0.0, z_max=1.0)


def run(update_ok):
    n = object.__new__(om6.ObjectManagerService)
    n.exploration_mode, n.robot_has_moved = True, False
    n.tracking_step_counter = n.exploration_frame_counter = 0
    n.last_room_check_time = 1e12
    n._vlm_status_counts, n.uncertain_objects, n.latest_bboxes = {}, [], {}
    n.object_services = NS(log_both=lambda *a, **k: None)
    n.get_logger = lambda: rosstub.Any()
    n.room_manager = NS(current_room_id="room_1", assign_room_by_geometry=lambda b: "room_1")
    n.tracking_activated_pub = n.kb_add_pub = rosstub.Any()
    n.decision_log = NS(write=lambda *a, **k: None)
    judged = []
    n.filter_hook = NS(name="t", judge=lambda p: judged.append(p) or NS(admitted=False, outcome="refused", reason="test", annotation={}))
    stored = NS(label="chair", object_id="obj_a", bbox=BOX)
    wm.persistent_perceptions.clear(); wm.persistent_perceptions.append(stored)  # noqa: E702
    n.check_tracking_transition = lambda *a: (True, stored, 1.0)
    n.modify_existing_object = lambda *a, **k: NS(success=update_ok, object_id="obj_a", message="refused")
    n.delete_uncertain_objects = lambda pov: False
    n.merge_duplicate_objects = lambda: False
    n.publish_kb_facts = lambda objs: []
    n.publish_kb_relation_facts = lambda: []
    n.flush_scan_summary = lambda frame_id=None: None
    n._record_sighting = lambda o, t: None
    n._note_update = lambda *a, **k: None
    n.update_spatial_relations = lambda: None
    n.room_manager.update_current_room_semantics = lambda objs: None
    for f in ("publish_persistent_bboxes", "publish_persistent_centroids", "publish_uncertain_bboxes",
              "publish_uncertain_centroids", "save_uncertain_objects", "save_persistent_perceptions"):
        setattr(om6, f, lambda *a, **k: None)
    box = NS(label="chair", has_orientation=False, has_bbox_2d=False, has_clip_embedding=False, **BOX)
    req = NS(bboxes=NS(header=NS(stamp=NS(sec=1, nanosec=0)), fov_x_max=0, fov_y_max=0, fov_z_max=0, boxes=[box]),
             descriptions=NS(descriptions=[NS(label="chair", color="", material="", description="", crop_path="", status="ok")]))
    resp = om6.ObjectManagerService.object_tracking_callback(n, req, NS())
    assert resp.tracking_mode_activated, "the transition must still fire"
    return judged


assert len(run(update_ok=False)) == 1, "a refused transition update must be offered to admission"
assert run(update_ok=True) == [], "an absorbed detection must not be proposed"
print("OK GA-10: refused transition update reaches admission; absorbed one does not")
