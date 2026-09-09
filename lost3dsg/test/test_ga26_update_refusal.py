"""GA-26 / GA-12 in _cb_update_object and _cb_delete_unseen_objects:
 - a rejected box is a refusal (success=False, message kept) and writes NO attributes;
 - a door gets measured distance/iou, not the constants 0.0 / 1.0;
 - an UNSTAMPED object is treated as young (updated in place, never replaced) -- GA-12;
 - the miss counter is reset by identity, not by label."""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import object_info  # noqa: E402
import object_manager_6 as om6  # noqa: E402
import object_services as osv  # noqa: E402
from world_model import wm  # noqa: E402

osv.save_persistent_perceptions = lambda node: None
osv.publish_pov_volume = lambda *a, **k: None
osv.bbox_centroid_in_volume = lambda bbox, vol: True


def box(x, s=1.0):
    return dict(x_min=x, x_max=x + s, y_min=0.0, y_max=s, z_min=0.0, z_max=s)


def svc():
    s = object.__new__(osv.ObjectServices)
    s.get_logger = lambda: rosstub.Any()
    s.log_both = s.log_operation = lambda *a, **k: None
    s.room_manager = NS(current_room_id="room_1", scene_graph={"room_1": {"objects": []}},
                        update_room_geometry=lambda *a: None, assign_room_by_geometry=lambda b: "room_1")
    s.decision_log = NS(write=lambda *a, **k: None)
    s.db = NS(on_object_moved=lambda *a, **k: None, on_uncertain_added=lambda *a, **k: None, on_object_deleted=lambda *a, **k: None)
    s.uncertain_objects, s.tracking_step_counter, s.considered_volume_pub = [], 0, None
    return s


def update(obj, bbox, **attrs):
    wm.persistent_perceptions.clear(); wm.persistent_perceptions.append(obj)  # noqa: E702
    req = NS(object_id=obj.object_id, update_bbox=True, has_orientation=False, description_embedding=None,
             description=attrs.get("description", ""), color=attrs.get("color", ""), material=attrs.get("material", ""), **bbox)
    return osv.ObjectServices._cb_update_object(svc(), req, NS())


# 1. rejected box: refusal, message kept, attributes NOT written
o = object_info.Object("mug", None, box(0.0, 0.2), description="a mug", color="white", material="ceramic")
o.object_id, o.creation_time = "obj_mug", 0.0
r = update(o, box(0.0, 5.0), description="a bookshelf", color="brown", material="wood")
assert r.success is False and "rejected" in r.message, (r.success, r.message)
assert (o.description, o.color, o.material) == ("a mug", "white", "ceramic"), "attributes leaked past the refusal"
assert o.bbox == box(0.0, 0.2)

# 2. door: measured, not assigned
d = object_info.Object("door", None, box(0.0), description="d", color="", material="")
d.object_id, d.creation_time = "obj_door", 0.0
r = update(d, box(2.7))
assert r.success and abs(r.distance - 2.7) < 1e-6 and r.iou == 0.0, (r.distance, r.iou)

# 3. GA-12: unstamped object, far box, no overlap -> updated in place (young), not replaced
u = object_info.Object("lamp", None, box(0.0), description="l", color="", material="")
u.object_id = "obj_lamp"  # no creation_time on purpose
r = update(u, box(2.0))
assert r.success and r.replaced is False and u.bbox == {**box(2.0), "has_orientation": False}, (r.success, r.message, r.replaced)
assert osv.OBJECT_STABILITY_TIMEOUT > 0 and om6.TRANSITION_MOVE_DISTANCE_M == 0.35

# 4. miss counter keyed on identity: chair#1 in view must not clear chair#2's tally
a = object_info.Object("chair", None, box(0.0), description="", color="", material="")
b = object_info.Object("chair", None, box(3.0), description="", color="", material="")
a.object_id, b.object_id = "obj_a", "obj_b"
b.not_seen_in_pov_frames = 3
wm.persistent_perceptions.clear(); wm.persistent_perceptions.extend([a, b])  # noqa: E702
req = NS(pov_volume_flat=[-10, 10, -10, 10, -10, 10], current_labels=["obj_a"], check_uncertain=False)
resp = osv.ObjectServices._cb_delete_unseen_objects(svc(), req, NS())
assert a.not_seen_in_pov_frames == 0 and b.not_seen_in_pov_frames == 4, (a.not_seen_in_pov_frames, b.not_seen_in_pov_frames)
print("OK GA-26/GA-12: refusal keeps attributes and message; door measured; unstamped=young; reset by identity")
