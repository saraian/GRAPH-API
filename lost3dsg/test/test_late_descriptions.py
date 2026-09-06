"""GA-108: a late describer answer reaches the object sighted in its origin frame, and only fills
what is still unknown; an unmatched answer is counted, not misattributed."""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "perception_module"))
import rosstub  # noqa: E402

rosstub.install()
import object_manager_6 as om  # noqa: E402
from world_model import wm  # noqa: E402


class Obs:
    def __init__(self, frame_id, bbox_2d=None):
        self.frame_id, self.bbox_2d = frame_id, bbox_2d


class Obj:
    def __init__(self, oid, label, frame, bbox_2d):
        self.object_id, self.label = oid, label
        self.description = self.color = self.material = self.shape = "unknown"
        self.observations = [Obs(frame, bbox_2d)]
        self.bbox = {}


class Log:
    def log_both(self, *_a, **_k): pass


def msg(label, frame, bbox, **fields):
    d = types.SimpleNamespace(label=label, origin_frame=frame, origin_bbox_2d=bbox,
                              description="unknown", color="unknown", material="unknown", shape="unknown")
    d.__dict__.update(fields)
    return types.SimpleNamespace(descriptions=[d])


node = object.__new__(om.ObjectManagerService if hasattr(om, "ObjectManagerService") else om.ObjectManager)
node.object_services = Log(); node._n_late_applied = 0; node._n_late_unmatched = 0
node._note_update = lambda *a, **k: None
t = 1788727156.528010731
a = Obj("obj_a", "pillow#1", t, [100, 100, 200, 200])      # sighted in the frame, left of the bed
b = Obj("obj_b", "pillow#2", t, [400, 100, 500, 200])      # same label, same frame, other box
c = Obj("obj_c", "bed#1", t, None)                         # no 2D box on its sighting
wm.persistent_perceptions[:] = [a, b, c]

node._late_descriptions_callback(msg("pillow#1", "1788727156_528010731", [105, 102, 198, 205], description="a white pillow", color="white"))
assert a.description == "a white pillow" and a.color == "white" and b.description == "unknown", (a.description, b.description)
node._late_descriptions_callback(msg("bed#1", "1788727156_528010731", [50, 300, 900, 700], description="a double bed"))
assert c.description == "a double bed", "sole same-label candidate without a box must match"
node._late_descriptions_callback(msg("pillow#3", "1788727156_528010731", [700, 700, 720, 720], description="stray"))
assert node._n_late_unmatched == 1 and b.description == "unknown", "a non-overlapping answer must not land on a namesake"
node._late_descriptions_callback(msg("pillow#1", "1788727156_528010731", [105, 102, 198, 205], description="a different answer"))
assert a.description == "a white pillow", "a filled field is never overwritten"
node._late_descriptions_callback(msg("pillow#1", "9999999999_000000000", [105, 102, 198, 205], description="wrong frame"))
assert node._n_late_unmatched == 2 and node._n_late_applied == 2
node.latest_bboxes = {"k": {"label": "bed#1", "bbox": {"bbox_2d": [1, 2, 3, 4]}}}
assert node._cycle_bbox_2d_for(c) == [1, 2, 3, 4], "GA-316: the cycle's box reaches the sighting"
print("OK GA-108/GA-316: late answers reach their origin object; unknown-only fill; unmatched counted; cycle box recovered")
