#!/usr/bin/env python3
"""GA-57: call each perception path once, on the host, and assert only that it does not raise.

Not a behaviour suite. Three of the four blockers that cost this project a robot run each
would have died here in seconds:

  'ObjectDescription' object has no attribute 'confirmed'   -> _publish_description_array
  Object(..., **desc) TypeError, one line deeper            -> _update_world_model
  save_uncertain_objects NameError / inside_area's six      -> first execution
  reassign_objects_by_geometry AttributeError               -> first execution

Every one of them is a name or an attribute that does not exist, on a path nothing executed.
Executing the path once is the whole of what was missing.

Run: python3 test_perception_smoke.py
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rosstub  # noqa: E402
rosstub.install()

# Import the modules NEXT TO THIS FILE. This is TS-05's lesson: a sibling test used to
# insert an absolute path to the other checkout, so a copy running in the vendored tree
# silently exercised the tree it was not testing.
import object_info      # noqa: E402
import object_services  # noqa: E402
import perception_2     # noqa: E402
import room_manager     # noqa: E402
from world_model import wm  # noqa: E402

for mod in (object_services, perception_2, room_manager):
    assert pathlib.Path(mod.__file__).resolve().parent == HERE, \
        f"testing the wrong tree: imported {mod.__file__}"

_started = __import__("time").time()

BOX = {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 1.0}
FAR = {"x_min": 9.0, "x_max": 10.0, "y_min": 9.0, "y_max": 10.0, "z_min": 0.0, "z_max": 1.0}
checks = []


def check(name, fn):
    try:
        fn()
        checks.append((name, None))
    except Exception as exc:                      # noqa: BLE001 - reporting, not handling
        checks.append((name, f"{type(exc).__name__}: {exc}"))


class Det:
    def __init__(self, label):
        self.label = self.instance_label = label
        self.is_confirmed = True
        self.mask = None


# --- the chain that broke every run ------------------------------------------------
def description_chain():
    node = perception_2.DetectObjectsNode.__new__(perception_2.DetectObjectsNode)
    node._undeliverable_fields = set()
    node.log_both = lambda *a, **k: None
    node.get_logger = lambda: rosstub.Any()
    node.make_header_msg = lambda t, stamp=None, frame_id="": t()
    node.pub_object_descriptions = rosstub.Any()

    dets = [Det("chair"), Det("desk")]
    descs = perception_2.DetectObjectsNode._build_descriptions(node, dets, {})
    assert descs and "confirmed" in descs[0], "the extra key is still produced -- that is the point"
    perception_2.DetectObjectsNode._publish_description_array(node, dets, descs, None)
    perception_2.DetectObjectsNode._update_world_model(
        node, dets, [None, None], [BOX, BOX], descs)


# --- the three names that did not exist --------------------------------------------
def inside_area():
    o = object_info.Object("chair", None, BOX)
    assert object_services.inside_area(o, (-1, 2, -1, 2, -1, 2)) is True
    assert object_services.inside_area(o, (5, 6, 5, 6, 5, 6)) is False


def save_uncertain():
    # GA-79: this writes a real file. Point the module's PROJECT_ROOT at a temp directory
    # for the call -- a test that mutates the tree it is testing is the defect class this
    # harness exists to catch, and it was writing output/uncertain_objects.txt on every run.
    import tempfile
    node = rosstub.Any()
    node.uncertain_objects = [object_info.Object("chair", None, BOX)]
    original = object_services.PROJECT_ROOT
    with tempfile.TemporaryDirectory() as tmp:
        object_services.PROJECT_ROOT = tmp
        try:
            object_services.save_uncertain_objects(node)
            written = pathlib.Path(tmp) / "output" / "uncertain_objects.txt"
            assert written.is_file(), "save_uncertain_objects wrote nothing"
        finally:
            object_services.PROJECT_ROOT = original


def reassign_rooms():
    rm = room_manager.RoomManager.__new__(room_manager.RoomManager)
    rm.scene_graph = {}
    rm.room_at_bbox = lambda bbox: None            # geometry cannot say
    obj = object_info.Object("chair", None, BOX)
    obj.room_id = "room_1"
    assert rm.reassign_objects_by_geometry([obj]) == []
    assert obj.room_id == "room_1", "an unplaceable object must not be re-filed"


# --- the merge path ----------------------------------------------------------------
def merge_path():
    svc = object_services.ObjectServices.__new__(object_services.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    svc.log_both = lambda *a, **k: None
    svc.room_manager = room_manager.RoomManager.__new__(room_manager.RoomManager)
    svc.room_manager.scene_graph = {}
    svc.room_manager.current_room_id = "room_1"
    svc.room_manager.room_at_bbox = lambda bbox: None
    svc.decision_log = rosstub.Any()

    a = object_info.Object("chair", None, BOX, description="a chair", color="red", material="wood")
    b = object_info.Object("chair", None, FAR, description="a chair", color="red", material="wood")
    a.object_id, b.object_id = "obj_a", "obj_b"
    a.creation_time, b.creation_time = 100.0, 200.0
    wm.persistent_perceptions.clear()
    wm.persistent_perceptions.extend([a, b])

    req = rosstub.Any()
    req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.75, True
    resp = rosstub.Any()
    object_services.ObjectServices._cb_merge_objects(svc, req, resp)


def empty_embedding():
    """GA-171: an EMPTY embedding must read as absent evidence, never crash the merge.

    `_serialize_embedding` encodes None as `[]` to cross the Graph API; the decoder turned
    that into an array of shape (0,), which passed every `is not None` guard and then raised
    `shapes (384,) and (0,) not aligned` inside the dot product -- in the merge callback, the
    first time the association stage ran long enough to update an object and then merge it.
    """
    import numpy as _np
    from nlp_utils import lost_similarity_detailed, world2vec
    from object_services import normalise_embedding
    V = _np.ones(384, dtype=_np.float32) / _np.sqrt(384)
    EMPTY = _np.asarray([], dtype=_np.float32)

    # the boundary restores absence
    assert normalise_embedding([]) is None
    assert normalise_embedding(None) is None
    assert normalise_embedding(V) is not None and len(normalise_embedding(V)) == 384

    # and the similarity function survives one anyway
    for d1, d2 in ((V, EMPTY), (EMPTY, V), (EMPTY, EMPTY),
                   (V, _np.ones(300, dtype=_np.float32))):
        score, ev = lost_similarity_detailed(world2vec, "chair", "chair", "red", "red",
                                             "wood", "wood", d1, d2)
        assert ev["description"] is False, "an uncomparable embedding must be ABSENT"
        assert 0.0 <= score <= 1.0


def install_list():
    """GA-128: every module imported by this package must be in CMakeLists' install list.

    Landed as a TEST, not a script, because the list is hand-maintained and nothing else
    compares it to the imports. perception_2 imported `detection_archive`, the module was in
    src/ and absent from lib/lost3dsg, and the node would have passed the preflight gate and
    then died at startup with ModuleNotFoundError -- a far worse place to find it than here.

    An import that works in src/ says nothing about what `ros2 run` can load.
    """
    import subprocess
    root = str(HERE.parent.parent)
    r = subprocess.run([sys.executable, str(HERE / "check_install_list.py"), root],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout.strip() or r.stderr.strip()


for name, fn in [("description chain (build -> publish -> world model)", description_chain),
                 ("install list covers every import (GA-128)", install_list),
                 ("empty embedding is absent, not a crash (GA-171)", empty_embedding),
                 ("inside_area", inside_area),
                 ("save_uncertain_objects", save_uncertain),
                 ("reassign_objects_by_geometry", reassign_rooms),
                 ("merge path (dry run)", merge_path)]:
    check(name, fn)

width = max(len(n) for n, _ in checks)
failed = 0
for name, err in checks:
    if err:
        failed += 1
        print(f"  FAIL  {name:<{width}}  {err}")
    else:
        print(f"  ok    {name}")
print(f"\n{len(checks) - failed}/{len(checks)} perception paths execute")

# The harness must leave the tree as it found it. It did not: it wrote a log per run and an
# uncertain_objects.txt, inside the tree under test (GA-79).
_stray = sorted(p.name for p in (HERE.parent.parent / "output").glob("*")
                if p.stat().st_mtime >= _started)
if _stray:
    print(f"  WARNING  this run wrote into the tree: {', '.join(_stray)}")
    failed += 1
sys.exit(1 if failed else 0)
