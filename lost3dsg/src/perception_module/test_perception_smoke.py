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


def scan_summary():
    """GA-190: the tracking scan emits ONE summary row per cycle, and resets.

    The merge path logs what candidate selection excluded, which is why 87% pruning could be
    measured there; this path logged nothing, so its cost was a code reading rather than a
    finding. One row per CYCLE, never one per comparison -- hook_decisions.jsonl already
    reached 1.8 GB at 99.98% merge_refused.
    """
    import importlib.util
    import json as _json
    import os as _os
    import tempfile

    from object_manager_6 import ObjectManagerService, _bbox_centre, _centre_distance

    def box(x, y, z, s=0.2):
        return {"x_min": x - s / 2, "x_max": x + s / 2, "y_min": y - s / 2,
                "y_max": y + s / 2, "z_min": z - s / 2, "z_max": z + s / 2}

    assert _bbox_centre(box(1, 2, 3)) == (1, 2, 3)
    assert _bbox_centre(None) is None and _bbox_centre({"x_min": 0}) is None
    assert abs(_centre_distance((0, 0, 0), box(3, 4, 0)) - 5.0) < 1e-9
    assert _centre_distance(None, box(1, 1, 1)) is None

    spec = importlib.util.spec_from_file_location("h_gа190", str(HERE / "hooks.py"))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    path = tempfile.mktemp(suffix=".jsonl")
    n = object.__new__(ObjectManagerService)
    n.decision_log = h.DecisionLog(path)
    n._scan_stats = None
    for _ in range(2):
        st = ObjectManagerService._scan_acc(n)
        st["visited"] += 3
        st["skipped_unstable"] += 1
        st["distances"] += [0.3, 2.4]
        st["scored"] += 2
        st["winner_distance"] = 0.3
        st["winners"] += 1
    ObjectManagerService.flush_scan_summary(n, frame_id="f1")
    rows = [_json.loads(x) for x in open(path)]
    assert len(rows) == 1 and rows[0]["kind"] == "tracking_scan_summary"
    assert rows[0]["detections"] == 2 and rows[0]["comparisons_scored"] == 4
    assert rows[0]["would_prune"]["1.0m"] == 2 and rows[0]["would_prune"]["3.0m"] == 0
    assert n._scan_stats is None, "the accumulator must reset after a flush"
    ObjectManagerService.flush_scan_summary(n, frame_id="f2")
    assert len(list(open(path))) == 1, "an empty cycle must emit NO row"
    _os.remove(path)


def tracking_gate():
    """GA-289: on the tracking path, locality before similarity and evidence before a win.

    A same-label object 9 m away must not be scored at all; one 0.3 m away must win; and a
    1.000 on the label alone (nothing else comparable) must not win even when near.
    """
    from object_manager_6 import ObjectManagerService, tracking_reach_m

    def box(x, y, z, s=0.4):
        return {"x_min": x - s / 2, "x_max": x + s / 2, "y_min": y - s / 2,
                "y_max": y + s / 2, "z_min": z - s / 2, "z_max": z + s / 2}

    def obj(label, b):
        o = object_info.Object(label, None, b, description="unknown", color="silver",
                               material="metal")
        o.creation_time = 0.0            # long past OBJECT_STABILITY_TIMEOUT
        return o

    near, far = obj("faucet#1", box(0.3, 0, 0)), obj("faucet#2", box(9.0, 0, 0))
    reach, basis = tracking_reach_m(far)
    assert "fallback" in basis and reach < 8.0, (reach, basis)

    n = object.__new__(ObjectManagerService)
    n.object_services = rosstub.Any()

    def scan(color, material, *objs):
        n._scan_stats = None
        wm.persistent_perceptions.clear()
        wm.persistent_perceptions.extend(objs)
        ObjectManagerService.check_tracking_transition(n, "faucet", color, material, None,
                                                       box(0, 0, 0))
        return n._scan_stats

    st = scan("silver", "metal", far)
    assert st["pruned_locality"] == 1 and st["scored"] == 0 and st["winners"] == 0, st
    st = scan("silver", "metal", far, near)
    assert st["winners"] == 1 and st["pruned_locality"] == 1, st
    assert abs(st["winner_distance"] - 0.3) < 1e-6, st
    st = scan("unknown", "unknown", near)          # label-only 1.000: near, but no evidence
    assert st["winners"] == 0 and st["refused_evidence"] == 1 and st["scored"] == 1, st
    wm.persistent_perceptions.clear()


def object_centroid():
    """GA-296: a world-model Object must carry the centroid its sighting record needs.

    Both Object(...) sites passed None, so `_record_sighting` returned early on every object
    ever created and `observations` stayed empty -- which made the covariance shell, the
    co-visibility channel and appearance re-id abstain by construction on BOTH association
    paths. Asserted here on the real constructor argument, not on the helper alone.
    """
    import inspect

    from object_services import _centroid_from_bbox

    c = _centroid_from_bbox(BOX)
    assert isinstance(c, list) and [round(x, 3) for x in c] == [0.5, 0.5, 0.5], c
    import json as _j
    _j.dumps(c)   # the object it lands on is serialised; a numpy array would raise here
    assert _centroid_from_bbox(None) is None
    assert _centroid_from_bbox({"x_min": 0}) is None

    src = inspect.getsource(object_services)
    assert "Object(label, None, bbox" not in src, "the add path still creates a centroid-less object"
    assert "_centroid_from_bbox(bbox)" in src

    # An object built the way the service builds one has a usable position for the sighting
    # guard: this is the exact condition _record_sighting tests before recording.
    o = object_info.Object("chair", _centroid_from_bbox(BOX), BOX)
    assert getattr(o, "centroid", None) is not None
    assert o.observations == [], "a fresh object records nothing until it is sighted"


def fov_transform_vectorised():
    """LAT-1: one lookup + _apply_transform must equal the per-point path exactly.

    The old branch computed `R.dot(p) + T` for each point, each with its own
    lookup_transform; the new one computes `p.dot(R.T) + T` once for the whole (N,3) array.
    The same product written the other way round, so the two agree to floating-point
    accumulation order and NOT to the last bit -- the first version of this check asserted
    exact equality and failed at 6.8e-14, which is how the plan's word "bit-identical" was
    caught. The bound below is tight enough that a wrong rotation convention (the failure
    this replaces) misses it by many orders of magnitude.
    """
    import numpy as _np
    from cv_utils import _apply_transform, _get_R_and_T

    class _V:
        def __init__(self, x, y, z, w=None):
            self.x, self.y, self.z = x, y, z
            if w is not None:
                self.w = w

    class _T:
        def __init__(self):
            self.transform = self
            # a real rotation (45 deg about z) with a translation, not identity: an identity
            # transform passes under either convention and would prove nothing.
            self.rotation = _V(0.0, 0.0, 0.3826834, 0.9238795)
            self.translation = _V(1.5, -2.0, 0.25)

    trans = _T()
    R, T = _get_R_and_T(trans)
    pts = _np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-4.5, 0.25, 7.0], [1e3, -1e3, 5.0]])

    per_point = _np.asarray([R.dot(p) + T for p in pts])
    vectorised = _apply_transform(pts, trans)
    assert vectorised.shape == per_point.shape
    worst = float(_np.abs(vectorised - per_point).max())
    assert worst < 1e-9, worst          # measured 6.8e-14 on coordinates up to 1e3 m

    # The value this function actually returns is the min/max per axis, in metres, and it
    # must not move enough to matter to a containment test with metre-scale thresholds.
    for i in range(3):
        assert abs(float(vectorised[:, i].min()) - float(per_point[:, i].min())) < 1e-9
        assert abs(float(vectorised[:, i].max()) - float(per_point[:, i].max())) < 1e-9


def dead_crop_publisher_gone():
    """LAT-5: nothing publishes to /cropped_image any more, and nothing calls publish_crops.

    Asserted against the real sources: the topic had no subscriber in either tree, so the
    encode-and-publish ran every cycle for nobody.
    """
    import inspect

    import input_output
    io_src = inspect.getsource(input_output)
    p2_src = inspect.getsource(perception_2)
    assert "def publish_crops" not in io_src
    assert "publish_crops(" not in p2_src
    assert "/cropped_image" not in p2_src
    assert not hasattr(input_output.PerceptionIOMixin, "publish_crops")


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
                 ("tracking scan summary: one row per cycle (GA-190)", scan_summary),
                 ("tracking gate: locality, then evidence (GA-289)", tracking_gate),
                 ("world-model object carries a centroid (GA-296)", object_centroid),
                 ("FOV transform: one lookup equals per-point (LAT-1)", fov_transform_vectorised),
                 ("dead crop publisher removed (LAT-5)", dead_crop_publisher_gone),
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
