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
import json
import os
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
    refused_rows = []

    class _Log:
        def write(self, kind, oid, **kw):
            if kind == "merge_refused":
                refused_rows.append(kw)

    svc.decision_log = _Log()

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

    # Joint rename with the ontology lane (their inbox 00004/00005). The typed key says the
    # UNIT -- metres here, unitless on the similarity arm. TRANSITION CLOSED 2026-09-06 on the
    # owner's authorisation: the legacy `threshold` is RETIRED on both arms, so its ABSENCE is
    # now the assertion. Retired only here: the evidence engine's rows keep `threshold` (a
    # third unit, log-odds) beside the new `threshold_log_odds`, because that is the arm that
    # actually runs and every existing reader of the word is reading it.
    dist_rows = [r for r in refused_rows if r.get("reason") == "distance"]
    assert dist_rows, f"the far pair must be refused on distance: {[r.get('reason') for r in refused_rows]}"
    dr = dist_rows[0]
    assert dr.get("threshold_distance_m") == 0.8, dr          # from request.max_distance
    assert "threshold" not in dr, "the legacy key is retired on the distance arm"
    assert "threshold_similarity" not in dr, "the distance arm must not carry the similarity key"

    # The similarity arm: same position, disagreeing attributes -> refused on similarity,
    # with the UNTYPED quantity on its own typed key.
    c = object_info.Object("chair", None, BOX, description="a chair", color="red", material="wood")
    d = object_info.Object("desk", None, BOX, description="a desk", color="blue", material="metal")
    c.object_id, d.object_id = "obj_c", "obj_d"
    c.creation_time, d.creation_time = 100.0, 200.0
    wm.persistent_perceptions.clear()
    wm.persistent_perceptions.extend([c, d])
    object_services.ObjectServices._cb_merge_objects(svc, req, resp)
    sim_rows = [r for r in refused_rows if r.get("reason") == "similarity"]
    assert sim_rows, f"the disagreeing pair must be refused on similarity: {[r.get('reason') for r in refused_rows]}"
    sr = sim_rows[0]
    assert sr.get("threshold_similarity") == 0.75, sr         # from request.min_similarity
    assert "threshold" not in sr, "the legacy key is retired on the similarity arm"
    assert "threshold_distance_m" not in sr, "the similarity arm must not carry the distance key"


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


def vlm_result_gating():
    """W2: a deferred VLM result attaches only to a detection whose 2D box overlaps the
    crop the result was computed from. instance_label is a per-frame ordinal reused
    across cycles, so the string alone cannot tell one object from another."""
    node = perception_2.DetectObjectsNode.__new__(perception_2.DetectObjectsNode)
    node._undeliverable_fields = set()
    node.log_both = lambda *a, **k: None
    node.get_logger = lambda: rosstub.Any()

    same = Det("chair")
    same.bbox = (10, 10, 50, 60)
    elsewhere = Det("chair")
    elsewhere.bbox = (400, 300, 450, 360)

    res = {"chair": {"description": "a red chair", "color": "red",
                     "material": "fabric", "shape": "rectangular",
                     "origin": {"frame": "frame_0001", "bbox": (11, 11, 49, 59)}}}

    kept = perception_2.DetectObjectsNode._build_descriptions(node, [same], dict(res))
    assert kept[0]["description"] == "a red chair", "a result for this object must attach"
    assert kept[0]["status"] == "ok", "an answered description must say so (W6)"

    refused = perception_2.DetectObjectsNode._build_descriptions(node, [elsewhere], dict(res))
    assert refused[0]["description"] == "unknown", \
        "a result for a different object must be refused, not attached"
    assert refused[0]["status"] == "unanswered", \
        "a refused result must read unanswered, not model_abstained (W6)"

    # the IoU helper itself: identical boxes 1.0, disjoint 0.0, malformed 0.0 (refuse)
    assert perception_2._bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert perception_2._bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert perception_2._bbox_iou(None, (0, 0, 10, 10)) == 0.0
    assert perception_2._bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == 0.3333333333333333


def vlm_status_split():
    """W6: call_failed / parse_failed / model_abstained are three different defects and
    must not land on the map as the same 'unknown'. parse failure is MARKED, never cached;
    a genuine answer IS cached; the status survives to _build_descriptions' output."""
    import tempfile

    import numpy as _np
    import vlm_call

    good = ('```json\n{"objects":[{"description":"a red chair","color":"red",'
            '"material":"fabric","shape":"rectangular"}]}\n```')

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("describe {LABEL}")
        prompt_path = fh.name

    img = _np.zeros((16, 16, 3), dtype=_np.uint8)

    def client(reply):
        calls = {"n": 0}

        def fn(prompt, b64):
            calls["n"] += 1
            if isinstance(reply, Exception):
                raise reply
            return reply
        c = vlm_call.VlmClient(vlm_call_fn=fn, image_encoder_fn=lambda im: "")
        return c, calls

    # parse failure: marked, not cached (the second call re-asks the model)
    c, calls = client("the model rambles, no json anywhere")
    r = c.call_crop_full(prompt_path, "chair", img)
    assert r["provenance"]["status"] == "parse_failed", r["provenance"]
    assert r["description"] == "unknown"
    assert r["provenance"].get("error") == "parse_failed"
    r2 = c.call_crop_full(prompt_path, "chair", img)
    assert calls["n"] == 2, "a parse failure must not enter the crop cache"

    # call failure: marked call_failed
    c, calls = client(RuntimeError("provider down"))
    r = c.call_crop_full(prompt_path, "chair", img)
    assert r["provenance"]["status"] == "call_failed", r["provenance"]

    # genuine answer: ok, cached
    c, calls = client(good)
    r = c.call_crop_full(prompt_path, "chair", img)
    assert r["provenance"]["status"] == "ok", r["provenance"]
    assert r["description"] == "a red chair"
    r2 = c.call_crop_full(prompt_path, "chair", img)
    assert calls["n"] == 1, "a genuine answer must be served from the cache"
    assert r2["description"] == "a red chair"

    # genuine abstention: model_abstained, cached (a refusal is an answer)
    abstain = '```json\n{"objects":[{"description":"unknown"}]}\n```'
    c, _ = client(abstain)
    r = c.call_crop_full(prompt_path, "chair", img)
    assert r["provenance"]["status"] == "model_abstained", r["provenance"]

    # the parser itself: malformed -> None (the caller decides), valid -> record
    c, _ = client("")
    assert c.parse_crop_response("garbage", "chair") is None
    parsed = c.parse_crop_response(good, "chair")
    assert parsed and parsed["description"] == "a red chair"

    # the status derivation used by _build_descriptions
    s = perception_2._description_status
    assert s({}) == "unanswered"
    assert s({"provenance": {"status": "parse_failed"}}) == "parse_failed"
    assert s({"provenance": {"error": "call_failed"}}) == "call_failed"  # older shape
    assert s({"description": "unknown"}) == "model_abstained"           # grid cell
    assert s({"description": "a red chair"}) == "ok"                    # grid cell


def cycle_ms_recorded():
    """WN1: the completed cycle's wall time is stamped beside the detection span, so a
    latency claim can quote the cycle (publish_objects entry->completion) and not the
    detection sub-span that `total_ms` actually measures (~2x off)."""
    import json as _json
    import tempfile

    node = perception_2.DetectObjectsNode.__new__(perception_2.DetectObjectsNode)
    node.latest_latencies = {"total_ms": 100.0}
    with tempfile.TemporaryDirectory() as tmp:
        saved = perception_2.LATENCY_JSON_PATHS
        perception_2.LATENCY_JSON_PATHS = (os.path.join(tmp, "perception_latencies.json"),)
        try:
            perception_2.DetectObjectsNode._record_cycle_ms(node, 13.7)
        finally:
            perception_2.LATENCY_JSON_PATHS = saved
        assert node.latest_latencies["cycle_ms"] == 13700.0, node.latest_latencies
        assert node.latest_latencies["total_ms"] == 100.0, "the detection span must survive"
        with open(os.path.join(tmp, "perception_latencies.json")) as fh:
            on_disk = _json.load(fh)
        assert on_disk["cycle_ms"] == 13700.0 and on_disk["total_ms"] == 100.0


def disappearance_removal_client():
    """GA-297: the disappearance-removal client returns the count+labels so the wiring
    can MEASURE the deletion rate, and reports a failed call as None -- not as a bool
    that erases the reason. (The wiring itself, inside object_tracking_callback, is
    WRITTEN, NOT TESTED: the callback needs a live bridge.)"""
    import object_manager_6 as om6mod

    node = om6mod.ObjectManagerService.__new__(om6mod.ObjectManagerService)
    node.get_logger = lambda: rosstub.Any()

    node._call_graph_api = lambda m, p, json_body: {
        "deleted_count": 2, "deleted_labels": ["chair", "desk"]}
    res = node.delete_undetected_objects(BOX, [Det("chair")], True)
    assert res and res["deleted_count"] == 2 and res["deleted_labels"] == ["chair", "desk"], res

    def boom(m, p, json_body):
        raise RuntimeError("bridge down")
    node._call_graph_api = boom
    assert node.delete_undetected_objects(BOX, [Det("chair")], True) is None, \
        "a failed removal call must report failure, not False"


def pca_gets_aabb_points():
    """W8: the PCA orientation must lift the SAME SOR'd point set the AABB was built
    from. The old 2k-point remove_outliers=False subsample kept mask-bleed far points
    in `oriented_extents` while the AABB dropped them -- two extents for one object,
    and the size gate PREFERS the oriented one, so the bleed could flip a verdict."""
    import detection_pipeline as dp
    import numpy as _np

    seen = {}

    def recorder(mask, depth, fx, fy, cx, cy, **kw):
        seen.update(kw)
        return _np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],
                           [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0],
                           [0, 2, 0], [1, 2, 0], [2, 2, 0], [3, 2, 0],
                           [1.5, 1.0, 0.5], [1.5, 1.0, 0.6], [1.5, 1.0, 0.7]])

    saved_fop, saved_at = dp._filter_object_points, dp._apply_transform
    dp._filter_object_points = recorder
    dp._apply_transform = lambda pts, t: pts
    try:
        node = dp.DetectionPipelineMixin.__new__(dp.DetectionPipelineMixin)
        node.log_both = lambda *a, **k: None
        det = Det("chair")
        det.mask = _np.zeros((4, 4, 1), dtype=_np.uint8)
        det.mask[0, 0, 0] = 1
        bbox = {"x_min": 0.0}

        class CI:
            k = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 0]

        dp.DetectionPipelineMixin._add_pca_orientation(node, [det], [bbox], None, CI(), "map")
    finally:
        dp._filter_object_points, dp._apply_transform = saved_fop, saved_at
    assert seen.get("remove_outliers") is True, f"PCA must not lift un-SOR'd points: {seen}"
    assert seen.get("sor_k") == 30 and seen.get("sor_std") == 1.5, seen
    assert seen.get("max_points_per_obj") == 20000, seen
    assert "oriented_extents" in bbox, f"the stubbed point set must still yield a box: {bbox}"


def crop_file_gets_describer_pixels():
    """W7: `prepare_crops` submits the SAME pixels to the file writer that the describer
    gets in `['cropped']`. The on-disk copy used to carry a 2 px green border the
    in-memory crop did not, so the gate (which reads the file) and the describer judged
    different pixels. cv2 is stubbed in this harness, so the check compares the SUBMITTED
    array against the describer's array -- the defect was a divergence between exactly
    those two, and that needs no real imwrite."""
    import numpy as _np
    import perception_2 as _p2

    submitted = {}

    class SyncExec:
        def submit(self, fn, *a, **k):
            if fn.__name__ == "save_crop_file":
                submitted["path"], submitted["image"] = a[0], a[1]

    node = _p2.DetectObjectsNode.__new__(_p2.DetectObjectsNode)
    node._io_executor = SyncExec()
    node.get_logger = lambda: rosstub.Any()
    node.log_both = lambda *a, **k: None

    det = Det("chair")
    det.bbox = (0, 0, 32, 32)
    det.mask = _np.ones((32, 32, 1), dtype=_np.uint8)

    img = _np.zeros((32, 32, 3), dtype=_np.uint8)
    img[:, :] = (200, 0, 0)                    # solid BLUE in BGR: no green pixel anywhere

    old = os.environ.get("GRAPH_API_OUTPUT_DIR")
    os.environ["GRAPH_API_OUTPUT_DIR"] = "/tmp/opencode_w7_crops"
    try:
        crops = _p2.DetectObjectsNode.prepare_crops(node, [det], img, "frame_w7")
    finally:
        if old is None:
            os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
        else:
            os.environ["GRAPH_API_OUTPUT_DIR"] = old
    assert crops and crops[0], "a crop must be produced"
    assert "image" in submitted, "the crop file write must be submitted"
    assert _np.array_equal(submitted["image"], crops[0]["cropped"]), \
        "the file and the describer must receive the same pixels (W7)"
    corner = submitted["image"][1, 1].astype(int)
    assert corner[1] < 80, f"no green border may be drawn: {tuple(corner)}"
    assert crops[0]["frame"] == "frame_w7" and len(crops[0]["bbox"]) == 4, \
        "the W2 provenance fields must ride along"


def save_persistent_roundtrip():
    """GA-107's save site: every LIVE object must survive the round-trip into
    persistent_perception.json. The old live-list iteration could silently skip an
    object under a concurrent removal -- and the skipped object was then DELETED from
    the stored JSON by the removed_ids pass while still on the map. (The race itself
    needs a live merge thread; this check pins the round-trip on a quiet list.)"""
    import tempfile

    original = object_services.PROJECT_ROOT
    wm.persistent_perceptions.clear()
    a = object_info.Object("chair", [0.0, 0.0, 0.0], BOX)
    b = object_info.Object("desk", [1.0, 1.0, 0.0], FAR)
    a.object_id, b.object_id = "obj_a", "obj_b"
    wm.persistent_perceptions.extend([a, b])
    with tempfile.TemporaryDirectory() as tmp:
        object_services.PROJECT_ROOT = tmp
        try:
            object_services.save_persistent_perceptions(rosstub.Any())
            with open(os.path.join(tmp, "output", "persistent_perception.json")) as fh:
                on_disk = json.load(fh)
        finally:
            object_services.PROJECT_ROOT = original
        ids = {e["object_id"] for e in on_disk}
        assert ids == {"obj_a", "obj_b"}, f"both live objects must survive: {ids}"
        # and a second save with one removed must drop exactly that one
        wm.persistent_perceptions.remove(b)
        with tempfile.TemporaryDirectory() as tmp2:
            object_services.PROJECT_ROOT = tmp2
            try:
                object_services.save_persistent_perceptions(rosstub.Any())
                with open(os.path.join(tmp2, "output", "persistent_perception.json")) as fh:
                    on_disk2 = json.load(fh)
            finally:
                object_services.PROJECT_ROOT = original
            ids2 = {e["object_id"] for e in on_disk2}
            assert ids2 == {"obj_a"}, f"the removal must be the only change: {ids2}"


for name, fn in [("description chain (build -> publish -> world model)", description_chain),
                 ("install list covers every import (GA-128)", install_list),
                 ("empty embedding is absent, not a crash (GA-171)", empty_embedding),
                 ("tracking scan summary: one row per cycle (GA-190)", scan_summary),
                 ("tracking gate: locality, then evidence (GA-289)", tracking_gate),
                 ("world-model object carries a centroid (GA-296)", object_centroid),
                  ("FOV transform: one lookup equals per-point (LAT-1)", fov_transform_vectorised),
                  ("dead crop publisher removed (LAT-5)", dead_crop_publisher_gone),
                  ("deferred VLM result refused off its object (W2)", vlm_result_gating),
                  ("description status split: failed != abstained (W6)", vlm_status_split),
                  ("cycle_ms stamped beside the detection span (WN1)", cycle_ms_recorded),
                  ("disappearance removal reports count+labels (GA-297)", disappearance_removal_client),
                  ("PCA orientation lifts the AABB's SOR'd points (W8)", pca_gets_aabb_points),
                  ("crop file gets the describer's pixels (W7)", crop_file_gets_describer_pixels),
                  ("persistent-JSON round-trip keeps every live object", save_persistent_roundtrip),
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
