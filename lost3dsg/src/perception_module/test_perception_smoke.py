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
import hashlib
import json
import os
import pathlib
import sys
import types

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
import perception_parallel  # noqa: E402
import room_manager     # noqa: E402
from bbox_fusion import add_fusion_view  # noqa: E402
from world_model import wm  # noqa: E402

for mod in (object_services, perception_2, perception_parallel, room_manager):
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


def bbox_fusion_message():
    """The measured voxel keys and view ID reach the typed bbox message."""
    published = []
    node = perception_2.DetectObjectsNode.__new__(perception_2.DetectObjectsNode)
    node.make_header_msg = lambda cls, **kwargs: cls()
    node.bbox_pub = types.SimpleNamespace(publish=published.append)
    det = types.SimpleNamespace(
        instance_label="chair",
        bbox=None,
        clip_embedding=None,
        fusion_voxel_keys=[0, 0, 0, 1, 2, 3],
    )

    perception_2.DetectObjectsNode._publish_bbox_array(
        node, [det], [dict(BOX)], None, None, cycle_id="cycle-17"
    )

    assert len(published) == 1 and published[0].cycle_id == "cycle-17"
    box = published[0].boxes[0]
    assert box.has_fusion_voxels is True
    assert box.fusion_voxel_size_m == perception_2.VOXEL_SIZE_M
    assert box.fusion_voxel_keys == [0, 0, 0, 1, 2, 3]


def parallel_perception_tracks_its_base():
    """The parallel copy names the exact perception_2.py revision it mirrors."""
    base_bytes = (HERE / "perception_2.py").read_bytes()
    digest = hashlib.sha256(base_bytes).hexdigest()
    assert digest == perception_parallel.PERCEPTION_2_BASELINE_SHA256, (
        "perception_2.py changed: audit and apply the same semantic change to "
        "perception_parallel.py before updating PERCEPTION_2_BASELINE_SHA256"
    )
    parallel_source = (HERE / "perception_parallel.py").read_text()
    begin = sum(line.lstrip().startswith("# PARALLEL_VARIANT_BEGIN:")
                for line in parallel_source.splitlines())
    end = sum(line.lstrip().startswith("# PARALLEL_VARIANT_END:")
              for line in parallel_source.splitlines())
    assert begin == end and begin >= 5, (begin, end)
    assert "ParallelFusionEncoder.from_config(CFG)" in parallel_source


def parallel_timing_names_the_backend():
    """The directly measured encoder time and identity reach the cycle row."""
    rows = []
    node = perception_parallel.DetectObjectsNode.__new__(
        perception_parallel.DetectObjectsNode
    )
    node.latest_latencies = {}
    node._bbox_fusion_measurement = {
        "component": "ParallelFusionEncoder",
        "backend": "cpu_processes",
        "elapsed_ms": 12.375,
        "cpu_workers": 4,
    }
    node._cycle_count = 0
    original_paths = perception_parallel.LATENCY_JSON_PATHS
    original_append = perception_parallel._append_cycle_row
    perception_parallel.LATENCY_JSON_PATHS = ()
    perception_parallel._append_cycle_row = rows.append
    try:
        perception_parallel.DetectObjectsNode._record_cycle_ms(
            node, 0.100, frame_id="frame-1", n_detections=3
        )
    finally:
        perception_parallel.LATENCY_JSON_PATHS = original_paths
        perception_parallel._append_cycle_row = original_append
    assert node.latest_latencies["bbox_fusion_encode_ms"] == 12.375
    assert node.latest_latencies["bbox_fusion_encoder"]["backend"] == "cpu_processes"
    assert rows[0]["bbox_fusion_encoder"]["cpu_workers"] == 4


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

    # The file belongs in the RUN'S BUNDLE, which is `GRAPH_API_OUTPUT_DIR`. `graph_api_bridge`
    # reads it from there for the on-hold / rejected / abstained audit and watches it in the
    # graph fingerprint, so a writer using PROJECT_ROOT agreed with those readers only when the
    # run's output directory happened to be the source tree's own `output/`.
    #
    # BOTH DIRECTIONS, because an assertion on the joined path alone would pass even if the
    # writer never ran: the file must appear in the bundle AND be absent from PROJECT_ROOT.
    original_env = os.environ.get("GRAPH_API_OUTPUT_DIR")
    with tempfile.TemporaryDirectory() as bundle, tempfile.TemporaryDirectory() as tree:
        object_services.PROJECT_ROOT = tree
        os.environ["GRAPH_API_OUTPUT_DIR"] = bundle
        try:
            object_services.save_uncertain_objects(node)
            in_bundle = pathlib.Path(bundle) / "uncertain_objects.txt"
            in_tree = pathlib.Path(tree) / "output" / "uncertain_objects.txt"
            assert in_bundle.is_file(), "the pool must land in the run's bundle"
            assert in_bundle.read_text().strip(), "the bundle copy must not be empty"
            assert not in_tree.exists(), "nothing may be written into the source tree"
        finally:
            object_services.PROJECT_ROOT = original
            if original_env is None:
                os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            else:
                os.environ["GRAPH_API_OUTPUT_DIR"] = original_env


def reassign_rooms():
    rm = room_manager.RoomManager.__new__(room_manager.RoomManager)
    rm.scene_graph = {}
    rm.room_at_bbox = lambda bbox: None            # geometry cannot say
    obj = object_info.Object("chair", None, BOX)
    obj.room_id = "room_1"
    assert rm.reassign_objects_by_geometry([obj]) == []
    assert obj.room_id == "room_1", "an unplaceable object must not be re-filed"


def merge_request_below_match_gate_refused():
    """GA-341: a merge request whose similarity floor sits at or below the match gate is
    REFUSED with the bound named -- never clamped, never merged at the bridge's old 0.75."""
    svc = object_services.ObjectServices.__new__(object_services.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    svc.log_both = lambda *a, **k: None
    svc.room_manager = room_manager.RoomManager.__new__(room_manager.RoomManager)
    svc.room_manager.scene_graph = {}
    svc.room_manager.room_at_bbox = lambda bbox: None
    svc.decision_log = rosstub.Any()
    wm.persistent_perceptions.clear()

    req, resp = rosstub.Any(), rosstub.Any()
    req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.75, True
    object_services.ObjectServices._cb_merge_objects(svc, req, resp)
    assert resp.success is False, "0.75 sits below sim_threshold 0.85 and must be refused"
    assert "sim_threshold" in resp.message and "0.85" in resp.message, resp.message
    req.min_similarity = object_services.SIM_THRESHOLD
    object_services.ObjectServices._cb_merge_objects(svc, req, resp)
    assert resp.success is False, "equal to the gate is not STRICTLY greater"
    req.min_similarity = object_services.MERGE_MIN_SIMILARITY
    object_services.ObjectServices._cb_merge_objects(svc, req, resp)
    assert resp.success is True, resp.message
    assert object_services.MERGE_MIN_SIMILARITY > object_services.SIM_THRESHOLD


def detector_failure_skips_the_cycle():
    """GA-427: a transient detector failure SKIPS the cycle, counted and named; it does not
    end the run on the first one and it does not pass silently. Run 20260909_004443 lost
    itself at 19m56s to one 180 s timeout on a service that answered 3 minutes later.

    NOT ASSERTED HERE, and the first draft of this check proved why: the run-ending path
    calls os._exit, so driving the counter to the limit kills the test process with no
    summary and no traceback. The limit is set above the number of failures driven, and the
    threshold itself needs a subprocess to exercise."""
    import detection_pipeline
    import numpy as _np

    class Boom:
        def __init__(self, exc):
            self.exc = exc

        def segment_scene(self, *a, **k):
            # RENAMED with _extract_scene_objects, and for the same reason: one structured
            # scene response now feeds every backend. A stub carrying the old name is not a
            # stub of anything -- the call raised AttributeError before reaching the failure
            # branch, so the strike counter this check exists for was never touched.
            raise self.exc

    class Node:
        """A REAL object, not rosstub.Any: the stub answers every attribute with a
        placeholder, so `getattr(self, "_det_strikes", 0)` would never take its default and
        the first-failure path -- the one that matters -- could not be exercised."""
        log_both = staticmethod(lambda *a, **k: None)

        def _extract_scene_objects(self, rgb):
            # RENAMED from _extract_detection_labels when detection moved to one structured
            # scene response. The stub kept the old name, so the real method fell through to
            # __getattr__, returned None, and run_detection took its "no objects" early return:
            # the backend was never called and NO STRIKE WAS COUNTED. The check reported a
            # missing attribute while the counter it tests was never reached.
            return [{"label": "chair"}]

        def _abort_if_moving(self, *a, **k):
            return False

        def __getattr__(self, name):
            # No-op for the pipeline's other collaborators, but NEVER for the strike
            # counter: that one must reach `getattr(..., 0)`'s default, which is the
            # first-failure path this check exists to exercise.
            if name.startswith("_det"):
                raise AttributeError(name)
            return lambda *a, **k: None

    node = Node()
    node.perception_backend = Boom(TimeoutError("read operation timed out"))
    cam = {"rgb": _np.zeros((4, 4, 3), _np.uint8)}

    original = dict(detection_pipeline.CFG.get("perception", {}))
    detection_pipeline.CFG["perception"] = {**original, "backend": "modal", "detector_strikes_max": 5}
    try:
        out = detection_pipeline.DetectionPipelineMixin.run_detection(node, cam)
        assert out == [], "a failed detector call must skip the cycle, not return junk"
        assert node._det_strikes == 1, node._det_strikes
        assert node._detector_status["status"] == "unreachable", node._detector_status
        assert node._detector_status["consecutive_failures"] == 1
        detection_pipeline.DetectionPipelineMixin.run_detection(node, cam)
        assert node._det_strikes == 2, "consecutive failures accumulate"
        # a DIFFERENT transport fault counts the same way
        node.perception_backend = Boom(ConnectionResetError("peer went away"))
        detection_pipeline.DetectionPipelineMixin.run_detection(node, cam)
        assert node._det_strikes == 3, node._det_strikes
        # guard disabled -> the old behaviour, crash on the first failure
        detection_pipeline.CFG["perception"] = {**original, "backend": "modal", "detector_strikes_max": 0}
        node._det_strikes = 0
        try:
            detection_pipeline.DetectionPipelineMixin.run_detection(node, cam)
            raise AssertionError("with the guard disabled the failure must propagate")
        except ConnectionResetError:
            pass
    finally:
        detection_pipeline.CFG["perception"] = original


def merge_lock_covers_writes_only():
    """GA-393 narrowed: the world-model lock is held for the WRITES and not for the sweep.

    Both halves can fail. If someone re-decorates the callback or widens the guard back over
    the comparison, the sweep probe sees the lock held. If someone drops the guard, the write
    probe sees it unheld. The merge callback was the only service callback in this file
    taking no lock at all, while add, update, delete and query all take one."""
    import threading

    import object_services as osv

    svc = osv.ObjectServices.__new__(osv.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    svc.log_both = lambda *a, **k: None
    svc.tracking_step_counter = 0
    svc.room_manager = room_manager.RoomManager.__new__(room_manager.RoomManager)
    svc.room_manager.scene_graph = {}
    svc.room_manager.current_room_id = "room_1"
    svc.room_manager.room_at_bbox = lambda bbox: None
    svc.room_manager.update_room_geometry = lambda *a, **k: None
    svc.decision_log = rosstub.Any()
    svc.persistent_bbox_pub = rosstub.Any()
    svc.persistent_centroids_pub = rosstub.Any()

    class CountingRLock:
        """Delegates to a REENTRANT lock, because the write block re-enters it through
        save_persistent_perceptions -> wm.snapshot(). Substituting a plain Lock here
        deadlocks the suite, which is how this test learned the property it now documents."""

        def __init__(self):
            self._lock = threading.RLock()
            self.depth = 0

        def __enter__(self):
            self._lock.acquire()
            self.depth += 1
            return self

        def __exit__(self, *exc):
            self.depth -= 1
            self._lock.release()
            return False

        def acquire(self, *a, **k):
            got = self._lock.acquire(*a, **k)
            if got:
                self.depth += 1
            return got

        def release(self):
            self.depth -= 1
            self._lock.release()

        @property
        def held(self):
            return self.depth > 0

    held_during_sweep = []
    real_lock, osv.wm.lock = osv.wm.lock, CountingRLock()
    real_sim = osv.lost_similarity_detailed

    def probe(*a, **k):
        held_during_sweep.append(osv.wm.lock.held)
        return real_sim(*a, **k)

    osv.lost_similarity_detailed = probe
    original_root = osv.PROJECT_ROOT
    try:
        a = object_info.Object("chair", None, BOX, description="a chair", color="red", material="wood")
        b = object_info.Object("chair", None, BOX, description="a chair", color="red", material="wood")
        a.object_id, b.object_id = "obj_keep", "obj_drop"
        a.creation_time, b.creation_time = 1.0, 2.0
        osv.wm.persistent_perceptions.clear()
        osv.wm.persistent_perceptions.extend([a, b])
        req, resp = rosstub.Any(), rosstub.Any()
        req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.95, False
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            osv.PROJECT_ROOT = tmp
            osv.ObjectServices._cb_merge_objects(svc, req, resp)
        assert held_during_sweep, "the probe never ran: the sweep did not compare the pair"
        assert not any(held_during_sweep), \
            f"the lock was held during the SWEEP on {sum(held_during_sweep)} comparison(s) — the guard is too wide"
        assert len(osv.wm.persistent_perceptions) == 1, osv.wm.persistent_perceptions
        assert osv.wm.persistent_perceptions[0].object_id == "obj_keep"
        assert not osv.wm.lock.held, "the lock must be released when the callback returns"
        assert resp.merged_count == 1, resp.merged_count
        # `merged_count` reports APPLIED merges. Run the race the locked re-check exists for:
        # the discard leaves the map during the unlocked sweep. The pair is decided, then
        # skipped as stale at the write -- so it must not be counted. Before the fix this
        # path fell through to the room-graph update and reported 1.
        c = object_info.Object("chair", None, BOX, description="a chair", color="red", material="wood")
        c.object_id, c.creation_time = "obj_gone", 3.0
        osv.wm.persistent_perceptions.append(c)

        def racing_probe(*a, **k):
            if c in osv.wm.persistent_perceptions:
                osv.wm.persistent_perceptions.remove(c)
            return real_sim(*a, **k)

        osv.lost_similarity_detailed = racing_probe
        resp2 = rosstub.Any()
        with tempfile.TemporaryDirectory() as tmp:
            osv.PROJECT_ROOT = tmp
            osv.ObjectServices._cb_merge_objects(svc, req, resp2)
        assert resp2.merged_count == 0, f"a stale pair was counted as a merge: {resp2.merged_count}"
        assert [o.object_id for o in osv.wm.persistent_perceptions] == ["obj_keep"]
    finally:
        osv.lost_similarity_detailed = real_sim
        osv.wm.lock = real_lock
        osv.PROJECT_ROOT = original_root
        osv.wm.persistent_perceptions.clear()


def every_config_key_read_is_declared():
    """A key the code reads and the file never declares silently takes the module fallback,
    so the config says one thing and the run does another. This swept 22 such keys out of
    association, habitat and perception in one pass -- including the five merge thresholds,
    where config.yaml documented `cost_ratio` and `min_consecutive` while object_services read
    `merge_cost_ratio` and `merge_min_consecutive` and got neither.

    Literal reads only: `CFG["section"]["key"]`, `CFG["section"].get("key")` and the
    `<name>_cfg.get("key")` locals. A dynamic read is invisible here and always will be."""
    import re

    import config as cfgmod

    declared = {k: set(v) for k, v in cfgmod._DEFAULTS.items() if isinstance(v, dict)}
    local_of = {"hab_cfg": "habitat", "assoc_cfg": "association", "p_cfg": "perception",
                "r_cfg": "rooms", "w_cfg": "walls", "v_cfg": "vlm", "f_cfg": "frames"}
    direct = re.compile(r'CFG\s*\[\s*["\'](\w+)["\']\s*\]\s*(?:\.get\(\s*|\[\s*)["\'](\w+)["\']')
    viavar = re.compile(r'(\w+_cfg)\.get\(\s*["\'](\w+)["\']')
    here = pathlib.Path(__file__).resolve().parent
    undeclared = []
    for path in sorted(here.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(errors="replace")
        for sec, key in direct.findall(text):
            if sec in declared and key not in declared[sec]:
                undeclared.append(f"{sec}.{key} in {path.name}")
        for var, key in viavar.findall(text):
            sec = local_of.get(var)
            if sec and sec in declared and key not in declared[sec]:
                undeclared.append(f"{sec}.{key} in {path.name}")
    assert not undeclared, ("config keys read but never declared, so each silently takes its "
                            "module fallback: " + "; ".join(sorted(set(undeclared))))


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

    summary_rows = []

    class _Log:
        def write(self, kind, oid, **kw):
            if kind == "merge_refused":
                refused_rows.append(kw)
            elif kind == "not_offered_summary":
                summary_rows.append(kw)

    svc.decision_log = _Log()

    # The legacy engine gained an AABB broad phase at `merge_aabb_margin_m` (0.8 m) in the
    # dev/lost3dsg-cleanup merge, so a pair far enough apart never reaches the distance gate
    # and no `merge_refused` row is written for it. BOTH populations are asserted here.
    #
    # `b` is the DISTANCE-ARM pair: 1.4 m between centres, which is past the 0.8 m criterion
    # but only 0.4 m between the boxes, so the broad phase offers it and the exact gate
    # refuses it. That keeps this arm testing what it was written to test -- the typed
    # threshold key -- instead of testing the broad phase by accident.
    # `far` is the BROAD-PHASE population: both of its pairs are dropped before comparison,
    # and must still be counted in the bundle.
    NEAR_BUT_PAST_GATE = {"x_min": 1.4, "x_max": 2.4, "y_min": 0.0, "y_max": 1.0,
                          "z_min": 0.0, "z_max": 1.0}
    a = object_info.Object("chair", None, BOX, description="a chair", color="red", material="wood")
    b = object_info.Object("chair", None, NEAR_BUT_PAST_GATE, description="a chair",
                           color="red", material="wood")
    far = object_info.Object("chair", None, FAR, description="a chair", color="red",
                             material="wood")
    a.object_id, b.object_id, far.object_id = "obj_a", "obj_b", "obj_far"
    a.creation_time, b.creation_time, far.creation_time = 100.0, 200.0, 300.0
    wm.persistent_perceptions.clear()
    wm.persistent_perceptions.extend([a, b, far])

    req = rosstub.Any()
    # GA-341: the request's floor must sit ABOVE sim_threshold (0.85) or the service refuses it.
    req.max_distance, req.min_similarity, req.dry_run = 0.8, 0.9, True
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

    # The pairs the broad phase dropped are COUNTED, never written one by one (GA-232: the
    # per-pair population is what took hook_decisions.jsonl to 1.71 GB). Without this row a
    # pruned pair leaves no trace in the bundle at all, which is the defect `_refused` was
    # introduced to end. Both of `far`'s pairs are out; the a-b pair is not.
    assert summary_rows, "the broad phase must record what it never offered"
    sm = summary_rows[0]
    assert sm.get("n_pairs") == 2, sm
    assert sm.get("by_reason") == {"aabb_margin": 2}, sm

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
    assert sr.get("threshold_similarity") == 0.9, sr          # from request.min_similarity
    assert "threshold" not in sr, "the legacy key is retired on the similarity arm"
    assert "threshold_distance_m" not in sr, "the similarity arm must not carry the distance key"


def merge_request_distance_survives_the_broad_phase():
    """The AABB broad phase must WIDEN to the request's distance, never narrow it.

    The load-time guard asserts `merge_aabb_margin_m >= merge_max_distance_m`, and its comment
    says the margin exists so the broad phase "cannot hide a pair that the legacy merge
    criterion would evaluate". It compares against the CONFIG value only. `_cb_merge_objects`
    sets MAX_DISTANCE from `request.max_distance`, and `_merge_candidates` was never told it,
    so a request asking for more than the config default was silently narrowed to the fixed
    margin -- the pair was dropped before comparison and no refusal was written.

    MEASURED before the fix, and reproduced here: centres 2.0 m apart, boxes 1.0 m apart, which
    is past the 0.8 m margin but well inside a 3.0 m request.
    """
    svc = object_services.ObjectServices.__new__(object_services.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    svc.log_both = lambda *a, **k: None
    svc.room_manager = room_manager.RoomManager.__new__(room_manager.RoomManager)
    svc.room_manager.scene_graph = {}
    svc.room_manager.current_room_id = "room_1"
    svc.room_manager.room_at_bbox = lambda bbox: None

    rows = []

    class _Log:
        def write(self, kind, oid, **kw):
            rows.append((kind, kw.get("reason"), kw.get("by_reason")))

    svc.decision_log = _Log()

    PAST_MARGIN = {"x_min": 2.0, "x_max": 3.0, "y_min": 0.0, "y_max": 1.0,
                   "z_min": 0.0, "z_max": 1.0}
    gap = PAST_MARGIN["x_min"] - BOX["x_max"]
    assert gap > object_services.MERGE_AABB_MARGIN_M, (
        f"the pair must sit OUTSIDE the config margin or this proves nothing: "
        f"gap {gap} m, margin {object_services.MERGE_AABB_MARGIN_M} m")

    def sweep(asked):
        rows.clear()
        a = object_info.Object("chair", None, BOX, description="a chair", color="red",
                               material="wood")
        b = object_info.Object("chair", None, PAST_MARGIN, description="a chair", color="red",
                               material="wood")
        a.object_id, b.object_id = "obj_a", "obj_b"
        a.creation_time, b.creation_time = 100.0, 200.0
        wm.persistent_perceptions.clear()
        wm.persistent_perceptions.extend([a, b])
        req = rosstub.Any()
        req.max_distance, req.min_similarity, req.dry_run = asked, 0.9, True
        object_services.ObjectServices._cb_merge_objects(svc, req, rosstub.Any())
        return list(rows)

    # The instrument must be able to say the OTHER thing: at the config distance this pair is
    # correctly dropped by the broad phase, and that is what the summary row is for.
    at_config = sweep(object_services.MERGE_AABB_MARGIN_M)
    assert any(k == "not_offered_summary" for k, _r, _b in at_config), at_config

    # The fix: a wider request must reach the exact gate, so the pair is COMPARED. Nothing may
    # be dropped by the broad phase, because the request asked for more than the margin.
    wider = sweep(3.0)
    assert not any(k == "not_offered_summary" for k, _r, _b in wider), (
        f"a request wider than the margin was narrowed by the broad phase: {wider}")
    assert any(k == "merge" or (k == "merge_refused" and r != "distance")
               for k, r, _b in wider), \
        f"the pair must reach the exact gate, not vanish: {wider}"
    wm.persistent_perceptions.clear()


def frame_queue_decouples_processing_from_the_motion_gate():
    """A queued snapshot is processed while the gate still says the robot is moving.

    MEASURED, and this is the whole reason the queue exists: two COMPLETE tours
    (20260911_133641 and _140421) of 4344 frames each produced ONE and TWO perception cycles,
    with no detections file in either bundle. The gate is not mis-tuned -- the tour's longest
    pause is 1 s, the same length as the gate's own sampling period, so a sampler measuring
    the delta since its last sample almost never lands inside a pause. A snapshot cannot be
    invalidated by motion that happened after it was taken, so the gate must not apply to one.

    Asserted here: OFF by default; capture ignores motion; a redundant viewpoint is dropped;
    translation and rotation are judged SEPARATELY; and the abort gate stands down for a
    queued frame but still fires for a live one.
    """
    from collections import deque

    N = perception_2.DetectObjectsNode

    # The bound is DERIVED here rather than written as a number, because the number I wrote
    # first was wrong and the first armed run refuted it. Since the queue discards the OLDEST
    # when full, a processed frame's age is depth x CAPTURE interval -- NOT depth x cycle
    # time, which is what the earlier bound of 9 came from. Measured on 20260911_150406:
    # 0.53 s per capture read, so depth 8 is about 4.2 s of staleness against a 30 s TF
    # buffer, and that run logged no TF failure at all.
    _CAPTURE_INTERVAL_S = 0.5     # the capture timer in _create_timers
    _TF_BUFFER_S = 30.0           # what compute_fov_volume_from_depth can still look up
    _q = object_services.CFG["perception"]
    _stale_s = _q["frame_queue_max"] * _CAPTURE_INTERVAL_S
    assert _stale_s <= _TF_BUFFER_S, (
        f"frame_queue_max {_q['frame_queue_max']} means a processed frame can be {_stale_s:.1f}s "
        f"old, past the {_TF_BUFFER_S:.0f}s TF buffer, so its transform lookup would fail")
    assert _q["frame_queue_max"] >= 0, _q["frame_queue_max"]
    if _q["frame_queue_max"] > 0:
        assert _q["frame_queue_min_translation_m"] > 0 and _q["frame_queue_min_rotation_rad"] > 0, \
            "an armed queue with a zero viewpoint threshold queues every frame it sees"

    class _Cam:
        def __init__(self, frames):
            self.frames = list(frames)

        def get_synced_data(self):
            # The real one CONSUMES its cache on every successful read.
            return self.frames.pop(0) if self.frames else None

    # REAL objects, not rosstub.Any(): Any answers every attribute, so the coordinates would
    # be stubs rather than numbers and the arithmetic under test would never run. Same trap
    # the GA-427 check hit.
    def snap(x, y=0.0, z=0.0, qz=0.0, qw=1.0):
        ns = types.SimpleNamespace
        return {"rgb": "px", "depth": "d", "camera_info": None,
                "transform": ns(transform=ns(translation=ns(x=x, y=y, z=z),
                                             rotation=ns(x=0.0, y=0.0, z=qz, w=qw)))}

    node = N.__new__(N)
    node.frame_queue = deque(maxlen=4)
    node.frame_queue_min_translation_m = 0.25
    node.frame_queue_min_rotation_rad = 0.26
    node._queued_frame = None
    node._queue_last_pose = None
    node._queue_captured = node._queue_redundant = node._queue_dropped = 0
    node.is_stationary = False          # the gate says MOVING for every call below
    node.processing_interrupted = False
    node.log_both = lambda *a, **k: None

    # 1. The first frame is always taken, and motion does not stop it.
    node.camera_data = _Cam([snap(0.0)])
    N._capture_frame_callback(node)
    assert node._queue_captured == 1 and len(node.frame_queue) == 1, node._queue_captured

    # 2. The same viewpoint again is redundant and is dropped, not queued.
    node.camera_data = _Cam([snap(0.01)])
    N._capture_frame_callback(node)
    assert node._queue_captured == 1 and node._queue_redundant == 1, \
        (node._queue_captured, node._queue_redundant)

    # 3. Translation alone past its own threshold is enough.
    node.camera_data = _Cam([snap(0.60)])
    N._capture_frame_callback(node)
    assert node._queue_captured == 2, node._queue_captured

    # 4. ROTATION ALONE is enough, with no translation at all. This is the case the motion
    #    gate cannot express: it sums metres and radians, so it cannot separate a turn in
    #    place from driving. 90 degrees about z.
    import math as _m
    half = _m.pi / 4.0
    node.camera_data = _Cam([snap(0.60, qz=_m.sin(half), qw=_m.cos(half))])
    N._capture_frame_callback(node)
    assert node._queue_captured == 3, \
        f"a pure rotation must be a new viewpoint: {node._queue_captured}"

    # 4b. THE CASE THAT DISCRIMINATES. A small shuffle AND a small turn, each below its own
    #     threshold: 0.20 m and 0.20 rad. Separate thresholds call this the same viewpoint.
    #     A SUMMED score -- which is exactly what the motion gate computes -- calls it new,
    #     because 0.40 clears 0.25. Without this case the test passes either way, and the
    #     separation it claims to protect is untested.
    tot = _m.pi / 2.0 + 0.20
    node.camera_data = _Cam([snap(0.80, qz=_m.sin(tot / 2.0), qw=_m.cos(tot / 2.0))])
    N._capture_frame_callback(node)
    assert node._queue_captured == 3, (
        "0.20 m and 0.20 rad are each below their own threshold, so this is the SAME "
        f"viewpoint; summing them into one score is the motion gate's error: {node._queue_captured}")

    # 5. An unreadable transform ABSTAINS from pruning rather than assuming redundancy.
    node.camera_data = _Cam([{"rgb": "px", "depth": "d", "camera_info": None, "transform": None}])
    N._capture_frame_callback(node)
    assert node._queue_captured == 4, node._queue_captured

    # 6. Full means the OLDEST goes, and the loss is COUNTED rather than silent.
    node.camera_data = _Cam([snap(9.0)])
    N._capture_frame_callback(node)
    assert node._queue_dropped == 1 and len(node.frame_queue) == 4, \
        (node._queue_dropped, len(node.frame_queue))

    # 7. The abort gate stands down for a queued snapshot and still fires for a live cycle.
    node._queued_frame = {"rgb": "px"}
    assert N._abort_if_moving(node, "detection") is False, \
        "a snapshot cannot be invalidated by motion after it was captured"
    node._queued_frame = None
    assert N._abort_if_moving(node, "detection") is True, \
        "the live path must still abort while the robot moves"


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
    node._cycle_count = 0          # what __init__ sets; GA-334 series
    node._io_executor = types.SimpleNamespace(submit=lambda fn, *a: fn(*a))
    with tempfile.TemporaryDirectory() as tmp:
        saved = perception_2.LATENCY_JSON_PATHS, perception_2.LATENCY_JSONL_PATHS
        perception_2.LATENCY_JSON_PATHS = (os.path.join(tmp, "perception_latencies.json"),)
        perception_2.LATENCY_JSONL_PATHS = (os.path.join(tmp, "perception_latencies.jsonl"),)
        try:
            perception_2.DetectObjectsNode._record_cycle_ms(node, 13.7, frame_id="f1", n_detections=2)
        finally:
            perception_2.LATENCY_JSON_PATHS, perception_2.LATENCY_JSONL_PATHS = saved
        with open(os.path.join(tmp, "perception_latencies.jsonl")) as fh:
            rows = [_json.loads(line) for line in fh]
        assert [(r["cycle"], r["frame_id"], r["n_detections"]) for r in rows] == [(1, "f1", 2)], rows
        assert node.latest_latencies["cycle_ms"] == 13700.0, node.latest_latencies
        assert node.latest_latencies["total_ms"] == 100.0, "the detection span must survive"
        with open(os.path.join(tmp, "perception_latencies.json")) as fh:
            on_disk = _json.load(fh)
        assert on_disk["cycle_ms"] == 13700.0 and on_disk["total_ms"] == 100.0

        # With no queue on this node every queue key is None, and None is the honest value:
        # "the queue was off" is not "the queue dropped zero frames".
        assert all(rows[0][k] is None for k in
                   ("queue_depth", "queue_captured", "queue_redundant", "queue_dropped",
                    "queue_age_s")), rows[0]

        # ARMED, the counters reach the ROW rather than only the log line. A counter that
        # lives only in the container log is a property of the launch, not of the bundle --
        # the defect the merge broad phase had when it logged what it pruned without
        # recording it, and the reason this was added on 2026-09-11.
        from collections import deque as _dq
        node.frame_queue = _dq([{"rgb": "a"}, {"rgb": "b"}], maxlen=8)
        node._queue_captured, node._queue_redundant, node._queue_dropped = 519, 174, 454
        node._last_queue_age_s = 4.2
        saved = perception_2.LATENCY_JSON_PATHS, perception_2.LATENCY_JSONL_PATHS
        perception_2.LATENCY_JSON_PATHS = (os.path.join(tmp, "b.json"),)
        perception_2.LATENCY_JSONL_PATHS = (os.path.join(tmp, "b.jsonl"),)
        try:
            perception_2.DetectObjectsNode._record_cycle_ms(node, 4.3, frame_id="f2",
                                                            n_detections=3)
        finally:
            perception_2.LATENCY_JSON_PATHS, perception_2.LATENCY_JSONL_PATHS = saved
        with open(os.path.join(tmp, "b.jsonl")) as fh:
            armed = [_json.loads(line) for line in fh][-1]
        assert (armed["queue_depth"], armed["queue_captured"], armed["queue_redundant"],
                armed["queue_dropped"], armed["queue_age_s"]) == (2, 519, 174, 454, 4.2), armed


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
        # GA-315 part 1: a mask pixel within 2 px of an image edge means "clipped" and gets
        # no PCA at all, so the in-frame case needs an interior pixel (contract from the
        # ontology lane, 2026-09-07). An 8x8 frame has an interior; a 4x4 one does not.
        det.mask = _np.zeros((8, 8, 1), dtype=_np.uint8)
        det.mask[4, 4, 0] = 1
        bbox = {"x_min": 0.0}

        class CI:
            k = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 0]

        dp.DetectionPipelineMixin._add_pca_orientation(node, [det], [bbox], None, CI(), "map")
        clipped = Det("chair")
        clipped.mask = _np.zeros((8, 8, 1), dtype=_np.uint8)
        clipped.mask[0, 0, 0] = 1
        clipped_bbox = {"x_min": 0.0}
        dp.DetectionPipelineMixin._add_pca_orientation(node, [clipped], [clipped_bbox], None, CI(), "map")
    finally:
        dp._filter_object_points, dp._apply_transform = saved_fop, saved_at
    assert seen.get("remove_outliers") is True, f"PCA must not lift un-SOR'd points: {seen}"
    assert seen.get("sor_k") == 30 and seen.get("sor_std") == 1.5, seen
    assert seen.get("max_points_per_obj") == 20000, seen
    assert "oriented_extents" in bbox, f"the stubbed point set must still yield a box: {bbox}"
    if hasattr(dp, "mask_touches_border"):   # GA-315 part 1 landed
        assert "oriented_extents" not in clipped_bbox and \
            clipped_bbox.get("orientation_skipped") == "mask_clipped", \
            f"a clipped mask must get no PCA keys and the skip marker: {clipped_bbox}"


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
    add_fusion_view(a, "chair", "cycle-17", 0.03, [0, 0, 0, 1, 2, 3])
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
        saved_a = next(e for e in on_disk if e["object_id"] == "obj_a")
        assert saved_a["bbox"] == BOX, "fusion must not replace association geometry"
        assert saved_a["fused_bbox"]["source"] == "multi_observation_voxel_agreement"
        assert saved_a["bbox_fusion"]["view_count"] == 1
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


def merge_survivor_by_grade():
    """GA-314: the merge survivor ranks credibility BEFORE age. A declined older object
    must not outlive an admitted newer one; with equal grades the GA-25 rule stands
    (older wins); an ungraded object ranks with no_grounds; and the grade must survive
    the persisted JSON so a bundle can show which side was the credible one."""
    import tempfile

    from object_services import merge_rank

    class Stub:
        description = "unknown"

    old, new = Stub(), Stub()
    old.label, old.object_id, old.creation_time, old.admission_grade = "doorway", "obj_old", 1.0, "decline"
    new.label, new.object_id, new.creation_time, new.admission_grade = "bed", "obj_new", 2.0, "admit"
    assert merge_rank(new) < merge_rank(old), "admit must outrank decline whatever the age"
    old.admission_grade = "admit"
    assert merge_rank(old) < merge_rank(new), "equal grades: the older identity survives (GA-25)"
    ungraded = Stub()
    ungraded.label, ungraded.object_id, ungraded.creation_time = "chair", "obj_u", 0.5
    assert merge_rank(new) < merge_rank(ungraded), "admit outranks an ungraded object"
    old.admission_grade = "decline"
    assert merge_rank(ungraded) < merge_rank(old), "an ungraded object outranks a declined one"

    # GA-372 (owner ruling 2026-09-08): equal grades fall to admission_filled before age.
    old.admission_grade, new.admission_grade = "admit", "admit"
    old.admission_filled, new.admission_filled = 3, 6
    assert merge_rank(new) < merge_rank(old), "equal grades: more filled slots beats age"
    new.admission_filled = 3
    assert merge_rank(old) < merge_rank(new), "equal grades and filled: the older identity survives"
    del new.admission_filled
    assert merge_rank(old) < merge_rank(new), "a missing filled count ranks as 0"
    old.admission_grade = "decline"
    assert merge_rank(new) < merge_rank(old), "the grade still comes before filled"

    original = object_services.PROJECT_ROOT
    wm.persistent_perceptions.clear()
    a = object_info.Object("chair", [0.0, 0.0, 0.0], BOX)
    a.object_id, a.admission_grade, a.admission_filled = "obj_g", "hold", 5
    wm.persistent_perceptions.append(a)
    with tempfile.TemporaryDirectory() as tmp:
        object_services.PROJECT_ROOT = tmp
        try:
            object_services.save_persistent_perceptions(rosstub.Any())
            with open(os.path.join(tmp, "output", "persistent_perception.json")) as fh:
                on_disk = json.load(fh)
        finally:
            object_services.PROJECT_ROOT = original
    assert on_disk[0]["admission_grade"] == "hold", "the grade must reach the bundle"
    assert on_disk[0]["admission_filled"] == 5, "the filled count must reach the bundle"


def orientation_fusion():
    """GA-315 part 2: the stored yaw is the axial mean of the accepted views, the extents
    come from ONE measured view (the one nearest that axis), a yaw-less view keeps the
    axis and updates the AABB, and the +-90 wrap does not fold the axis to zero."""
    import math

    from object_services import fuse_orientation

    def view(yaw_deg, x=0.0):
        return {"x_min": x, "x_max": x + 2.0, "y_min": 0.0, "y_max": 1.0, "z_min": 0.0, "z_max": 0.5,
                "yaw": math.radians(yaw_deg), "oriented_center": [x + 1.0, 0.5, 0.25],
                "oriented_extents": [2.0, 1.0, 0.5]}

    class Obj:
        bbox = None

    # the audit's real bed: four far views, then the clipped 45-degree wedge
    o = Obj()
    for deg in (-86.6, -82.2, -77.7, -68.7):
        o.bbox, o._yaw_acc = fuse_orientation(o, view(deg))
    fused = math.degrees(o.bbox["yaw"])
    assert abs(fused - (-78.9)) < 1.0, f"axial mean of the four views: {fused}"
    assert o.bbox["yaw_views"] == 4 and o.bbox["has_orientation"] is True
    # the representative is chosen as the views arrive (greedy, documented): it is a REAL
    # view, within a few degrees of the axis, and its extents are the ones stored
    rep = math.degrees(o.bbox["yaw_view"])
    assert rep in (-86.6, -82.2, -77.7, -68.7) or abs(rep - (-82.2)) < 1e-6, rep
    assert abs(rep - fused) < 5.0, f"the representative must sit near the axis: {rep} vs {fused}"
    o.bbox, o._yaw_acc = fuse_orientation(o, view(46.0))
    pulled = math.degrees(o.bbox["yaw"])
    assert abs(pulled - (-78.9)) < 10.0 and o.bbox["yaw_views"] == 5, \
        f"one wedge view must not capture the axis: {pulled}"
    assert abs(math.degrees(o.bbox["yaw_view"]) - 46.0) > 30.0, "the wedge is not the representative"

    # a clipped (yaw-less) view: AABB moves, axis stays, count stays
    unoriented = {k: v for k, v in view(0.0, x=0.3).items()
                  if k not in ("yaw", "oriented_center", "oriented_extents")}
    unoriented["has_orientation"] = False
    o.bbox, o._yaw_acc = fuse_orientation(o, unoriented)
    assert o.bbox["x_min"] == 0.3 and abs(math.degrees(o.bbox["yaw"]) - pulled) < 1e-9
    assert o.bbox["yaw_views"] == 5 and o.bbox["has_orientation"] is True and "oriented_extents" in o.bbox

    # the wrap: +85 and -85 are the SAME axis, 10 degrees apart; a scalar mean says 0
    w = Obj()
    for deg in (85.0, -85.0):
        w.bbox, w._yaw_acc = fuse_orientation(w, view(deg))
    d = (math.degrees(w.bbox["yaw"]) - 90.0) % 180.0
    assert min(d, 180.0 - d) < 1e-6, f"the fused axis must sit at +-90, not 0: {math.degrees(w.bbox['yaw'])}"

    # an object with no views and an unoriented arrival: nothing invented
    n = Obj()
    n.bbox, n._yaw_acc = fuse_orientation(n, dict(unoriented))
    assert "yaw" not in n.bbox and n._yaw_acc["n"] == 0

    # A persisted/previously accepted oriented box followed by a clipped view used
    # to crash because initialization incremented the accumulator without storing
    # its representative view.
    prior = Obj()
    prior.bbox = view(15.0)
    prior.bbox, prior._yaw_acc = fuse_orientation(prior, dict(unoriented))
    assert prior.bbox["has_orientation"] is True
    assert "oriented_center" in prior.bbox and prior._yaw_acc["view"] is not None

    # Also tolerate a legacy accumulator that has n/c/s but no representative.
    legacy = Obj()
    legacy.bbox = view(20.0)
    legacy._yaw_acc = {"n": 1, "c": 1.0, "s": 0.0, "view": None}
    legacy.bbox, legacy._yaw_acc = fuse_orientation(legacy, dict(unoriented))
    assert legacy.bbox["has_orientation"] is True


def ontology_veto_seam():
    """GA-309: the hook's optional `disjoint` reaches the ontology channel and the record
    names which component answered; an absent attribute reproduces today's record exactly
    (no `disjoint_source`, "disjoint": False as the literal default); a raise is not caught."""
    import association as assoc

    def obj(oid, typ):
        o = assoc.AssocObject(object_id=oid, bbox=dict(BOX), centroid=[0.5, 0.5, 0.5], label=oid,
                              onto_type=typ)
        o.onto_aligned = True
        return o

    a, b = obj("a", "Bed"), obj("b", "Pillow")

    class Hook:
        name = "kg-test"

        def disjoint(self, x, y):
            return {x, y} == {"Bed", "Pillow"}

    svc = object_services.ObjectServices.__new__(object_services.ObjectServices)
    svc.filter_hook = Hook()
    ctx, _built = object_services.ObjectServices._assoc_build(svc, [])
    assert ctx.disjoint_fn is not None and ctx.disjoint_source == "kg-test", ctx.disjoint_source
    ps = assoc.score_pair(a, b, ctx)
    assert ps.vetoed and ps.vetoed_by == ["ontology"], ps.vetoed_by
    assert ps.channels["ontology"]["disjoint_source"] == "kg-test", ps.channels["ontology"]

    class NoAttr:
        name = "exemplar"

    svc.filter_hook = NoAttr()
    ctx0, _ = object_services.ObjectServices._assoc_build(svc, [])
    assert ctx0.disjoint_fn is None and ctx0.disjoint_source is None
    ps0 = assoc.score_pair(a, b, ctx0)
    # the record every bundle before 2026-09-07 carried, byte for byte: no disjoint_source
    assert not ps0.vetoed and ps0.channels["ontology"] == {
        "log_odds": 0.0, "type_a": "Bed", "type_b": "Pillow", "disjoint": False}, ps0.channels["ontology"]

    class Raises:
        name = "kg-strict"

        def disjoint(self, x, y):
            raise KeyError(x)

    svc.filter_hook = Raises()
    ctx1, _ = object_services.ObjectServices._assoc_build(svc, [])
    try:
        assoc.score_pair(a, b, ctx1)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown class name must raise through the channel (rule 14)")


def overlap_null_uses_larger_box():
    """GA-307: the overlap null is the LARGER box's share of the map, not the intersection's,
    so a sliver of contact scores less than a real containment of a big fragment."""
    import math

    import association as assoc

    vmap = 100.0
    door = (0.0, 1.0, 0.0, 0.1, 0.0, 2.0)                  # 0.2 m3
    light = (0.5, 0.55, 0.0, 0.05, 1.9, 2.0)               # ~250 cm3, fully inside the door
    lo_sliver, d = assoc.channel_overlap(door, light, vmap)
    assert abs(d["larger_m3"] - 0.2) < 1e-9, d
    assert abs(lo_sliver - math.log(vmap / 0.2) * 1.0) < 1e-9, (lo_sliver, d)
    big_a = (0.0, 2.0, 0.0, 1.0, 0.0, 1.0)                 # 2 m3
    big_b = (0.1, 2.1, 0.0, 1.0, 0.0, 1.0)                 # 2 m3, 1.9 m3 shared
    lo_same, _ = assoc.channel_overlap(big_a, big_b, vmap)
    assert lo_same < lo_sliver, "two views of one large object score below a tiny fragment: the null says a small box is the likelier chance overlap"
    # a genuine fragment/whole pair keeps a strong positive term
    assert lo_sliver > 3.0 and lo_same > 3.0, (lo_sliver, lo_same)


def merge_pending_blob():
    """GA-339 (a)+(b): needs_max rides in merge_pending.json (0 when nothing is pending), and a
    non-serialisable value RAISES instead of being warned away (rule 14).

    GA-341 follow-up: the blob also carries the EFFECTIVE floor of the arm that ran
    (engine, min_similarity, max_distance_m, sim_threshold), because run 1 showed the
    similarity floor reaches no other artefact — the evidence arm emits threshold_log_odds
    and a criterion naming threshold_similarity had no subject to test."""
    import tempfile

    import association as assoc

    svc = object_services.ObjectServices.__new__(object_services.ObjectServices)
    svc.get_logger = lambda: rosstub.Any()
    svc._merge_sweep = 7
    h = assoc.Hypothesis(("a", "b"))
    h.state = {"overlap": 5.0}
    h._streak = 0
    h2 = assoc.Hypothesis(("c", "d"))
    h2.state = {"overlap": 4.0}
    h2._streak = 1
    svc._hypotheses = {("a", "b"): h, ("c", "d"): h2}
    old = os.environ.get("GRAPH_API_OUTPUT_DIR")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["GRAPH_API_OUTPUT_DIR"] = tmp
        try:
            object_services.ObjectServices._publish_merge_pending(svc)
            with open(os.path.join(tmp, "merge_pending.json")) as fh:
                blob = json.load(fh)
            assert blob["pending"] == 2 and blob["needs_max"] == object_services.MERGE_MIN_CONSECUTIVE, blob
            # GA-341 follow-up: the blob carries the EFFECTIVE floor of the arm that ran.
            for k, want in (("engine", object_services.MERGE_ENGINE),
                            ("min_similarity", object_services.MERGE_MIN_SIMILARITY),
                            ("max_distance_m", object_services.MERGE_MAX_DISTANCE),
                            ("sim_threshold", object_services.SIM_THRESHOLD)):
                assert blob[k] == want, (k, blob.get(k), want)
            assert blob["min_similarity"] > blob["sim_threshold"], blob
            svc._hypotheses = {}
            object_services.ObjectServices._publish_merge_pending(svc)
            with open(os.path.join(tmp, "merge_pending.json")) as fh:
                assert json.load(fh)["needs_max"] == 0
            svc._merge_sweep = object()   # not JSON-serialisable
            try:
                object_services.ObjectServices._publish_merge_pending(svc)
            except TypeError:
                pass
            else:
                raise AssertionError("a non-serialisable blob must raise, not warn")
        finally:
            if old is None:
                os.environ.pop("GRAPH_API_OUTPUT_DIR", None)
            else:
                os.environ["GRAPH_API_OUTPUT_DIR"] = old


def gt_semantic_cache_depth():
    """GA-353: the GT semantic cache keeps the compressed blob for GT_SEMANTIC_CACHE_FRAMES
    arrivals and decodes at lookup, so a frame looked up 100 arrivals later is still exact;
    the 9th-oldest of 8 (the old depth) must NOT be the eviction point any more; a frame
    beyond the depth is None, never a neighbour; and the cycle row says whether it hit."""
    import gt_codec
    import numpy as _np
    from detection_archive import frame_id_from_stamp

    node = perception_2.DetectObjectsNode.__new__(perception_2.DetectObjectsNode)
    node.log_both = lambda *a, **k: None
    node._gt_semantic = {}

    class Stamp:
        def __init__(self, i):
            self.sec, self.nanosec = 1700000000 + i, 123
    frames = {}
    for i in range(perception_2.GT_SEMANTIC_CACHE_FRAMES + 30):
        arr = _np.full((6, 8), i, dtype=_np.int32)
        arr[0, 0] = 7
        msg = rosstub.Any()
        msg.header = rosstub.Any()
        msg.header.stamp = Stamp(i)
        msg.data = gt_codec.encode(arr)
        perception_2.DetectObjectsNode._gt_semantic_callback(node, msg)
        frames[i] = (frame_id_from_stamp(Stamp(i)), arr)
    n = perception_2.GT_SEMANTIC_CACHE_FRAMES
    assert len(node._gt_semantic) == n, len(node._gt_semantic)
    newest = n + 30 - 1
    fid, arr = frames[newest - 100]                      # 100 arrivals ago (9 was the old cliff)
    got = perception_2.DetectObjectsNode._gt_semantic_for(node, fid)
    assert got is not None and _np.array_equal(got, arr), "a frame 100 arrivals back must decode exactly"
    assert isinstance(next(iter(node._gt_semantic.values())), bytes), "the cache holds blobs, not arrays"
    fid_old, _ = frames[newest - n]                      # one past the depth
    assert perception_2.DetectObjectsNode._gt_semantic_for(node, fid_old) is None, "beyond the depth is None, never a neighbour"
    assert n >= 100, "the depth must cover 11 f/s x a 9.4 s p95 lookup latency"


def oriented_boxes_are_yaw_only():
    """GA-360: an emitted orientation is a yaw about map z and nothing else. The PCA output
    carries exactly yaw / oriented_center / oriented_extents (no roll, pitch or quaternion),
    and the corner builder every renderer uses yields a box whose eight corners sit on two z
    levels with vertical edges parallel to z -- for ANY yaw. A rotation about another axis
    breaks both assertions."""
    import math

    import detection_pipeline as dp
    import numpy as _np
    from box_view import box_corners_map

    rng = _np.random.default_rng(360)
    pts = rng.normal(size=(400, 3)) * _np.array([1.5, 0.3, 0.4]) + _np.array([2.0, 1.0, 0.7])
    ob = dp.pca_oriented_box(pts)
    assert ob is not None and set(ob) == {"yaw", "oriented_center", "oriented_extents"}, ob
    assert isinstance(ob["yaw"], float) and -math.pi / 2 <= ob["yaw"] < math.pi / 2, ob["yaw"]
    for yaw_deg in (0.0, 17.0, -45.0, 89.0, 123.0):
        box = {"oriented_center": [2.0, 1.0, 0.7], "oriented_extents": [3.0, 0.6, 0.8],
               "yaw": math.radians(yaw_deg)}
        corners, oriented = box_corners_map(box)
        assert oriented and len(corners) == 8
        zs = sorted({round(float(c[2]), 9) for c in corners})
        assert zs == [round(0.7 - 0.4, 9), round(0.7 + 0.4, 9)], f"yaw {yaw_deg}: z levels {zs}"
        # the corner list pairs (sx, sy, -ez) with (sx, sy, +ez): each vertical edge is pure z
        for lo, hi in zip(corners[0::2], corners[1::2]):
            assert abs(lo[0] - hi[0]) < 1e-9 and abs(lo[1] - hi[1]) < 1e-9, f"yaw {yaw_deg}: tilted edge"
    for forbidden in ("roll", "pitch", "quat", "quaternion", "rotation"):
        assert forbidden not in ob, forbidden


def localisation_gate():
    """GA-359: under pose_source rtabmap a cycle runs only with a /localization_pose younger
    than localization_max_age_s; a stale or absent pose skips the cycle and counts it; under
    simulator the gate is open and the counter stays a measured 0; the latency row carries both."""
    import time as _t

    node = perception_2.DetectObjectsNode.__new__(perception_2.DetectObjectsNode)
    node.log_both = lambda *a, **k: None
    node.pose_source = "rtabmap"
    node.localization_max_age_s = 5.0
    node._last_localization_time = None
    node.cycles_skipped_unlocalised = 0
    ran = []
    node.publish_objects = lambda frame=None: ran.append(1)
    node._perception_lock = __import__("threading").Lock()
    node.get_clock = lambda: rosstub.Any()
    node.first_detection_done = False
    node.is_stationary = True
    node.manual_trigger_requested = False
    perception_2.DetectObjectsNode._run_perception_cycle(node)          # no pose ever: skip
    assert ran == [] and node.cycles_skipped_unlocalised == 1
    node._last_localization_time = _t.monotonic() - 6.0                 # stale: skip
    perception_2.DetectObjectsNode._run_perception_cycle(node)
    assert ran == [] and node.cycles_skipped_unlocalised == 2
    node._last_localization_time = _t.monotonic() - 1.0                 # fresh: run
    perception_2.DetectObjectsNode._run_perception_cycle(node)
    assert ran == [1] and node.cycles_skipped_unlocalised == 2
    node.pose_source = "simulator"
    node._last_localization_time = None
    perception_2.DetectObjectsNode._run_perception_cycle(node)           # gate off
    assert ran == [1, 1] and node.cycles_skipped_unlocalised == 2


for name, fn in [("description chain (build -> publish -> world model)", description_chain),
                 ("bbox fusion keys and view ID reach the typed message", bbox_fusion_message),
                 ("parallel perception mirrors the named base", parallel_perception_tracks_its_base),
                 ("parallel timing names the measured backend", parallel_timing_names_the_backend),
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
                 ("merge survivor: credibility, then filled slots, before age (GA-314, GA-372)", merge_survivor_by_grade),
                 ("yaw fused over accepted views, extents from one (GA-315 part 2)", orientation_fusion),
                 ("ontology veto reaches the channel and names its source (GA-309)", ontology_veto_seam),
                 ("overlap null charges the larger box, not the sliver (GA-307)", overlap_null_uses_larger_box),
                 ("GT semantic cache: 120 compressed frames, exact lookup (GA-353)", gt_semantic_cache_depth),
                 ("oriented boxes are yaw-only: two z levels, vertical edges (GA-360)", oriented_boxes_are_yaw_only),
                 ("localisation gate: stale pose skips and counts, simulator arm open (GA-359)", localisation_gate),
                 ("merge_pending carries needs_max; a bad blob raises (GA-339)", merge_pending_blob),
                 ("merge request below the match gate is refused (GA-341)", merge_request_below_match_gate_refused),
                 ("detector failure skips the cycle, counted (GA-427)", detector_failure_skips_the_cycle),
                 ("merge lock covers the writes, not the sweep (GA-393)", merge_lock_covers_writes_only),
                 ("every config key the code reads is declared", every_config_key_read_is_declared),
                 ("broad phase widens to the request's distance, never narrows",
                  merge_request_distance_survives_the_broad_phase),
                 ("frame queue processes a snapshot while the gate says moving",
                  frame_queue_decouples_processing_from_the_motion_gate),
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
