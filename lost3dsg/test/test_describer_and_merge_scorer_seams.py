"""Two generic seams.

perception.describer.emit_attributes (default true): false asks the VLM for label and box only,
marks colour/material/shape/description as NOT_REQUESTED in the parsed record and in the
published descriptions, and the text-field similarity drops those terms.

hooks.merge_scorer (default null): a zero-argument factory loaded like hooks.filter; its
score(left, right) decides same/different/defer inside the legacy merge loop, after the
locality gates and before the text-field similarity. Every call is a `merge_score` row.
"""
import os
import re
import sys
import tempfile
from types import SimpleNamespace as NS

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "perception_module")
sys.path.insert(0, SRC)
import rosstub  # noqa: E402

rosstub.install()
import detection_pipeline  # noqa: E402
import hooks  # noqa: E402
import nlp_utils  # noqa: E402
import object_info  # noqa: E402
import object_manager_6 as om6  # noqa: E402
import object_services as osv  # noqa: E402
import perception_2 as p2  # noqa: E402
import scene_analysis  # noqa: E402
from config import CFG, NOT_REQUESTED, describer_emits_attributes  # noqa: E402
from vlm_call import VlmClient  # noqa: E402
from world_model import wm  # noqa: E402

ATTR_WORDS = re.compile(r"colou?r|material|shape|descri", re.I)
FOUR = ("description", "color", "material", "shape")
SCENE_REPLY = ('{"objects": [{"label": "chair", "description": "a chair", "color": "red", '
               '"material": "wood", "shape": "rectangular", '
               '"bbox": {"x_min": 10, "y_min": 10, "x_max": 500, "y_max": 500}}]}')


def set_emit(value):
    CFG["perception"].setdefault("describer", {})["emit_attributes"] = value


def scene_call(client_calls):
    def call(prompt, image, **kw):
        client_calls.append((prompt, kw))
        return SCENE_REPLY
    return VlmClient(vlm_call_fn=call, image_encoder_fn=lambda im: "")


crop_prompt = os.path.join(tempfile.mkdtemp(), "crop.txt")
open(crop_prompt, "w").write(open(os.path.join(SRC, "prompts", "visual_prompt.txt")).read())
rgb = np.zeros((100, 100, 3), dtype=np.uint8)

# ------------------------------------------------------------------ 1. defaults unchanged
assert CFG["perception"]["describer"]["emit_attributes"] is True and describer_emits_attributes()
assert CFG["hooks"]["merge_scorer"] is None
assert detection_pipeline.scene_prompt_path() == os.path.join(
    os.path.dirname(detection_pipeline.__file__), "prompts", "scene_analysis_prompt.txt")
calls = []
objs = scene_call(calls).call_scene(detection_pipeline.scene_prompt_path(), rgb)
assert calls[0][1]["response_format"] is scene_analysis.SCENE_ANALYSIS_RESPONSE_FORMAT
assert (objs[0].color, objs[0].material, objs[0].shape) == ("red", "wood", "rectangular")
crop_calls = []
c = VlmClient(vlm_call_fn=None, image_encoder_fn=lambda im: "",
              crop_call_fn=lambda p, i: crop_calls.append(p) or '{"objects": [{"color": "red"}]}')
r = c.call_crop_full(crop_prompt, "chair", rgb)
assert len(crop_calls) == 1 and r["color"] == "red" and r["material"] == "unknown"
fake = NS(log_both=lambda *a, **k: None)
d = p2.DetectObjectsNode._build_descriptions(fake, [NS(instance_label="chair", bbox=(0, 0, 1, 1))], {})
assert all(d[0][f] == "unknown" for f in FOUR) and d[0]["status"] == "unanswered", d
print("ok defaults: full prompt, full schema, crop call made, absent fields stay 'unknown'")

# ------------------------------------------------------------------ 2. emit_attributes: false
set_emit(False)
path = detection_pipeline.scene_prompt_path()
assert path.endswith("scene_analysis_prompt_no_attributes.txt"), path
text = open(path).read()
assert not ATTR_WORDS.search(text), ATTR_WORDS.findall(text)
full = open(os.path.join(SRC, "prompts", "scene_analysis_prompt.txt")).read().splitlines()
assert [ln for ln in full if not ATTR_WORDS.search(ln.split(":")[0])] == text.splitlines(), \
    "the label/bbox part must be byte-identical to the full prompt"
CFG["paths"]["scene_analysis_prompt"] = "/elsewhere/custom.txt"
try:
    detection_pipeline.scene_prompt_path()
    raise AssertionError("a custom scene prompt cannot be silently replaced")
except ValueError:
    pass
CFG["paths"]["scene_analysis_prompt"] = ""

calls = []
objs = scene_call(calls).call_scene(path, rgb)
item = calls[0][1]["response_format"]["json_schema"]["schema"]["properties"]["objects"]["items"]
assert set(item["properties"]) == {"label", "bbox"} and set(item["required"]) == {"label", "bbox"}, item
assert not ATTR_WORDS.search(calls[0][0])
assert objs[0].label == "chair" and all(getattr(objs[0], f) == NOT_REQUESTED for f in FOUR), objs

crop_calls = []
r = c.call_crop_full(crop_prompt, "chair", rgb)
assert crop_calls == [], "the crop describer has nothing to ask: no call"
assert all(r[f] == NOT_REQUESTED for f in FOUR) and r["provenance"]["status"] == NOT_REQUESTED, r


def no_call(*a, **k):
    raise AssertionError("the grid describer must not call the VLM")


grid = p2.DetectObjectsNode.process_crop_grid(
    NS(vlm=NS(call_image_prompt=no_call)), [{"label": "chair", "cropped": rgb}, {"label": "bed", "cropped": rgb}])
assert all(grid["__grid__"][lab][f] == NOT_REQUESTED for lab in ("chair", "bed") for f in FOUR), grid

det = NS(instance_label="chair", bbox=(0, 0, 1, 1), **{f: getattr(objs[0], f) for f in FOUR})
res = p2.DetectObjectsNode._unified_scene_description_results(fake, [det])
assert all(res["chair"][f] == NOT_REQUESTED for f in FOUR) and res["chair"]["provenance"]["status"] == NOT_REQUESTED
d = p2.DetectObjectsNode._build_descriptions(fake, [det, NS(instance_label="bed", bbox=(0, 0, 1, 1))], res)
assert all(x[f] == NOT_REQUESTED for x in d for f in FOUR), d
assert [x["status"] for x in d] == [NOT_REQUESTED, NOT_REQUESTED], d

published = []
late = NS(_late_descriptions=[("lamp", {"frame": "1_0", "bbox": (0, 0, 1, 1)}, {})],
          pub_object_descriptions_late=NS(publish=published.append),
          make_header_msg=lambda *a, **k: NS(descriptions=[]), log_both=lambda *a, **k: None)
p2.DetectObjectsNode._publish_late_descriptions(late, None)
m = published[0].descriptions[0]
assert all(getattr(m, f) == NOT_REQUESTED for f in FOUR) and m.status == NOT_REQUESTED
print("ok emit_attributes=false: label/box prompt and schema, no crop or grid call, NOT_REQUESTED published")

# ------------------------------------------------------------------ 3. similarity drops the terms
NR = NOT_REQUESTED
s_nr, ev_nr = nlp_utils.lost_similarity_detailed(None, "chair", "chair", NR, NR, NR, NR, None, None)
s_unk, ev_unk = nlp_utils.lost_similarity_detailed(None, "chair", "chair", "unknown", "unknown",
                                                   "unknown", "unknown", None, None)
assert (s_nr, ev_nr) == (s_unk, ev_unk) and ev_nr["optional_count"] == 0, (s_nr, ev_nr)
_, ev_one = nlp_utils.lost_similarity_detailed(None, "chair", "chair", NR, NR, "wood", NR, None, None)
assert ev_one["material"] is False, "NOT_REQUESTED against a value is neither match nor mismatch"
_, ev_wood = nlp_utils.lost_similarity_detailed(None, "chair", "chair", NR, NR, "wood", "wood", None, None)
assert ev_wood["material"] is True, "a real value is still scored"
assert nlp_utils.get_embedding(None, NR) is None
set_emit(True)
print("ok similarity: NOT_REQUESTED terms drop out exactly like 'unknown'")

# ------------------------------------------------------------------ 4. merge scorer: loading
assert hooks.load_merge_scorer({"hooks": {"merge_scorer": ""}}) is None
pkg = tempfile.mkdtemp()
open(os.path.join(pkg, "stub_scorer.py"), "w").write(
    "class Scorer:\n"
    "    def __init__(self):\n        self.calls, self.ticks, self.plan = [], 0, []\n"
    "    def score(self, left, right):\n"
    "        self.calls.append((left['object_id'], right['object_id']))\n"
    "        d = self.plan.pop(0)\n"
    "        if isinstance(d, Exception):\n            raise d\n"
    "        return {'decision': d, 'why': 'stub'}\n"
    "    def tick(self):\n        self.ticks += 1\n"
    "def make():\n    return Scorer()\n"
    "def make_bad():\n    return object()\n")
scorer = hooks.load_merge_scorer({"hooks": {"search_paths": [pkg], "merge_scorer": "stub_scorer:make"}})
for spec, exc in (("stub_scorer:make_bad", TypeError), ("stub_scorer:absent", AttributeError),
                  ("no_such_module_xyz:make", ModuleNotFoundError)):
    try:
        hooks.load_merge_scorer({"hooks": {"search_paths": [pkg], "merge_scorer": spec}})
        raise AssertionError(f"{spec} must raise at start-up")
    except exc:
        pass
print("ok load: null -> None; a factory resolves once; a bad spec raises")

# ------------------------------------------------------------------ 5. merge scorer drives the loop
osv.save_persistent_perceptions = lambda node: None
osv.publish_persistent_bboxes = osv.publish_persistent_centroids = lambda *a, **k: None
rows = []
similarity_calls = []
real_similarity = osv.lost_similarity_detailed


def fake_similarity(*a):
    similarity_calls.append(a)
    return 0.99, {"optional_count": 2}


osv.lost_similarity_detailed = fake_similarity
osv.get_embedding = lambda model, text: None


def box(x):
    return dict(x_min=x, x_max=x + 0.5, y_min=0.0, y_max=0.5, z_min=0.0, z_max=0.5)


def objects():
    a = object_info.Object("chair", None, box(0.0), description="a chair", color="red", material="wood")
    b = object_info.Object("chair#2", None, box(0.1), description="a chair", color="red", material="wood")
    a.object_id, b.object_id, a.creation_time, b.creation_time = "obj_a", "obj_b", 0.0, 1.0
    wm.persistent_perceptions[:] = [a, b]
    return a, b


def svc(merge_scorer):
    s = object.__new__(osv.ObjectServices)
    s.get_logger = lambda: rosstub.Any()
    s.log_both = lambda *a, **k: None
    s.room_manager = NS(current_room_id="room_1", scene_graph={}, update_room_geometry=lambda *a: None,
                        room_at_bbox=lambda b: "room_1")
    s.decision_log = NS(write=lambda kind, obj, **kw: rows.append(dict(kind=kind, object=obj, **kw)))
    s.tracking_step_counter = 0
    s.persistent_bbox_pub = s.persistent_centroids_pub = None
    if merge_scorer is not None:
        s.merge_scorer = merge_scorer
    return s


def merge(merge_scorer):
    rows.clear()
    similarity_calls.clear()
    r = osv.ObjectServices._cb_merge_objects(svc(merge_scorer), NS(max_distance=0.8, min_similarity=0.95,
                                                                  dry_run=False), NS())
    assert r.success, r.message
    return r


objects()
r = merge(None)                                   # default: no scorer, similarity decides
assert r.merged_count == 1 and similarity_calls and not [x for x in rows if x["kind"] == "merge_score"]
assert '"decision_reason": "similarity"' in r.merge_log_json

a, b = objects()
scorer.plan = ["same"]
r = merge(scorer)
assert r.merged_count == 1 and not similarity_calls, "same merges without the text-field similarity"
assert '"decision_reason": "merge_scorer"' in r.merge_log_json and wm.persistent_perceptions == [a]
score_rows = [x for x in rows if x["kind"] == "merge_score"]
assert len(score_rows) == 1 and score_rows[0]["decision"] == "same" and score_rows[0]["object"] == "obj_a"
assert score_rows[0]["candidate"] == "obj_b" and score_rows[0]["elapsed_ms"] >= 0

objects()
scorer.plan = ["different"]
r = merge(scorer)
assert r.merged_count == 0 and len(wm.persistent_perceptions) == 2 and not similarity_calls
assert [x["reason"] for x in rows if x["kind"] == "merge_refused"] == ["scorer_different"]

a, b = objects()
scorer.plan, scorer.calls = ["defer", "same"], []
assert merge(scorer).merged_count == 0 and len(wm.persistent_perceptions) == 2
assert [x["reason"] for x in rows if x["kind"] == "merge_refused"] == ["scorer_defer"]
assert merge(scorer).merged_count == 1 and scorer.calls == [("obj_a", "obj_b")] * 2, \
    "a deferred pair is offered again at the next merge evaluation"

objects()
scorer.plan = [RuntimeError("boom")]
r = merge(scorer)
row = [x for x in rows if x["kind"] == "merge_score"][0]
assert r.merged_count == 0 and row["decision"] == "defer" and "boom" in row["error"], row
scorer.plan = ["maybe"]
objects()
assert merge(scorer).merged_count == 0 and [x for x in rows if x["kind"] == "merge_score"][0]["decision"] == "defer"
osv.lost_similarity_detailed = real_similarity
print("ok merge loop: same merges without similarity, different refuses, defer re-offers, errors defer")

# ------------------------------------------------------------------ 6. tick from the periodic drain
node = object.__new__(om6.ObjectManagerService)
node.get_logger = lambda: rosstub.Any()
om6.ObjectManagerService._tick_merge_scorer(node)            # no scorer: no-op
node.merge_scorer = scorer
ticks = scorer.ticks
om6.ObjectManagerService._tick_merge_scorer(node)
assert scorer.ticks == ticks + 1
node.merge_scorer = NS(score=lambda left, right: {}, tick=lambda: 1 / 0)
om6.ObjectManagerService._tick_merge_scorer(node)            # logged, not raised
node.merge_scorer = NS(score=lambda left, right: {})
om6.ObjectManagerService._tick_merge_scorer(node)            # no tick(): no-op
assert om6.ObjectManagerService._node_dict is not None and om6.ObjectManagerService._node_dict(
    NS(object_id="x", label="l", color="c", material="m", description="d", bbox={}, room_id="r")) == {
    "object_id": "x", "label": "l", "color": "c", "material": "m", "description": "d", "bbox": {},
    "room_id": "r"}
print("ok tick: called when present, errors logged, absent tick is a no-op")
print("test_describer_and_merge_scorer_seams: ok")
