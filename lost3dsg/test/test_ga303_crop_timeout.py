"""GA-303: cfg vlm.crop_timeout bounds the describer calls (single crop and crop grid) and
nothing else; before this the key was declared and read by nothing."""
import os
import sys
import tempfile
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "perception_module")
sys.path.insert(0, SRC)
import rosstub  # noqa: E402

rosstub.install()
import cv_utils  # noqa: E402
from config import CFG  # noqa: E402
from vlm_call import VlmClient  # noqa: E402

seen = []


def _create(**kw):
    seen.append(kw.get("timeout", "absent"))
    return NS(choices=[NS(message=NS(content='{"description": "d", "color": "c", "material": "m", "shape": "s"}'))])


cv_utils._client = NS(chat=NS(completions=NS(create=_create)))
assert cv_utils.vlm_call("p", "img") == '{"description": "d", "color": "c", "material": "m", "shape": "s"}'
cv_utils.vlm_call("p", "img", timeout=15.0)
assert seen == ["absent", 15.0], seen

trace_events = []
assert cv_utils.vlm_call(
    "p",
    "img",
    trace_fn=trace_events.append,
    request_kind="unified_scene",
) == '{"description": "d", "color": "c", "material": "m", "shape": "s"}'
assert len(trace_events) == 1, trace_events
trace = trace_events[0]
assert trace["request_kind"] == "unified_scene", trace
assert trace["status"] == "ok", trace
assert trace["attempt"] == 1 and trace["attempts_total"] == CFG["vlm"]["retries"] + 1, trace
assert trace["request_ms"] >= 0 and trace["call_ms"] >= 0, trace

calls = []
client = VlmClient(vlm_call_fn=lambda p, i: calls.append("general") or '["chair"]',
                   image_encoder_fn=lambda im: "",
                   crop_call_fn=lambda p, i: calls.append("crop") or '{"description": "d"}')
prompt = os.path.join(tempfile.mkdtemp(), "crop.txt")
open(prompt, "w").write("describe {LABEL}")
client.call_image_prompt(None, "grid prompt")
client.call_crop_full(prompt, "chair", cropped=__import__("numpy").zeros((2, 2, 3), "uint8"))
assert calls == ["crop", "crop"], calls
assert VlmClient(vlm_call_fn=lambda p, i: "x", image_encoder_fn=lambda im: "")._crop_call("p", "i") == "x"

src = open(os.path.join(SRC, "perception_2.py")).read()
assert 'timeout=CFG["vlm"]["crop_timeout"]' in src
assert CFG["vlm"]["crop_timeout"] == 15.0, CFG["vlm"]["crop_timeout"]
print("test_ga303_crop_timeout: ok")
