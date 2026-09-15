"""Focused coverage for the Gemini generateContent VLM transport."""

import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "perception_module")
sys.path.insert(0, SRC)

import rosstub  # noqa: E402

rosstub.install()
# The test exercises the direct Gemini transport and does not need the optional SDK used by
# the legacy OpenAI-compatible transport. Keep this focused test runnable in the lightweight
# repository test environment as well as in the full runtime image.
try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = object
    sys.modules["openai"] = openai_stub
import cv_utils  # noqa: E402
from scene_analysis import SCENE_ANALYSIS_RESPONSE_FORMAT  # noqa: E402


class _Response:
    headers = {"x-request-id": "gemini-test-request"}

    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def main():
    vlm = cv_utils.CFG["vlm"]
    original = dict(vlm)
    old_key = os.environ.get("GEMINI_API_KEY")
    old_urlopen = cv_utils.urllib.request.urlopen
    seen = {}

    try:
        vlm.update({
            "provider": "auto",
            "base_url": "https://aiplatform.googleapis.com/v1",
            "model": "gemini-3.8-flash",
            "thinking_level": "low",
            "image_mime_type": "image/png",
            "retries": 0,
            "timeout": 7.5,
            "api_key": "",
        })
        os.environ["GEMINI_API_KEY"] = "test-gemini-key"

        assert cv_utils._is_gemini_vlm()
        assert cv_utils._gemini_endpoint() == (
            "https://aiplatform.googleapis.com/v1/publishers/google/models/"
            "gemini-3.8-flash:generateContent"
        )

        def fake_urlopen(request, timeout):
            seen["request"] = request
            seen["timeout"] = timeout
            return _Response(json.dumps({
                "candidates": [{
                    "content": {
                        "parts": [
                            {"text": "internal thought", "thought": True},
                            {"text": '{"objects": []}'},
                        ]
                    },
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 15,
                },
            }).encode("utf-8"))

        cv_utils.urllib.request.urlopen = fake_urlopen
        traces = []
        result = cv_utils.vlm_call(
            "Test.",
            encoded_image="aGVsbG8=",
            response_format=SCENE_ANALYSIS_RESPONSE_FORMAT,
            image_detail="high",
            trace_fn=traces.append,
            request_kind="unified_scene",
        )
        assert result == '{"objects": []}', result
        assert seen["timeout"] == 7.5, seen

        request = seen["request"]
        assert request.full_url == cv_utils._gemini_endpoint()
        headers = {key.lower(): value for key, value in request.header_items()}
        assert headers["x-goog-api-key"] == "test-gemini-key", headers
        assert headers["content-type"] == "application/json", headers
        body = json.loads(request.data.decode("utf-8"))
        assert body["contents"] == [{
            "role": "user",
            "parts": [
                {"text": "Test."},
                {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
            ],
        }], body
        generation = body["generationConfig"]
        assert generation["thinkingConfig"] == {"thinkingLevel": "low"}, generation
        assert generation["responseMimeType"] == "application/json", generation
        assert generation["responseSchema"]["type"] == "OBJECT", generation
        assert "additionalProperties" not in generation["responseSchema"], generation

        assert traces and traces[0]["status"] == "ok", traces
        assert traces[0]["prompt_tokens"] == 11, traces
        assert traces[0]["completion_tokens"] == 4, traces
        assert traces[0]["total_tokens"] == 15, traces
        assert traces[0]["request_id"] == "gemini-test-request", traces
        print("test_gemini_vlm: ok")
    finally:
        vlm.clear()
        vlm.update(original)
        cv_utils.urllib.request.urlopen = old_urlopen
        if old_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = old_key


if __name__ == "__main__":
    main()
