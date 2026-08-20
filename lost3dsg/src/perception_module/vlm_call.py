#!/usr/bin/env python3
import re
import json


class VlmClient:
    def __init__(self, vlm_call_fn, image_encoder_fn):
        self._vlm_call = vlm_call_fn
        self._encode = image_encoder_fn

    def call_labels(self, prompt_path, rgb):
        prompt = open(prompt_path).read()
        raw = self._vlm_call(prompt, self._encode(rgb))
        cleaned = (
            raw.replace("[", "")
            .replace("]", "")
            .replace('"', "")
            .replace("\n", "")
            .strip()
        )
        return [l.strip() for l in cleaned.split(",") if l.strip()]

    def call_crop(self, prompt_path, label):
        return open(prompt_path).read().strip().replace("{LABEL}", label)

    def parse_crop_response(self, raw, label):
        default_result = {k: "unknown" for k in ("description", "color", "material", "shape")}
        default_result.update({"label": label, "json_answer": "{}"})

        try:
            cleaned = re.sub(r"```json|```", "", raw).strip()
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            obj_data = json.loads(match.group(0) if match else "{}").get("objects", [{}])[0]
            result = dict(default_result)
            result.update({k: obj_data.get(k, "unknown") for k in ("description", "color", "material", "shape")})
            result["json_answer"] = match.group(0) if match else "{}"
            return result
        except Exception:
            return default_result

    def call_crop_full(self, prompt_path, label, cropped):
        """Chiamata completa: costruisce prompt, chiama la VLM, parsa la risposta. Sincrona."""
        prompt = self.call_crop(prompt_path, label)
        try:
            raw = self._vlm_call(prompt, self._encode(cropped))
            return self.parse_crop_response(raw, label)
        except Exception:
            default_result = {k: "unknown" for k in ("description", "color", "material", "shape")}
            default_result.update({"label": label, "json_answer": "{}"})
            return default_result