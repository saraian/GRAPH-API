#!/usr/bin/env python3
import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
from config import CFG


def compute_crop_hash(cropped_image) -> str:
    """Compute robust MD5 hash of downscaled crop to ignore micro-pixel jitter."""
    if cropped_image is None or getattr(cropped_image, "size", 0) == 0:
        return "empty"
    try:
        import cv2
        small = cv2.resize(cropped_image, (32, 32), interpolation=cv2.INTER_AREA)
        return hashlib.md5(small.tobytes()).hexdigest()[:16]
    except Exception:
        return hashlib.md5(cropped_image.tobytes()[:2048]).hexdigest()[:16]


def discretize_viewpoint(yaw: float = 0.0, distance: float = 1.0) -> Tuple[int, int]:
    """Bucket viewpoint into 8 angular sectors (45 deg) and 0.5m range bins."""
    angle_norm = (float(yaw) + np.pi) % (2.0 * np.pi)
    pose_bucket = int(angle_norm / (np.pi / 4.0)) % 8
    range_bucket = min(max(0, int(float(distance) / 0.5)), 4)
    return pose_bucket, range_bucket


class CropVlmCache:
    """Per-object crop VLM cache with viewpoint-bucketed keys and provenance tracking."""

    def __init__(self):
        self._cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, cache_key: Tuple[str, int, int]) -> Any:
        if cache_key in self._cache:
            self.hits += 1
            entry = self._cache[cache_key]
            res = dict(entry["result"])
            res["provenance"] = dict(entry["provenance"])
            res["provenance"]["cached"] = True
            return res
        self.misses += 1
        return None

    def put(self, cache_key: Tuple[str, int, int], result: dict, provenance: dict):
        self._cache[cache_key] = {
            "result": dict(result),
            "provenance": dict(provenance),
        }

    def invalidate(self, crop_hash=None, label=None):
        """Conflict-triggered invalidation: purge cached entries on ontological dispute."""
        if crop_hash:
            keys_to_del = [k for k in self._cache if k[0] == crop_hash]
            for k in keys_to_del:
                del self._cache[k]
        elif label:
            keys_to_del = [k for k, v in self._cache.items() if v["result"].get("label") == label]
            for k in keys_to_del:
                del self._cache[k]

    @property
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / float(total), 3) if total > 0 else 0.0,
            "size": len(self._cache),
        }


class VlmClient:
    def __init__(self, vlm_call_fn, image_encoder_fn):
        self._vlm_call = vlm_call_fn
        self._encode = image_encoder_fn
        self.last_room_belief = None
        self.cache = CropVlmCache()

    def call_labels(self, prompt_path, rgb, current_room="unknown", room_evidence=""):
        prompt = open(prompt_path).read()
        prompt = prompt.replace("{CURRENT_ROOM}", current_room or "unknown")
        prompt = prompt.replace("{ROOM_EVIDENCE}", room_evidence or "none")
        raw = self._vlm_call(prompt, self._encode(rgb))
        return self.parse_labels_response(raw)

    def parse_labels_response(self, raw: str):
        if not raw:
            return []
        cleaned = re.sub(r"```json|```", "", raw).strip()

        # 1. Try parsing full JSON dict with "objects" and "room_belief"
        try:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    # TODO (PROV-O): Wire last_room_belief to PROV-O provenance layer (wasDerivedFrom / wasAttributedTo)
                    self.last_room_belief = data.get("room_belief")
                    objects = data.get("objects", [])
                    if isinstance(objects, list) and objects:
                        return [str(x).strip().lower() for x in objects if str(x).strip()]
        except Exception:
            pass

        # 2. Try parsing JSON array directly
        try:
            match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
            if match:
                items = json.loads(match.group(0))
                if isinstance(items, list):
                    return [str(x).strip().lower() for x in items if str(x).strip()]
        except Exception:
            pass

        # 3. Parse bullet points or comma-separated tokens
        labels = []
        for line in cleaned.split("\n"):
            line = re.sub(r"^\s*[-*•\d\.\)]+\s*", "", line).strip()
            if not line:
                continue
            for part in line.split(","):
                part = re.sub(r"[^\w\s-]", "", part).strip().lower()
                if part and part not in labels:
                    labels.append(part)
        return labels

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

    def call_crop_full(self, prompt_path, label, cropped, yaw: float = 0.0, distance: float = 1.0, image_id: str = ""):
        """Single-object VLM description with viewpoint-bucketed crop cache and provenance retention."""
        crop_hash = compute_crop_hash(cropped)
        pose_bucket, range_bucket = discretize_viewpoint(yaw, distance)
        cache_key = (crop_hash, pose_bucket, range_bucket)

        # Cache check: Hit retains original derivation provenance
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = self.call_crop(prompt_path, label)
        now_iso = datetime.now().isoformat()
        model_name = CFG.get("vlm", {}).get("model", "unknown")

        try:
            raw = self._vlm_call(prompt, self._encode(cropped))
            parsed = self.parse_crop_response(raw, label)
            provenance = {
                "model": model_name,
                "image_id": image_id,
                "timestamp": now_iso,
                "cached": False,
                "viewpoint": {"pose_bucket": pose_bucket, "range_bucket": range_bucket},
            }
            parsed["provenance"] = provenance
            self.cache.put(cache_key, parsed, provenance)
            return parsed
        except Exception:
            default_result = {k: "unknown" for k in ("description", "color", "material", "shape")}
            default_result.update({
                "label": label,
                "json_answer": "{}",
                "provenance": {
                    "model": model_name,
                    "image_id": image_id,
                    "timestamp": now_iso,
                    "cached": False,
                    "error": "call_failed"
                }
            })
            return default_result