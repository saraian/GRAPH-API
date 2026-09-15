#!/usr/bin/env python3
import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
from config import CFG
from crop_context import (
    CALL_FAILED,
    MODEL_ABSTAINED,
    OK,
    PARSE_FAILED,
    classify_description_result,
)


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
    def __init__(self, vlm_call_fn, image_encoder_fn, crop_call_fn=None):
        self._vlm_call = vlm_call_fn
        # GA-303: the describer calls (single crop, crop grid) go through this one, which
        # perception_2 binds to cfg vlm.crop_timeout; the label call keeps vlm.timeout.
        self._crop_call = crop_call_fn or vlm_call_fn
        self._encode = image_encoder_fn
        self.last_room_belief = None
        self.cache = CropVlmCache()

    def call_labels(self, prompt_path, rgb, current_room="unknown", room_evidence=""):
        prompt = open(prompt_path).read()
        prompt = prompt.replace("{CURRENT_ROOM}", current_room or "unknown")
        prompt = prompt.replace("{ROOM_EVIDENCE}", room_evidence or "none")
        raw = self._vlm_call(prompt, self._encode(rgb))
        return self.parse_labels_response(raw)

    def call_scene(self, prompt_path, rgb, excluded=None):
        """Analyze the complete frame in one structured VLM request.

        The returned boxes are converted to coordinates in the original image,
        even when the encoder downsizes the image before transport: the response
        uses aspect-preserving 0..1000 normalized coordinates.
        """
        from scene_analysis import (
            SCENE_ANALYSIS_RESPONSE_FORMAT,
            normalize_excluded_labels,
            parse_scene_analysis,
        )

        if excluded is None:
            excluded = (CFG.get("perception", {}) or {}).get("excluded_labels", [])
        excluded = normalize_excluded_labels(excluded)

        with open(prompt_path, encoding="utf-8") as prompt_file:
            prompt = prompt_file.read()
        raw = self._vlm_call(
            prompt,
            self._encode(rgb),
            response_format=SCENE_ANALYSIS_RESPONSE_FORMAT,
            image_detail="high",
        )
        height, width = rgb.shape[:2]
        return parse_scene_analysis(
            raw,
            image_width=width,
            image_height=height,
            excluded_labels=excluded,
        )

    def _clean_labels(self, raw_labels):
        """Lemmatise and de-duplicate, logging what was collapsed. GA-285.

        A dropped label is PRINTED, not swallowed: a term the VLM asked for and the detector
        never saw is a fact about the run, and silently shrinking the request is how a
        vocabulary gap becomes invisible.
        """
        try:
            import label_norm
        except ImportError:
            # No normaliser on the path is not a reason to ask for duplicates, but it is also
            # not a reason to fail: fall back to the previous behaviour and say so once.
            if not getattr(self, "_label_norm_warned", False):
                self._label_norm_warned = True
                print("[vlm] label_norm unavailable; label list NOT de-duplicated", flush=True)
            return [str(x).strip().lower() for x in raw_labels if str(x).strip()]
        kept, dropped = label_norm.clean(raw_labels)
        if dropped:
            print("[vlm] label list: " + ", ".join(
                f"{d!r} -> {k!r} ({why})" for d, k, why in dropped), flush=True)
        return kept

    def parse_labels_response(self, raw: str):
        """VLM reply -> the label list the detector is asked for.

        GA-285. EVERY return path goes through label_norm.clean(). Two of the three did no
        de-duplication at all, and none collapsed synonyms -- so a reply containing both
        "door" and "doorway" asked the detector for both, and it returned the same pixels
        twice at IoS 1.00 in every frame. Class-aware NMS cannot merge them afterwards; the
        duplicates have to not be created.
        """
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
                        return self._clean_labels(objects)
        except Exception:
            pass

        # 2. Try parsing JSON array directly
        try:
            match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
            if match:
                items = json.loads(match.group(0))
                if isinstance(items, list):
                    return self._clean_labels(items)
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
                if part:
                    labels.append(part)
        return self._clean_labels(labels)

    def call_image_prompt(self, image, prompt):
        """One image, one prompt, RAW text back. GA-209.

        The grid describer needs a call that does not assume a single object: no per-crop
        cache (a grid is a composite that never repeats), no crop-response parsing (the
        reply is a list of cells, parsed by crop_grid.parse), and no default-result
        swallowing -- the CALLER decides what a failure means, because for a grid the right
        answer to a bad reply is to retry the missing cells individually, not to return
        "unknown" for all of them.

        Raising rather than returning a default is deliberate: `call_crop_full` catches its
        own failure and hands back a filled-in "unknown" record, which is right when one
        object is at stake and wrong when six are.
        """
        return self._crop_call(prompt, self._encode(image))

    def call_crop(self, prompt_path, label):
        return open(prompt_path).read().strip().replace("{LABEL}", label)

    def parse_crop_response(self, raw, label):
        """-> the parsed record, or None when the reply could not be read. W6.

        The previous version returned an all-"unknown" default on a parse failure, making a
        malformed reply byte-identical to a genuine model "unknown" -- the conflation that
        inflated the reported unknown-description rate (49%/59%) with parse failures the
        model never made. `detection_archive` itself calls that outcome "unrecoverable".
        The CALLER now decides what a parse failure means and marks it.
        """
        default_result = {k: "unknown" for k in ("description", "color", "material", "shape")}
        default_result.update({"label": label, "json_answer": "{}"})

        try:
            cleaned = re.sub(r"```json|```", "", raw).strip()
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match is None:
                # No JSON object anywhere in the reply is a PARSE failure, not an
                # abstention: the old `else "{}"` arm turned pure prose into an
                # all-"unknown" record byte-identical to a genuine refusal -- the exact
                # conflation this function exists to remove (caught by the W6 smoke check).
                return None
            obj_data = json.loads(match.group(0)).get("objects", [{}])[0]
            result = dict(default_result)
            result.update({k: obj_data.get(k, "unknown") for k in ("description", "color", "material", "shape")})
            result["json_answer"] = match.group(0)
            return result
        except Exception:
            return None

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
            raw = self._crop_call(prompt, self._encode(cropped))
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
                    "status": CALL_FAILED,
                    "error": "call_failed"
                }
            })
            return default_result

        # W6. classify_description_result (crop_context, decision 6) was built for exactly
        # this split and until now was never called on the live path: a failed call is an
        # infrastructure problem, a failed parse a brittleness problem, an abstention a
        # genuine refusal -- three different fixes that all read as the same "unknown".
        parsed = self.parse_crop_response(raw, label)
        status = classify_description_result(raw, parsed, call_error=None)
        if parsed is None:
            # PARSE_FAILED: the call succeeded, the reply could not be read. The fields stay
            # "unknown" -- an unparseable answer is no answer -- but the record now SAYS so.
            parsed = {k: "unknown" for k in ("description", "color", "material", "shape")}
            parsed.update({"label": label, "json_answer": "{}"})
        provenance = {
            "model": model_name,
            "image_id": image_id,
            "timestamp": now_iso,
            "cached": False,
            "status": status,
            "viewpoint": {"pose_bucket": pose_bucket, "range_bucket": range_bucket},
        }
        if status == PARSE_FAILED:
            provenance["error"] = PARSE_FAILED
        parsed["provenance"] = provenance
        if status in (OK, MODEL_ABSTAINED):
            # Only genuine answers are cached. A parse failure used to enter the cache too,
            # serving the same non-answer on every later view of the same crop.
            self.cache.put(cache_key, parsed, provenance)
        return parsed
