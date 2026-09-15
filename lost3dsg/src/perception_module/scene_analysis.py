"""Structured whole-scene VLM response contract and validation helpers."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


NORMALIZED_COORDINATE_MAX = 1000.0


def normalize_excluded_labels(excluded):
    """Return configured labels in a form suitable for exact and plural matching."""
    return {
        str(value).strip().lower()
        for value in (excluded or ())
        if str(value).strip()
    }


def is_excluded_label(label: str, excluded) -> bool:
    """Whether *label* is configured as excluded, accepting a simple plural form."""
    normalized = str(label).strip().lower()
    excluded = normalize_excluded_labels(excluded)
    return normalized in excluded or (
        normalized.endswith("s") and normalized[:-1] in excluded
    )


SCENE_ANALYSIS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "scene_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                            "color": {"type": "string"},
                            "material": {"type": "string"},
                            "shape": {
                                "type": "string",
                                "enum": [
                                    "cube",
                                    "sphere",
                                    "cylinder",
                                    "rectangular",
                                    "irregular",
                                    "flat",
                                    "elongated",
                                    "curved",
                                    "angular",
                                    "conical",
                                    "unknown",
                                ],
                            },
                            "bbox": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "x_min": {"type": "integer"},
                                    "y_min": {"type": "integer"},
                                    "x_max": {"type": "integer"},
                                    "y_max": {"type": "integer"},
                                },
                                "required": ["x_min", "y_min", "x_max", "y_max"],
                            },
                        },
                        "required": [
                            "label",
                            "description",
                            "color",
                            "material",
                            "shape",
                            "bbox",
                        ],
                    },
                }
            },
            "required": ["objects"],
        },
    },
}


@dataclass(frozen=True)
class SceneObject:
    """One VLM-detected object with a pixel-space bounding box."""

    label: str
    description: str
    color: str
    material: str
    shape: str
    bbox: tuple[float, float, float, float]


def _text(value: Any, default: str = "unknown") -> str:
    if not isinstance(value, str):
        return default
    value = value.strip()
    return value if value else default


def clip_pixel_bbox(bbox: Any, image_width: int, image_height: int):
    """Validate and clip an ``(x_min, y_min, x_max, y_max)`` pixel box.

    The pipeline uses half-open image intervals: ``x_max == image_width`` and
    ``y_max == image_height`` are valid right/bottom edges, while the first pixel
    inside the box is ``x_min, y_min``. Keeping that convention in one helper avoids
    the old mixture of inclusive OpenCV endpoints and exclusive crop endpoints.
    """
    if image_width <= 0 or image_height <= 0:
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return None
    x_min, y_min, x_max, y_max = values
    if x_min >= x_max or y_min >= y_max:
        return None
    # Intersect both endpoints with the half-open image domain. In particular, a
    # box wholly to the right/bottom must become ``x_min == x_max == width``
    # (and be rejected below), rather than a fake one-pixel strip at the edge.
    x_min = max(0.0, min(x_min, float(image_width)))
    y_min = max(0.0, min(y_min, float(image_height)))
    x_max = max(0.0, min(x_max, float(image_width)))
    y_max = max(0.0, min(y_max, float(image_height)))
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, y_min, x_max, y_max


def _pixel_bbox(bbox: Any, image_width: int, image_height: int):
    if not isinstance(bbox, dict) or image_width <= 0 or image_height <= 0:
        return None

    try:
        x_min = float(bbox["x_min"])
        y_min = float(bbox["y_min"])
        x_max = float(bbox["x_max"])
        y_max = float(bbox["y_max"])
    except (KeyError, TypeError, ValueError):
        return None

    values = (x_min, y_min, x_max, y_max)
    if not all(0.0 <= value <= NORMALIZED_COORDINATE_MAX for value in values):
        return None
    if x_min >= x_max or y_min >= y_max:
        return None

    # Max coordinates may equal the image dimensions: SAM accepts a box whose
    # right or bottom edge lies on the image boundary. The returned pixel box
    # remains half-open, matching NumPy crops and the SAM prompt transform.
    return clip_pixel_bbox(
        (x_min * image_width / NORMALIZED_COORDINATE_MAX,
         y_min * image_height / NORMALIZED_COORDINATE_MAX,
         x_max * image_width / NORMALIZED_COORDINATE_MAX,
         y_max * image_height / NORMALIZED_COORDINATE_MAX),
        image_width,
        image_height,
    )


def _json_text(content: str) -> str:
    """Accept strict JSON and the common fenced form returned by some gateways."""
    if not isinstance(content, str):
        raise ValueError("VLM response is not text")
    value = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else value


def parse_scene_analysis(
    content: str,
    image_width: int,
    image_height: int,
    excluded_labels=None,
):
    """Validate a scene response and convert its normalized boxes to pixels.

    One invalid object is skipped without discarding valid siblings. A malformed
    top-level response, or a non-empty response in which every box is invalid,
    raises so the caller records a failed VLM cycle instead of publishing false
    empty-scene evidence. Configured excluded labels are removed after validation;
    a frame containing only excluded objects is a valid empty scene, not a failed
    VLM response.
    """

    try:
        payload = json.loads(_json_text(content))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("VLM response is not valid JSON") from exc

    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list):
        raise TypeError("VLM response must contain an objects array")

    excluded = normalize_excluded_labels(excluded_labels)
    result = []
    valid_objects = 0
    for item in objects:
        if not isinstance(item, dict):
            continue

        label = _text(item.get("label"), default="")
        bbox = _pixel_bbox(item.get("bbox"), image_width, image_height)
        if not label or bbox is None:
            continue

        valid_objects += 1
        label = label.lower()
        if is_excluded_label(label, excluded):
            continue

        result.append(
            SceneObject(
                label=label,
                description=_text(item.get("description")),
                color=_text(item.get("color")),
                material=_text(item.get("material")),
                shape=_text(item.get("shape")),
                bbox=bbox,
            )
        )

    if objects and valid_objects == 0:
        raise ValueError("VLM response contained objects but no valid detections")

    return result
