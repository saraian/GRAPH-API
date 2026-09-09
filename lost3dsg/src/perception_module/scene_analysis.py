"""Structured whole-scene VLM response contract and validation helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


NORMALIZED_COORDINATE_MAX = 1000.0


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
    # right or bottom edge lies on the image boundary.
    return (
        max(0.0, min(x_min * image_width / NORMALIZED_COORDINATE_MAX, image_width - 1.0)),
        max(0.0, min(y_min * image_height / NORMALIZED_COORDINATE_MAX, image_height - 1.0)),
        max(1.0, min(x_max * image_width / NORMALIZED_COORDINATE_MAX, float(image_width))),
        max(1.0, min(y_max * image_height / NORMALIZED_COORDINATE_MAX, float(image_height))),
    )


def _json_text(content: str) -> str:
    """Accept strict JSON and the common fenced form returned by some gateways."""
    if not isinstance(content, str):
        raise ValueError("VLM response is not text")
    value = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else value


def parse_scene_analysis(content: str, image_width: int, image_height: int):
    """Validate a scene response and convert its normalized boxes to pixels.

    One invalid object is skipped without discarding valid siblings. A malformed
    top-level response, or a non-empty response in which every box is invalid,
    raises so the caller records a failed VLM cycle instead of publishing false
    empty-scene evidence.
    """

    try:
        payload = json.loads(_json_text(content))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("VLM response is not valid JSON") from exc

    objects = payload.get("objects") if isinstance(payload, dict) else None
    if not isinstance(objects, list):
        raise TypeError("VLM response must contain an objects array")

    result = []
    for item in objects:
        if not isinstance(item, dict):
            continue

        label = _text(item.get("label"), default="")
        bbox = _pixel_bbox(item.get("bbox"), image_width, image_height)
        if not label or bbox is None:
            continue

        result.append(
            SceneObject(
                label=label.lower(),
                description=_text(item.get("description")),
                color=_text(item.get("color")),
                material=_text(item.get("material")),
                shape=_text(item.get("shape")),
                bbox=bbox,
            )
        )

    if objects and not result:
        raise ValueError("VLM response contained objects but no valid detections")

    return result
