#!/usr/bin/env python3
"""Resolve one run's camera contract and write calibration.json."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import yaml

DEFAULTS = {"width": 1280, "height": 960, "hfov": 90.0}
ENV_KEYS = {"width": "FEED_WIDTH", "height": "FEED_HEIGHT", "hfov": "FEED_HFOV"}


def _choose(name: str, habitat: dict, environ: dict[str, str]):
    env_value = environ.get(ENV_KEYS[name])
    if env_value is not None and str(env_value).strip() != "":
        return env_value
    value = habitat.get(name)
    return DEFAULTS[name] if value is None or str(value).strip() == "" else value


def resolve_sensor(config_path: Path, environ: dict[str, str] | None = None) -> dict:
    environ = os.environ if environ is None else environ
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("run configuration must be a mapping")
    habitat = loaded.get("habitat") or {}
    if not isinstance(habitat, dict):
        raise ValueError("habitat configuration must be a mapping")

    raw_width = _choose("width", habitat, environ)
    raw_height = _choose("height", habitat, environ)
    raw_hfov = _choose("hfov", habitat, environ)
    try:
        width_f, height_f, hfov = float(raw_width), float(raw_height), float(raw_hfov)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera width, height and hfov must be numeric") from exc
    if not all(math.isfinite(v) for v in (width_f, height_f, hfov)):
        raise ValueError("camera width, height and hfov must be finite")
    if width_f <= 0 or height_f <= 0 or not width_f.is_integer() or not height_f.is_integer():
        raise ValueError("camera width and height must be positive integers")
    if not 0.0 < hfov < 180.0:
        raise ValueError("camera hfov must be between 0 and 180 degrees")

    width, height = int(width_f), int(height_f)
    fx = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)
    return {
        "width": width,
        "height": height,
        "hfov": hfov,
        "calibration": {
            "camera_name": "habitat_camera_optical",
            "resolution": {"width": width, "height": height},
            "hfov_deg": hfov,
            "intrinsics": {
                "fx": round(fx, 4), "fy": round(fx, 4),
                "cx": width / 2.0, "cy": height / 2.0,
            },
            "distortion_model": "plumb_bob",
            "distortion_coefficients": [0.0] * 5,
            "source": "resolved_config",
            "source_note": (
                "GA-233. One validated resolution supplies calibration, run metadata and "
                "the feed environment. Non-empty FEED_* overrides YAML habitat values."
            ),
        },
    }


def write_calibration(path: Path, calibration: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        json.dump(calibration, stream, indent=2)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("calibration", type=Path)
    args = parser.parse_args()
    result = resolve_sensor(args.config)
    write_calibration(args.calibration, result["calibration"])
    print(f"{result['width']}\t{result['height']}\t{result['hfov']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
