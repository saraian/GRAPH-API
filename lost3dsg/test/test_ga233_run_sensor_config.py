import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

SCRIPT = Path(__file__).with_name("run_sensor_config.py")
SPEC = importlib.util.spec_from_file_location("run_sensor_config", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config(tmp_path, habitat=None):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"habitat": habitat or {}}))
    return path


def test_yaml_values_drive_calibration_and_export_tuple(tmp_path):
    cfg = config(tmp_path, {"width": 640, "height": 480, "hfov": 90})
    out = tmp_path / "calibration.json"
    proc = subprocess.run([sys.executable, str(SCRIPT), str(cfg), str(out)],
                          text=True, capture_output=True, check=True, env={})
    assert proc.stdout.strip().split("\t") == ["640", "480", "90.0"]
    cal = json.loads(out.read_text())
    assert cal["resolution"] == {"width": 640, "height": 480}
    assert cal["intrinsics"] == {"fx": 320.0, "fy": 320.0, "cx": 320.0, "cy": 240.0}


def test_nonempty_environment_overrides_yaml(tmp_path):
    got = MODULE.resolve_sensor(
        config(tmp_path, {"width": 640, "height": 480, "hfov": 90}),
        {"FEED_WIDTH": "800", "FEED_HEIGHT": "600", "FEED_HFOV": "75"},
    )
    assert (got["width"], got["height"], got["hfov"]) == (800, 600, 75.0)
    expected_fx = 400.0 / math.tan(math.radians(75.0) / 2.0)
    assert got["calibration"]["intrinsics"]["fx"] == round(expected_fx, 4)


def test_empty_environment_values_use_yaml(tmp_path):
    got = MODULE.resolve_sensor(config(tmp_path, {"width": 320, "height": 200, "hfov": 70}),
                                {"FEED_WIDTH": "", "FEED_HEIGHT": "  ", "FEED_HFOV": ""})
    assert (got["width"], got["height"], got["hfov"]) == (320, 200, 70.0)


def test_missing_values_use_defaults(tmp_path):
    got = MODULE.resolve_sensor(config(tmp_path), {})
    assert (got["width"], got["height"], got["hfov"]) == (1280, 960, 90.0)


@pytest.mark.parametrize("values", [
    {"width": 0}, {"height": -1}, {"width": 640.5},
    {"hfov": 0}, {"hfov": 180}, {"hfov": "nan"}, {"width": "bad"},
])
def test_invalid_sensor_values_are_refused(tmp_path, values):
    habitat = {"width": 640, "height": 480, "hfov": 90}
    habitat.update(values)
    with pytest.raises(ValueError):
        MODULE.resolve_sensor(config(tmp_path, habitat), {})
