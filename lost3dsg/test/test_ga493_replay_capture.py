import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "src/perception_module/ga493_replay_capture.py"
SPEC = importlib.util.spec_from_file_location("ga493_replay_capture", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_disabled_mode_has_no_side_effect(monkeypatch, tmp_path):
    monkeypatch.delenv("GA493_REPLAY_CAPTURE_DIR", raising=False)
    assert MOD.ReplayCapture.from_environment("producer") is None
    assert list(tmp_path.iterdir()) == []


def test_gt_enabled_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("GA493_REPLAY_CAPTURE_DIR", str(tmp_path / "capture"))
    monkeypatch.setenv("FEED_GT_SEMANTIC", "1")
    with pytest.raises(RuntimeError, match="refuses"):
        MOD.ReplayCapture.from_environment("producer")


def test_lossless_cycle_round_trip_and_completion(tmp_path):
    root = tmp_path / "capture"
    capture = MOD.ReplayCapture(root, "producer", 2, 10_000_000,
                                identity={"model": "open-vocabulary-test"},
                                source_files=[SCRIPT])
    depth = np.array([[1.0004, 2.123456], [np.nan, 4.0]], dtype=np.float32)
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    det = SimpleNamespace(mask=np.array([[1, 0], [0, 1]], dtype=bool), label="object",
                          instance_label="object", bbox=[0, 0, 2, 2], observation=None,
                          clip_embedding=[0.1, 0.2])
    camera = SimpleNamespace(__slots__=("width", "height", "k"), width=2, height=2,
                             k=[1.0] * 9)
    transform = {"stamp": {"sec": 1, "nanosec": 2}, "matrix": list(range(16))}
    capture.producer_cycle("cycle-a", rgb, depth, camera, transform, [det],
                           [{"x_min": 0.0}], [[1.0, 2.0, 3.0]],
                           [{"description": "an object", "color": "blue"}])
    capture.finalize()
    saved = np.load(next((root / "producer/cycles").glob("*.npz")))
    assert saved["depth"].dtype == np.float32
    np.testing.assert_array_equal(saved["depth"], depth)
    np.testing.assert_array_equal(saved["rgb"], rgb)
    np.testing.assert_array_equal(saved["masks"][0], det.mask)
    meta = json.loads(next((root / "producer/cycles").glob("*.json")).read_text())
    assert meta["depth"] == {"dtype": "float32", "shape": [2, 2], "units": "metres"}
    assert meta["detections"][0]["description"]["color"] == "blue"
    complete = json.loads((root / "producer/complete.json").read_text())
    assert complete["cycles"] == 1 and complete["events"] == 1


def test_float64_depth_is_preserved_at_the_actual_consumer_boundary(tmp_path):
    root = tmp_path / "capture"
    capture = MOD.ReplayCapture(root, "producer", 1, 10_000_000)
    depth = np.array([[1.0000000001, np.nan]], dtype=np.float64)
    rgb = np.zeros((1, 2, 3), dtype=np.uint8)
    capture.producer_cycle("cycle-f64", rgb, depth, {}, {}, [], [], [], [])
    capture.finalize()
    saved = np.load(next((root / "producer/cycles").glob("*.npz")))
    assert saved["depth"].dtype == np.float64
    np.testing.assert_array_equal(saved["depth"], depth)
    meta = json.loads(next((root / "producer/cycles").glob("*.json")).read_text())
    assert meta["depth"]["dtype"] == "float64"


def test_gt_fields_are_rejected(tmp_path):
    capture = MOD.ReplayCapture(tmp_path / "capture", "consumer", 1, 10000)
    with pytest.raises(ValueError, match="GT field"):
        capture.event("input", {"ground_truth": [1]})


def test_nonempty_role_directory_is_refused(tmp_path):
    role_dir = tmp_path / "capture/producer"
    role_dir.mkdir(parents=True)
    (role_dir / "existing.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        MOD.ReplayCapture(tmp_path / "capture", "producer", 1, 10000)


def test_byte_limit_fails_closed(tmp_path):
    capture = MOD.ReplayCapture(tmp_path / "capture", "consumer", 1, 10)
    with pytest.raises(RuntimeError, match="byte limit"):
        capture.event("large", {"value": "x" * 100})


def test_completion_refuses_tampered_cycle(tmp_path):
    root = tmp_path / "capture"
    capture = MOD.ReplayCapture(root, "producer", 1, 10_000_000)
    depth = np.ones((2, 2), dtype=np.float32)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    capture.producer_cycle("cycle-a", rgb, depth, {}, {}, [], [], [], [])
    npz_path = next((root / "producer/cycles").glob("*.npz"))
    npz_path.write_bytes(npz_path.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="NPZ hash mismatch"):
        capture.finalize()
    assert not (root / "producer/complete.json").exists()
