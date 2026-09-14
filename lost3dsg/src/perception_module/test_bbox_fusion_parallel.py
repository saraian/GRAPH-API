#!/usr/bin/env python3
"""Exact-output checks for the parallel bbox voxel encoders."""
import importlib.util
import os
import signal
import subprocess
import sys

import numpy as np
from bbox_fusion import VOXEL_SIZE_M, fusion_payload_from_points
from bbox_fusion_parallel import ParallelFusionEncoder

CUDA_RESULT = {"status": "not_run", "devices": []}


def worker_sigint_is_ignored():
    """Run inside the spawned pool and report its signal ownership."""
    return signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def fixture():
    rng = np.random.default_rng(907)
    points = []
    labels = []
    for index in range(8):
        cloud = rng.normal(size=(5000 + index * 137, 3)).astype(np.float64)
        cloud = np.concatenate((cloud, cloud[:31]), axis=0)
        cloud[index, index % 3] = np.nan
        points.append(cloud)
        labels.append(f"object-{index}")
    points.extend((np.array([[30.0, 30.0, 30.0]]), None, np.empty((0, 3))))
    labels.extend(("wall", "lamp", "table"))
    return points, labels


def reference(points, labels):
    return [fusion_payload_from_points(value, label, VOXEL_SIZE_M)
            for value, label in zip(points, labels)]


def test_sequential_identity_and_measurement():
    points, labels = fixture()
    encoder = ParallelFusionEncoder(backend="sequential")
    try:
        assert encoder.encode(points, labels) == reference(points, labels)
        measured = encoder.last_measurement
        assert measured["backend"] == "sequential"
        assert measured["verified_against_sequential"] is True
        assert measured["objects"] == len(points)
        assert measured["elapsed_ms"] >= 0.0
    finally:
        encoder.shutdown()


def test_spawned_cpu_processes_match_and_propagate_worker_failure():
    points, labels = fixture()
    encoder = ParallelFusionEncoder(
        backend="cpu_processes", cpu_workers=4, cpu_chunksize=1
    )
    try:
        assert encoder.encode(points, labels) == reference(points, labels)
        measured = encoder.last_measurement
        assert measured["backend"] == "cpu_processes"
        assert measured["cpu_workers"] == 4
        assert measured["verified_against_sequential"] is True
        assert encoder._process_pool.submit(worker_sigint_is_ignored).result() is True
        try:
            encoder.encode([np.zeros((4, 2))], ["chair"])
        except ValueError as exc:
            assert "shape (N, 3+)" in str(exc)
        else:
            raise AssertionError("a worker failure returned a partial or empty batch")
    finally:
        encoder.shutdown()


def test_cuda_matches_when_available():
    if importlib.util.find_spec("torch") is None:
        CUDA_RESULT["status"] = "skipped: torch is not installed"
        return
    import torch

    if not torch.cuda.is_available():
        CUDA_RESULT["status"] = "skipped: PyTorch CUDA is unavailable"
        return
    points, labels = fixture()
    device_sets = [(0,)]
    if torch.cuda.device_count() >= 2:
        device_sets.append((0, 1))
    for devices in device_sets:
        encoder = ParallelFusionEncoder(
            backend="cuda",
            cuda_devices=devices,
            cuda_max_batch_points=12000,
        )
        try:
            assert encoder.encode(points, labels) == reference(points, labels)
            measured = encoder.last_measurement
            assert measured["backend"] == "cuda"
            assert measured["cuda_devices"] == list(devices)
            assert measured["verified_against_sequential"] is True
        finally:
            encoder.shutdown()
    CUDA_RESULT["status"] = "passed"
    CUDA_RESULT["devices"] = [list(devices) for devices in device_sets]


def test_cuda_selection_stops_when_devices_are_hidden():
    code = """
from bbox_fusion_parallel import ParallelFusionEncoder
try:
    ParallelFusionEncoder(backend='cuda', cuda_devices=(0,))
except RuntimeError as exc:
    assert 'PyTorch' in str(exc), str(exc)
else:
    raise AssertionError('cuda selection silently used another backend')
"""
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"parallel bbox fusion self-check OK ({len(tests)} checks)")
    print(f"CUDA check: {CUDA_RESULT['status']}; devices={CUDA_RESULT['devices']}")
