#!/usr/bin/env python3
"""Exact CPU-process and CUDA encoders for one frame of bbox fusion views.

The sequential NumPy encoder in :mod:`bbox_fusion` defines the result.  These
backends may change where and how the work runs.  They must not change voxel
size, float precision, filtering, output order, or the frame boundary.
"""
import multiprocessing
import signal
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from bbox_fusion import (
    VOXEL_SIZE_M,
    fusion_eligible_label,
    fusion_payload_from_points,
)

_INT32_MIN = np.iinfo(np.int32).min
_INT32_MAX = np.iinfo(np.int32).max
BACKENDS = frozenset({"sequential", "cpu_processes", "cuda"})


def _cpu_payload_worker(task):
    """Picklable entry point used by persistent spawned worker processes."""
    points, label, voxel_m = task
    return fusion_payload_from_points(points, label, voxel_m)


def _configure_cpu_worker():
    """Keep the parent ROS process responsible for an orderly SIGINT shutdown."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _normalise_cuda_items(points, labels):
    """Apply the reference input policy before data moves to a CUDA device."""
    payloads = [[] for _ in points]
    active = []
    for index, (point_set, label) in enumerate(zip(points, labels)):
        if point_set is None or not fusion_eligible_label(label):
            continue
        array = np.asarray(point_set, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] < 3:
            raise ValueError("points must have shape (N, 3+)")
        array = array[:, :3]
        array = array[np.all(np.isfinite(array), axis=1)]
        if len(array):
            active.append((index, np.ascontiguousarray(array)))
    return payloads, active


def _point_batches(items, max_points):
    """Keep each object whole and keep the input order inside every CUDA batch."""
    if max_points == 0:
        return [items] if items else []
    batches = []
    current = []
    current_points = 0
    for item in items:
        size = len(item[1])
        if current and current_points + size > max_points:
            batches.append(current)
            current = []
            current_points = 0
        current.append(item)
        current_points += size
        if current_points >= max_points:
            batches.append(current)
            current = []
            current_points = 0
    if current:
        batches.append(current)
    return batches


def _cuda_payloads_on_device(torch_module, items, device_index, voxel_m, max_points):
    """Encode assigned objects on one CUDA device and return index/payload pairs."""
    by_index = {}
    device = torch_module.device(f"cuda:{device_index}")
    with torch_module.inference_mode(), torch_module.cuda.device(device):
        for batch in _point_batches(items, max_points):
            lengths = [len(array) for _, array in batch]
            packed_numpy = np.concatenate([array for _, array in batch], axis=0)
            packed = torch_module.from_numpy(packed_numpy).to(
                device=device, dtype=torch_module.float64
            )
            owners = torch_module.repeat_interleave(
                torch_module.arange(len(batch), device=device, dtype=torch_module.int64),
                torch_module.tensor(lengths, device=device, dtype=torch_module.int64),
            )
            keys = torch_module.floor(packed / voxel_m).to(torch_module.int64)
            if bool(torch_module.any(keys < _INT32_MIN).item()) or bool(
                    torch_module.any(keys > _INT32_MAX).item()):
                raise ValueError("voxel key exceeds int32 transport range")
            combined = torch_module.cat((owners[:, None], keys), dim=1)
            unique = torch_module.unique(combined, sorted=True, dim=0).cpu().numpy()
            for local_index, (source_index, _) in enumerate(batch):
                selected = unique[unique[:, 0] == local_index, 1:]
                if len(selected) > 1:
                    order = np.lexsort((selected[:, 2], selected[:, 1], selected[:, 0]))
                    selected = selected[order]
                by_index[source_index] = selected.astype(
                    np.int32, copy=False
                ).ravel().tolist()
        torch_module.cuda.synchronize(device)
    return by_index


def _split_by_point_count(items, count):
    """Balance whole objects across devices without changing returned order."""
    groups = [[] for _ in range(count)]
    totals = [0] * count
    for item in sorted(items, key=lambda entry: len(entry[1]), reverse=True):
        target = min(range(count), key=totals.__getitem__)
        groups[target].append(item)
        totals[target] += len(item[1])
    return groups


class ParallelFusionEncoder:
    """Persistent execution backend for one perception cycle's voxel payloads."""

    def __init__(self, backend="cpu_processes", cpu_workers=4, cpu_chunksize=1,
                 cuda_devices=(0,), cuda_max_batch_points=0,
                 voxel_m=VOXEL_SIZE_M, verify=True):
        self.backend = str(backend).strip().lower()
        self.cpu_workers = int(cpu_workers)
        self.cpu_chunksize = int(cpu_chunksize)
        self.cuda_devices = tuple(int(device) for device in cuda_devices)
        self.cuda_max_batch_points = int(cuda_max_batch_points)
        self.voxel_m = float(voxel_m)
        if self.backend not in BACKENDS:
            raise ValueError(
                f"perception_parallel.bbox_backend must be one of {sorted(BACKENDS)}"
            )
        if self.voxel_m <= 0.0:
            raise ValueError("voxel_m must be positive")
        if self.cpu_chunksize < 1:
            raise ValueError("bbox_cpu_chunksize must be at least 1")
        if self.cuda_max_batch_points < 0:
            raise ValueError("bbox_cuda_max_batch_points must be non-negative")

        self._process_pool = None
        self._device_pool = None
        self._torch = None
        self._cuda_names = []
        if self.backend == "cpu_processes":
            if self.cpu_workers < 2:
                raise ValueError("cpu_processes requires at least two workers")
            self._process_pool = ProcessPoolExecutor(
                max_workers=self.cpu_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_configure_cpu_worker,
            )
        elif self.backend == "cuda":
            if not self.cuda_devices:
                raise ValueError("cuda backend requires at least one device")
            if len(set(self.cuda_devices)) != len(self.cuda_devices):
                raise ValueError("bbox_cuda_devices must not contain duplicates")
            try:
                import torch
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "cuda backend selected but PyTorch is not installed"
                ) from exc

            if not torch.cuda.is_available():
                raise RuntimeError("cuda backend selected but PyTorch CUDA is unavailable")
            device_count = torch.cuda.device_count()
            invalid = [device for device in self.cuda_devices
                       if device < 0 or device >= device_count]
            if invalid:
                raise ValueError(
                    f"CUDA device indices {invalid} are outside 0..{device_count - 1}"
                )
            self._torch = torch
            self._cuda_names = [torch.cuda.get_device_name(device)
                                for device in self.cuda_devices]
            if len(self.cuda_devices) > 1:
                self._device_pool = ThreadPoolExecutor(
                    max_workers=len(self.cuda_devices),
                    thread_name_prefix="bbox_cuda",
                )

        self.last_measurement = None
        self.verified = False
        if verify:
            self._verify_against_reference()

    @classmethod
    def from_config(cls, config):
        """Build from the explicit ``perception_parallel`` configuration block."""
        section = config.get("perception_parallel", {}) or {}
        return cls(
            backend=section.get("bbox_backend", "cpu_processes"),
            cpu_workers=section.get("bbox_cpu_workers", 4),
            cpu_chunksize=section.get("bbox_cpu_chunksize", 1),
            cuda_devices=section.get("bbox_cuda_devices", [0]),
            cuda_max_batch_points=section.get("bbox_cuda_max_batch_points", 0),
            voxel_m=VOXEL_SIZE_M,
            verify=True,
        )

    @property
    def identity(self):
        """Return the backend identity that must accompany latency measurements."""
        return {
            "component": type(self).__name__,
            "backend": self.backend,
            "voxel_m": self.voxel_m,
            "cpu_workers": self.cpu_workers if self.backend == "cpu_processes" else None,
            "cpu_chunksize": self.cpu_chunksize if self.backend == "cpu_processes" else None,
            "cpu_start_method": "spawn" if self.backend == "cpu_processes" else None,
            "cuda_devices": list(self.cuda_devices) if self.backend == "cuda" else [],
            "cuda_device_names": list(self._cuda_names),
            "cuda_max_batch_points": (
                self.cuda_max_batch_points if self.backend == "cuda" else None
            ),
            "verified_against_sequential": self.verified,
        }

    def _encode_unmeasured(self, points, labels):
        if len(points) != len(labels):
            raise ValueError("points and labels must have the same length")
        if self.backend == "sequential":
            return [fusion_payload_from_points(point_set, label, self.voxel_m)
                    for point_set, label in zip(points, labels)]
        if self.backend == "cpu_processes":
            tasks = [(point_set, label, self.voxel_m)
                     for point_set, label in zip(points, labels)]
            return list(self._process_pool.map(
                _cpu_payload_worker, tasks, chunksize=self.cpu_chunksize
            ))

        payloads, active = _normalise_cuda_items(points, labels)
        if len(self.cuda_devices) == 1:
            encoded = _cuda_payloads_on_device(
                self._torch, active, self.cuda_devices[0], self.voxel_m,
                self.cuda_max_batch_points,
            )
        else:
            groups = _split_by_point_count(active, len(self.cuda_devices))
            futures = []
            for device, group in zip(self.cuda_devices, groups):
                if group:
                    futures.append(self._device_pool.submit(
                        _cuda_payloads_on_device,
                        self._torch,
                        group,
                        device,
                        self.voxel_m,
                        self.cuda_max_batch_points,
                    ))
            encoded = {}
            for future in futures:
                encoded.update(future.result())
        for index, payload in encoded.items():
            payloads[index] = payload
        return payloads

    def encode(self, points, labels):
        """Encode one frame and retain its directly measured execution data."""
        started = time.perf_counter()
        payloads = self._encode_unmeasured(points, labels)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_measurement = {
            **self.identity,
            "elapsed_ms": round(elapsed_ms, 3),
            "objects": len(points),
            "eligible_objects": sum(
                point_set is not None and fusion_eligible_label(label)
                for point_set, label in zip(points, labels)
            ),
            "output_voxels": sum(len(payload) // 3 for payload in payloads),
        }
        return payloads

    def _verify_against_reference(self):
        points = [
            np.array([
                [-0.061, 0.000, 0.031],
                [-0.061, 0.000, 0.031],
                [0.001, 0.029, 0.030],
                [0.031, 0.000, 0.000],
                [np.nan, 1.000, 2.000],
            ], dtype=np.float64),
            np.array([[9.0, 9.0, 9.0]], dtype=np.float64),
            None,
            np.empty((0, 3), dtype=np.float64),
        ]
        labels = ["chair#2", "wall", "lamp", "table"]
        expected = [fusion_payload_from_points(value, label, self.voxel_m)
                    for value, label in zip(points, labels)]
        actual = self._encode_unmeasured(points, labels)
        if actual != expected:
            raise RuntimeError(
                f"{self.backend} bbox encoder failed its sequential equivalence check"
            )
        self.verified = True

    def shutdown(self):
        """Stop persistent workers. A shutdown error must reach the caller."""
        if self._device_pool is not None:
            self._device_pool.shutdown(wait=True, cancel_futures=True)
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True, cancel_futures=True)
