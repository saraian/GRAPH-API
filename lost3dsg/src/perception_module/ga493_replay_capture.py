"""Opt-in, GT-isolated capture for deterministic perception replay."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_FORBIDDEN_KEYS = {
    "gt", "ground_truth", "semantic_frame", "habitat_gt_instance_id",
    "habitat_id", "gt_instance", "gt_box", "gt_inventory",
}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    fields = getattr(value, "get_fields_and_field_types", None)
    if callable(fields):
        return {name: _json_value(getattr(value, name)) for name in fields()}
    slots = getattr(value, "__slots__", None)
    if slots:
        return {name.lstrip("_"): _json_value(getattr(value, name)) for name in slots}
    return str(value)


def _assert_gt_free(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_KEYS or normalized.startswith("ground_truth"):
                raise ValueError(f"GT field is forbidden in replay input: {path}.{key}")
            _assert_gt_free(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_gt_free(child, f"{path}[{index}]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
        temp = Path(stream.name)
    temp.replace(path)


class ReplayCapture:
    """Append-only debug evidence. Disabled unless GA493_REPLAY_CAPTURE_DIR is set."""

    def __init__(self, root: Path, role: str, max_cycles: int, max_bytes: int,
                 identity: dict | None = None, source_files: list[Path] | None = None):
        if not root.is_absolute():
            raise ValueError("GA493_REPLAY_CAPTURE_DIR must be absolute")
        if role not in {"producer", "consumer"}:
            raise ValueError("replay role must be producer or consumer")
        if max_cycles <= 0 or max_bytes <= 0:
            raise ValueError("replay limits must be positive")
        self.root = root
        self.role = role
        self.max_cycles = max_cycles
        self.max_bytes = max_bytes
        self.role_dir = root / role
        self.cycles_dir = self.role_dir / "cycles"
        if self.role_dir.exists() and any(self.role_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty replay role directory: {self.role_dir}")
        self.role_dir.mkdir(parents=True, exist_ok=True)
        self.cycles_dir.mkdir(exist_ok=True)
        self.events_path = self.role_dir / "events.jsonl"
        # Callbacks in different ROS callback groups call event() concurrently; the number and
        # the append are one operation (see event()).
        self._event_lock = threading.Lock()
        self._sequence = 0
        self._cycles = 0
        self._bytes = 0
        self._closed = False
        sources = {}
        for source in source_files or []:
            source = Path(source)
            if source.is_file():
                sources[str(source)] = _sha256(source)
        manifest = {
            "schema": "graph_api.ga493_replay_capture.v1",
            "role": role,
            "created_at": time.time(),
            "max_cycles": max_cycles,
            "max_bytes": max_bytes,
            "gt_isolation": {
                "runtime_gt_enabled": False,
                "forbidden_from_replay": sorted(_FORBIDDEN_KEYS),
                "evaluation": "separate post-run process and directory",
            },
            "initial_state": "empty" if role == "consumer" else "not_applicable",
            "identity": _json_value(identity or {}),
            "source_sha256": sources,
        }
        _assert_gt_free(manifest["identity"], "identity")
        _atomic_json(self.role_dir / "manifest.json", manifest)

    @classmethod
    def from_environment(cls, role: str, identity: dict | None = None,
                         source_files: list[Path] | None = None):
        raw = os.environ.get("GA493_REPLAY_CAPTURE_DIR", "").strip()
        if not raw:
            return None
        if os.environ.get("FEED_GT_SEMANTIC", "0").strip().lower() not in {"", "0", "false", "no", "off"}:
            raise RuntimeError("GA-493 replay capture refuses while FEED_GT_SEMANTIC is enabled")
        return cls(
            Path(raw), role,
            int(os.environ.get("GA493_REPLAY_CAPTURE_MAX_CYCLES", "12")),
            int(os.environ.get("GA493_REPLAY_CAPTURE_MAX_BYTES", str(5 * 1024**3))),
            identity=identity, source_files=source_files,
        )

    def _reserve(self, size: int) -> None:
        if self._closed:
            raise RuntimeError("replay capture is already complete")
        if self._bytes + size > self.max_bytes:
            raise RuntimeError("GA-493 replay capture byte limit exceeded")
        self._bytes += size

    def event(self, kind: str, payload: dict | None = None, cycle_id: str | None = None) -> None:
        # ONE LOCK ROUND THE NUMBER AND THE WRITE. 2026-09-14: this read `self._sequence`, did an
        # fsync'd append, then incremented, with nothing serialising it -- and `movement_callback`
        # runs in its own callback group beside the default-group callbacks, so two events took
        # the same number. MEASURED: 1 duplicate on 20260914_170517, 3 on 20260914_170945 (rows
        # 26/27, 54/55, 79/80, each a `movement_state_changed` sharing with a `walls_arrived` or
        # `periodic_tick`). `finalize` then raised "event sequence is not contiguous", the replay
        # artefact was void and object_manager_6 exited 1 at teardown in BOTH runs.
        with self._event_lock:
            row = {
                "sequence": self._sequence,
                "recorded_at": time.time(),
                "kind": str(kind),
                "cycle_id": cycle_id,
                "payload": _json_value(payload or {}),
            }
            _assert_gt_free(row)
            encoded = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            self._reserve(len(encoded))
            with self.events_path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            self._sequence += 1

    def producer_cycle(self, cycle_id: str, rgb, depth, camera_info, transform,
                       detections, bboxes_3d, centroids_3d, descriptions) -> None:
        if self._cycles >= self.max_cycles:
            return
        depth_array = np.asarray(depth)
        if not np.issubdtype(depth_array.dtype, np.floating):
            raise ValueError(f"depth must use a floating dtype, got {depth_array.dtype}")
        masks = []
        detection_rows = []
        for index, det in enumerate(detections):
            mask = np.asarray(getattr(det, "mask", None))
            while mask.ndim > 2:
                mask = mask[..., 0]
            if mask.shape != depth_array.shape:
                raise ValueError(f"mask/depth shape mismatch at detection {index}")
            masks.append(mask.astype(np.bool_, copy=False))
            embedding = getattr(det, "clip_embedding", None)
            detection_rows.append({
                "index": index,
                "label": getattr(det, "label", None),
                "instance_label": getattr(det, "instance_label", None),
                "bbox_2d": _json_value(getattr(det, "bbox", None)),
                "observation": _json_value(getattr(det, "observation", None)),
                "bbox_3d": _json_value(bboxes_3d[index] if index < len(bboxes_3d) else None),
                "centroid_3d": _json_value(centroids_3d[index] if index < len(centroids_3d) else None),
                "description": _json_value(descriptions[index] if index < len(descriptions) else None),
                "clip_embedding": _json_value(embedding),
                "clip_embedding_present": embedding is not None,
            })
        metadata = {
            "cycle_id": cycle_id,
            "depth": {"dtype": str(depth_array.dtype), "shape": list(depth_array.shape), "units": "metres"},
            "rgb": {"dtype": str(np.asarray(rgb).dtype), "shape": list(np.asarray(rgb).shape), "channel_order": "BGR"},
            "camera_info": _json_value(camera_info),
            "map_from_camera": _json_value(transform),
            "detections": detection_rows,
        }
        _assert_gt_free(metadata)
        stem = f"{self._cycles:04d}_{cycle_id}"
        npz_path = self.cycles_dir / f"{stem}.npz"
        meta_path = self.cycles_dir / f"{stem}.json"
        with tempfile.NamedTemporaryFile("wb", dir=self.cycles_dir,
                                         prefix=f".{stem}.", delete=False) as stream:
            np.savez_compressed(
                stream,
                rgb=np.asarray(rgb, dtype=np.uint8),
                depth=depth_array,
                masks=np.stack(masks) if masks else np.empty((0,) + depth_array.shape, dtype=np.bool_),
            )
            temp_npz = Path(stream.name)
        encoded_meta = (json.dumps(metadata, sort_keys=True, indent=2) + "\n").encode()
        size = temp_npz.stat().st_size + len(encoded_meta)
        self._reserve(size)
        temp_npz.replace(npz_path)
        with tempfile.NamedTemporaryFile("wb", dir=self.cycles_dir,
                                         prefix=f".{stem}.", delete=False) as stream:
            stream.write(encoded_meta)
            temp_meta = Path(stream.name)
        temp_meta.replace(meta_path)
        self._cycles += 1
        self.event("producer_cycle", {
            "npz": npz_path.name, "metadata": meta_path.name,
            "npz_sha256": _sha256(npz_path), "metadata_sha256": _sha256(meta_path),
            "detections": len(detection_rows),
        }, cycle_id=cycle_id)

    def finalize(self) -> None:
        if self._closed:
            return
        event_rows = []
        if self.events_path.exists():
            with self.events_path.open("r", encoding="utf-8") as stream:
                event_rows = [json.loads(line) for line in stream if line.strip()]
        if [row["sequence"] for row in event_rows] != list(range(len(event_rows))):
            raise RuntimeError("GA-493 replay event sequence is not contiguous")
        if len(event_rows) != self._sequence:
            raise RuntimeError("GA-493 replay event count does not match capture state")

        cycle_rows = [row for row in event_rows if row["kind"] == "producer_cycle"]
        referenced_files = set()
        for row in cycle_rows:
            detail = row["payload"]
            npz_path = self.cycles_dir / detail["npz"]
            metadata_path = self.cycles_dir / detail["metadata"]
            if not npz_path.is_file() or not metadata_path.is_file():
                raise RuntimeError("GA-493 replay cycle is incomplete")
            if _sha256(npz_path) != detail["npz_sha256"]:
                raise RuntimeError("GA-493 replay NPZ hash mismatch")
            if _sha256(metadata_path) != detail["metadata_sha256"]:
                raise RuntimeError("GA-493 replay metadata hash mismatch")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata["cycle_id"] != row["cycle_id"]:
                raise RuntimeError("GA-493 replay cycle identity mismatch")
            referenced_files.update({npz_path.name, metadata_path.name})
        actual_files = {path.name for path in self.cycles_dir.iterdir() if path.is_file()}
        if actual_files != referenced_files:
            raise RuntimeError("GA-493 replay cycle file set does not match its event ledger")
        if len(cycle_rows) != self._cycles:
            raise RuntimeError("GA-493 replay cycle count does not match capture state")
        payload = {
            "schema": "graph_api.ga493_replay_capture.complete.v1",
            "role": self.role,
            "cycles": self._cycles,
            "events": self._sequence,
            "bytes_accounted": self._bytes,
            "events_sha256": _sha256(self.events_path) if self.events_path.exists() else None,
            "reconciliation": {
                "event_sequence_contiguous": True,
                "cycle_events": len(cycle_rows),
                "cycle_files": len(actual_files),
                "cycle_hashes_verified": True,
            },
            "completed_at": time.time(),
        }
        _atomic_json(self.role_dir / "complete.json", payload)
        self._closed = True
