from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List
import numpy as np


@dataclass(frozen=True)
class ObservationRefData:
    """Pipeline identity for one detector output; contains no semantic ground truth."""

    schema_version: int
    run_id: str
    producer_id: str
    capture_id: str
    capture_identity_kind: str
    capture_sec: int
    capture_nanosec: int
    camera_frame_id: str
    cycle_id: str
    detection_index: int
    observation_id: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "producer_id": self.producer_id,
            "capture_id": self.capture_id,
            "capture_identity_kind": self.capture_identity_kind,
            "capture_stamp": {
                "sec": self.capture_sec,
                "nanosec": self.capture_nanosec,
            },
            "camera_frame_id": self.camera_frame_id,
            "cycle_id": self.cycle_id,
            "detection_index": self.detection_index,
            "observation_id": self.observation_id,
        }


def make_observation_ref(run_id, producer_id, cycle_id, detection_index,
                         stamp, camera_frame_id=""):
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", 0))
    capture_id = f"{producer_id}:{sec}:{nanosec}"
    observation_id = f"{producer_id}:{cycle_id}:{int(detection_index)}"
    return ObservationRefData(
        schema_version=1,
        run_id=str(run_id or ""),
        producer_id=str(producer_id),
        capture_id=capture_id,
        capture_identity_kind="stamp_within_producer",
        capture_sec=sec,
        capture_nanosec=nanosec,
        camera_frame_id=str(camera_frame_id or ""),
        cycle_id=str(cycle_id),
        detection_index=int(detection_index),
        observation_id=observation_id,
    )


def write_observation_msg(target, ref):
    """Write one reference to a generated ROS message without guessing legacy identity."""
    if ref is None:
        target.schema_version = 0
        return
    data = ref if isinstance(ref, dict) else ref.as_dict()
    stamp = data.get("capture_stamp") or {}
    target.schema_version = int(data["schema_version"])
    target.run_id = str(data.get("run_id") or "")
    target.producer_id = str(data["producer_id"])
    target.capture_id = str(data["capture_id"])
    target.capture_identity_kind = str(data["capture_identity_kind"])
    target.capture_stamp.sec = int(stamp.get("sec", 0))
    target.capture_stamp.nanosec = int(stamp.get("nanosec", 0))
    target.camera_frame_id = str(data.get("camera_frame_id") or "")
    target.cycle_id = str(data["cycle_id"])
    target.detection_index = int(data["detection_index"])
    target.observation_id = str(data["observation_id"])


def observation_dict_from_msg(source):
    if isinstance(source, ObservationRefData):
        return source.as_dict()
    if isinstance(source, dict):
        if int(source.get("schema_version", 0)) == 0:
            return None
        # The caller owns no part of the returned nested mapping.
        result = dict(source)
        result["capture_stamp"] = dict(source.get("capture_stamp") or {})
        return result
    if source is None or int(getattr(source, "schema_version", 0)) == 0:
        return None
    stamp = getattr(source, "capture_stamp", None)
    return {
        "schema_version": int(source.schema_version),
        "run_id": str(source.run_id),
        "producer_id": str(source.producer_id),
        "capture_id": str(source.capture_id),
        "capture_identity_kind": str(source.capture_identity_kind),
        "capture_stamp": {
            "sec": int(getattr(stamp, "sec", 0)),
            "nanosec": int(getattr(stamp, "nanosec", 0)),
        },
        "camera_frame_id": str(source.camera_frame_id),
        "cycle_id": str(source.cycle_id),
        "detection_index": int(source.detection_index),
        "observation_id": str(source.observation_id),
    }


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    label: str
    score: Optional[float]
    mask: np.ndarray
    # Whole-scene VLM attributes. Detector/cloud detections retain the historical
    # unknown defaults; the unified local path fills them from the same request
    # that produced bbox, so no later crop-VLM request is needed.
    description: str = field(default="unknown")
    color: str = field(default="unknown")
    material: str = field(default="unknown")
    shape: str = field(default="unknown")
    instance_label: Optional[str] = field(default=None)
    is_confirmed: bool = field(default=True)
    clip_embedding: Optional[List[float]] = field(default=None)
    observation: Optional[ObservationRefData] = field(default=None)
