from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np


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
    # Explicit calibration join key. It is assigned by perception_2 once the exact
    # frame stamp is known and is carried through the archive and both ROS arrays.
    detection_id: Optional[str] = field(default=None)
