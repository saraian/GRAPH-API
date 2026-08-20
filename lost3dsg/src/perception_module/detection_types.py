from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    label: str
    score: float
    mask: np.ndarray
    instance_label: Optional[str] = field(default=None)
