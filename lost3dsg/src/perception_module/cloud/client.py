"""Perception Backend Adapters for Local and Cloud Inference.

Provides unified interface for:
1. Modal Serverless GPU Endpoint (`ModalPerceptionBackend`)
2. Managed APIs (`ManagedPerceptionBackend` - Fal.ai / Replicate)
3. Local Onboard Execution (`LocalPerceptionBackend`)
"""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from detection_types import Detection
except ImportError:
    from ..detection_types import Detection


def rle_encode(mask_binary: np.ndarray) -> Dict[str, Any]:
    """Fast run-length encoding for binary boolean mask."""
    m = (mask_binary > 0).astype(np.uint8)
    h, w = m.shape[:2]
    flat = m.flatten()
    if len(flat) == 0:
        return {"size": [int(h), int(w)], "first_val": 0, "counts": []}
    diffs = np.diff(flat)
    change_indices = np.where(diffs != 0)[0] + 1
    split_indices = np.concatenate([[0], change_indices, [len(flat)]])
    run_lengths = np.diff(split_indices)
    return {
        "size": [int(h), int(w)],
        "first_val": int(flat[0]),
        "counts": [int(c) for c in run_lengths],
    }


def rle_decode(rle_dict: Dict[str, Any]) -> np.ndarray:
    """Decode run-length encoding into binary mask (HxW)."""
    h, w = rle_dict["size"]
    counts = rle_dict["counts"]
    if not counts:
        return np.zeros((h, w), dtype=np.uint8)
    cur_val = rle_dict.get("first_val", 0)
    flat = np.empty(h * w, dtype=np.uint8)
    idx = 0
    for count in counts:
        flat[idx:idx + count] = cur_val
        idx += count
        cur_val = 1 - cur_val
    return flat.reshape((h, w))


class PerceptionBackend(ABC):
    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Check reachability and readiness of the backend."""
        pass

    @abstractmethod
    def detect_and_segment(
        self,
        rgb_image: np.ndarray,
        labels: List[str],
        score_threshold: float = 0.15,
        nms_threshold: float = 0.50,
    ) -> Tuple[List[Detection], Dict[str, float]]:
        """Run open-vocab detection, segmentation, and crop embeddings.

        Returns:
            (detections_list, timing_breakdown_dict)
        """
        pass


class ModalPerceptionBackend(PerceptionBackend):
    """Client for the Modal-hosted Perception Microservice."""

    def __init__(self, endpoint_url: str, timeout_seconds: float = 35.0):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.last_health = None

        if "-predict.modal.run" in self.endpoint_url:
            self.predict_url = self.endpoint_url
            self.health_url = self.endpoint_url.replace("-predict.modal.run", "-health.modal.run")
        elif "-health.modal.run" in self.endpoint_url:
            self.health_url = self.endpoint_url
            self.predict_url = self.endpoint_url.replace("-health.modal.run", "-predict.modal.run")
        else:
            self.predict_url = f"{self.endpoint_url}/predict"
            self.health_url = f"{self.endpoint_url}/health"

    def health(self) -> Dict[str, Any]:
        try:
            req = urllib.request.Request(self.health_url, headers={"User-Agent": "GraphAPI-PerceptionClient/1.0"})
            with urllib.request.urlopen(req, timeout=25.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.last_health = {"reachable": True, "details": data}
                return self.last_health
        except Exception as exc:
            self.last_health = {"reachable": False, "error": str(exc)}
            return self.last_health

    def detect_and_segment(
        self,
        rgb_image: np.ndarray,
        labels: List[str],
        score_threshold: float = 0.15,
        nms_threshold: float = 0.50,
    ) -> Tuple[List[Detection], Dict[str, float]]:
        if not labels:
            return [], {}

        # 1. Encode image to JPEG base64
        success, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode RGB frame to JPEG")
        b64_image = base64.b64encode(buffer).decode("utf-8")

        payload = {
            "image_b64": b64_image,
            "labels": labels,
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
        }

        # 2. Call Modal predict endpoint
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.predict_url,
            data=req_data,
            headers={"Content-Type": "application/json", "User-Agent": "GraphAPI-PerceptionClient/1.0"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        detections = []
        for item in result.get("detections", []):
            mask_np = rle_decode(item["mask_rle"])
            det = Detection(
                bbox=tuple(item["bbox"]),
                label=item["label"],
                score=float(item["score"]),
                mask=mask_np[..., None],
            )
            det.clip_embedding = item.get("clip_embedding")
            detections.append(det)

        timings = result.get("timings_ms", {})
        return detections, timings


class ManagedPerceptionBackend(PerceptionBackend):
    """Adapter for managed third-party perception endpoints (Fal.ai / Replicate)."""

    def __init__(self, provider: str = "fal", api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("FAL_KEY") or os.getenv("REPLICATE_API_TOKEN")

    def health(self) -> Dict[str, Any]:
        return {
            "reachable": bool(self.api_key),
            "provider": self.provider,
            "has_api_key": bool(self.api_key),
        }

    def detect_and_segment(
        self,
        rgb_image: np.ndarray,
        labels: List[str],
        score_threshold: float = 0.15,
        nms_threshold: float = 0.50,
    ) -> Tuple[List[Detection], Dict[str, float]]:
        t_start = time.time()
        if not self.api_key:
            raise RuntimeError(f"Missing API key for managed provider '{self.provider}'")

        if self.provider == "fal":
            # Example Fal.ai SAM2 / Florence-2 call
            try:
                import fal_client
            except ImportError:
                raise ImportError("Please install fal-client: pip install fal-client")

            # Encode image to JPEG base64
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            data_uri = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

            # 1. Florence-2 / Grounding DINO object detection
            _res = fal_client.subscribe(
                "fal-ai/florence-2-large/grounding",
                arguments={"image_url": data_uri, "text_input": ", ".join(labels)},
            )
            # Parse detections and SAM masks
            # (Stubbed adapter format for managed providers)
            detections = []
            return detections, {"total": round((time.time() - t_start) * 1000, 1)}

        raise NotImplementedError(f"Provider '{self.provider}' not implemented yet")


class LocalPerceptionBackend(PerceptionBackend):
    """Wraps local onboard PyTorch models (OWLv2 + VitSam)."""

    def __init__(self, detector=None, vitsam=None):
        self.detector = detector
        self.vitsam = vitsam

    def health(self) -> Dict[str, Any]:
        return {
            "reachable": True,
            "type": "local",
            "models_loaded": (self.detector is not None and self.vitsam is not None),
        }

    def detect_and_segment(
        self,
        rgb_image: np.ndarray,
        labels: List[str],
        score_threshold: float = 0.15,
        nms_threshold: float = 0.50,
    ) -> Tuple[List[Detection], Dict[str, float]]:
        # Handled in-line by standard detection_pipeline.py methods
        return [], {}


def get_perception_backend(cfg: Dict[str, Any]) -> PerceptionBackend:
    """Factory creating configured perception backend."""
    p_cfg = cfg.get("perception", {})
    backend_type = p_cfg.get("backend", "local").lower()

    if backend_type == "modal":
        endpoint = p_cfg.get("modal_endpoint") or os.getenv("MODAL_PERCEPTION_URL", "")
        if not endpoint:
            print("[PerceptionBackend] Warning: 'modal_endpoint' not set in config; using local placeholder.")
        return ModalPerceptionBackend(endpoint_url=endpoint)
    elif backend_type in ("managed", "fal", "replicate"):
        provider = p_cfg.get("provider", "fal")
        return ManagedPerceptionBackend(provider=provider)
    else:
        return LocalPerceptionBackend()
