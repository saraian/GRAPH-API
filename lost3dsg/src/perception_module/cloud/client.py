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
    total = int(sum(counts))
    if total != h * w:
        # GA-16: this allocated with np.empty and filled only sum(counts) entries, so a
        # short payload left UNINITIALISED HEAP in the tail and the mask contained
        # whatever had been there. np.zeros would be the wrong repair: a short payload is
        # a corrupt mask, not something to pad, and zero-filling turns a transport error
        # into a plausible mask that nothing downstream can tell from a real one.
        raise ValueError(
            f"RLE payload covers {total} of {h * w} pixels ({h}x{w}); the mask is "
            "incomplete and cannot be decoded."
        )
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

    # WHAT WAS MEASURED, and it is not the path this timeout governs.
    #
    # n=7 on 2026-08-30, one before each launch:
    #     25.3  42.3  45.8  41.1  26.6  47.7  45.6      min 25.3   max 47.7
    #
    # Those seven are a `curl` GET against the endpoint — a request that carries no
    # payload and asks for no inference. The call this timeout governs sends an image
    # and waits for detection and segmentation. They are not the same distribution, and
    # 60.0 has since been exceeded TWICE on the inference path.
    #
    # 60.0 stands because two exceedances are not a distribution and no measured
    # replacement exists yet — not because these seven support it. They do not.
    #
    # For whoever sets this properly: measure on the INFERENCE path, with a real image,
    # from a cold container. Re-running the curl probe will reproduce 47.7, make 60.0
    # look comfortable, and be wrong for the same reason it is wrong here.
    def __init__(self, endpoint_url: str, timeout_seconds: float = 60.0):
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
            # The field exists and this call site did not reach it: a hardcoded 25.0,
            # below every one of the seven measured cold starts, so the first health
            # check after the container went cold could not succeed. Run 10 died here.
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
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

        t_encode = time.time()
        # 1. Encode image to JPEG base64
        # GA-13: this applied COLOR_RGB2BGR before encoding. The array is ALREADY BGR —
        # utils.py:161 asks cv_bridge for 'bgr8' — and imencode expects BGR, so the
        # conversion swapped red and blue in every JPEG sent to the cloud, and every
        # VLM label and embedding from that path was computed on a colour-swapped
        # image. The parameter is named `rgb_image` and the dict key is "rgb"; neither
        # is. The names are what misled the author, and renaming them is a separate
        # change across three files.
        success, buffer = cv2.imencode(".jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
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

        # The elapsed time is attached to whatever goes wrong, and the exception is
        # RE-RAISED unchanged in type. Three runs died on this line and not one recorded
        # how long it had waited — so "it timed out" was all anyone had, and a timeout
        # with no duration cannot be told from a hang, a slow cold start, or a payload
        # the server never began. This does not catch anything: `raise ... from exc`
        # preserves the type and the traceback, and adds the one number that makes the
        # next failure informative.
        #
        # It is the in-run number that is wanted. Measured in isolation this call is
        # 43.6 s cold and 1.2-1.4 s warm, comfortably inside the timeout; in-run it has
        # exceeded 60 s three times on the same line. An isolated perception call is not
        # an in-run perception call, and only the failing run can say by how much.
        t_encode_ms = (time.time() - t_encode) * 1000.0
        t_request = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read()
                result = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            waited = time.time() - t_request
            note = (f"{exc} [perception request waited {waited:.1f}s of a "
                    f"{self.timeout_seconds:.1f}s timeout, {len(labels)} labels, "
                    f"{len(req_data) / 1024:.0f} KiB payload]")
            try:
                enriched = type(exc)(note)
            except Exception:
                # Not every exception takes a single string — HTTPError wants five
                # arguments. Losing the type is bad; replacing a timeout with a
                # TypeError raised inside the error handler is far worse, and that is
                # what an unguarded `type(exc)(msg)` does on those types.
                raise RuntimeError(note) from exc
            raise enriched from exc

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
        # SUCCESS-PATH MEASUREMENT. The elapsed-on-raise above made a FAILURE informative;
        # this makes a success informative, and item 13 needs the second one.
        #
        # What it settles. The pipeline's `t_cloud` wraps this whole method — JPEG encode,
        # base64, json.dumps, the HTTP call, and RLE-decoding every mask. The server's own
        # `total` covers only what it computed. Everything between was one lump called
        # `wire`, and in run 19 that lump was 42,256 ms of 42,275 while the server's
        # non-stage time was 19 ms. Three other cloud runs read 892, 974 and 2,524 ms on
        # the same line, so run 19 is a 47x OUTLIER, not a standing cost.
        #
        # `request_ms` brackets urlopen ALONE, so the lump splits three ways and each part
        # names a different cause:
        #     t_cloud - request_ms   client-side encode and mask decode  (ours)
        #     request_ms - total     connection setup, Modal queueing, COLD START
        #     total                  the server's own compute
        # A 36.6 KiB body cannot take 42 s on any network, so transfer was never a
        # plausible explanation for run 19 — but "not transfer" was an inference, and this
        # makes it an observation. `payload_bytes` is recorded MEASURED rather than assumed
        # because perception's falsifiable prediction rides on it: at 1280x960 the body
        # goes to 95.5 KiB, 2.6x not 4x, because JPEG absorbs the upscale. If the overhead
        # scales ~2.6x the cost is transfer-bound; if it does not move, cold start stands.
        #
        # UNDER `client`, NOT beside the stage timings. Those keys are the SERVER's, and
        # GA-14 raises on a missing one while printing `sorted(cloud_timings)`. A
        # client-side number sitting in that namespace could be read as a stage the server
        # reported, which is the conflation GA-14 exists to prevent — so the nesting is the
        # point, not tidiness.
        timings["client"] = {
            "encode_ms": round(t_encode_ms, 1),
            "request_ms": round((time.time() - t_request) * 1000.0, 1),
            "payload_bytes": len(req_data),
            "response_bytes": len(raw),
            "labels": len(labels),
        }
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
            _, buffer = cv2.imencode(".jpg", rgb_image)   # GA-13: already BGR, see above
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
            # This printed a warning saying it would use a "local placeholder" and then
            # returned a Modal backend pointed at an empty URL — so the run continued
            # with a backend that cannot answer, and the message described something
            # the code does not do. Working rule 14: a missing component stops the run.
            raise RuntimeError(
                "perception.backend is 'modal' but 'modal_endpoint' is unset and "
                "MODAL_PERCEPTION_URL is not in the environment. Set one, or set "
                "perception.backend to 'local'."
            )
        return ModalPerceptionBackend(
            endpoint_url=endpoint,
            timeout_seconds=float(p_cfg.get("cloud_timeout_s", 60.0)),
        )
    elif backend_type in ("managed", "fal", "replicate"):
        provider = p_cfg.get("provider", "fal")
        return ManagedPerceptionBackend(provider=provider)
    else:
        return LocalPerceptionBackend()
