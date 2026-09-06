"""Perception Backend Adapters for Local and Cloud Inference.

Provides unified interface for:
1. Modal Serverless GPU Endpoint (`ModalPerceptionBackend`)
2. Managed APIs (`ManagedPerceptionBackend` - Fal.ai / Replicate)
3. Local Onboard Execution (`LocalPerceptionBackend`)
"""

import base64
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.parse
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
        self._http = None   # one keep-alive HTTPS connection for the predict path

        if "-predict.modal.run" in self.endpoint_url:
            self.predict_url = self.endpoint_url
            self.health_url = self.endpoint_url.replace("-predict.modal.run", "-health.modal.run")
        elif "-health.modal.run" in self.endpoint_url:
            self.health_url = self.endpoint_url
            self.predict_url = self.endpoint_url.replace("-health.modal.run", "-predict.modal.run")
        else:
            self.predict_url = f"{self.endpoint_url}/predict"
            self.health_url = f"{self.endpoint_url}/health"

    def _post_keepalive(self, body: bytes) -> bytes:
        """POST `body` to the predict URL over ONE persistent HTTPS connection.

        MEASURED against the live endpoint 2026-09-06: a fresh connection costs 0.20 s to
        connect and 0.34 s to finish TLS before the first byte moves; the same request on
        an already-open connection answers in 0.15 s. urlopen opened a new connection per
        cycle, so every cycle paid ~0.35 s for nothing. The server is unchanged and the
        bytes on the wire are the same, so the detections are the same.

        A keep-alive connection can be closed by the far side while the client idles
        between cycles; that surfaces as RemoteDisconnected / BadStatusLine / a reset on
        the next request, BEFORE any inference ran. Exactly that case reconnects once. A
        timeout is NOT retried: it would double the wait on a request the server may be
        working on, and the caller already reports the waited time.
        """
        u = urllib.parse.urlsplit(self.predict_url)
        path = u.path or "/"
        if u.query:
            path += "?" + u.query
        headers = {"Content-Type": "application/json",
                   "User-Agent": "GraphAPI-PerceptionClient/1.0",
                   "Connection": "keep-alive"}
        for attempt in (0, 1):
            if self._http is None:
                self._http = http.client.HTTPSConnection(
                    u.hostname, u.port or 443, timeout=self.timeout_seconds)
            conn = self._http
            try:
                conn.request("POST", path, body=body, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
            except (socket.timeout, TimeoutError):
                self._drop_connection()
                raise
            except (http.client.HTTPException, ConnectionError, OSError):
                self._drop_connection()
                if attempt == 1:
                    raise
                continue
            if resp.status >= 400:
                # Same exception type urlopen raised, so the caller's handling is unchanged.
                self._drop_connection()
                raise urllib.error.HTTPError(self.predict_url, resp.status, resp.reason,
                                             resp.headers, None)
            if resp.getheader("Connection", "").lower() == "close":
                self._drop_connection()
            return raw
        raise RuntimeError("unreachable")   # both attempts either returned or raised

    def _drop_connection(self):
        conn, self._http = self._http, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

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
        containment_threshold: float = None,
    ) -> Tuple[List[Detection], Dict[str, float]]:
        if not labels:
            return [], {}

        # GA-286. Defaulted from config rather than in the signature, so the LOCAL path
        # (utils.apply_nms) and the CLOUD path read the same number from the same place. Two
        # NMS implementations disagreeing about what counts as a duplicate is a defect
        # waiting to happen, and this one already happened once in the other direction: the
        # containment fix went into the local path only and was inert for every cloud run.
        if containment_threshold is None:
            try:
                from config import CFG
                containment_threshold = float(
                    (CFG.get("perception", {}) or {}).get("containment_threshold", 0.85))
            except Exception:
                containment_threshold = 0.85

        t_encode = time.time()
        # 1. Encode image to JPEG base64
        # GA-13: this applied COLOR_RGB2BGR before encoding. The array is ALREADY BGR —
        # utils.py:161 asks cv_bridge for 'bgr8' — and imencode expects BGR, so the
        # conversion swapped red and blue in every JPEG sent to the cloud, and every
        # VLM label and embedding from that path was computed on a colour-swapped
        # image. The parameter is named `rgb_image` and the dict key is "rgb"; neither
        # is. The names are what misled the author, and renaming them is a separate
        # change across three files.
        # GA-278. DOWNSCALE BEFORE THE WIRE. Measured 2026-09-02 against this endpoint:
        # the request is BANDWIDTH-bound, not latency-bound, up to a floor around 1.76 s --
        # 1280x960 (180 KB) took 2820 ms, 960x720 (87 KB) took 1793 ms, and 640x480 (46 KB)
        # took 1759 ms. So 0.75x captures the whole saving and going smaller buys 34 ms.
        #
        # THE QUALITY COST, MEASURED RATHER THAN ASSUMED. One frame showed 6 detections at
        # full and 4 at 0.75x, which looked like a third of them lost. Across SIX frames it
        # is 45 vs 43 (-4%) for -21% request time, and two frames GAINED detections at the
        # smaller size (6->8, 3->5). The per-frame variance is larger than the difference,
        # so this is noise, not a systematic loss -- but n=6, and it is a trade, not a
        # free win. `cloud.send_scale: 1.0` restores the previous behaviour exactly.
        #
        # The backend resizes to its models' native inputs anyway (OWLv2 960x960, SAM
        # 1024x1024), so the pixels dropped here were being discarded server-side.
        try:
            from config import CFG
            _scale = float((CFG.get("cloud", {}) or {}).get("send_scale", 0.75))
        except Exception:
            _scale = 0.75
        _orig_h, _orig_w = rgb_image.shape[:2]
        _sent_scale = 1.0
        if 0.1 < _scale < 0.999:
            rgb_image = cv2.resize(rgb_image, (int(_orig_w * _scale), int(_orig_h * _scale)),
                                   interpolation=cv2.INTER_AREA)
            # The ACTUAL scale, recomputed from the integer size the resize produced. Using
            # the requested float instead would drift by up to a pixel per axis, and that
            # error lands straight in the 3D projection.
            _sent_scale = rgb_image.shape[1] / float(_orig_w)
        success, buffer = cv2.imencode(".jpg", rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode RGB frame to JPEG")
        b64_image = base64.b64encode(buffer).decode("utf-8")

        payload = {
            "image_b64": b64_image,
            "labels": labels,
            "score_threshold": score_threshold,
            "nms_threshold": nms_threshold,
            # GA-286. SENT EXPLICITLY. Without this the server's own default governs, and the
            # bundle's config would record a containment threshold the run did not use -- the
            # same shape as the corpus_order field that recorded an order no run applied.
            #
            # INERT UNTIL MODAL IS REDEPLOYED, and that is measured, not assumed: the deployed
            # PerceptionRequest does not declare this field, so pydantic IGNORES it. Verified
            # 2026-09-03 against the live endpoint -- requests with and without it both
            # succeed and return the same detections. So sending it is SAFE now and takes
            # effect only when modal_perception.py is redeployed. Until then the cloud path
            # runs plain IoU NMS and nested duplicates survive; the LOCAL path (utils.apply_nms)
            # already suppresses them.
            "containment_threshold": containment_threshold,
        }

        # 2. Call Modal predict endpoint
        req_data = json.dumps(payload).encode("utf-8")

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
            raw = self._post_keepalive(req_data)
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
        # BACK TO THE ORIGINAL FRAME. The backend answered in the coordinates of the image
        # it was SENT. Everything downstream -- crop construction, the segmentation overlay,
        # and the 3D projection through the full-resolution camera intrinsics -- works in
        # the ORIGINAL frame. Returning a 0.75x box unscaled would shrink every object by a
        # third and shift its centroid, silently, in a way no gate could detect: the boxes
        # would still be plausible boxes, just of the wrong things in the wrong places.
        _inv = 1.0 / _sent_scale if _sent_scale else 1.0
        for item in result.get("detections", []):
            mask_np = rle_decode(item["mask_rle"])
            if _sent_scale != 1.0:
                bx = [float(v) * _inv for v in item["bbox"]]
                # INTER_NEAREST: a mask is a label field, not an image. Interpolating it
                # would invent fractional membership at every boundary pixel.
                mask_np = cv2.resize(mask_np.astype("uint8"), (_orig_w, _orig_h),
                                     interpolation=cv2.INTER_NEAREST)
            else:
                bx = [float(v) for v in item["bbox"]]
            det = Detection(
                bbox=tuple(bx),
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

        # GA-19. The `fal` branch used to pay for a real Florence-2 grounding call
        # (`fal_client.subscribe`), bind the answer to `_res` and return `[]` -- a paid
        # request whose result was thrown away, read downstream as an empty scene. No
        # provider has a parser yet, so no provider may place a call: refuse before the
        # network, not after. Wiring one needs the provider's box/label envelope parsed
        # into `Detection` AND a mask source (Florence-2 grounding returns boxes only).
        del t_start
        raise NotImplementedError(
            f"managed perception provider '{self.provider}' has no response parser; "
            "set perception.backend to 'modal' or 'local'"
        )


class LocalPerceptionBackend(PerceptionBackend):
    """Wraps local onboard PyTorch models (OWLv2 + VitSam)."""

    def health(self) -> Dict[str, Any]:
        # GA-19. `models_loaded` was computed from constructor arguments the only
        # construction site (get_perception_backend) never passed, so it read False by
        # accident. It IS False: this backend holds no models -- detect_and_segment is a
        # stub, and test/preflight_gate.py refuses it by class name.
        return {"reachable": True, "type": "local", "models_loaded": False}

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
    elif backend_type == "local":
        return LocalPerceptionBackend()
    # GA-19. Any other string used to fall through to the local stub, whose
    # detect_and_segment is `return [], {}` -- so a typo in perception.backend produced
    # an empty scene that read as "nothing was there". Working rule 14: refuse.
    raise ValueError(
        f"perception.backend={backend_type!r} is not one of "
        "'modal', 'local', 'managed', 'fal', 'replicate'"
    )
