"""Modal Serverless Perception Microservice with YOLO-World-L and SAM 2.1.

Deploy with:
    modal deploy lost3dsg/src/perception_module/cloud/modal_perception.py

Runs:
1. YOLO-World-L open-vocabulary object detection (fast, tight indoor bounding boxes)
2. SAM 2.1 (Segment Anything 2.1 Hiera-Small) instance segmentation producing exact pixel masks
3. CLIP ViT-B/32 crop embeddings (512-d, one per detection, `null` for a sliver) -- GA-342

DEPLOY SKEW (GA-85, GA-211, GA-342). What runs is whatever `modal deploy` was last given, and
nothing in a bundle said which source that was. This module now bakes the deploying tree's
commit into the image at deploy time (`LOST3DSG_SRC_SHA` / `LOST3DSG_SRC_DIRTY`, computed on the
host when `modal deploy` imports this file) and `health()` reports both, so a bundle's health
record names the source. Rules: deploy from a COMMITTED tree (dirty = "true" is a finding), never
switch the live service while a run is in flight, and re-read health() after every deploy.

GA-342. df98138 (2026-09-06) rewrote this service around YOLO-World and dropped the crop
embedder while keeping the header line that claimed it; every modal-backend run since carried
zero embeddings and `association.channel_appearance` measured 0 rows (192014 before it: 33 of
437). Restored here by owner ruling (2026-09-07 ~15:30, "A: restore it in the Modal service").
YOLO-World's CLIP is a text tower only, so the image encoder is loaded explicitly: CLIP
ViT-B/32 gives the same 512-d space the archived embeddings are in.

Deployed on an NVIDIA T4 GPU on Modal with scale-to-zero.
"""

import base64
import io
import os
import subprocess
import time
from typing import Any, Dict, List

import modal
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

APP_NAME = "lost3dsg-perception"
CLIP_MODEL = "openai/clip-vit-base-patch32"   # 512-d image features, the archived space
MIN_CROP_PX = 4
"""Below this, in EITHER dimension, a crop carries no usable appearance (GA-17)."""


def _source_stamp():
    """(sha, dirty) of the tree this file is deployed from -- computed on the HOST at deploy
    time, where git exists; inside the container both read "unknown" and the baked env wins."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", __file__], cwd=here,
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return sha, ("true" if dirty else "false")
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"


def crop_regions(boxes, rgb_np, w, h, min_px=MIN_CROP_PX):
    """-> [(box_index, crop_array)] for the boxes whose crop is at least `min_px` a side.

    The BOX INDEX travels with the crop. The embeddings are computed in one batch, so
    dropping a crop shifts every later position in that batch while the assembly step still
    indexes `crop_embeddings` by box; carrying the index is what keeps the two from drifting.
    Split out of the endpoint so this can be tested without a GPU, weights, or Modal -- the
    misalignment it prevents is silent, and would attach one object's appearance to another.
    """
    out = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = rgb_np[y1:y2, x1:x2]
        if crop.shape[0] < min_px or crop.shape[1] < min_px:
            continue
        out.append((i, crop))
    return out


_SRC_SHA, _SRC_DIRTY = _source_stamp()

perception_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "ultralytics>=8.2.70",
        "transformers>=4.44.0",
        "accelerate>=0.30.0",
        "opencv-python-headless>=4.9.0",
        "pillow>=10.2.0",
        "numpy>=1.24.0",
        "fastapi>=0.110.0",
        "pycocotools>=2.0.7",
        "sentencepiece",
        "git+https://github.com/facebookresearch/sam2.git",
    )
    .run_commands(
        # Pre-cache YOLO-World-L, SAM 2.1 and CLIP ViT-B/32 weights in the image build layer
        "python3 -c \"from ultralytics import YOLOWorld; YOLOWorld('yolov8l-worldv2.pt')\"",
        "python3 -c \"from sam2.sam2_image_predictor import SAM2ImagePredictor; SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-small', device='cpu')\"",
        f"python3 -c \"from transformers import CLIPModel, CLIPProcessor; CLIPModel.from_pretrained('{CLIP_MODEL}'); CLIPProcessor.from_pretrained('{CLIP_MODEL}')\"",
    )
    # Deploy-skew stamp: evaluated on the host at deploy time, read by health() in the container.
    .env({"LOST3DSG_SRC_SHA": _SRC_SHA, "LOST3DSG_SRC_DIRTY": _SRC_DIRTY})
)

app = modal.App(APP_NAME, image=perception_image)


class PerceptionRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded JPEG/PNG image")
    labels: List[str] = Field(..., description="Candidate open-vocabulary labels from VLM")
    score_threshold: float = Field(default=0.15, description="Confidence threshold for detections")
    nms_threshold: float = Field(default=0.45, description="IoU threshold for non-maximum suppression")


def rle_encode(mask_binary: np.ndarray) -> Dict[str, Any]:
    """Fast run-length encoding for binary boolean mask (HxW)."""
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


@app.cls(
    gpu="T4",               # T4 GPU tier ($0.000164/sec)
    # 600 s, not 20. GA-207 measured what a short window does: one cycle that idles past
    # it pays a ~40 s cold start, which makes the next cycle idle past it too, and the run
    # settles at 54 s/cycle (20260901_151714: wire_ms 38,695 of 54,404). A warm cycle is
    # 3-5 s, so 20 s held only while nothing hiccupped. 600 s costs nothing while the
    # robot is not running and closes the trap.
    scaledown_window=600,
    max_containers=1,       # Guardrail: strictly 1 GPU instance
    timeout=45,             # Timeout after 45s
)
class PerceptionService:
    @modal.enter()
    def setup(self):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from ultralytics import YOLOWorld

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Modal Perception] Initializing YOLO-World-L and SAM 2.1 on {self.device}...")

        # 1. YOLO-World-L Detector
        self.detector = YOLOWorld("yolov8l-worldv2.pt")
        # ponytail: move to GPU BEFORE the first set_classes(). ultralytics builds and caches
        # the CLIP text model on whatever device the YOLO weights sit on at that moment; if the
        # first set_classes() runs on CPU, predict(device=cuda) then moves the cached text model
        # to cuda while its tokenizer keeps device="cpu" -> every subsequent call dies with
        # "index is on cpu, different from other tensors on cuda:0".
        self.detector.to(self.device)
        self.detector.set_classes(["chair"])  # warm the CLIP text head on-device
        print("[Modal Perception] YOLO-World-L detector ready ✅")

        # 2. SAM 2.1 Predictor
        # ponytail: no bbox-mask fallback. A rectangular "mask" silently poisons the 3D
        # lift — the depth crop then includes wall and floor pixels, extents inflate, and
        # the envelope check declines a real object for being the wrong size. A dead
        # segmentator must stop the service, not quietly change what the geometry means.
        self.sam2 = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small")
        self.has_sam2 = True
        print("[Modal Perception] SAM 2.1 Hiera-Small segmentation ready ✅")

        # 3. CLIP image encoder for the crop embeddings (GA-342). fp16 on the GPU: ~150 MB of
        # weights, a batch of crops in a few ms. A dead embedder must stop the service, not
        # quietly return `null` for every crop and read as "all slivers" downstream.
        from transformers import CLIPModel, CLIPProcessor
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        if self.device.type == "cuda":
            self.clip = self.clip.half()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        print(f"[Modal Perception] CLIP {CLIP_MODEL} crop embedder ready ✅")

        # weights-resident VRAM, measured once both models are on-device (peak
        # during inference is reported separately by health())
        self.vram_weights_mb = (
            round(torch.cuda.memory_allocated() / 1048576, 1) if torch.cuda.is_available() else 0.0
        )
        print(f"[Modal Perception] VRAM after load: {self.vram_weights_mb} MB")

        # First-inference warmup. MEASURED 2026-09-06 on a fresh container: the first
        # predict reported server total 2,766 ms against 285 ms warm -- CUDA kernel
        # selection and the first SAM 2.1 image encode, paid by the run's first cycle.
        # Paying it here moves it into the cold start, where the launcher's health probe
        # already waits. A blank image with one box exercises both models end to end.
        try:
            blank = np.zeros((256, 256, 3), dtype=np.uint8)
            self.detector.predict(Image.fromarray(blank), conf=0.15, iou=0.5, verbose=False,
                                  device=self.device)
            with torch.inference_mode():
                self.sam2.set_image(blank)
                self.sam2.predict(box=np.array([[64.0, 64.0, 192.0, 192.0]]), multimask_output=False)
            print("[Modal Perception] warmup inference done ✅")
        except Exception as exc:   # warmup is an optimisation; a failure must not stop the service
            print(f"[Modal Perception] warmup skipped: {type(exc).__name__}: {exc}")
        self._classes_set = ["chair"]   # matches the set_classes() above; predict() compares against it

    # GA-319 follow-up (2026-09-06). Explicit labels: Modal derives the subdomain from
    # "<workspace>--<app>-<class>-<method>" and truncates it past 63 characters with a hash. The
    # new workspace name is long enough that both endpoints came back as "...perceptionser-477194"
    # style addresses, and client.py (:118-125) recognises a pair ONLY by the "-predict.modal.run" /
    # "-health.modal.run" suffixes; its fallback appends /predict and /health to a base URL, which a
    # Modal endpoint answers with 404. Short fixed labels keep the pair recognisable regardless of
    # the workspace name.
    @modal.fastapi_endpoint(method="GET", label="lost3dsg-health")
    def health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        import torch
        return {
            "status": "ready",
            "service": APP_NAME,
            "models": {
                "detector": "YOLO-World-L (v2)",
                "segmentor": "SAM 2.1 Hiera-Small" if getattr(self, "has_sam2", False) else "fallback",
                "embedder": f"CLIP {CLIP_MODEL} (512-d, min crop {MIN_CROP_PX} px)",
            },
            # Deploy-skew stamp (module docstring): the tree `modal deploy` ran from.
            "source": {"sha": os.environ.get("LOST3DSG_SRC_SHA", "unknown"),
                       "dirty": os.environ.get("LOST3DSG_SRC_DIRTY", "unknown")},
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "has_sam2": getattr(self, "has_sam2", False),
            "vram": {
                # weights at load, current allocation, and the high-water mark since
                # start (a served frame is the peak) — all MB
                "weights_mb": getattr(self, "vram_weights_mb", 0.0),
                "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1) if torch.cuda.is_available() else 0.0,
                "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1) if torch.cuda.is_available() else 0.0,
                "peak_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1) if torch.cuda.is_available() else 0.0,
                "device_total_mb": (
                    round(torch.cuda.get_device_properties(0).total_memory / 1048576, 1)
                    if torch.cuda.is_available() else 0.0
                ),
            },
        }

    @modal.fastapi_endpoint(method="POST", label="lost3dsg-predict")
    def predict(self, req: PerceptionRequest) -> Dict[str, Any]:
        """Full open-vocabulary perception: YOLO-World-L + SAM 2.1 Masking."""
        import torch

        t_start = time.time()
        # Decode image
        img_bytes = base64.b64decode(req.image_b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        rgb_np = np.array(pil_image)
        h, w = rgb_np.shape[:2]

        if not req.labels:
            return {"detections": [], "timings_ms": {"total": 0.0}}

        # 1. YOLO-World-L Detection
        t0 = time.time()
        clean_labels = [lbl.strip().lower() for lbl in req.labels if lbl.strip()]
        # set_classes re-encodes the label list with the CLIP text tower on every call.
        # The same list yields the same class embeddings, so it is skipped when the
        # request repeats the previous list (consecutive cycles at one waypoint often do).
        if clean_labels != getattr(self, "_classes_set", None):
            self.detector.set_classes(clean_labels)
            self._classes_set = list(clean_labels)
        
        results = self.detector.predict(
            pil_image,
            conf=req.score_threshold,
            iou=req.nms_threshold,
            verbose=False,
            device=self.device
        )[0]
        
        boxes_xyxy = results.boxes.xyxy.cpu().numpy().astype(np.float64) if len(results.boxes) > 0 else np.empty((0, 4))
        scores = results.boxes.conf.cpu().numpy().astype(np.float64) if len(results.boxes) > 0 else np.empty((0,))
        cls_indices = results.boxes.cls.cpu().numpy().astype(int) if len(results.boxes) > 0 else np.empty((0,), dtype=int)
        
        pred_labels = [clean_labels[idx] for idx in cls_indices] if len(cls_indices) > 0 else []
        t_det = time.time() - t0

        if len(boxes_xyxy) == 0:
            return {
                "detections": [],
                "timings_ms": {
                    "detector": round(t_det * 1000, 1),  # same key as the non-empty path
                    "total": round((time.time() - t_start) * 1000, 1)
                }
            }

        # 2. SAM 2.1 Pixel Mask Segmentation
        t0 = time.time()
        masks = []
        with torch.inference_mode():
            self.sam2.set_image(rgb_np)
            sam_masks, sam_scores, _ = self.sam2.predict(
                box=boxes_xyxy,
                multimask_output=False,
            )
            # sam_masks: (N, 1, H, W) or (N, H, W)
            for i in range(len(boxes_xyxy)):
                m = sam_masks[i]
                if m.ndim == 3:
                    m = m[0]
                masks.append((m > 0.0).astype(np.uint8))
        if len(masks) != len(boxes_xyxy):
            raise RuntimeError(
                f"SAM 2.1 returned {len(masks)} masks for {len(boxes_xyxy)} boxes")

        t_sam = time.time() - t0

        # 3. Crop CLIP embeddings (GA-342, restoring GA-17's shape). A crop under MIN_CROP_PX
        # in either dimension gets `clip_embedding: null` -- never a placeholder image: every
        # sliver used to receive the SAME embedding (a black square's) and any two slivers then
        # scored 1.0 against each other. The box, label, score and mask are still returned.
        # `crop_regions` carries the BOX INDEX because the batch skips slivers, and the
        # assembly below reads the list by box index; pre-filled to len(boxes) so the two
        # indexings cannot drift.
        t0 = time.time()
        regions = crop_regions(boxes_xyxy, rgb_np, w, h)
        crop_embeddings = [None] * len(boxes_xyxy)
        if regions:
            inputs = self.clip_processor(images=[Image.fromarray(c) for _, c in regions],
                                         return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            if self.clip.dtype == torch.float16:
                pixel_values = pixel_values.half()
            with torch.inference_mode():
                feats = self.clip.get_image_features(pixel_values=pixel_values)
                # transformers returns a tensor or a BaseModelOutputWithPooling depending on
                # the version (measured: this image's version returns the latter; the first
                # deploy of this file 500'd on `.float()`). The pooled projection is the
                # 512-d vector either way.
                if not torch.is_tensor(feats):
                    feats = feats.pooler_output
                feats = torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()
            for k, (box_i, _) in enumerate(regions):
                crop_embeddings[box_i] = [round(float(v), 5) for v in feats[k]]
        t_clip = time.time() - t0

        # Build output response with RLE encoded binary masks
        detections_out = []
        for i, (box, label, score, mask) in enumerate(zip(boxes_xyxy, pred_labels, scores, masks)):
            detections_out.append({
                "bbox": [round(float(v), 2) for v in box],
                "label": label,
                "score": round(float(score), 4),
                "mask_rle": rle_encode(mask),
                "clip_embedding": crop_embeddings[i],
            })

        t_total = time.time() - t_start
        return {
            "detections": detections_out,
            # `detector` and `sam2` are the keys detection_pipeline requires (GA-211); `clip` is
            # an extra measured stage, kept because it is a real measurement.
            "timings_ms": {
                "detector": round(t_det * 1000, 1),
                "sam2": round(t_sam * 1000, 1),
                "clip": round(t_clip * 1000, 1),
                "total": round(t_total * 1000, 1)
            }
        }
