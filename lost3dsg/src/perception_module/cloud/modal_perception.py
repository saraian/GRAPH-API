"""Modal Serverless Perception Microservice with YOLO-World-L and SAM 2.1.

Deploy with:
    modal deploy lost3dsg/src/perception_module/cloud/modal_perception.py

Runs:
1. YOLO-World-L open-vocabulary object detection (fast, tight indoor bounding boxes)
2. SAM 2.1 (Segment Anything 2.1 Hiera-Small) instance segmentation producing exact pixel masks
3. Semantic Cross-Class NMS and crop feature embeddings

Deployed on an NVIDIA T4 / L4 GPU on Modal with scale-to-zero.
"""

import base64
import io
import time
from typing import Any, Dict, List

import modal
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

APP_NAME = "lost3dsg-perception"

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
        # Pre-cache YOLO-World-L and SAM 2.1 model weights in the image build layer
        "python3 -c \"from ultralytics import YOLOWorld; YOLOWorld('yolov8l-worldv2.pt')\"",
        "python3 -c \"from sam2.sam2_image_predictor import SAM2ImagePredictor; SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-small', device='cpu')\"",
    )
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
    scaledown_window=20,    # Scale-to-zero after 20s idle
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

        # weights-resident VRAM, measured once both models are on-device (peak
        # during inference is reported separately by health())
        self.vram_weights_mb = (
            round(torch.cuda.memory_allocated() / 1048576, 1) if torch.cuda.is_available() else 0.0
        )
        print(f"[Modal Perception] VRAM after load: {self.vram_weights_mb} MB")

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
            },
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
        self.detector.set_classes(clean_labels)
        
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

        # Build output response with RLE encoded binary masks
        detections_out = []
        for i, (box, label, score, mask) in enumerate(zip(boxes_xyxy, pred_labels, scores, masks)):
            detections_out.append({
                "bbox": [round(float(v), 2) for v in box],
                "label": label,
                "score": round(float(score), 4),
                "mask_rle": rle_encode(mask),
            })

        t_total = time.time() - t_start
        return {
            "detections": detections_out,
            "timings_ms": {
                "detector": round(t_det * 1000, 1),
                "sam2": round(t_sam * 1000, 1),
                "total": round(t_total * 1000, 1)
            }
        }
