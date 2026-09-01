"""Modal Serverless Perception Microservice.

Deploy with:
    modal deploy lost3dsg/src/perception_module/cloud/modal_perception.py

Runs OWLv2 open-vocabulary detection + SAM instance segmentation + CLIP feature extraction
on an NVIDIA L4 GPU in the cloud with per-second billing and scale-to-zero.

THIS FILE IS NOT WHAT IS DEPLOYED, AND THE THING THAT IS DEPLOYED HAS NO COMMIT.
Established 2026-09-01; recorded here so it is not re-derived from timing keys a third time.

  what runs        YOLO-World-L (v2) + SAM 2.1 Hiera-Small. Not inferred — the health record
                   in every run bundle says so verbatim:
                   perception_latencies.json .components.perception_backend.details.models
  where it lives   /DATA/GRAPH-API working tree ONLY, as an UNCOMMITTED edit to this same
                   path. `git show HEAD:` in BOTH repositories returns THIS OWLv2 file;
                   neither HEAD contains the string `YOLOWorld`.
  what a clone     the OWLv2 server below. A fresh checkout of either repository, deployed,
  would deploy     REPLACES the running detector.

Two consequences, and they point opposite ways:

1. DO NOT redeploy this file to make GA-17's fix live. It would swap the detector mid-
   experiment, and the fix it would carry protects a field (`clip_embedding`) that nothing
   reads: measured over the whole tree, clip_embedding reaches only detection_types.py,
   client.py:221 and perception_2.py, which stores it and dumps output/clip_embeddings.json.
   No association, no merge, no map, no viewer.

2. DO NOT discard GA-17's fix either. This file is what both repositories hold COMMITTED, so
   it is what any fresh clone deploys — the defect is in the version with the widest reach,
   not a dead branch.

The provenance hole is the finding, and it is worse than a stale build. A stale build has a
commit that describes it. THE RUNNING SERVICE HAS NONE: it was deployed from an uncommitted
working tree, so if that tree is lost the running service cannot be reproduced from either
repository. GA-85 named a deploy skew between builds; this is a deploy from nothing.
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
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "opencv-python-headless>=4.9.0",
        "pillow>=10.2.0",
        "numpy>=1.24.0",
        "fastapi>=0.110.0",
        "pycocotools>=2.0.7",
    )
    .run_commands(
        # Pre-cache OWLv2 and SAM model weights in the image build layer
        "python3 -c \"from transformers import Owlv2Processor, Owlv2ForObjectDetection, SamModel, SamProcessor; "
        "Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble'); "
        "Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble'); "
        "SamProcessor.from_pretrained('facebook/sam-vit-base'); "
        "SamModel.from_pretrained('facebook/sam-vit-base')\"",
    )
)

app = modal.App(APP_NAME, image=perception_image)


class PerceptionRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded JPEG/PNG image")
    labels: List[str] = Field(..., description="Candidate open-vocabulary labels from VLM")
    score_threshold: float = Field(default=0.15, description="Confidence threshold for detections")
    nms_threshold: float = Field(default=0.50, description="IoU threshold for non-maximum suppression")


MIN_CROP_PX = 4
"""Below this, in EITHER dimension, a crop carries no usable appearance."""


def crop_regions(boxes, rgb_np, w, h, min_px=MIN_CROP_PX):
    """-> [(box_index, crop_array)] for the boxes whose crop is at least `min_px` a side.

    The BOX INDEX travels with the crop. The embeddings are computed in one batch, so
    dropping a crop shifts every later position in that batch while the assembly step still
    indexes `crop_embeddings` by box; carrying the index is what keeps the two from drifting.
    Split out of the endpoint so this can be tested without a GPU, weights, or Modal — the
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


@app.cls(
    gpu="T4",               # Economical GPU tier ($0.000164/sec) to conserve credits
    scaledown_window=15,    # Aggressive scale-to-zero: stop container after 15s idle
    max_containers=1,       # Budget guardrail: strictly at most 1 GPU instance
    timeout=30,             # Kill hung jobs after 30s
)
class PerceptionService:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Modal Perception] Initializing models on {self.device}...")

        # 1. OWLv2 Detector & CLIP backbone
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.detector = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device).eval()

        # 2. SAM Segmentation Model
        try:
            from transformers import SamModel, SamProcessor
            self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            self.sam_model = SamModel.from_pretrained(
                "facebook/sam-vit-base",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device).eval()
            self.has_sam = True
        except Exception as exc:
            print(f"[Modal Perception] SAM load warning: {exc}. Segmentation will use bbox fallback.")
            self.has_sam = False

        print("[Modal Perception] Models successfully loaded and ready.")

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> Dict[str, Any]:
        """Health and readiness check."""
        import torch
        return {
            "status": "ready",
            "service": APP_NAME,
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "has_sam": getattr(self, "has_sam", False),
        }

    @modal.fastapi_endpoint(method="POST")
    def predict(self, req: PerceptionRequest) -> Dict[str, Any]:
        """Full open-vocabulary perception: Detection + SAM Masking + Crop Embedding."""
        import torch

        t_start = time.time()
        # Decode image
        img_bytes = base64.b64decode(req.image_b64)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        rgb_np = np.array(pil_image)
        h, w = rgb_np.shape[:2]

        if not req.labels:
            return {"detections": [], "timings_ms": {"total": 0.0}}

        # 1. OWLv2 Open-Vocab Detection
        t0 = time.time()
        text_queries = [f"a photo of a {label}" for label in req.labels]
        inputs = self.processor(
            text=[text_queries],
            images=pil_image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        if "pixel_values" in inputs and self.detector.dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

        with torch.inference_mode():
            outputs = self.detector(**inputs)
            target_sizes = torch.tensor([[h, w]], device=self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=req.score_threshold,
                text_labels=[req.labels],
            )[0]

        boxes = results["boxes"].cpu().numpy().astype(np.float64)
        scores = results["scores"].cpu().numpy().astype(np.float64)
        labels = results["text_labels"]
        t_owlv2 = time.time() - t0

        if len(boxes) == 0:
            return {
                "detections": [],
                "timings_ms": {
                    "owlv2": round(t_owlv2 * 1000, 1),
                    "total": round((time.time() - t_start) * 1000, 1)
                }
            }

        # 2. NMS
        t0 = time.time()
        keep_indices = []
        # Per-class NMS
        for unique_label in set(labels):
            cls_mask = [i for i, lbl in enumerate(labels) if lbl == unique_label]
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = cls_scores.argsort()[::-1]

            while order.size > 0:
                i = order[0]
                keep_indices.append(cls_mask[i])
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w_inter = np.maximum(0.0, xx2 - xx1)
                h_inter = np.maximum(0.0, yy2 - yy1)
                inter = w_inter * h_inter
                iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
                inds = np.where(iou <= req.nms_threshold)[0]
                order = order[inds + 1]

        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = [labels[i] for i in keep_indices]
        t_nms = time.time() - t0

        # 3. SAM Instance Segmentation
        t0 = time.time()
        masks = []
        if self.has_sam and len(boxes) > 0:
            sam_boxes = [[[float(b[0]), float(b[1]), float(b[2]), float(b[3])] for b in boxes]]
            sam_inputs = self.sam_processor(pil_image, input_boxes=sam_boxes, return_tensors="pt")
            sam_inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in sam_inputs.items()}
            if "pixel_values" in sam_inputs and self.sam_model.dtype == torch.float16:
                sam_inputs["pixel_values"] = sam_inputs["pixel_values"].to(torch.float16)
            with torch.inference_mode():
                sam_outputs = self.sam_model(**sam_inputs)
                raw_masks = self.sam_processor.image_processor.post_process_masks(
                    sam_outputs.pred_masks.cpu(),
                    sam_inputs["original_sizes"].cpu(),
                    sam_inputs["reshaped_input_sizes"].cpu()
                )[0]
            for m in raw_masks:
                mask_np = m[0].numpy().astype(np.uint8)  # Best predicted IoU mask
                masks.append(mask_np)
        else:
            # Fallback rectangular mask
            for b in boxes:
                m = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, b)
                m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
                masks.append(m)
        t_sam = time.time() - t0

        # 4. Crop CLIP Feature Extraction (using OWLv2 image features)
        t0 = time.time()
        # GA-17. A crop under MIN_CROP_PX in either dimension was replaced by
        # `np.zeros((64, 64, 3))` and embedded like a real crop. Every sliver in a frame
        # therefore received the SAME embedding — the encoding of a black square — so any
        # two slivers scored 1.0 against each other whatever the threshold was set to. A
        # substitute that is indistinguishable downstream from a measurement is the same
        # fault as an estimated stage timing (GA-14), in the same service.
        #
        # The remedy is `clip_embedding: null` for that detection: the box, label, score and
        # mask are all still real and are still returned. `detection_types.Detection` already
        # declares the field Optional and the assembly at :271 already emits None, so the
        # absence travels without a consumer change.
        #
        # `crop_index` exists because the embeddings are batched: skipping a crop shifts
        # every later index, and the assembly reads crop_embeddings[i] against the BOX index.
        # The list is pre-filled to len(boxes) so the two indexings cannot drift apart.
        regions = crop_regions(boxes, rgb_np, w, h)
        crops = [Image.fromarray(c) for _, c in regions]

        crop_embeddings = [None] * len(boxes)
        if crops:
            crop_inputs = self.processor(images=crops, return_tensors="pt")
            crop_inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in crop_inputs.items()}
            if "pixel_values" in crop_inputs and self.detector.dtype == torch.float16:
                crop_inputs["pixel_values"] = crop_inputs["pixel_values"].to(torch.float16)
            with torch.inference_mode():
                feats = self.detector.owlv2.get_image_features(pixel_values=crop_inputs["pixel_values"])
                if not torch.is_tensor(feats):
                    feats = feats.pooler_output
                feats = torch.nn.functional.normalize(feats, dim=-1).cpu().numpy()
                for k, (box_i, _) in enumerate(regions):
                    crop_embeddings[box_i] = [round(float(v), 5) for v in feats[k]]
        t_clip = time.time() - t0

        # Assemble Detections
        detections_out = []
        for i in range(len(boxes)):
            b = boxes[i].tolist()
            detections_out.append({
                "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                "label": str(labels[i]),
                "score": float(scores[i]),
                "mask_rle": rle_encode(masks[i]),
                "clip_embedding": crop_embeddings[i] if i < len(crop_embeddings) else None,
            })

        t_total = time.time() - t_start
        return {
            "status": "ok",
            "detections": detections_out,
            "timings_ms": {
                "owlv2": round(t_owlv2 * 1000, 1),
                "nms": round(t_nms * 1000, 1),
                "sam": round(t_sam * 1000, 1),
                "clip": round(t_clip * 1000, 1),
                "total": round(t_total * 1000, 1),
            }
        }
