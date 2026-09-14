"""Local CLIP image embeddings for the unified-VLM perception path.

The cloud backend already returns a normalized CLIP ViT-B/32 image vector for every
usable detection crop.  The local path used to stop after Regolo's boxes and VitSAM's
mask, leaving ``Detection.clip_embedding`` unset.  This module keeps the same model,
crop convention, and 512-dimensional feature space on the local path.

Imports of torch/transformers are intentionally lazy.  The geometry and parser tests
must be able to import the perception package without loading a GPU model.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional, Sequence

import numpy as np

from scene_analysis import clip_pixel_bbox


CLIP_MODEL = "openai/clip-vit-base-patch32"
MIN_CROP_PX = 4


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _offline_hub() -> bool:
    return _truthy(os.environ.get("HF_HUB_OFFLINE")) or _truthy(
        os.environ.get("TRANSFORMERS_OFFLINE")
    )


def pixel_crop_rgb(
    image_bgr: np.ndarray,
    bbox: Sequence[float],
    min_crop_px: int = MIN_CROP_PX,
):
    """Return one clipped BGR image box as a PIL RGB image, or ``None``.

    Scene analysis and the ROS message use half-open ``(x_min, y_min, x_max, y_max)``
    pixel boxes.  The integer conversion intentionally matches the cloud CLIP path in
    ``cloud/modal_perception.py``: truncate the floating-point endpoints, clamp to the
    image, and then apply the minimum-size check.  Keeping both backends on one slicing
    convention prevents the same VLM box from receiving different appearance features
    depending on where segmentation ran.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        return None
    height, width = image.shape[:2]
    clipped = clip_pixel_bbox(bbox, width, height)
    if clipped is None:
        return None

    x_min, y_min, x_max, y_max = clipped
    x0 = max(0, min(width, int(x_min)))
    y0 = max(0, min(height, int(y_min)))
    x1 = max(0, min(width, int(x_max)))
    y1 = max(0, min(height, int(y_max)))
    minimum = max(int(min_crop_px), 1)
    if x1 <= x0 or y1 <= y0 or x1 - x0 < minimum or y1 - y0 < minimum:
        return None

    # cv_bridge and the rest of the local pipeline use BGR.  CLIPProcessor expects
    # RGB PIL input, so convert explicitly instead of relying on a caller's name.
    crop_bgr = np.ascontiguousarray(image[y0:y1, x0:x1, :3])
    crop_rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
    from PIL import Image

    return Image.fromarray(crop_rgb.astype(np.uint8), mode="RGB")


class ClipEmbedder:
    """Batch-normalized CLIP image encoder used by local detections."""

    def __init__(
        self,
        model_id: str = CLIP_MODEL,
        device: str = "auto",
        require_cuda: bool = False,
        min_crop_px: int = MIN_CROP_PX,
        batch_size: int = 8,
    ):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        requested = str(device or "auto").strip().lower()
        if requested in ("", "auto"):
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            selected = str(device).strip()
        if _truthy(require_cuda) and not selected.lower().startswith("cuda"):
            raise RuntimeError(
                "CLIP appearance embeddings require CUDA, but the configured device is "
                f"{selected!r}"
            )
        if selected.lower().startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CLIP appearance embeddings requested {selected!r}, but PyTorch CUDA "
                "is unavailable"
            )

        self.model_id = str(model_id)
        self.device = torch.device(selected)
        self.min_crop_px = max(int(min_crop_px), 1)
        self.batch_size = max(int(batch_size), 1)
        local_files_only = _offline_hub()
        self.processor = CLIPProcessor.from_pretrained(
            self.model_id, local_files_only=local_files_only
        )
        self.model = CLIPModel.from_pretrained(
            self.model_id, local_files_only=local_files_only
        ).to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.dimension = int(getattr(self.model.config, "projection_dim", 512))

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> Optional["ClipEmbedder"]:
        settings = cfg.get("appearance", {}) if isinstance(cfg, dict) else {}
        settings = settings if isinstance(settings, dict) else {}
        if not _truthy(settings.get("enabled", True)):
            return None

        # VITSAM_REQUIRE_CUDA is the deployment-wide GPU gate already used by the
        # launcher.  An explicit appearance.require_cuda can only strengthen it;
        # setting VITSAM_REQUIRE_CUDA=0 remains the documented CPU escape hatch.
        require_cuda = _truthy(settings.get("require_cuda", False)) or _truthy(
            os.environ.get("VITSAM_REQUIRE_CUDA")
        )
        return cls(
            model_id=str(settings.get("model_id", CLIP_MODEL)),
            device=str(settings.get("device", "auto")),
            require_cuda=require_cuda,
            min_crop_px=int(settings.get("min_crop_px", MIN_CROP_PX)),
            batch_size=int(settings.get("batch_size", 8)),
        )

    def embed_boxes(
        self,
        image_bgr: np.ndarray,
        boxes: Sequence[Sequence[float]],
    ) -> list[Optional[list[float]]]:
        """Return one normalized vector slot per input box.

        Invalid boxes and crops below the minimum size remain ``None`` at their original
        index.  A dropped sliver must never shift another object's appearance onto it.
        """
        result: list[Optional[list[float]]] = [None] * len(boxes)
        prepared = []
        for index, bbox in enumerate(boxes):
            crop = pixel_crop_rgb(image_bgr, bbox, self.min_crop_px)
            if crop is not None:
                prepared.append((index, crop))
        if not prepared:
            return result

        torch = self.torch
        model_dtype = next(self.model.parameters()).dtype
        for start in range(0, len(prepared), self.batch_size):
            batch = prepared[start : start + self.batch_size]
            inputs = self.processor(
                images=[crop for _, crop in batch], return_tensors="pt"
            )
            pixel_values = inputs["pixel_values"].to(self.device)
            if model_dtype == torch.float16:
                pixel_values = pixel_values.half()
            with torch.inference_mode():
                features = self.model.get_image_features(pixel_values=pixel_values)
                # Transformers versions used by the cloud service have returned both
                # a tensor and a pooled output wrapper. Accept both forms explicitly.
                if not torch.is_tensor(features):
                    features = features.pooler_output
                features = torch.nn.functional.normalize(features.float(), dim=-1)

            vectors = features.detach().cpu().numpy()
            if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
                raise RuntimeError(
                    f"CLIP returned shape {vectors.shape}, expected (*, {self.dimension})"
                )
            for row, (index, _) in zip(vectors, batch):
                if not np.all(np.isfinite(row)):
                    continue
                result[index] = [float(value) for value in row]
        return result
