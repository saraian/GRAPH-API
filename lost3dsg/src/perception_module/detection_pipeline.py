import os
import time
import torch
import numpy as np

from utils import apply_nms
from detection_types import Detection


class DetectionPipelineMixin:
    def run_detection(self, camera_data):
        self.log_both("info", "=== START DETECTION ===")
        self._refresh_room_geometry_if_available()

        if self._abort_if_moving("detection startup"):
            return []

        labels = self._extract_detection_labels(camera_data["rgb"])
        if not labels:
            return []

        bboxs, labels, scores = self._run_open_vocab_detector(camera_data["rgb"], labels)
        if len(bboxs) == 0:
            self.log_both("info", "OWLv2 found no objects")
            return []

        bboxs, labels, scores = self._apply_detection_nms(bboxs, labels, scores)
        if len(bboxs) == 0:
            self.log_both("info", "OWLv2 found no objects after NMS")
            return []

        detections = self._segment_detections(camera_data["rgb"], bboxs, labels, scores)
        if self._abort_if_moving("SAM segmentation"):
            return []

        self._publish_detection_pointclouds(detections, camera_data)
        self._refresh_room_geometry_if_available()
        self.log_both("info", f"Detection complete: {len(detections)} objects")
        return detections

    def _refresh_room_geometry_if_available(self):
        refresh_fn = getattr(self, "refresh_current_room_geometry", None)
        if not callable(refresh_fn):
            return

        try:
            refresh_fn()
        except Exception as exc:
            if hasattr(self, "log_both"):
                self.log_both("warn", f"Room geometry refresh skipped: {exc}")

    def _extract_detection_labels(self, rgb_image):
        t0 = time.time()
        prompt_path = os.path.join(os.path.dirname(__file__), "object_identification_prompt.txt")
        labels = self.vlm.call_labels(prompt_path, rgb_image)
        self.log_both("info", f"[PROFILE] VLM labels: {time.time() - t0:.3f}s")
        self.log_both("info", f"[PROFILE] Labels: {labels}")

        if self._abort_if_moving("VLM label extraction"):
            return []
        if not labels:
            self.log_both("warn", "VLM returned no labels")
            return []
        return labels

    def _run_open_vocab_detector(self, rgb_image, labels):
        self.detector.set_classes(labels)
        t0 = time.time()
        with torch.inference_mode():
            bboxs, detected_labels, scores = self.detector.predict(rgb_image)
        self.log_both("info", f"[PROFILE] OWLv2 predict: {time.time() - t0:.3f}s")

        if self._abort_if_moving("OWLv2 detection"):
            return [], [], []
        return bboxs, detected_labels, scores

    def _apply_detection_nms(self, bboxs, labels, scores):
        t0 = time.time()
        bboxs, labels, scores = apply_nms(bboxs, labels, scores, iou_threshold=0.5)
        self.log_both("info", f"[PROFILE] NMS: {time.time() - t0:.3f}s")
        return bboxs, labels, scores

    def _segment_detections(self, rgb_image, bboxs, labels, scores):
        t0 = time.time()
        with torch.inference_mode():
            detections = self._run_vitsam(rgb_image, bboxs, labels, scores)
        self.log_both("info", f"[PROFILE] SAM/mask + Detection build: {time.time() - t0:.3f}s")
        return detections

    def _publish_detection_pointclouds(self, detections, camera_data):
        self.pcl_object_id_counter = 0
        t0 = time.time()
        self.color_pcl(detections, camera_data)
        self.log_both("info", f"[PROFILE] PointCloud2 (color_pcl): {time.time() - t0:.3f}s")

    def _run_vitsam(self, rgb_image, bboxs, labels, scores):
        detections = []
        if len(bboxs) == 0:
            return detections

        for bbox, label_name, score in zip(bboxs, labels, scores):
            masks, _ = self.vitsam(rgb_image, bbox)
            masks_np = np.asarray(masks)

            if masks_np.ndim == 2:
                masks_np = masks_np[None, :, :]
            elif masks_np.ndim == 3 and masks_np.shape[-1] == 1:
                masks_np = np.transpose(masks_np, (2, 0, 1))

            for mask in masks_np:
                mask = np.asarray(mask)
                mask = np.squeeze(mask)
                mask = (mask > 0).astype(np.uint8)
                detections.append(
                    Detection(
                        bbox=tuple(bbox),
                        label=label_name,
                        score=float(score),
                        mask=mask[..., None],
                    )
                )
        return detections
