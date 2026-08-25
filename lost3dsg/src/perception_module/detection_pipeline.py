import os
import time

import numpy as np
import torch
from config import CFG
from cv_utils import _apply_transform, _filter_object_points
from detection_types import Detection
from utils import apply_nms


def pca_oriented_box(pts_map, min_anisotropy=1.2):
    """Yaw-about-z oriented box from object points in the map frame — the optional
    PCA keys (yaw / oriented_center / oriented_extents) that box_corners_map and
    the map store already consume. Returns None when the XY spread is too small or
    too isotropic for a stable orientation (the AABB alone is then the honest box)."""
    pts = np.asarray(pts_map, dtype=np.float64)
    if pts.shape[0] < 10:
        return None
    xy = pts[:, :2]
    cov = np.cov(xy, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return None
    evals, evecs = np.linalg.eigh(cov)  # ascending
    if evals[0] <= 1e-10 or evals[1] / evals[0] < min_anisotropy ** 2:
        return None
    major = evecs[:, 1]
    yaw = float(np.arctan2(major[1], major[0]))
    # a box is symmetric under 180°: keep yaw in [-pi/2, pi/2)
    if yaw < -np.pi / 2:
        yaw += np.pi
    elif yaw >= np.pi / 2:
        yaw -= np.pi

    c, s = np.cos(yaw), np.sin(yaw)
    u = xy[:, 0] * c + xy[:, 1] * s      # box frame
    v = -xy[:, 0] * s + xy[:, 1] * c
    z = pts[:, 2]
    # ponytail: 1/99 percentile trim instead of SOR — the AABB pass already
    # validated these points; this only guards the extents against stragglers
    lo_u, hi_u = np.percentile(u, [1, 99])
    lo_v, hi_v = np.percentile(v, [1, 99])
    lo_z, hi_z = np.percentile(z, [1, 99])
    if min(hi_u - lo_u, hi_v - lo_v, hi_z - lo_z) <= 1e-4:
        return None
    uc, vc = (lo_u + hi_u) / 2.0, (lo_v + hi_v) / 2.0
    return {
        "yaw": yaw,
        "oriented_center": [float(uc * c - vc * s), float(uc * s + vc * c), float((lo_z + hi_z) / 2.0)],
        "oriented_extents": [float(hi_u - lo_u), float(hi_v - lo_v), float(hi_z - lo_z)],
    }


class DetectionPipelineMixin:
    def run_detection(self, camera_data):
        self.log_both("info", "=== START DETECTION ===")
        t_cycle_start = time.time()
        self._refresh_room_geometry_if_available()

        if self._abort_if_moving("detection startup"):
            return []

        t0 = time.time()
        labels = self._extract_detection_labels(camera_data["rgb"])
        t_vlm = time.time() - t0
        if not labels:
            return []

        backend = getattr(self, "perception_backend", None)
        backend_type = CFG.get("perception", {}).get("backend", "local").lower()
        if backend and backend_type != "local":
            t0 = time.time()
            detections, cloud_timings = backend.detect_and_segment(
                camera_data["rgb"],
                labels,
                score_threshold=CFG.get("perception", {}).get("score_threshold", 0.15),
                nms_threshold=CFG.get("perception", {}).get("nms_threshold", 0.50),
            )
            t_cloud = time.time() - t0
            t_owlv2 = cloud_timings.get("owlv2", t_cloud * 0.4 * 1000.0) / 1000.0
            t_nms = cloud_timings.get("nms", 10.0) / 1000.0
            t_sam = cloud_timings.get("sam", t_cloud * 0.5 * 1000.0) / 1000.0
            if len(detections) == 0:
                self.log_both("info", "Cloud perception backend found no objects")
                return []
        else:
            t0 = time.time()
            bboxs, labels, scores = self._run_open_vocab_detector(camera_data["rgb"], labels)
            t_owlv2 = time.time() - t0
            if len(bboxs) == 0:
                self.log_both("info", "OWLv2 found no objects")
                return []

            t0 = time.time()
            bboxs, labels, scores = self._apply_detection_nms(bboxs, labels, scores)
            t_nms = time.time() - t0
            if len(bboxs) == 0:
                self.log_both("info", "OWLv2 found no objects after NMS")
                return []

            t0 = time.time()
            detections = self._segment_detections(camera_data["rgb"], bboxs, labels, scores)
            t_sam = time.time() - t0

        if self._abort_if_moving("SAM segmentation"):
            return []

        t0 = time.time()
        self._publish_detection_pointclouds(detections, camera_data)
        t_proj = time.time() - t0

        t_total = time.time() - t_cycle_start
        self._refresh_room_geometry_if_available()

        vlm_info = dict(getattr(self, "_vlm_status", {"status": "unknown"}))
        if hasattr(self, "vlm") and hasattr(self.vlm, "cache"):
            vlm_info["crop_cache"] = self.vlm.cache.stats
            vlm_info["crop_concurrency"] = CFG.get("vlm", {}).get("crop_concurrency", 4)

        latencies = {
            "vlm_ms": round(t_vlm * 1000.0, 1),
            "owlv2_ms": round(t_owlv2 * 1000.0, 1),
            "nms_ms": round(t_nms * 1000.0, 1),
            "sam_ms": round(t_sam * 1000.0, 1),
            "projection_ms": round(t_proj * 1000.0, 1),
            "total_ms": round(t_total * 1000.0, 1),
            "last_updated": time.time(),
            "components": {
                "vlm": vlm_info,
                "perception_backend": (
                    backend.health() if backend and hasattr(backend, "health")
                    else {"reachable": True, "type": backend_type}
                ),
            }
        }
        self.latest_latencies = latencies
        try:
            import json
            for target_path in ("/tmp/perception_latencies.json", "/ws/output/perception_latencies.json"):
                try:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, "w") as f:
                        json.dump(latencies, f, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

        self.log_both("info", f"Detection complete: {len(detections)} objects (total cycle: {t_total:.3f}s)")
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
        prompt_path = CFG["paths"]["identification_prompt"] or os.path.join(
            os.path.dirname(__file__), "prompts", "object_identification_prompt.txt")
        current_room = getattr(self, "current_room_id", "unknown")
        room_evidence = getattr(self, "current_room_labels_str", "none")
        try:
            labels = self.vlm.call_labels(
                prompt_path,
                rgb_image,
                current_room=current_room,
                room_evidence=room_evidence,
            )
            self._vlm_status = {
                "status": "ok",
                "model": CFG.get("vlm", {}).get("model", "unknown"),
                "latency_ms": round((time.time() - t0) * 1000.0, 1),
                "room_belief": getattr(self.vlm, "last_room_belief", None),
            }
        except Exception as exc:
            # Config seam: with vlm.fallback_labels set, an unreachable VLM
            # degrades to a static open-vocab list (loudly) instead of killing
            # the detection cycle. Empty list (default) = raise as before.
            fallback = CFG["vlm"].get("fallback_labels") or []
            self._vlm_status = {
                "status": "degraded_fallback" if fallback else "unreachable",
                "model": CFG.get("vlm", {}).get("model", "unknown"),
                "error": str(exc),
            }
            if not fallback:
                raise
            self.log_both("warn", f"VLM labels unavailable ({exc}); using {len(fallback)} fallback labels")
            labels = list(fallback)
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

    def _add_pca_orientation(self, detections, bboxes_3d, depth, camera_info, transform):
        """Add the optional PCA keys to each valid bbox dict, in place.
        # ponytail: re-lifts a 2k-point subsample per mask instead of editing
        # cv_utils (outside the Agent-1 boundary); fold into
        # mask_list_to_centroid_and_bbox if projection_ms ever hurts."""
        if transform is None:
            return
        fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
        for det, bbox in zip(detections, bboxes_3d):
            if not bbox:
                continue
            try:
                pts = _filter_object_points(
                    det.mask[:, :, 0], depth, fx, fy, cx, cy,
                    max_points_per_obj=2000, remove_outliers=False,
                )
                if pts is None:
                    continue
                obb = pca_oriented_box(_apply_transform(pts, transform))
                if obb:
                    bbox.update(obb)
            except Exception as exc:
                self.log_both("warn", f"PCA orientation failed for {det.instance_label}: {exc}")

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


if __name__ == "__main__":
    # Self-check for pca_oriented_box (run inside the perception container).
    yaw_true = 0.5
    xs, ys, zs = np.meshgrid(np.arange(-1.0, 1.0, 0.05), np.arange(-0.25, 0.25, 0.05), np.arange(0.0, 0.5, 0.1))
    pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1)
    c, s = np.cos(yaw_true), np.sin(yaw_true)
    pts[:, :2] = pts[:, :2] @ np.array([[c, s], [-s, c]])  # rotate xy by yaw_true
    pts += np.array([3.0, 4.0, 0.2])
    box = pca_oriented_box(pts)
    assert box is not None and abs(box["yaw"] - yaw_true) < 0.05, box
    assert np.allclose(box["oriented_extents"], [2.0, 0.5, 0.5], atol=0.15), box
    assert np.allclose(box["oriented_center"], [3.0, 4.0, 0.45], atol=0.1), box
    theta = np.linspace(0, 2 * np.pi, 500)
    circle = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    assert pca_oriented_box(circle) is None          # isotropic -> keep AABB
    assert pca_oriented_box(pts[:5]) is None         # too few points
    print("detection_pipeline PCA self-check OK")
