import logging
import http.client
import os
import sys
import time

import numpy as np
import torch
from config import CFG
from cv_utils import _apply_transform, _filter_object_points
from detection_types import Detection


def mask_touches_border(mask, margin_px=2):
    """True when the mask has a pixel within `margin_px` of any image edge — i.e. the
    object very likely continues OUTSIDE the frame and the mask is a clipped wedge.

    GA-315. PCA on a wedge returns its hypotenuse: a right isosceles triangle has an
    eigenvalue ratio of 3.0 (anisotropy 1.73, past the 1.2 gate) and a principal axis at
    45 degrees, whatever the object's real axis. Measured over 25 bundles' detections.jsonl
    (2026-09-07): beds longer than 1.7 m whose mask touches a border are diagonal
    (25-65 degrees mod 90) in 64 of 118; fully-in-frame beds in 9 of 43. No single-view
    estimator can recover the axes from a wedge that holds one bed edge, so the honest
    answer for a clipped mask is NO orientation, not a guessed one.

    The 2D box could stand in for the mask, but the wedge is in the raw mask (re-lifts of
    the depth PNGs reproduce the archived yaws to 1.8 degrees), so the mask is the instrument."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    k = max(int(margin_px), 1)
    return bool(m[:k].any() or m[-k:].any() or m[:, :k].any() or m[:, -k:].any())


def _rectangle_support_yaw(xy, tol=0.04, step_deg=1.0):
    """Choose the yaw with the most support on the fitted rectangle perimeter."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 10:
        return None
    best_n, best_th = -1, None
    for deg in np.arange(-90.0, 90.0, step_deg):
        th = np.radians(deg)
        c, s = np.cos(th), np.sin(th)
        u = xy[:, 0] * c + xy[:, 1] * s
        v = -xy[:, 0] * s + xy[:, 1] * c
        lu, hu = np.percentile(u, [1, 99])
        lv, hv = np.percentile(v, [1, 99])
        n = int(((np.abs(u - lu) <= tol) | (np.abs(u - hu) <= tol)
                 | (np.abs(v - lv) <= tol) | (np.abs(v - hv) <= tol)).sum())
        if n > best_n:
            best_n, best_th = n, float(th)
    return best_th


def pca_oriented_box(pts_map, min_anisotropy=1.2, top_fraction=0.2, top_min_points=30):
    """Yaw-about-z oriented box from object points in the map frame — the optional
    PCA keys (yaw / oriented_center / oriented_extents) that box_corners_map and
    the map store already consume. Returns None when the XY spread is too small or
    too isotropic for a stable orientation (the AABB alone is then the honest box).

    The top surface determines the perimeter-supported yaw; all points determine extents.
    Near-square or poorly supported sets return None so the AABB remains authoritative."""
    pts = np.asarray(pts_map, dtype=np.float64)
    if pts.shape[0] < 10:
        return None
    xy = pts[:, :2]
    z_all = pts[:, 2]
    top = z_all >= z_all.max() - top_fraction * max(z_all.max() - z_all.min(), 1e-6)
    fit_xy = xy[top] if int(top.sum()) >= top_min_points else xy
    if not np.all(np.isfinite(fit_xy)):
        return None
    yaw = _rectangle_support_yaw(fit_xy)
    if yaw is None:
        return None
    # a box is symmetric under 180°: keep yaw in [-pi/2, pi/2)
    if yaw < -np.pi / 2:
        yaw += np.pi
    elif yaw >= np.pi / 2:
        yaw -= np.pi

    c, s = np.cos(yaw), np.sin(yaw)
    u = xy[:, 0] * c + xy[:, 1] * s      # box frame
    v = -xy[:, 0] * s + xy[:, 1] * c
    z = pts[:, 2]
    lo_u, hi_u = np.percentile(u, [5, 95])
    lo_v, hi_v = np.percentile(v, [5, 95])
    lo_z, hi_z = np.percentile(z, [5, 95])
    if min(hi_u - lo_u, hi_v - lo_v, hi_z - lo_z) <= 1e-4:
        return None
    du, dv = hi_u - lo_u, hi_v - lo_v
    if dv > du:
        yaw = yaw + np.pi / 2 if yaw < 0 else yaw - np.pi / 2
        c, s = np.cos(yaw), np.sin(yaw)
        u, v = xy[:, 0] * c + xy[:, 1] * s, -xy[:, 0] * s + xy[:, 1] * c
        lo_u, hi_u = np.percentile(u, [5, 95])
        lo_v, hi_v = np.percentile(v, [5, 95])
        du, dv = hi_u - lo_u, hi_v - lo_v
    if du / max(dv, 1e-6) < min_anisotropy:
        return None
    uc, vc = (lo_u + hi_u) / 2.0, (lo_v + hi_v) / 2.0
    return {
        "yaw": float(yaw),
        "oriented_center": [float(uc * c - vc * s), float(uc * s + vc * c), float((lo_z + hi_z) / 2.0)],
        "oriented_extents": [float(du), float(dv), float(hi_z - lo_z)],
    }


def _backend_health_no_roundtrip(backend, backend_type):
    """LAT-4. Telemetry must never add a network round trip to the detect path.

    `backend.health()` was called once per cycle to fill one telemetry field. On the HTTP
    client that is a SECOND Modal request beside the detection itself -- ~1.5 s warm, and a
    cold container turns it into tens of seconds for a field nobody controls anything with.

    The HTTP client already caches its last result in `last_health`, refreshed by whoever
    probes it deliberately, so the field is served from that. With no probe yet the value is
    reported as UNKNOWN rather than as reachable: this path cannot tell, and writing `True`
    here would be the telemetry asserting something it did not measure.

    Local and provider backends compute their health from state already in memory, with no
    I/O at all, so they are still called directly.
    """
    if backend is None or not hasattr(backend, "health"):
        return {"reachable": True, "type": backend_type}
    if hasattr(backend, "health_url"):          # the HTTP client: cached only, never probed here
        cached = getattr(backend, "last_health", None)
        if cached:
            return dict(cached, source="cached (LAT-4: not probed on the detect path)")
        return {"reachable": None, "type": backend_type,
                "source": "not probed on the detect path (LAT-4)"}
    return backend.health()


class DetectionPipelineMixin:
    def run_detection(self, camera_data):
        self.log_both("info", "=== START DETECTION ===")
        t_cycle_start = time.time()
        self._refresh_room_geometry_if_available()

        if self._abort_if_moving("detection startup"):
            return []

        # None on the local path, and None is the honest value there: a local backend makes
        # no request, so there is no wire time to report. 0.0 would read as "measured zero".
        client_timings = None
        backend = getattr(self, "perception_backend", None)
        backend_type = CFG.get("perception", {}).get("backend", "local").lower()

        t0 = time.time()
        # Scene analysis is a VLM operation, independent of where detection and
        # segmentation run. Every perception backend therefore starts from the same
        # single structured VLM response rather than selecting a VLM workflow by
        # backend type.
        scene_objects = self._extract_scene_objects(camera_data["rgb"])
        if not scene_objects:
            return []
        t_vlm = time.time() - t0

        if backend_type != "local":
            if backend is None:
                raise RuntimeError(
                    f"perception backend {backend_type!r} was selected but no backend was constructed"
                )
            t0 = time.time()
            try:
                detections, cloud_timings = backend.segment_scene(
                    camera_data["rgb"], scene_objects)
            except (TimeoutError, OSError, http.client.HTTPException, RuntimeError) as exc:
                # A transient remote segmentation failure skips one cycle and is counted;
                # repeated failures terminate the run instead of silently substituting data.
                self._det_strikes = getattr(self, "_det_strikes", 0) + 1
                strikes_max = int(CFG.get("perception", {}).get("detector_strikes_max", 3))
                self._detector_status = {
                    "status": "unreachable",
                    "backend": backend_type,
                    "error": f"{type(exc).__name__}: {str(exc)[:260]}",
                    "consecutive_failures": self._det_strikes,
                    "strikes_max": strikes_max,
                }
                self.log_both(
                    "error",
                    f"[SEGMENTATION] segment_scene FAILED ({type(exc).__name__}: "
                    f"{str(exc)[:160]}); cycle skipped; strike "
                    f"{self._det_strikes}/{strikes_max}",
                )
                if strikes_max > 0 and self._det_strikes >= strikes_max:
                    self.log_both(
                        "error",
                        f"[SEGMENTATION] ENDING THE RUN: the perception service failed "
                        f"on {self._det_strikes} consecutive cycles.",
                    )
                    for handler in list(logging.getLogger().handlers):
                        try:
                            handler.flush()
                        except Exception:
                            pass
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os._exit(1)
                if strikes_max <= 0:
                    raise
                return []
            self._det_strikes = 0
            t_cloud = time.time() - t0
            client_timings = cloud_timings.get("client")
            if len(detections) == 0:
                raise RuntimeError(
                    "segmentation backend returned no masks for a non-empty VLM scene"
                )
            # Masks without a segmentation timing are a broken backend contract. There
            # is deliberately no detector timing: the VLM supplied the only boxes.
            missing = [k for k in ("sam2",) if k not in cloud_timings]
            if missing:
                raise RuntimeError(
                    f"perception backend returned no timing for {missing}; refusing to "
                    f"estimate it from the wall clock "
                    f"(reported keys: {sorted(cloud_timings)})"
                )
            t_owlv2 = 0.0
            t_nms = 0.0
            nms_reported = False
            t_sam = cloud_timings["sam2"] / 1000.0
            t_backend_overhead = max(0.0, t_cloud - (t_owlv2 + t_nms + t_sam))
            server_total = cloud_timings.get("total")
            server_total_reported = server_total is not None
            t_wire = max(0.0, t_cloud - server_total / 1000.0) if server_total_reported else None
        else:
            # The VLM already produced one box per object instance. Keep the legacy
            # timing fields well-defined for stages that were deliberately not run.
            t_owlv2 = 0.0
            t_nms = 0.0
            nms_reported = False
            t_backend_overhead = 0.0   # no round trip on the local path: measured, not assumed
            t_wire = 0.0               # likewise: no wire
            server_total_reported = False

            t0 = time.time()
            detections = self._segment_scene_objects(camera_data["rgb"], scene_objects)
            t_sam = time.time() - t0
            if not detections:
                self.log_both("warn", "VitSAM returned no masks for the VLM scene boxes")
                return []

        if self._abort_if_moving("SAM segmentation"):
            return []

        t0 = time.time()
        self._publish_detection_pointclouds(detections, camera_data)
        t_proj = time.time() - t0

        t_total = time.time() - t_cycle_start
        self._refresh_room_geometry_if_available()

        vlm_info = dict(getattr(self, "_vlm_status", {"status": "unknown"}))

        latencies = {
            "vlm_ms": round(t_vlm * 1000.0, 1),
            "owlv2_ms": round(t_owlv2 * 1000.0, 1),
            "nms_ms": round(t_nms * 1000.0, 1),
            # GA-14: 0.0 in nms_ms means BOTH "measured as zero" and "not reported
            # separately by the backend". This says which, so a reader can tell them
            # apart. A field that distinguishes two cases is worth nothing until a
            # reader distinguishes them — so any consumer quoting nms_ms must read this.
            "nms_reported": nms_reported,
            "backend_overhead_ms": round(t_backend_overhead * 1000.0, 1),
            "wire_ms": round(t_wire * 1000.0, 1) if t_wire is not None else None,
            "server_total_reported": server_total_reported,
            # The CLIENT's own decomposition of what `wire_ms` lumps together, forwarded
            # from cloud/client.py. Item 13 turns on being able to split it:
            #     total_ms - client.request_ms  our encode and mask decode
            #     client.request_ms - server total   connection, queueing, COLD START
            # Nested under its own key because everything above it is the SERVER's number
            # and these are ours; run 19 put 42,256 ms of 42,275 into `wire` and nothing
            # could say which half it was. `payload_bytes` is here because it is MEASURED:
            # a 36.6 KiB body cannot take 42 s, and that was an inference until now.
            "client": client_timings,
            "sam_ms": round(t_sam * 1000.0, 1),
            "projection_ms": round(t_proj * 1000.0, 1),
            # WN1. This is the DETECTION sub-span only (entry of run_detection to here) --
            # NOT the cycle: `publish_objects` wraps it with FOV computation, 3D geometry,
            # PCA orientation, crops, archiving and the world-model write, which together
            # roughly double the wall time. Kept under this name because summarize_run and
            # the viewer read it; the true cycle is `cycle_ms`, written by publish_objects
            # at cycle completion. Any latency table must quote `cycle_ms`.
            "total_ms": round(t_total * 1000.0, 1),
            "last_updated": time.time(),
            "components": {
                "vlm": vlm_info,
                "perception_backend": _backend_health_no_roundtrip(backend, backend_type),
            }
        }
        self.latest_latencies = latencies
        try:
            import json
            metrics_root = os.environ.get("GRAPH_API_OUTPUT_DIR", "/root/exchange/output")
            for target_path in (
                    "/tmp/perception_latencies.json",
                    os.path.join(metrics_root, "perception_latencies.json")):
                try:
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with open(target_path, "w") as f:
                        json.dump(latencies, f, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

        self.log_both("info", f"Detection complete: {len(detections)} objects (detection span: {t_total:.3f}s; "
                              f"the full cycle is `cycle_ms`, written at publish_objects completion -- WN1)")
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

    def _extract_scene_objects(self, rgb_image):
        """Return boxes and attributes from exactly one whole-frame VLM call."""
        t0 = time.time()
        prompt_path = CFG["paths"].get("scene_analysis_prompt") or os.path.join(
            os.path.dirname(__file__), "prompts", "scene_analysis_prompt.txt")
        try:
            scene_objects = self.vlm.call_scene(prompt_path, rgb_image)
            self._vlm_status = {
                "status": "ok",
                "model": CFG.get("vlm", {}).get("model", "unknown"),
                "latency_ms": round((time.time() - t0) * 1000.0, 1),
            }
        except Exception as exc:
            # Match the current label-call outage policy: a transient failure skips
            # this cycle loudly, while repeated failures stop a run. Crucially, no
            # fallback detector or static labels substitute for a failed scene call.
            self._vlm_strikes = getattr(self, "_vlm_strikes", 0) + 1
            strikes_max = int(CFG.get("perception", {}).get("vlm_strikes_max", 3))
            self._vlm_status = {
                "status": "unreachable",
                "model": CFG.get("vlm", {}).get("model", "unknown"),
                "error": str(exc)[:300],
                "consecutive_failures": self._vlm_strikes,
                "strikes_max": strikes_max,
            }
            self.log_both(
                "error",
                f"[VLM] unified scene call FAILED ({type(exc).__name__}: "
                f"{str(exc)[:160]}); cycle skipped; strike "
                f"{self._vlm_strikes}/{strikes_max}",
            )
            if strikes_max > 0 and self._vlm_strikes >= strikes_max:
                self.log_both(
                    "error",
                    f"[VLM] ENDING THE RUN: the unified scene call failed on "
                    f"{self._vlm_strikes} consecutive cycles.",
                )
                for handler in list(logging.getLogger().handlers):
                    try:
                        handler.flush()
                    except Exception:
                        pass
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(1)
            if strikes_max <= 0:
                raise
            return []

        self._vlm_strikes = 0
        self.log_both(
            "info",
            f"[PROFILE] Unified VLM scene analysis: {time.time() - t0:.3f}s; "
            f"{len(scene_objects)} object(s)",
        )
        if self._abort_if_moving("unified VLM scene analysis"):
            return []
        if not scene_objects:
            self.log_both("warn", "Unified VLM scene analysis returned no objects")
            return []
        return scene_objects

    def _segment_scene_objects(self, rgb_image, scene_objects):
        """Run local VitSAM on VLM boxes and retain their same-call attributes."""
        t0 = time.time()
        detections = []
        with torch.inference_mode():
            for scene_object in scene_objects:
                masks, _ = self.vitsam(rgb_image, scene_object.bbox)
                masks_np = np.asarray(masks)
                if masks_np.ndim == 2:
                    masks_np = masks_np[None, :, :]
                elif masks_np.ndim == 3 and masks_np.shape[-1] == 1:
                    masks_np = np.transpose(masks_np, (2, 0, 1))

                for mask in masks_np:
                    mask = (np.squeeze(np.asarray(mask)) > 0).astype(np.uint8)
                    detections.append(
                        Detection(
                            bbox=tuple(scene_object.bbox),
                            label=scene_object.label,
                            # This VLM schema provides no calibrated detector
                            # confidence. None is honest; 1.0 would be invented.
                            score=None,
                            mask=mask[..., None],
                            description=scene_object.description,
                            color=scene_object.color,
                            material=scene_object.material,
                            shape=scene_object.shape,
                        )
                    )
        self.log_both(
            "info",
            f"[PROFILE] VitSAM on unified VLM boxes: {time.time() - t0:.3f}s",
        )
        return detections

    def _publish_detection_pointclouds(self, detections, camera_data):
        self.pcl_object_id_counter = 0
        t0 = time.time()
        self.color_pcl(detections, camera_data)
        self.log_both("info", f"[PROFILE] PointCloud2 (color_pcl): {time.time() - t0:.3f}s")

    def _add_pca_orientation(self, detections, bboxes_3d, depth, camera_info, transform):
        """Add the optional PCA keys to each valid bbox dict, in place.
        Reads the points the geometry stage kept on each detection (2026-09-06); the
        re-lift it used to do per mask was pure duplication of the AABB pass (W8)."""
        if transform is None:
            return
        fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
        for det, bbox in zip(detections, bboxes_3d):
            if not bbox:
                continue
            try:
                # GA-315. A mask that runs off the image edge is a wedge of the object, and a
                # wedge's principal axis is its hypotenuse. No PCA for it: the AABB stays,
                # and the reason is written beside it (a key the msg builder ignores and the
                # detections.jsonl archive keeps, so the skip count is readable per bundle).
                if mask_touches_border(det.mask):
                    bbox["orientation_skipped"] = "mask_clipped"
                    continue
                # The geometry stage now keeps the map-frame points it measured the box
                # from on the detection (cv_utils points_out), and this reads them. The
                # re-lift below is IDENTICAL work (W8, same mask, same parameters, same
                # transform) and runs only for a caller that skipped _compute_3d_geometry.
                pts_map = getattr(det, "points_map", None)
                if pts_map is None:
                    pts = _filter_object_points(
                        det.mask[:, :, 0], depth, fx, fy, cx, cy,
                        # W8. The SAME parameters `mask_list_to_centroid_and_bbox` (the AABB
                        # pass) uses. The previous 2k-point remove_outliers=False subsample
                        # did NOT "already have validated points" -- it re-lifted a FRESH
                        # subsample with no SOR, so a mask bleeding onto the wall/floor kept
                        # its far points in `oriented_extents` while the AABB dropped them:
                        # two extents for one object, and the size gate PREFERS the oriented
                        # one, so the bleed could flip admit/hold/decline. With the same
                        # arguments the subsample is deterministic, so this is literally the
                        # same kept point set the AABB was built from.
                        max_points_per_obj=20000, remove_outliers=True,
                        sor_k=30, sor_std=1.5,
                    )
                    if pts is None:
                        continue
                    pts_map = _apply_transform(pts, transform)
                obb = pca_oriented_box(pts_map)
                if obb:
                    bbox.update(obb)
            except Exception as exc:
                self.log_both("warn", f"PCA orientation failed for {det.instance_label}: {exc}")

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
    assert np.allclose(box["oriented_extents"], [1.755, 0.45, 0.40], atol=0.02), box
    assert np.allclose(box["oriented_center"], [3.0, 4.0, 0.45], atol=0.1), box
    theta = np.linspace(0, 2 * np.pi, 500)
    circle = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    assert pca_oriented_box(circle) is None          # isotropic -> keep AABB
    assert pca_oriented_box(pts[:5]) is None         # too few points
    # GA-315: a mask running off the bottom edge is clipped; an interior one is not.
    m = np.zeros((960, 1280, 1), dtype=np.uint8)
    m[300:600, 400:900] = 1
    assert not mask_touches_border(m)
    m[958:, 400:900] = 1
    assert mask_touches_border(m)
    print("detection_pipeline PCA self-check OK")
