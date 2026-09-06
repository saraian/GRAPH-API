import logging
import os
import sys
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
    # ponytail: 1/99 percentile trim. Since W8 these points ARE the AABB pass's SOR'd
    # set (same _filter_object_points arguments), so the claim below finally holds;
    # the trim stays as a residual-straggler guard, not as the only one.
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

        t0 = time.time()
        labels = self._extract_detection_labels(camera_data["rgb"])
        t_vlm = time.time() - t0
        if not labels:
            return []

        # None on the local path, and None is the honest value there: a local backend makes
        # no request, so there is no wire time to report. 0.0 would read as "measured zero".
        client_timings = None
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
            client_timings = cloud_timings.get("client")
            # GA-14: these defaulted to 40% of the wall clock, a constant 10 ms, and 50%
            # of the wall clock — three invented numbers written into the same record as
            # real measurements, where the fractions even sum to 0.9. A missing stage
            # timing is a broken contract with the perception service, not a value to
            # guess, and a measurement path is where a silent estimate does most damage.
            #
            # SETTLED by the raise below, run 15, first cycle: the server emits
            # `detector`, `sam2` and `total`. The two checkouts disagreed on these names
            # and nothing on disk said which was right, so this read THIS tree's names
            # and put `sorted(cloud_timings)` in the error. The first mismatch then named
            # the truth on its first occurrence rather than being papered over by an
            # estimate — which is what the message is for, and it is the only reason the
            # question is now answered instead of guessed.
            # This return used to sit AFTER the stage-timing check below, and that order
            # killed run 19 on its fourth cycle. When the detector finds nothing the server
            # returns early and SAM never runs, so there IS no `sam2` stage to report — the
            # check demanded a measurement of work that was not done. Zero detections is a
            # result, not a contract violation, and the run must not die on one.
            #
            # The reported keys go in the log line because they are the discriminant. Run 19
            # reported ['total', 'yolo_world'] here while its three earlier cycles reported
            # detector/sam2 on the non-empty path — the deployed Modal build names the empty
            # path's detector timing `yolo_world`, which `modal_perception.py:188` renames to
            # `detector` ("same key as the non-empty path") in a build that is not deployed.
            # Reading no key on this path makes the client correct under BOTH builds, so no
            # redeploy is needed; without the log line the next skew is invisible again.
            #
            # NO stage timing is recorded for this cycle. GA-14's rule is unchanged: a value
            # that is not a measurement must not sit where measurements live, and that holds
            # for a cycle with nothing to measure as much as for one with a missing number.
            if len(detections) == 0:
                self.log_both("info", "Cloud perception backend found no objects "
                                      f"(reported timing keys: {sorted(cloud_timings)})")
                return []
            # Detections WITHOUT stage timings is the genuine contract violation and still
            # raises: the server did the work and did not say what it cost.
            missing = [k for k in ("detector", "sam2") if k not in cloud_timings]
            if missing:
                raise RuntimeError(
                    f"perception backend returned no timing for {missing}; refusing to "
                    f"estimate it from the wall clock "
                    f"(reported keys: {sorted(cloud_timings)})"
                )
            t_owlv2 = cloud_timings["detector"] / 1000.0   # output field stays owlv2_ms
            # NMS does NOT raise: on this backend it runs inside the detector and is never
            # reported separately, so 0.0 is the correct value. But 0.0 alone means BOTH
            # "measured as zero" and "not reported", and a reader cannot tell them apart —
            # so which one it is is recorded as its own key. Additive, because three
            # consumers read this record by key.
            nms_reported = "nms" in cloud_timings
            t_nms = cloud_timings.get("nms", 0.0) / 1000.0
            t_sam = cloud_timings["sam2"] / 1000.0         # output field stays sam_ms
            # The unattributed majority of a cloud cycle lives here, and it was invisible
            # while the three stage timings above were invented: those are the SERVER's
            # numbers, t_cloud is the wall clock, and nobody recorded the difference. On
            # the 26 Aug run the stages summed to 328 ms of a 1,288 ms median cycle; the
            # missing 857 ms was never a slow function, it was this subtraction. Keep
            # both — the server times say what compute cost, the remainder says what the
            # round trip cost, and only the second responds to moving the backend.
            t_backend_overhead = max(0.0, t_cloud - (t_owlv2 + t_nms + t_sam))
            # The server also reports `total`, which splits that remainder in two: what
            # the server spent outside the three stages, and what the wire cost. Ours
            # minus theirs is network and serialisation; only that half responds to
            # moving the backend. Recorded as None rather than 0.0 when the server does
            # not report `total`, with a flag beside it, because a 0.0 there would mean
            # both "no wire time" and "not reported" — the ambiguity this file just
            # removed from nms_ms.
            server_total = cloud_timings.get("total")
            server_total_reported = server_total is not None
            t_wire = max(0.0, t_cloud - server_total / 1000.0) if server_total_reported else None
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
            nms_reported = True        # measured directly on this path
            t_backend_overhead = 0.0   # no round trip on the local path: measured, not assumed
            t_wire = 0.0               # likewise: no wire
            server_total_reported = False
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
            for target_path in ("/tmp/perception_latencies.json", "/ws/output/perception_latencies.json"):
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
            # GA-53: a config seam stood here. With `vlm.fallback_labels` set, an
            # unreachable VLM was replaced by a static open-vocabulary list and the
            # cycle continued — so every downstream detection, association and verdict
            # came from labels no model produced, and the bundle recorded a completed
            # run. Working rule 14: a handler that SUBSTITUTES is a mute. That fix made
            # this branch re-raise unconditionally.
            #
            # GA-288 (owner ruling 2026-09-03): re-raising unconditionally was the OTHER
            # extreme. Run 20260903_135823 died at 18 cycles on a sub-second DNS blip --
            # the exception propagated through the timer callback into executor.spin()
            # and the node exited 1. A transient fault ended a 55-minute run.
            #
            # So this now mirrors object_manager_6's input-silence watchdog, which the
            # owner approved for the same shape: a failed cycle is SKIPPED, LOUDLY --
            # logged at ERROR, counted, and recorded in the bundle's vlm status -- and
            # the next cycle retries. `perception.vlm_strikes_max` consecutive failures
            # end the run the same way om6 does. Nothing substitutes for the VLM and
            # nothing is muted; what changed is that one failure is no longer fatal.
            self._vlm_strikes = getattr(self, "_vlm_strikes", 0) + 1
            strikes_max = int(CFG.get("perception", {}).get("vlm_strikes_max", 3))
            self._vlm_status = {
                "status": "unreachable",
                "model": CFG.get("vlm", {}).get("model", "unknown"),
                "error": str(exc)[:300],
                "consecutive_failures": self._vlm_strikes,
                "strikes_max": strikes_max,
            }
            self.log_both("error", f"[VLM] label call FAILED ({type(exc).__name__}: "
                                   f"{str(exc)[:160]}); cycle skipped; strike "
                                   f"{self._vlm_strikes}/{strikes_max}")
            if strikes_max > 0 and self._vlm_strikes >= strikes_max:
                self.log_both("error", f"[VLM] ENDING THE RUN: the VLM label call failed on "
                                       f"{self._vlm_strikes} consecutive cycles. The VLM is "
                                       f"gone, not blinking, and this node cannot make "
                                       f"progress without it.")
                for h in list(logging.getLogger().handlers):
                    try:
                        h.flush()
                    except Exception:
                        pass
                sys.stdout.flush(); sys.stderr.flush()
                # os._exit, as in om6: this runs on an executor thread, where SystemExit
                # unwinds that thread only and leaves the process spinning.
                os._exit(1)
            if strikes_max <= 0:
                raise          # the guard is disabled: crash-on-first-failure, as before
            return []
        self._vlm_strikes = 0
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
        # H10. The LOCAL backend was detecting at predict()'s hardcoded 0.35 while the
        # cloud path forwards `perception.score_threshold` (0.15) -- the same run config
        # meant different detections on different backends. The config value is passed
        # here now, same read as the cloud path.
        with torch.inference_mode():
            bboxs, detected_labels, scores = self.detector.predict(
                rgb_image, box_threshold=CFG.get("perception", {}).get("score_threshold", 0.15))
        self.log_both("info", f"[PROFILE] OWLv2 predict: {time.time() - t0:.3f}s")

        if self._abort_if_moving("OWLv2 detection"):
            return [], [], []
        return bboxs, detected_labels, scores

    def _apply_detection_nms(self, bboxs, labels, scores):
        t0 = time.time()
        # H11. The local path hardcoded IoU 0.5 while the cloud backend reads
        # `perception.nms_threshold` -- same config, different suppression per backend.
        # Same read as the cloud path now.
        bboxs, labels, scores = apply_nms(
            bboxs, labels, scores,
            iou_threshold=CFG.get("perception", {}).get("nms_threshold", 0.50))
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
        Reads the points the geometry stage kept on each detection (2026-09-06); the
        re-lift it used to do per mask was pure duplication of the AABB pass (W8)."""
        if transform is None:
            return
        fx, fy, cx, cy = camera_info.k[0], camera_info.k[4], camera_info.k[2], camera_info.k[5]
        for det, bbox in zip(detections, bboxes_3d):
            if not bbox:
                continue
            try:
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
