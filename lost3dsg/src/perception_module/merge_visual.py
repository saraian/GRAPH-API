"""Review ambiguous merge candidates using the archived detector boxes and RGB frames."""
import base64
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
from time import monotonic

import cv2
import numpy as np


VERDICTS = {"same_whole_object", "part_whole", "different_objects", "uncertain"}
PROMPT = """These two views come from the same mapped environment. Compare only the objects
outlined in green. Decide whether the outlines refer to the same physical instance at
the same level of detail. Use visible markings, shape, boundaries and nearby landmarks.
Matching category or appearance alone is insufficient. A component and its containing
object are part_whole. If identity cannot be established, choose uncertain.
Reply with exactly one token: same_whole_object, part_whole, different_objects, uncertain."""


def best_view(obj):
    best = None
    largest = 0
    for obs in getattr(obj, "covisibility_observations", ()) or getattr(obj, "observations", ()) or ():
        try:
            _, sec, nsec = obs.capture_id.rsplit(":", 2)
            frame = f"{int(sec)}_{int(nsec):09d}"
            coords = tuple(float(v) for v in obs.bbox_2d)
            x1, y1, x2, y2 = coords
        except (AttributeError, ValueError, TypeError):
            continue
        if np.isfinite(coords).all() and x2 > x1 and y2 > y1:
            area = (x2 - x1) * (y2 - y1)
            if area > largest:
                largest = area
                best = (frame, coords)
    return best


def panel(frames, view):
    frame, coords = view
    image = cv2.imread(str(frames / (frame + ".jpg")))
    if image is None:
        raise ValueError("missing_frame")
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.clip(np.rint(coords), 0, [w, h, w, h]).astype(int)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("empty_box")
    pad = max(16, max(x2 - x1, y2 - y1) // 2)
    cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
    crop = image[max(0, y1-pad):y2+pad, max(0, x1-pad):x2+pad]
    scale = min(480 / crop.shape[0], 480 / crop.shape[1])
    crop = cv2.resize(crop, (max(1, round(crop.shape[1]*scale)),
                             max(1, round(crop.shape[0]*scale))))
    tile = np.full((480, 480, 3), 40, dtype=np.uint8)
    y, x = (480-crop.shape[0])//2, (480-crop.shape[1])//2
    tile[y:y+crop.shape[0], x:x+crop.shape[1]] = crop
    return tile


class VisualMergeReview:
    def __init__(self, archive, call, logger):
        self.frames = Path(archive) / "frames"
        self.call = call
        self.logger = logger
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="merge_review")
        self.jobs = {}
        self.results = OrderedDict()

    def _review(self, views):
        pair = np.hstack([panel(self.frames, view) for view in views])
        ok, jpg = cv2.imencode(".jpg", pair)
        if not ok:
            raise ValueError("jpeg_encoding")
        answer = self.call(PROMPT, base64.b64encode(jpg).decode("ascii"),
                           image_mime_type="image/jpeg", timeout=15, request_kind="merge_review")
        verdict = str(answer).strip().casefold()
        return verdict if verdict in VERDICTS else "uncertain"

    def check(self, a, b):
        views = (best_view(a), best_view(b))
        if None in views:
            return "missing_view", views
        key = tuple(sorted(views))
        # Collect at most one bounded queue of completed calls per check. The worker
        # only reads immutable frame references and never touches the world model.
        for completed, future in list(self.jobs.items()):
            if future.done():
                try:
                    self.results[completed] = (future.result(), float("inf"))
                except Exception as exc:
                    self.logger.warning(f"Visual merge review failed: {exc}")
                    self.results[completed] = ("unavailable", monotonic() + 30)
                del self.jobs[completed]
                if len(self.results) > 2048:
                    self.results.popitem(last=False)
        if key in self.results:
            verdict, expiry = self.results[key]
            if monotonic() < expiry:
                self.results.move_to_end(key)
                return verdict, views
            del self.results[key]
        if key not in self.jobs and len(self.jobs) < 16:
            self.jobs[key] = self.worker.submit(self._review, key)
        return "pending", views

    def close(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
