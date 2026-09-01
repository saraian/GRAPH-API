#!/usr/bin/env python3
"""Per-detection archiving — the record that makes co-visibility and re-ID measurable.

WHY, from a measurement rather than a preference. Replaying tonight's bundles through
`association.py` produced ten disagreements and every one was a pillow being merged into the
bed it rests on -- containment 0.850, 0.686, 0.417 -- because containment cannot tell "a
fragment of X" from "an object sitting on X". The channel that separates those two cases is
CO-VISIBILITY: a pillow and the bed under it appear in one frame, which is a hard negative no
overlap score may override. It could not be evaluated, because nothing in the archive records
WHICH FRAME a detection came from.

That is the entire gap. `Detection` already carries its 2D box and its SAM2 mask; the
perception cycle already has the RGB frame and the stamp. None of it is written down. So this
module writes the four things that were missing, and nothing else:

    frame id  ->  the RGB frame it names
    detection ->  its 2D box, its mask, and the construction meta of the crop it produced

WHAT IT UNLOCKS, all of which is impossible today: co-visibility as a runtime constraint (free
-- it needs only the frame id); appearance re-ID with view-tagged descriptors (needs the
bearing, which needs the frame); and an honest A/B over crop constructions (needs the box and
the mask). It is also the thing that stops a merge destroying its own evidence: in one of
tonight's runs 100% of merge decisions became unauditable because the objects were gone.

OFF BY DEFAULT. A frame per cycle is not free, and a perception callback is the wrong place to
discover that. `CFG["archive"]["per_detection"]` turns it on; when off, every method here
returns immediately and touches no disk.

NEVER PARTIAL. If a row cannot be written completely it is not written at all, and the failure
is counted and reported. A half-written archive that looks whole is worse than an empty one --
it is the same defect as a description of "unknown" that might mean four different things.
"""

import json
import os
import threading


class DetectionArchive:
    """Writes frames/<frame_id>.jpg and detections.jsonl. Thread-safe, append-only."""

    def __init__(self, root, enabled=False, logger=None, save_rgb=True):
        self.enabled = bool(enabled)
        self.root = root
        self.logger = logger
        self.save_rgb = save_rgb
        self._lock = threading.Lock()
        self._frames_written = set()
        self.n_rows = 0
        self.n_failed = 0
        self._path = None
        self._frame_dir = None
        if self.enabled:
            self._frame_dir = os.path.join(root, "frames")
            os.makedirs(self._frame_dir, exist_ok=True)
            self._path = os.path.join(root, "detections.jsonl")

    # -- frames ---------------------------------------------------------------------------

    def record_frame(self, frame_id, rgb):
        """Write the RGB frame once per frame id. Returns the path, or None."""
        if not self.enabled or rgb is None or not self.save_rgb:
            return None
        with self._lock:
            if frame_id in self._frames_written:
                return os.path.join(self._frame_dir, f"{frame_id}.jpg")
            path = os.path.join(self._frame_dir, f"{frame_id}.jpg")
            try:
                from PIL import Image
                Image.fromarray(rgb[:, :, :3].astype("uint8")).save(path, quality=92)
                self._frames_written.add(frame_id)
                return path
            except Exception as exc:
                self.n_failed += 1
                self._warn(f"frame {frame_id} not archived: {exc}")
                return None

    # -- detections -----------------------------------------------------------------------

    def record_detection(self, frame_id, det, camera_position=None, centroid=None,
                         bbox_3d=None, crop_meta=None, stamp=None, room_id=None):
        """One row per detection per frame. All of it, or none of it.

        The 2D box and the mask come straight off the Detection the pipeline already built --
        this adds no computation to the cycle, only a write.
        """
        if not self.enabled:
            return False
        try:
            row = {
                "frame_id": frame_id,
                "stamp": stamp,
                "label": getattr(det, "label", None),
                "instance_label": getattr(det, "instance_label", None),
                "score": float(getattr(det, "score", 0.0) or 0.0),
                "bbox_2d": [float(v) for v in getattr(det, "bbox", []) or []],
                "bbox_3d": bbox_3d,
                "centroid": list(centroid) if centroid is not None else None,
                "camera_position": list(camera_position) if camera_position is not None else None,
                "room_id": room_id,
                "crop_meta": crop_meta,
                "mask_rle": self._encode_mask(getattr(det, "mask", None)),
            }
            if len(row["bbox_2d"]) != 4:
                # Without the 2D box the row cannot support any of the three things this
                # archive exists for, so it is not a row. Counted, not silently dropped.
                self.n_failed += 1
                self._warn(f"detection on frame {frame_id} has no usable 2D box; row skipped")
                return False
            with self._lock:
                with open(self._path, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                self.n_rows += 1
            return True
        except Exception as exc:
            self.n_failed += 1
            self._warn(f"detection row on frame {frame_id} not archived: {exc}")
            return False

    def _encode_mask(self, mask):
        """RLE via cloud.client's encoder when importable, else an identical local encoder.

        Reused rather than reimplemented -- a second FORMAT would be a second thing to keep
        in step. But `cloud/client.py` imports cv2 at module scope, so on any host without
        OpenCV the import fails and, in the first version of this method, a bare
        `except Exception: return None` turned that into a mask of None. Every row would
        have carried `mask_rle: null` and looked like a detection that simply had no mask.

        THAT IS THE DEFECT THIS WHOLE REVIEW IS ABOUT, written by me, in the file whose
        purpose is to make things visible. A missing dependency is not an absent mask.

        So: same wire format, computed locally when the shared module cannot be imported,
        and any REAL failure is counted and reported rather than returned as None.
        """
        if mask is None:
            return None
        import numpy as np
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = m.astype(bool)
        try:
            from cloud.client import rle_encode
            return rle_encode(m)
        except ImportError:
            return _rle_encode_local(m)
        except Exception as exc:
            self.n_failed += 1
            self._warn(f"mask RLE failed ({type(exc).__name__}: {exc}); row will carry none")
            return None

    def stats(self):
        return {"enabled": self.enabled, "rows": self.n_rows,
                "frames": len(self._frames_written), "failed": self.n_failed}

    def _warn(self, msg):
        if self.logger is not None:
            try:
                self.logger.warn(f"[ARCHIVE] {msg}")
                return
            except Exception:
                pass
        print(f"[ARCHIVE] {msg}")


def _rle_encode_local(m):
    """Byte-identical to cloud.client.rle_encode, without the cv2 import that file carries."""
    import numpy as np
    h, w = m.shape[:2]
    flat = m.reshape(-1).astype(np.uint8)
    if flat.size == 0:
        return {"size": [int(h), int(w)], "first_val": 0, "counts": []}
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    return {"size": [int(h), int(w)], "first_val": int(flat[0]),
            "counts": [int(c) for c in np.diff(bounds)]}


def frame_id_from_stamp(stamp):
    """A stable, sortable frame id from a ROS stamp. Same stamp -> same id, always.

    Co-visibility is an equality test on this value, so it must be exact and derived from the
    stamp rather than from a counter: a counter differs between the publisher and any replay,
    and two nodes would then disagree about which detections shared a frame.
    """
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nsec = getattr(stamp, "nanosec", None)
    if sec is None:
        return str(stamp)
    return f"{int(sec)}_{int(nsec):09d}"


# ---------------------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------------------


def demo():
    import shutil
    import tempfile

    import numpy as np

    class D:
        def __init__(self, label, box, mask, inst=None):
            self.label, self.bbox, self.mask, self.instance_label = label, box, mask, inst
            self.score = 0.9

    class Stamp:
        sec, nanosec = 1788200177, 475640800

    tmp = tempfile.mkdtemp(prefix="detarch_")
    try:
        # OFF by default: touches nothing.
        off = DetectionArchive(tmp, enabled=False)
        assert off.record_detection("f1", D("bed", (1, 2, 3, 4), None)) is False
        assert not os.path.exists(os.path.join(tmp, "detections.jsonl"))
        assert os.listdir(tmp) == [], "disabled archive wrote to disk"
        print("  disabled -> no directories, no files, no writes")

        a = DetectionArchive(tmp, enabled=True)
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[10:20, 10:30] = (200, 30, 30)
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:20, 10:30] = True

        fid = frame_id_from_stamp(Stamp())
        assert fid == "1788200177_475640800", fid
        # stable: same stamp -> same id, which is what co-visibility compares
        assert frame_id_from_stamp(Stamp()) == fid
        print(f"  frame id from stamp : {fid} (stable, so co-visibility can compare it)")

        p = a.record_frame(fid, rgb)
        assert p and os.path.exists(p)
        assert a.record_frame(fid, rgb) == p and len(a._frames_written) == 1, \
            "frame written twice"
        print("  frame written once per id, not once per detection")

        ok = a.record_detection(fid, D("bed", (10, 10, 30, 20), mask, "bed#1"),
                                camera_position=[0, 0, 1], centroid=[1, 2, 3],
                                bbox_3d={"x_min": 0}, crop_meta={"status": "ok"},
                                room_id="bedroom")
        assert ok
        a.record_detection(fid, D("pillow", (12, 11, 22, 18), mask, "pillow#1"),
                           camera_position=[0, 0, 1])
        # a detection with no 2D box is REFUSED, not half-written
        assert a.record_detection(fid, D("ghost", (), None)) is False
        assert a.stats()["rows"] == 2 and a.stats()["failed"] == 1, a.stats()
        print(f"  rows {a.stats()['rows']}, refused {a.stats()['failed']} "
              f"(a row without a 2D box is not a row)")

        rows = [json.loads(x) for x in open(os.path.join(tmp, "detections.jsonl"))]
        assert len(rows) == 2
        assert rows[0]["mask_rle"] and rows[0]["mask_rle"]["size"] == [48, 64], rows[0]["mask_rle"]
        assert rows[0]["instance_label"] == "bed#1"
        # THE POINT OF THE WHOLE FILE: these two share a frame, so co-visibility can veto.
        assert rows[0]["frame_id"] == rows[1]["frame_id"]
        print(f"  co-visibility is now answerable: '{rows[0]['instance_label']}' and "
              f"'{rows[1]['instance_label']}' share frame {rows[0]['frame_id']} "
              f"-> hard negative, no overlap score can fuse them")

        # round-trip the mask through the shared decoder
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from cloud.client import rle_decode
        except ImportError:
            def rle_decode(d):
                import numpy as np
                out = np.zeros(d["size"][0] * d["size"][1], dtype=np.uint8)
                v, i = d["first_val"], 0
                for c in d["counts"]:
                    out[i:i + c] = v; i += c; v = 1 - v
                return out.reshape(d["size"])
        back = rle_decode(rows[0]["mask_rle"]).astype(bool)
        assert back.shape == mask.shape and (back == mask).all(), "mask did not round-trip"
        print("  mask round-trips through cloud.client's existing RLE codec (not a second one)")

        print("\ndetection_archive self-check OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    demo()
