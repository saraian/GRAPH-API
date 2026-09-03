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
                         bbox_3d=None, crop_meta=None, stamp=None, room_id=None,
                         semantic_frame=None):
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
            # GROUND TRUTH, and named so nothing can mistake it for a perception output.
            # `instance_label` above is the PERCEPTION label ("mirror#2"); this is habitat's
            # own id for the object the mask actually covers. Both are recorded because the
            # whole point of the analysis is to compare them.
            gt_id, gt_note = dominant_gt_instance(getattr(det, "mask", None), semantic_frame)
            row["habitat_gt_instance_id"] = gt_id
            if gt_id is None:
                # Why it is absent, so "no GT" can be told from "GT said nothing here".
                row["habitat_gt_absent_reason"] = gt_note
            else:
                row["habitat_gt_mask_coverage"] = round(float(gt_note), 4)
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


def dominant_gt_instance(mask, semantic_frame, min_coverage=0.5):
    """-> (habitat instance id, coverage) for a detection's mask, or (None, reason).

    GROUND TRUTH, VALIDATION ONLY. Nothing on the runtime path calls this and nothing on the
    runtime path may: the pipeline must never see an instance id it did not infer. It is
    computed here, at archive time, so a labelled pair exists for the analyses that need one
    -- box error against GT, same-instance vs different-instance re-ID separation, and
    GT-labelled association pairs -- none of which the runtime is allowed to know about.

    The mechanism is an intersection: habitat's semantic sensor renders its own instance id
    per PIXEL, so the id under a detection's mask is the object the detector actually hit.
    The DOMINANT id is taken, with its coverage, because a mask spanning two objects is a
    real and interesting case rather than an error -- and a mask that is only half on its
    object should not be recorded as a clean label.

    RETURNS A REASON RATHER THAN A GUESS when it cannot answer: no semantic frame, a shape
    mismatch, an empty mask, or a dominant id below `min_coverage`. A wrong GT label is worse
    than a missing one, because everything downstream treats GT as the thing being measured
    against.
    """
    if semantic_frame is None:
        return None, "no semantic frame"
    m = mask
    if m is None:
        return None, "no mask"
    import numpy as np
    m = np.asarray(m)
    while m.ndim > 2:
        m = m[..., 0]
    m = m.astype(bool)
    try:
        sem = np.asarray(semantic_frame)
    except Exception as exc:
        # GA-172b: a GT problem must never cost the DETECTION row. The row's purpose is the
        # detection; ground truth is an addition to it. Before this, a semantic frame that
        # would not convert raised inside record_detection's broad handler and the whole row
        # was dropped -- the archive losing a real measurement because an optional
        # validation field was malformed.
        return None, f"semantic frame not array-like ({type(exc).__name__})"
    if sem.ndim < 2:
        return None, f"semantic frame has {sem.ndim} dimension(s)"
    if sem.shape[:2] != m.shape[:2]:
        return None, f"shape mismatch mask{m.shape[:2]} semantic{sem.shape[:2]}"
    if not m.any():
        return None, "empty mask"
    ids = sem[m]
    vals, counts = np.unique(ids, return_counts=True)
    k = int(np.argmax(counts))
    coverage = float(counts[k]) / float(ids.size)
    if coverage < float(min_coverage):
        return None, f"dominant id covers only {coverage:.2f} of the mask"
    return int(vals[k]), coverage


def resolve_archive_dir(cfg, project_root=None):
    """Where the per-detection archive is written. REFUSES rather than guessing.

    GA-166. This used to be `CFG["archive"]["dir"] or dirname(CFG["paths"]["operations_log"])`,
    written as a convenience so the archive would land beside the run's other output. In run
    042828 `operations_log` was `/tmp/operations.txt`, so the archive wrote 18 rows and 3
    frames into `/tmp` -- outside the bundle, collected by nobody, and destroyed with the
    container. It was recovered only because someone copied it out of a live container.

    I DERIVED A DATA PATH FROM A LOG PATH. The directory that happens to hold a log is not
    "where the run's data goes": they are the same directory in the defaults and different
    directories in every real run, which is exactly the case the default was written for and
    exactly the case it got wrong.

    So the order is now explicit and the fallback is a REFUSAL:
      1. `archive.dir`, if set -- the caller said where.
      2. `paths.output_dir`, if set -- the run's output directory, which is what a bundle is.
      3. Otherwise raise. A per-detection archive written somewhere nobody collects is worse
         than none, because it looks like it worked -- and it fails at node startup, loudly,
         instead of at collection time, silently.

    `operations_log`'s directory is deliberately NOT consulted any more.
    """
    d = (cfg.get("archive", {}) or {}).get("dir") or ""
    if d:
        return d
    # GA-166b. ASK THE STACK, do not guess. `live_stack_container.sh:49` exports
    # GRAPH_API_OUTPUT_DIR to the directory the bundle is bind-mounted at (/ws/output in
    # this layout), and it is where hook_decisions.jsonl, room.json and knowledge_graph.ttl
    # all land. Reading it makes the archive correct BY CONSTRUCTION rather than by a
    # constant that happens to match a container layout.
    #
    # This is the second wrong answer here and the two failed the same way. The first
    # derived the path from operations_log's DIRECTORY -- wrong because a log's directory is
    # not where data goes. The second defaulted to /root/exchange/output -- wrong because it
    # is a plausible-looking constant this layout does not use. BOTH PRODUCED AN ARCHIVE THAT
    # IS WRITTEN, LOOKS WRITTEN, AND IS COLLECTED BY NOBODY. A guess that resembles the right
    # answer fails in exactly the same way as one that does not.
    d = os.environ.get("GRAPH_API_OUTPUT_DIR") or ""
    if d:
        return d
    d = (cfg.get("paths", {}) or {}).get("output_dir") or ""
    if d:
        return d
    raise ValueError(
        "per-detection archiving is enabled but no output directory is configured. Set "
        "archive.dir, or export GRAPH_API_OUTPUT_DIR, or set paths.output_dir. Refusing to "
        "guess: two previous defaults both wrote an entire run's archive somewhere nothing "
        "collected it -- operations_log's directory (/tmp) and a hardcoded "
        "/root/exchange/output that this container layout does not use.")


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

        # --- GA-166: the path must be EXPLICIT, and refuse rather than guess ------------
        assert resolve_archive_dir({"archive": {"dir": "/x"}}) == "/x"
        assert resolve_archive_dir({"paths": {"output_dir": "/bundle"}}) == "/bundle"
        # operations_log's directory is deliberately NOT consulted -- this is the exact
        # config that sent run 042828's archive to /tmp
        try:
            resolve_archive_dir({"paths": {"operations_log": "/tmp/operations.txt"}})
            raise AssertionError("must REFUSE, not fall back to the log's directory")
        except ValueError as e:
            assert "Refusing to guess" in str(e)
        print("  operations_log's dir is NOT used; no output dir -> ValueError, not /tmp")

        # GA-166b: the STACK's own value wins over any constant we might pick.
        os.environ["GRAPH_API_OUTPUT_DIR"] = "/ws/output"
        try:
            assert resolve_archive_dir({}) == "/ws/output"
            assert resolve_archive_dir({"paths": {"output_dir": "/root/exchange/output"}}) \
                == "/ws/output", "the env var must beat a plausible-looking constant"
            assert resolve_archive_dir({"archive": {"dir": "/explicit"}}) == "/explicit"
            print("  GRAPH_API_OUTPUT_DIR beats paths.output_dir; archive.dir still wins")
        finally:
            del os.environ["GRAPH_API_OUTPUT_DIR"]

        # --- GA-167: the archive must be built ONCE per node -----------------------------
        import re
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "perception_2.py")).read().splitlines()
        cur, sites = None, []
        for i, line in enumerate(src, 1):
            m = re.match(r"    def (\w+)", line)
            if m:
                cur = m.group(1)
            if "DetectionArchive(" in line and "import" not in line:
                sites.append((cur, i))
        assert len(sites) == 1, f"DetectionArchive built at {len(sites)} sites: {sites}"
        assert sites[0][0] == "__init__", (
            f"DetectionArchive is built in {sites[0][0]}() at line {sites[0][1]}, not "
            f"__init__ -- that is GA-167: once per crop instead of once per node, which "
            f"lost cycle 1 and silently disabled the frame dedupe")
        print(f"  built exactly once, in {sites[0][0]}() at perception_2.py:{sites[0][1]} "
              f"— asserted by FUNCTION NAME, not by anchor uniqueness")

        # --- the GT join: dominant instance under the mask, and REFUSAL when unsure ------
        import numpy as _np
        sem = _np.zeros((48, 64), dtype=_np.int32)
        sem[10:20, 10:30] = 77            # the object the detector hit
        sem[10:20, 30:40] = 99            # a neighbour it partly overlaps
        clean = _np.zeros((48, 64), bool); clean[10:20, 10:30] = True
        gid, cov = dominant_gt_instance(clean, sem)
        assert gid == 77 and cov == 1.0, (gid, cov)
        print(f"  GT join: clean mask -> instance {gid}, coverage {cov:.2f}")

        # a mask spanning two objects: dominant id wins and the coverage says it is impure
        split = _np.zeros((48, 64), bool); split[10:20, 10:36] = True
        gid2, cov2 = dominant_gt_instance(split, sem)
        assert gid2 == 77 and 0.5 < cov2 < 1.0, (gid2, cov2)
        print(f"  mask spanning two objects -> instance {gid2}, coverage {cov2:.2f} "
              f"(recorded, not hidden)")

        # below the coverage floor it REFUSES rather than labelling
        mostly_other = _np.zeros((48, 64), bool); mostly_other[10:20, 26:40] = True
        gid3, why = dominant_gt_instance(mostly_other, sem, min_coverage=0.8)
        assert gid3 is None and "covers only" in why, (gid3, why)
        print(f"  below the coverage floor -> None ({why})")

        # and every other way it can fail returns a REASON, never a guess
        assert dominant_gt_instance(clean, None)[1] == "no semantic frame"
        assert dominant_gt_instance(clean, _np.zeros((10, 10), _np.int32))[0] is None
        assert dominant_gt_instance(_np.zeros((48, 64), bool), sem)[1] == "empty mask"
        print("  no frame / shape mismatch / empty mask -> a reason, never a guess")

        # the row carries GT under an unmistakable name, beside the perception label
        a2 = DetectionArchive(tmp, enabled=True)
        a2.record_detection("f9", D("mirror", (10, 10, 30, 20), clean, "mirror#2"),
                            semantic_frame=sem)
        r = [json.loads(x) for x in open(os.path.join(tmp, "detections.jsonl"))][-1]
        assert r["instance_label"] == "mirror#2", "perception label must be untouched"
        assert r["habitat_gt_instance_id"] == 77 and r["habitat_gt_mask_coverage"] == 1.0
        a2.record_detection("f9", D("mirror", (10, 10, 30, 20), clean, "mirror#3"))
        r2 = [json.loads(x) for x in open(os.path.join(tmp, "detections.jsonl"))][-1]
        assert r2["habitat_gt_instance_id"] is None
        assert r2["habitat_gt_absent_reason"] == "no semantic frame"
        print(f"  row: instance_label={r['instance_label']!r} (perception) beside "
              f"habitat_gt_instance_id={r['habitat_gt_instance_id']} (truth); absent rows "
              f"say WHY")

        # --- GA-172: crop_meta must REACH the row, and a GT failure must cost GT only ----
        a3 = DetectionArchive(tmp, enabled=True)
        meta = {"status": "ok", "construction": "contour", "upsampled": True}
        a3.record_detection("f10", D("ac", (10, 10, 30, 20), clean, "ac#1"),
                            crop_meta=meta, semantic_frame=sem)
        rr = [json.loads(x) for x in open(os.path.join(tmp, "detections.jsonl"))][-1]
        assert rr["crop_meta"] == meta, (
            "crop_meta is null on the row -- it is computed in prepare_crops, so the archive "
            "must run AFTER it; running before means the field does not exist yet and the "
            "four-way unanswerable/model_abstained status is unrecoverable")
        print(f"  crop_meta reaches the row: {rr['crop_meta']['construction']}, "
              f"status {rr['crop_meta']['status']}")

        a3.record_detection("f11", D("ac", (10, 10, 30, 20), clean, "ac#2"),
                            crop_meta=meta, semantic_frame=object())
        rr2 = [json.loads(x) for x in open(os.path.join(tmp, "detections.jsonl"))][-1]
        assert rr2["instance_label"] == "ac#2", "a GT failure must not cost the DETECTION row"
        assert rr2["crop_meta"] == meta and rr2["habitat_gt_instance_id"] is None
        print(f"  malformed GT -> row survives, reason {rr2['habitat_gt_absent_reason']!r} "
              f"(the row's purpose is the detection; GT is an addition to it)")

        print("\ndetection_archive self-check OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    demo()
