#!/usr/bin/env python3
"""Write the provenance sidecar of a published map, measured at publish time. GA-295(c).

Called by live_run.sh's publish step:
    stamp_map_provenance.py <published.db> <scene> <source_run> <integrity_marker> [params_sha]

WHY THIS EXISTS. mp3d_17DRP was published with NO provenance file at all, so when GA-295 found
the hm3d_00861 library had drifted 24 MB, the mp3d map's drift could not be checked, only
guessed (register GA-295, WIDENS). The owner ruled 4 Sep: publish COPY, not hard-link, and a
provenance file must cover mp3d too. hm3d_00861's sidecar was written BY HAND after the fact;
this makes provenance a publish-time measurement so every published map carries it from birth.

WHAT IT RECORDS, AND WHAT IT DOES NOT. `bytes` and `sha256_db` are measured on the PUBLISHED
copy, after `cp`, and they are its drift check — the hm3d drift (24 MB against provenance) was
found by exactly that comparison. `integrity_check` and `nodes` come from the BUNDLE's own gate
marker (`rtabmap.db.INTEGRITY_OK`, the publish precondition), NOT from a check of the
published copy: that check runs only when the container is gone (GA-104's lesson — never take
a SQLite lock against a live writer), and until someone runs PRAGMA integrity_check on the
PUBLISHED file the copy is unverified. The sidecar says so in `verified`.

The provenance is OVERWRITTEN on republish, exactly as the db itself is — a sidecar that
described a previous copy while the file beside it moved would be worse than none.
"""
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys

MARKER_PAT = re.compile(r"integrity=(\S+)\s+nodes=(\d+)")


def sha256_of(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) < 5:
        print(f"usage: {pathlib.Path(sys.argv[0]).name} <published.db> <scene> <source_run>"
              " <integrity_marker> [params_sha]", file=sys.stderr)
        return 2
    db = pathlib.Path(sys.argv[1])
    scene, source_run = sys.argv[2], sys.argv[3]
    marker_path = pathlib.Path(sys.argv[4])
    params_file = pathlib.Path(sys.argv[5]) if len(sys.argv) > 5 else None

    if not db.is_file():
        print(f"!! provenance NOT written: {db} does not exist", file=sys.stderr)
        return 1
    marker = marker_path.read_text(errors="replace") if marker_path.is_file() else ""
    m = MARKER_PAT.search(marker)
    if not m:
        # The publish precondition is the marker (checked by the caller before the copy); a
        # malformed marker here means the db was copied without a certified integrity check.
        # Say so and fail loud — an uncertified provenance file is worse than a failed publish.
        print(f"!! provenance NOT written: malformed integrity marker ({marker_path}: {marker!r})",
              file=sys.stderr)
        return 1
    integrity, nodes = m.group(1), int(m.group(2))

    # The container path is what a localization run will see; derived from the library layout,
    # not hardcoded, so a future floor or scene is covered the same way.
    container_path = "/found/" + str(db).split("/maps/", 1)[-1]

    params_sha = params_file.read_text(errors="replace").strip() if (
        params_file and params_file.is_file()) else ""
    # GA-359 / owner 2026-09-07 ("rebuild the map at the correct resolution"). The camera model a map
    # was BUILT with decides whether a run can localise against it, and nothing recorded it: the
    # 31 Aug map was found to be 640x480 / fx 320 only by parsing its Data.calibration blob while
    # runs streamed 1280x960 / fx 640. Two instruments, both stamped: the source run's own
    # calibration.json (beside the integrity marker) and the width/height the db itself stores.
    camera = None
    calib = marker_path.parent / "calibration.json"
    if calib.is_file():
        try:
            c = json.loads(calib.read_text())
            camera = {"resolution": c.get("resolution"), "intrinsics": c.get("intrinsics"),
                      "hfov_deg": c.get("hfov_deg"), "source": str(calib)}
        except (OSError, ValueError) as exc:
            camera = {"error": f"calibration.json unreadable: {exc}"}
    db_camera = None
    try:
        import sqlite3
        import struct
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        blob = con.execute("select calibration from Data limit 1").fetchone()
        con.close()
        if blob and blob[0]:
            ints = struct.unpack("<12i", bytes(blob[0])[:48])
            db_camera = {"width": ints[4], "height": ints[5],
                         "note": "Data.calibration header ints [4],[5] of the first node (rtabmap CameraModel serialisation)"}
    except Exception as exc:  # a stamp must not fail the publish over a blob it cannot parse; the null says so (rule 5)
        db_camera = {"error": f"{type(exc).__name__}: {exc}"}
    sidecar = {
        "scene": scene,
        "source_run": source_run,
        "published_at": datetime.datetime.now().astimezone().isoformat(),
        "published_by": "live_run.sh publish path (stamp_map_provenance.py)",
        "integrity_check": integrity,
        "nodes": nodes,
        "bytes": db.stat().st_size,
        "sha256_db": sha256_of(db),
        "params_sha": params_sha or None,
        "params_sha_note": (None if params_sha else
                            "NO params-sha beside this map — the localization gate REFUSES it "
                            "(see hm3d_00861's rtabmap_runA_insurance_1755.db precedent)."),
        "verified": ("integrity and node count are the SOURCE BUNDLE's gate marker, the publish "
                     "precondition — NOT a check of this copy. This copy is UNVERIFIED until "
                     "PRAGMA integrity_check is run on the PUBLISHED file after the container "
                     "is gone; when that is done, record it here."),
        "drift_check": "bytes and sha256_db are of the PUBLISHED copy as measured at publish "
                       "time. Compare against the file as found — a mismatch is drift (GA-295).",
        "container_path": container_path,
        "camera": camera,
        "camera_db": db_camera,
        "pose_source": os.environ.get("FEED_POSE_SOURCE") or None,
        "pose_source_note": ("the odometry the map was BUILT with: simulator = Habitat's true pose as /odom "
                             "(the 31 Aug map was built the same way, undeclared). Declared, not hidden (GA-359)."),
        "camera_note": ("camera = the source run's calibration.json (what the feed rendered); camera_db = "
                        "the model stored in the db's first node. A localisation run at another resolution "
                        "matches features but verifies few closures (measured 2026-09-07: 640x480 map, 1280x960 "
                        "runs, 1-17 accepted vs 1-78 rejected per run)."),
    }
    out = db.parent / (db.name + ".provenance.json")
    out.write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"    provenance stamped: {out.name} ({sidecar['bytes']} bytes, "
          f"sha256 {sidecar['sha256_db'][:16]}…)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
