#!/usr/bin/env python3
"""Class A pre-flight gate — wiring probes, run INSIDE the container immediately before a
measured run. Aborts loudly so a bad run is never produced, rather than detected afterwards.

    python3 preflight_gate.py --out /ws/output/preflight.json [--only a1,a2] [--allow-skip]

Exit 0 = every probe that ran passed. Exit 1 = at least one failed, or a probe could not run
and --allow-skip was not given.

FIRST PRINCIPLE, and every probe here is an instance of it:
    assert the IDENTITY of the component that answered, never that something answered.

Six defects on 26 Aug were all of the second shape. A config was named and never loaded. An
aligner was configured and never ran, so a baseline masqueraded as the real thing for weeks.
A policy was exported on the host and never reached the container. Each would have been caught
by asking "which one answered?" instead of "did something answer?".

ponytail: the probes deliberately do NOT try to repair anything. A gate that fixes what it
finds is a gate that hides what it found; the run is cheap to restart and the bundle is not.
"""
import argparse
import json
import os
import sys
import time

# Probes return (ok: bool|None, detail: dict). ok=None means "could not run" — distinct from
# False, because a probe that could not run has asserted NOTHING and must never read as a pass.
SKIPPED = None

# ext4 truncates mtime to whole seconds; see a5_bundle_clean.
FS_MTIME_TOL = 2.0

# a1's golden pair. The positive is on record from a live run on 26 Aug (couch -> Sofa, 0.903).
# The negative is the half that catches an aligner answering everything; it must be a string no
# ontology can plausibly contain.
GOLDEN_LABEL = "couch"
GOLDEN_EXPECT = "Sofa"
NONSENSE_LABEL = "zzqx_not_a_real_object_kind"


# ---------------------------------------------------------------------------------------
# Digests. ONE implementation, used by the probes here AND by live_run.sh via --print-*,
# so the launcher never restates the hashing. A check that restates its subject drifts from
# it in silence, which is how the tautological a2 survived.
#
# Deny-list, not allow-list: COLCON_IGNORE has no extension, so no extension list can catch
# it, and its presence decides whether the package builds. The principle for the exclusions
# is "exclude what the system writes; keep what a person wrote, however dead" — build/,
# install/, __pycache__/ and grafici_output/ are outputs; src/perception_module/old/ is not.
DENY_DIRS = {"build", "install", "__pycache__", ".git", "grafici_output", ".ruff_cache"}
DENY_EXTS = {".pyc", ".pyo", ".log", ".db"}


def _tree_files(root):
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in DENY_DIRS]
        for fn in files:
            if os.path.splitext(fn)[1] not in DENY_EXTS:
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def file_sha16(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def tree_sha(root):
    """Digest a tree by (relative name, content) pairs.

    Names are IN the digest: the old `find | xargs cat | sha256sum` concatenated bytes with
    no separator, so a rename was invisible and so was a new empty file.

    A root matching no files RAISES. The old form wrote its error to /dev/null and hashed
    nothing, yielding e3b0c44298fc1c14 — the sha256 of emptiness, a plausible sixteen-hex
    provenance stamp for a hash that covered zero files.
    """
    import hashlib
    files = _tree_files(root)
    if not files:
        raise ValueError(f"hash root matched no files: {root!r}")
    h = hashlib.sha256()
    for path in files:
        h.update(os.path.relpath(path, root).encode())
        h.update(b"\0")
        h.update(file_sha16(path).encode())
        h.update(b"\n")
    return h.hexdigest()[:16], len(files)


def merged_cfg_sha(cfg):
    """Hash the MERGED configuration as loaded. config.CFG is _merge(_DEFAULTS, yaml) and
    takes no environment input, so it reproduces on the host from the same two files — which
    is what makes the comparison a statement about the container boundary."""
    import hashlib
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def a1_aligner_identity():
    """WHICH aligner answered — not that one exists.

    found.kg_align.make_aligner returns KGAlignerAdapter for FOUND_ALIGNER=kg and a plain
    Aligner for exemplar/lexical. Constructing it is not enough: a KGAlignerAdapter that
    raises on first use still reports as 'kg'. So the probe also aligns a golden label whose
    correct answer is on record from a live run — couch -> Sofa at 0.903.
    """
    from found.dims import DimensionDB
    from found.kg_align import make_aligner

    want = os.environ.get("FOUND_ALIGNER", "kg")
    aligner = make_aligner(DimensionDB().types())
    got = type(aligner).__name__
    expect_cls = "KGAlignerAdapter" if want == "kg" else "Aligner"
    if got != expect_cls:
        return False, {"requested": want, "constructed": got, "expected": expect_cls}

    def _name(res):
        if not res:
            return None
        return getattr(res, "name", None) or (res.get("name") if isinstance(res, dict) else None)

    # POSITIVE: a label whose correct answer is on record from a live run on 26 Aug.
    # The name is compared; the score is recorded and NOT asserted at 0.903, because a gate
    # that demands an exact float is brittle, and a brittle gate gets switched off.
    golden = aligner.align(GOLDEN_LABEL)
    got_name = _name(golden)

    # NEGATIVE: a label that is in no ontology. This is the half the probe was missing —
    # without it an aligner that returns a non-empty name for EVERYTHING passes, which is
    # exactly the fallback behaviour the probe exists to detect.
    nonsense = aligner.align(NONSENSE_LABEL)
    nonsense_name = _name(nonsense)

    detail = {
        "requested": want, "constructed": got,
        "golden": {"label": GOLDEN_LABEL, "expected": GOLDEN_EXPECT,
                   "aligned_to": got_name,
                   "score": getattr(golden, "score", None) if golden else None,
                   "score_note": "recorded 0.903 live 26 Aug; recorded here, not asserted"},
        "negative": {"label": NONSENSE_LABEL, "aligned_to": nonsense_name,
                     "expected": None},
    }
    if got_name != GOLDEN_EXPECT:
        return False, dict(detail, why=(
            f"{GOLDEN_LABEL!r} aligned to {got_name!r}, expected {GOLDEN_EXPECT!r}. "
            "CHECK THE ENVIRONMENT FIRST — is /kb mounted and first on PYTHONPATH? "
            "Updating the golden when the real cause was the mount calibrates this probe "
            "against the failure it exists to detect, and it then passes forever."))
    if nonsense_name is not None:
        return False, dict(detail, why=(
            f"a label in no ontology aligned to {nonsense_name!r}. An aligner that answers "
            "everything is the fallback this probe exists to catch."))
    return True, detail


def a2_config_identity(expect_name=None, expect_sha=None, expect_merged=None):
    """Hash the config the process LOADED, not the path it was handed.

    config.CFG is the merged dict after yaml + defaults. Hashing the file on the host says
    nothing about what the process resolved: the CFG_NAME bug was exactly a correct path
    recorded beside a different config in use.
    """
    sys.path.insert(0, "/ws/install/lost3dsg/lib/lost3dsg")
    import config as cfgmod

    env_path = os.environ.get("GRAPH_API_CONFIG", "<unset>")
    # CFG_PATH is the file config.py ACTUALLY read, or None when the defaults are in force.
    # The environment variable only says what it was asked to read. A run on the defaults is
    # a run with hooks.filter empty and FOUND out of the loop, and it is silent.
    loaded_path = getattr(cfgmod, "CFG_PATH", None)
    merged = merged_cfg_sha(cfgmod.CFG)
    file_sha = file_sha16(loaded_path) if loaded_path else None
    detail = {"env_path": env_path, "loaded_path": loaded_path,
              "merged_cfg_sha256_16": merged, "config_file_sha256_16": file_sha,
              "perception_backend": (cfgmod.CFG.get("perception") or {}).get("backend"),
              "hooks_filter": (cfgmod.CFG.get("hooks") or {}).get("filter")}

    if loaded_path is None:
        return False, dict(detail, why=(
            f"no config file was loaded — config.py fell back to its defaults. "
            f"GRAPH_API_CONFIG={env_path!r} does not exist. hooks.filter is empty, so FOUND "
            "is not in the loop. This is a wiring fault, not a configuration choice."))
    # The old comparison was basename(GRAPH_API_CONFIG) against CFG_NAME, and the container
    # builds GRAPH_API_CONFIG from CFG_NAME — true for every value, so it could never fail.
    if expect_name and os.path.basename(loaded_path) != expect_name:
        return False, dict(detail, expected_name=expect_name)
    if expect_sha and file_sha != expect_sha:
        return False, dict(detail, expected_file_sha=expect_sha, why=(
            "the config file differs from the one the launcher hashed — the mounted tree is "
            "not the tree the launcher read. CHECK WHICH CHECKOUT $REPO POINTS AT before "
            "re-stamping the expectation; re-stamping turns this probe back into a tautology."))
    if expect_merged and merged != expect_merged:
        return False, dict(detail, expected_merged_sha=expect_merged, why=(
            "the file matches but the MERGED configuration does not — config.py's _DEFAULTS "
            "differ between the host and the container. Every key the yaml never writes is "
            "decided there."))
    return True, detail


def a3_policy_reached_container(expect_policy):
    """Compare the launcher's INTENDED values against what is set in here.

    Not an echo of whatever is present — that is what made an "enforcing" run a
    pass-through: FOUND_ENFORCE was absent from the -e list, so exporting it on the host
    did nothing and the run printed the value it had never delivered.
    """
    if not expect_policy:
        return SKIPPED, {"reason": "no --expect-policy given; nothing to compare against"}
    mismatches, seen = {}, {}
    for k, want in expect_policy.items():
        got = os.environ.get(k)
        seen[k] = got
        if got != want:
            mismatches[k] = {"intended": want, "in_container": got}
    return (not mismatches), {"checked": seen, "mismatches": mismatches}


def a4_perception_twice():
    """Probe the backend TWICE. A one-shot liveness check passes a broken service.

    The CLIP-on-CPU fault returned success on the first request and 500 on every one after,
    because the text head was built on the first call and never reused. health() is not
    enough either — it need not exercise the path that broke. Two real inferences.
    """
    import numpy as np

    sys.path.insert(0, "/ws/install/lost3dsg/lib/lost3dsg")
    import config as cfgmod
    from cloud.client import get_perception_backend

    backend = get_perception_backend(cfgmod.CFG)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    attempts = []
    for i in (1, 2):
        t0 = time.time()
        try:
            backend.detect_and_segment(frame, ["chair"])
            attempts.append({"attempt": i, "ok": True, "ms": round((time.time() - t0) * 1000)})
        except Exception as exc:
            attempts.append({"attempt": i, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return all(a["ok"] for a in attempts), {"backend": type(backend).__name__,
                                            "attempts": attempts}


# Only these reach the bundle: live_run.sh copies *.json, *.jsonl and *.log out of the
# scratch directory. Leftover detection_*.png are never archived, and a gate that fails on a
# file nobody copies gets skipped.
ARCHIVED_GLOBS = (".json", ".jsonl", ".log")


def a5_bundle_clean(run_dir, run_start_epoch, scratch_dir=None):
    """No bundle artefact may predate run start.

    Catches two different lies with one check: a scratch directory left dirty from the
    previous run (nothing clears $OUT_DIR between runs, so a run that dies early archives
    its predecessor's files under its own source hashes), and a file copied in by hand.
    """
    if not run_start_epoch:
        return SKIPPED, {"reason": "no --run-start given"}
    # Filesystem mtimes are truncated to the second on ext4, so a file written a
    # fraction of a second AFTER run start reads as older than it and every run would
    # fail the gate on its own fresh output. The tolerance costs nothing against the
    # real target — an artefact left by the previous run is hours stale, and `cp`
    # without -p stamps a copied file with the time it was copied.
    threshold = run_start_epoch - FS_MTIME_TOL
    stale = []

    def _scan(base, archived_only):
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if archived_only and os.path.splitext(fn)[1] not in ARCHIVED_GLOBS:
                    continue
                path = os.path.join(root, fn)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                if mtime < threshold:
                    stale.append({"dir": base, "path": os.path.relpath(path, base),
                                  "mtime": round(mtime), "threshold": round(threshold)})

    # The bundle directory is created by mkdir -p immediately before the run, so it is empty
    # by construction and this scan catches only a file placed by hand.
    _scan(run_dir, archived_only=False)
    # The scratch directory is the one that is never cleared, and its archived extensions are
    # copied into the bundle. This is what the probe is actually for.
    if scratch_dir and os.path.isdir(scratch_dir):
        _scan(scratch_dir, archived_only=True)

    detail = {"run_dir": run_dir, "scratch_dir": scratch_dir,
              "stale_count": len(stale), "stale": stale[:20]}
    if stale:
        detail["why"] = ("these artefacts predate this run and would be archived into its "
                         "bundle under its source hashes. live_run.sh renames $OUT_DIR before "
                         "each run; that did not happen. Move /tmp/graphapi_live by hand, then "
                         "restart.")
    return (not stale), detail


def a6_camera_pose_offset(expect_height_m=1.5, tol=0.25):
    """The camera pose MUST differ from the base pose by the mount height.

    base_link sits on the floor, habitat_camera 1.5 m up (live_stack_container.sh, rtabmap
    grid args). A zero offset means something is publishing the base pose where the camera
    pose belongs — which is what made the per-frame viewpoint series describe a viewpoint
    1.5 m below the rendering eye, and the frustum denominator with it.
    """
    import rclpy
    from rclpy.duration import Duration
    from tf2_ros import Buffer, TransformListener

    rclpy.init(args=None)
    try:
        node = rclpy.create_node("preflight_pose_probe")
        buf = Buffer()
        TransformListener(buf, node)
        deadline = time.time() + 20.0
        tf = None
        while time.time() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
            try:
                tf = buf.lookup_transform("base_link", "habitat_camera",
                                          rclpy.time.Time(), Duration(seconds=0.5))
                break
            except Exception:
                continue
        if tf is None:
            return SKIPPED, {"reason": "base_link -> habitat_camera not published within 20 s"}
        dz = float(tf.transform.translation.z)
        return abs(dz - expect_height_m) <= tol, {
            "dz_m": round(dz, 4), "expected_m": expect_height_m, "tol_m": tol}
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


# Roots, typed by MOUNT DISCIPLINE. Only /graph_api is copied into the container's install
# tree at startup, so only it has a freeze point. /found reaches the process through
# hooks.search_paths -> sys.path.insert and /kb through PYTHONPATH: both are live for the
# whole run, so a comparison taken at container start asserts a property that cannot hold
# even when it passes — an edit at T2, mid-run, still changes the code that executes.
FROZEN_ROOTS = {"graph_api": "/graph_api/lost3dsg"}
LIVE_ROOTS = {"found": "/found/found", "kb": "/kb"}


def a7_source_frozen(expect):
    """The copied tree must match what the launcher hashed. The live mounts are SAMPLED.

    Rule 9 made mechanical: the container copies its sources once at startup, so anything
    edited between the launcher's stamp and that copy is executed by the run and absent from
    its record. That voided every number produced before 26 August.
    """
    # An expectation that never arrived is the failure this gate exists for. Passing on an
    # empty --expect-src-sha would make a7 assert nothing while reporting PASS — the same
    # shape as the config comparison that could never fail. Nothing to compare = SKIPPED.
    if not expect:
        return SKIPPED, {"reason": "no --expect-src-sha given; the launcher's digests did not "
                                   "reach the container, so there is nothing to compare against"}
    frozen, live, mismatches = {}, {}, {}
    for name, root in FROZEN_ROOTS.items():
        if not os.path.isdir(root):
            continue
        sha, n = tree_sha(root)
        frozen[name] = {"sha256_16": sha, "files": n}
        want = (expect or {}).get(name)
        if want and want != sha:
            mismatches[name] = {"launcher": want, "container": sha}
    for name, root in LIVE_ROOTS.items():
        if os.path.isdir(root):
            sha, n = tree_sha(root)
            live[name] = {"sha256_16": sha, "files": n,
                          "note": "live mount, not copied — sampled again at teardown"}
    detail = {"frozen": frozen, "live_sampled": live, "mismatches": mismatches}
    if mismatches:
        moved = ", ".join(mismatches)
        detail["why"] = (
            f"an edit landed between the provenance stamp and the container start ({moved}). "
            "This run would execute code that its own bundle does not describe. That is not a "
            "fault in the code — it is the freeze rule. Restart the run: it costs minutes, and "
            "the alternative voided every number produced before 26 August.")
    if not frozen:
        return SKIPPED, {"reason": "no frozen root present; not running in the container?"}
    return (not mismatches), detail


PROBES = {
    "a1": ("aligner_identity", a1_aligner_identity),
    "a2": ("config_identity", a2_config_identity),
    "a3": ("policy_reached_container", a3_policy_reached_container),
    "a4": ("perception_twice", a4_perception_twice),
    "a5": ("bundle_clean", a5_bundle_clean),
    "a6": ("camera_pose_offset", a6_camera_pose_offset),
    "a7": ("source_frozen", a7_source_frozen),
}


def _kv(s):
    out = {}
    for part in (s or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Class A pre-flight gate")
    ap.add_argument("--out", default="/ws/output/preflight.json")
    ap.add_argument("--only", help="comma-separated probe ids, e.g. a2,a3")
    ap.add_argument("--expect-config-name")
    ap.add_argument("--expect-config-sha", help="sha of the config FILE, from the launcher")
    ap.add_argument("--expect-merged-sha", help="sha of the MERGED cfg, from the launcher")
    ap.add_argument("--expect-src-sha", help="K=V,K=V tree digests the launcher recorded")
    ap.add_argument("--scratch-dir", default="/out",
                    help="the directory that is NOT cleared between runs")
    # The launcher calls these instead of restating the hashing in shell.
    ap.add_argument("--print-merged-sha", action="store_true",
                    help="print the merged-config sha for the CURRENT GRAPH_API_CONFIG and exit")
    ap.add_argument("--print-tree-sha", metavar="ROOT",
                    help="print '<sha> <file count>' for ROOT and exit")
    ap.add_argument("--expect-policy", help="K=V,K=V of values the launcher INTENDED")
    ap.add_argument("--run-dir", default="/ws/output")
    ap.add_argument("--run-start", type=float, default=0.0, help="epoch seconds")
    ap.add_argument("--camera-height", type=float, default=1.5)
    ap.add_argument("--allow-skip", action="store_true",
                    help="treat a probe that could not run as non-fatal (NOT for a measured run)")
    args = ap.parse_args(argv)

    # --- launcher helpers. They exit; they never run a probe. ---------------------------
    if args.print_tree_sha:
        try:
            sha, n = tree_sha(args.print_tree_sha)
        except ValueError as exc:
            print(f"!! {exc}", file=sys.stderr)
            return 1
        print(f"{sha} {n}")
        return 0
    if args.print_merged_sha:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in ("/ws/install/lost3dsg/lib/lost3dsg",
                     os.path.join(here, "..", "src", "perception_module")):
            if os.path.isdir(cand):
                sys.path.insert(0, os.path.abspath(cand))
        import config as cfgmod
        if getattr(cfgmod, "CFG_PATH", None) is None:
            print("!! no config file loaded — config.py fell back to its defaults; refusing "
                  "to print a sha for a configuration nobody chose", file=sys.stderr)
            return 1
        print(merged_cfg_sha(cfgmod.CFG))
        return 0

    wanted = [p.strip() for p in args.only.split(",")] if args.only else list(PROBES)
    bound = {
        "a1": a1_aligner_identity,
        "a2": lambda: a2_config_identity(args.expect_config_name, args.expect_config_sha,
                                         args.expect_merged_sha),
        "a3": lambda: a3_policy_reached_container(_kv(args.expect_policy)),
        "a4": a4_perception_twice,
        "a5": lambda: a5_bundle_clean(args.run_dir, args.run_start, args.scratch_dir),
        "a6": lambda: a6_camera_pose_offset(args.camera_height),
        "a7": lambda: a7_source_frozen(_kv(args.expect_src_sha)),
    }

    results, failed, skipped = [], [], []
    for pid in wanted:
        name = PROBES[pid][0]
        try:
            ok, detail = bound[pid]()
        except Exception as exc:
            ok, detail = SKIPPED, {"reason": f"probe raised: {type(exc).__name__}: {exc}"}
        results.append({"id": pid, "name": name, "ok": ok, "detail": detail})
        mark = "PASS" if ok else ("SKIP" if ok is SKIPPED else "FAIL")
        print(f"  [{mark}] {pid} {name}: {json.dumps(detail, default=str)[:200]}")
        if ok is False:
            failed.append(pid)
        elif ok is SKIPPED:
            skipped.append(pid)

    verdict = "pass" if not failed and (args.allow_skip or not skipped) else "fail"
    report = {"verdict": verdict, "failed": failed, "skipped": skipped, "probes": results}
    try:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, default=str)
    except OSError as exc:
        print(f"!! could not write {args.out}: {exc}")

    if verdict != "pass":
        # A skipped probe asserted nothing. Failing on it is the whole point: the aligner
        # that "could not run" is exactly how the exemplar baseline shipped as the KG result.
        print(f"!! PRE-FLIGHT FAILED — failed={failed} skipped={skipped}")
        return 1
    print(f"  pre-flight OK ({len(results)} probes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
