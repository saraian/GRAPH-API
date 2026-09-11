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
import glob
import os
import sys
import time

# Probes return (ok: bool|None, detail: dict). ok=None means "could not run" — distinct from
# False, because a probe that could not run has asserted NOTHING and must never read as a pass.
SKIPPED = None

# Backends whose segment_scene returns an empty result rather than raising. a4 cannot
# distinguish "answered instantly" from "did nothing" by timing, so it refuses them by name.
STUB_BACKENDS = frozenset({"LocalPerceptionBackend"})

# The per-stage keys detection_pipeline requires and refuses to invent.
#
# A COPY, and a test asserts it equals the pipeline's own list — see
# test_a4s_timing_keys_match_the_pipelines. The two must not drift: a4 exists to catch this
# failure before a run, and a4 asking for different names than the pipeline would make it
# certify a run that then dies at its first detection.
#
# The names were `("owlv2", "sam")` here and in the vendored pipeline. The server actually
# emits `['detector', 'sam2', 'total']` — measured from a live refusal, not inferred — and
# /DATA/GRAPH-API's pipeline reads detector/sam2 with the reason at the site: "Modal reports
# detector/sam2 (NMS runs inside the detector there)". So this was never a backend-contract
# fault. It is a port that was never made, in one checkout.
_PIPELINE_TIMING_FALLBACK = ("detector", "sam2")


def pipeline_timing_keys():
    """Whatever `detection_pipeline` demands — READ, not copied.

    a4 exists to predict the pipeline's refusal before a run pays for it. A copy of the key
    names can drift from the pipeline's, and then a4 either certifies a run that dies at its
    first detection or refuses one that would have worked. Reading the pipeline's own list
    makes a4 wrong exactly when the pipeline is wrong, which is the only correct behaviour for
    a predictor.

    The vendored tree asks for ("owlv2", "sam"); the server emits ['detector', 'sam2', 'total']
    — measured from a live refusal — and /DATA/GRAPH-API's pipeline reads detector/sam2 with
    the reason at the site. So a4 reading the vendored list will FAIL, correctly: that run does
    die. The remedy is to port the two names, not to teach a4 different ones.
    """
    import re
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in ("/ws/install/lost3dsg/lib/lost3dsg/detection_pipeline.py",
                 os.path.join(here, "..", "src", "perception_module", "detection_pipeline.py")):
        if os.path.exists(cand):
            m = re.search(r"missing\s*=\s*\[k for k in \(([^)]*)\)", open(cand).read())
            if m:
                keys = tuple(x.strip().strip("\"'") for x in m.group(1).split(",") if x.strip())
                if keys:
                    return keys, cand
    return _PIPELINE_TIMING_FALLBACK, None

# ext4 truncates mtime to whole seconds; see a5_bundle_clean.
FS_MTIME_TOL = 2.0

# a1's golden pair. The positive is on record from a live run on 26 Aug (couch -> Sofa, 0.903).
# The negative is the half that catches an aligner answering everything; it must be a string no
# ontology can plausibly contain.
GOLDEN_LABEL = "couch"
GOLDEN_EXPECT = "Sofa"
NONSENSE_LABEL = "zzqx_not_a_real_object_kind"


# ---------------------------------------------------------------------------------------
# Digests. ONE implementation, used by the probes here AND by run_sim.sh via --print-*,
# so the launcher never restates the hashing. A check that restates its subject drifts from
# it in silence, which is how the tautological a2 survived.
#
# Deny-list, not allow-list: COLCON_IGNORE has no extension, so no extension list can catch
# it, and its presence decides whether the package builds. The principle for the exclusions
# is "exclude what the system writes; keep what a person wrote, however dead" — build/,
# install/, __pycache__/ and grafici_output/ are outputs; a superseded module is not.
# Excluded: everything the SYSTEM writes. `output/` is where the running stack writes —
# hook_decisions.jsonl defaults to <package>/output/ — and `.mypy_cache` is written by anyone
# running the type checker. With those inside the set the frozen root COULD NOT HOLD STILL BY
# CONSTRUCTION: running the stack moved it, and a lane running mypy moved it. Three files, and
# they are why a 137->139 change of source read as 140.
DENY_DIRS = {"build", "install", "__pycache__", ".git", "grafici_output",
             ".ruff_cache", ".mypy_cache", ".pytest_cache", "output"}
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


_MISSING = object()


def _aligned_name(res):
    """Return `(name, source)` for an alignment result, or RAISE if its shape is unrecognised.

    a probe read `res.name` for a whole day. The alignment result it read has fields
    `label, aligned, score, evidence` — **there is no `name`** — so `getattr(res, "name", None)`
    returned None for every input the aligner could produce. The consequence is worse than a
    wrong field:

      * the positive case could never match "Sofa"  -> a1 could never PASS
      * the negative control always saw None        -> it could never FAIL

    **A check that cannot succeed, whose control cannot fail**, inside the gate built to stop
    exactly that. The aligner was right the entire time: it computed `Sofa`, returned it in
    `.aligned`, and wrote it into `.evidence`.

    So this raises rather than defaulting. `getattr(x, "field", None)` cannot tell a renamed
    field from a genuine null, and that difference is the whole content here — an unrecognised
    result shape means the probe cannot assert anything, and the harness turns a raise into
    SKIPPED, which is a failed verdict. Silence is what cost the day.
    """
    if res is None:
        return None, "no result"
    for attr in ("aligned",):                      # the alignment result of an extension
        val = getattr(res, attr, _MISSING)
        if val is not _MISSING:
            return val, f".{attr}"
    if isinstance(res, dict):
        for key in ("aligned", "name"):
            if key in res:
                return res[key], f"[{key!r}]"
        raise AttributeError(
            f"alignment result is a dict with none of the expected keys: {sorted(res)}")
    raise AttributeError(
        f"{type(res).__name__} exposes no recognised alignment field. Known: .aligned, or a "
        f"dict with 'aligned'/'name'. Has: {sorted(a for a in dir(res) if not a.startswith('_'))}")


def _discriminants(aligner, label, result):
    """The values that DECIDE the verdict, not just the ones that describe it.

    Recorded because a1 once produced `score 0.9032032489776611 -> aligned_to null` while the
    same pair on the same host, same model and same ontology gave `top=0.9032 z=5.22 n=170 ->
    ACCEPT`. The score agreed to the last digit; the CANDIDATE SET differed, and z is what the
    rule compares. **The probe recorded the number that agreed and not the number that decided**,
    so telling the two environments apart took an investigation instead of a subtraction.

    Everything here is optional: the exemplar and lexical aligners have none of it, and a probe
    that crashes collecting diagnostics is worse than one that reports fewer.
    """
    out = {}
    for key, attr in (("n_candidates", "_n"), ("top_min", "_top_min"), ("z_min", "_z_min")):
        val = getattr(aligner, attr, None)
        if val is not None:
            out[key] = val
    # `evidence` carries the z the rule actually used, formatted by the aligner itself rather
    # than recomputed here — a second implementation of the statistic could disagree with the
    # one that decided, which is the failure this whole gate is about.
    if result is not None and getattr(result, "evidence", None):
        out["evidence"] = result.evidence
    # The five nearest classes turn "it refused" into "it refused because these looked alike".
    embedder = getattr(aligner, "_embedder", None)
    if embedder is not None and hasattr(embedder, "rank"):
        try:
            ranked = embedder.rank(label.strip().lower(), k=5)
            out["top5"] = [{"class": str(iri).rsplit("#", 1)[-1].rsplit("/", 1)[-1],
                            "score": round(float(sc), 4)} for iri, sc in ranked]
        except Exception as exc:      # noqa: BLE001 - diagnostics must never fail the probe
            out["top5_error"] = f"{type(exc).__name__}: {exc}"
    return out


# a1 IS NOT HERE, and no probe that asserts about a belief layer belongs in this file.
# It named an ontology-specific golden pair and imported an extension package, so outside the
# one deployment that ships that package it could NEVER pass -- and this harness turns "cannot
# pass" into "failed". A gate that always fails gets switched off, and every check it carried
# goes with it.
#
# An extension supplies such a probe through the `Probe` seam below, the same way it extends
# Filter and Refiner. The implementation moved; the copy here was simply never deleted, so
# this file carried an ontology probe long after it had one elsewhere.


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
    # a run with hooks.filter empty and the extension out of the loop, and it is silent.
    loaded_path = getattr(cfgmod, "CFG_PATH", None)
    merged = merged_cfg_sha(cfgmod.CFG)
    file_sha = file_sha16(loaded_path) if loaded_path else None
    # THE LOCAL OVERRIDE IS RECORDED HERE, and it has to be. It used to be folded into CFG_PATH as
    # "<config> + <local>", which is not a path: this probe opens loaded_path to hash it, so with an
    # override in force a2 raised FileNotFoundError and SKIPPED -- and a skipped probe fails the
    # gate. Splitting the two values fixed that, and this is where the override must reappear, or
    # an overridden run and a plain one are identical in the bundle.
    local_path = getattr(cfgmod, "CFG_LOCAL_PATH", None)
    detail = {"env_path": env_path, "loaded_path": loaded_path,
              "local_override_path": local_path,
              "local_override_sha256_16": file_sha16(local_path) if local_path else None,
              "merged_cfg_sha256_16": merged, "config_file_sha256_16": file_sha,
              "perception_backend": (cfgmod.CFG.get("perception") or {}).get("backend"),
              "hooks_filter": (cfgmod.CFG.get("hooks") or {}).get("filter")}

    if loaded_path is None:
        return False, dict(detail, why=(
            f"no config file was loaded — config.py fell back to its defaults. "
            f"GRAPH_API_CONFIG={env_path!r} does not exist. hooks.filter is empty, so the extension "
            "is not in the loop. This is a wiring fault, not a configuration choice."))
    # The old comparison was basename(GRAPH_API_CONFIG) against CFG_NAME, and the container
    # builds GRAPH_API_CONFIG from CFG_NAME — true for every value, so it could never fail.
    # BASENAMES ON BOTH SIDES. CFG_NAME may now be a path relative to lost3dsg/test/ rather than a
    # bare filename -- `--config` accepts a config anywhere, and the launcher resolves the name
    # against its own directory. Comparing a basename against "../src/perception_module/config.yaml"
    # failed with no reason attached, which is how this probe reported a WORKING run as broken on
    # Gin, 2026-09-10. The directory half is already covered by the file-sha comparison below, which
    # is the stronger check: it proves the CONTENT, not the spelling.
    if expect_name and os.path.basename(loaded_path) != os.path.basename(expect_name):
        return False, dict(detail, expected_name=expect_name, why=(
            f"the container loaded {os.path.basename(loaded_path)!r} but the launcher named "
            f"{os.path.basename(expect_name)!r} — the two are different files."))
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
    pass-through: the enforcement variable was absent from the `-e` list, so exporting it on
    the HOST did nothing, and the run printed the value it had never delivered.

    The probe names no variable of its own. Which keys matter is the CALLER's declaration,
    passed as `expect_policy`, so a deployment with a different policy layer checks its own
    keys and this file learns nothing about any of them.
    """
    if not expect_policy:
        # GA-477 (2026-09-10, found by the ontology lane, whose third launch this cost). NOTHING TO
        # COMPARE IS AN ANSWER, NOT AN ABSENCE. A skip fails the gate (:1574, and rightly -- a probe
        # that did not run must not read as green), so a3 skipping made the gate UNPASSABLE for any
        # stack with no policy layer: the launcher builds the expectation by looping over
        # EXT_ENV_PASS, which is empty without an extension, so a3 skipped for graphapi_only_config,
        # for smoke_config, and for every GRAPH-API-only deployment.
        #
        # This is the same finding GA-435 fixed for a7 four hours earlier -- "on a deployment with
        # NO extension ... a GRAPH-API-only stack could not pass the gate at all" -- which survived
        # one probe over because the fix was made where it was found rather than where it applied.
        #
        # SO THE DISCRIMINATOR IS WHETHER A POLICY LAYER EXISTS, not whether an expectation arrived.
        # No extension declares one: there is nothing to deliver and nothing to check, and the probe
        # says so. An extension IS configured and the expectation is missing: that is the launcher
        # failing to compute what it promised, which is the fault a3 was written for, and it still
        # skips.
        if _found_exercised("auto"):
            return SKIPPED, {"reason": "an extension is configured but no --expect-policy arrived; "
                                       "the launcher builds it from EXT_ENV_PASS, so this is a gap "
                                       "in the launcher, not an absent policy layer"}
        return True, {"checked": {}, "mismatches": {},
                      "reason": "no policy layer in this deployment: the config names no filter "
                                "outside this repository, so no extension variable has to reach "
                                "the container. Vacuously true, and recorded rather than skipped."}
    mismatches, seen = {}, {}
    for k, want in expect_policy.items():
        got = os.environ.get(k)
        seen[k] = got
        if got != want:
            mismatches[k] = {"intended": want, "in_container": got}
    return (not mismatches), {"checked": seen, "mismatches": mismatches}


# GA-438 (2026-09-10). THE HUB MODELS A RUN LOADS, enumerated from the source because no list
# existed anywhere. Owner policy, 2026-09-10: "models should be already downloaded and cached
# beforehand", so a download DURING a run is a fault and a4 asserts these instead of recording them.
# The previous note here -- "a cache miss is a first-run download and not a fault" -- was right when
# it was written and is not right now.
#
# THREE OF THE FOUR LOAD ON EVERY RUN, WHATEVER THE BACKEND, and that is the part the old check
# missed: it sat inside `if backend == "local"`, so a cloud-backend run could pass a4 and then
# download three models. visual_reid picks dinov2-base over dinov2-small on MEASURED VRAM, so both
# must be present or the run downloads whichever one it chooses.
HF_MODELS_EVERY_RUN = (
    ("sentence-transformers/all-MiniLM-L6-v2", "nlp_utils.py, semantic matching"),
    ("facebook/dinov2-small", "visual_reid.py, crop embedder (ViT-S/14)"),
    ("facebook/dinov2-base", "visual_reid.py, chosen over -small on measured VRAM"),
)
HF_MODELS_LOCAL_BACKEND = (
    ("google/owlv2-base-patch16-ensemble", "models.py, the in-line detector"),
)


def _hf_models_present(local_backend):
    """-> (cache root, [missing model ids], {present model id: snapshot path}).

    FILES ON DISK, never a load: a4 runs before the stack and loading OWLv2 plus the SAM pair would
    cost minutes and a lot of memory in the one probe whose job is to be cheap.

    A DIRECTORY IS NOT A CACHED MODEL. huggingface creates models--<org>--<name> before it has
    finished fetching, and an interrupted download leaves the directory with an empty snapshots/.
    So this requires a snapshot holding at least one file, which is the difference between "the
    hub was here once" and "this run can load it offline".
    """
    hf = (os.environ.get("PREFLIGHT_HF_CACHE") or os.environ.get("HF_HOME")
          or os.environ.get("TRANSFORMERS_CACHE") or "/models/hf")
    # READ THE CACHE THE CONTAINER WILL USE, and say so when this is not it. HF_HOME on the host is
    # not HF_HOME in the container -- that difference IS the /ext fault this probe now exists to
    # catch, and a probe that reads the wrong cache becomes the defect it was written for. Same
    # shape as a8's install-tree test: outside the container, assert nothing rather than assert
    # about the wrong machine. PREFLIGHT_HF_CACHE names the cache explicitly when it is known.
    wanted = list(HF_MODELS_EVERY_RUN) + (list(HF_MODELS_LOCAL_BACKEND) if local_backend else [])
    missing, seen = [], {}
    # THE LOADER READS $HF_HOME/hub, SO ONLY THAT COUNTS. A recursive search found a stale
    # sentence-transformers directory at the cache ROOT -- blobs and refs, no snapshots dir the
    # loader would use -- and reported the model present while an offline load of it raised
    # LocalEntryNotFoundError. Measured 2026-09-10 against /DATA/huggingface_cache. A probe that
    # searches more widely than the loader does will pass a model the run cannot open.
    hub = os.path.join(hf, "hub") if os.path.isdir(os.path.join(hf, "hub")) else hf
    for model_id, _why in wanted:
        stem = "models--" + model_id.replace("/", "--")
        hit = None
        cand = os.path.join(hub, stem)
        if glob.glob(os.path.join(cand, "snapshots", "*", "*")):
            hit = cand
        if hit:
            seen[model_id] = hit
        else:
            missing.append(model_id)
    return hf, missing, seen


def _hub_why(missing, hf):
    why = {m: w for m, w in list(HF_MODELS_EVERY_RUN) + list(HF_MODELS_LOCAL_BACKEND)}
    return ("these hub models are NOT cached under " + hf + ": "
            + "; ".join(f"{m} ({why.get(m, 'loaded by this stack')})" for m in missing)
            + ". Owner policy 2026-09-10: models are cached beforehand, so a download inside a run "
              "is a fault -- it lands in the first perception cycle, which is the cycle a cold-start "
              "number is read from. Fetch them into the cache the container mounts, or point "
              "PREFLIGHT_HF_CACHE at the cache this run will use.")


def a4_perception_twice(frame=None):
    """Probe the backend TWICE. A one-shot liveness check passes a broken service.

    The CLIP-on-CPU fault returned success on the first request and 500 on every one after,
    because the text head was built on the first call and never reused. health() is not
    enough either — it need not exercise the path that broke. Two real inferences.
    """

    sys.path.insert(0, "/ws/install/lost3dsg/lib/lost3dsg")
    import config as cfgmod
    from cloud.client import get_perception_backend

    backend = get_perception_backend(cfgmod.CFG)
    name = type(backend).__name__

    # GA-438. THE ALWAYS-LOADED MODELS ARE ASSERTED FOR EVERY BACKEND, above the local/cloud split.
    # The cache check used to sit INSIDE `backend == "local"`, so a cloud run asserted nothing about
    # models it loads regardless: nlp_utils loads MiniLM on every run and visual_reid loads one of
    # the two dinov2 sizes on every run. A cloud-backend run could pass a4 and then download three
    # models inside its first perception cycle.
    _hf_root, _always_missing, _always_seen = _hf_models_present(local_backend=False)
    _in_container = os.path.isdir(INSTALL_TREE)
    if _always_missing and _in_container:
        return False, {"backend": name, "hf_cache_root": _hf_root,
                       "hub_models_present": _always_seen,
                       "hub_models_missing": _always_missing,
                       "scope": "loaded on EVERY run, whatever perception.backend is",
                       "why": _hub_why(_always_missing, _hf_root)}

    # GA-81. `LocalPerceptionBackend.segment_scene` is `return [], {}` — it does not
    # raise, so three calls against it record three passes having run NOTHING. a4 exists to
    # catch a backend that answers once and fails after; a backend that answers instantly and
    # always is the same defect with the sign flipped, and a4 could not see it.
    #
    # So a4 asserts WHICH backend answered before asking it anything — the gate's first
    # principle applied to a4 itself. It has never fired: both launcher configs set `modal`,
    # and today's 36,521 ms first call proves Modal answered. Latent, not historical.
    # GA-434 (finding: experiment lane, 2026-09-09; verified in the source here). THE NAME CHECK
    # BELOW WOULD REFUSE A LEGITIMATE LOCAL RUN. On `perception.backend: "local"` the factory returns
    # LocalPerceptionBackend, but the pipeline NEVER ASKS IT ANYTHING: detection_pipeline gates the
    # backend call on `backend_type != "local"`, and perception_2 builds OWLv2() and VitSam() in-line
    # instead. So the stub's empty return is not on the detection path, and a4's premise is true of
    # the OBJECT it inspects and false of the CODE that runs (rule 51).
    #
    # What the local path actually depends on is FILES, so that is what is asserted. Deliberately NOT
    # by loading the models: a4 runs before the stack, and loading OWLv2 and the SAM pair here would
    # cost minutes and a lot of memory in the one probe whose job is to be cheap.
    local_cfg = (getattr(cfgmod, "CFG", {}) or {}).get("perception", {}).get("backend", "local")
    if str(local_cfg).lower() == "local":
        paths = (cfgmod.CFG.get("paths") or {})
        want = {"vitsam_encoder": paths.get("vitsam_encoder"), "vitsam_decoder": paths.get("vitsam_decoder")}
        missing = {k: v for k, v in want.items() if not (v and os.path.isfile(v) and os.path.getsize(v) > 0)}
        hf, hub_missing, hub_seen = _hf_models_present(local_backend=True)
        cached = "google/owlv2-base-patch16-ensemble" not in hub_missing
        detail = {"backend": name, "path": "in-line (OWLv2 + VitSam), the stub is never called",
                  "vitsam": {k: (v, os.path.getsize(v) if v and os.path.isfile(v) else None)
                             for k, v in want.items()},
                  "owlv2_weights_cached": cached, "hf_cache_root": hf,
                  "hub_models_present": hub_seen, "hub_models_missing": hub_missing,
                  "note": "the segmenter's files and the hub models are ASSERTED, both from disk and "
                          "never by loading them. Whether the models actually detect is the first "
                          "cycle's answer, not this probe's."}
        if missing:
            detail["why"] = (f"perception.backend is 'local' and the segmenter files do not resolve: "
                             f"{missing}. The in-line path cannot start without them, and there is no "
                             f"fallback — a4 refuses rather than letting the run discover it.")
            return False, detail
        # AFTER the segmenter files, on purpose. The segmenter is what decides whether the in-line
        # path can start at all, and it was a4's contract before today; putting the hub check first
        # would change which reason a4 gives for a fault it already caught.
        if hub_missing and _in_container:
            detail["why"] = _hub_why(hub_missing, hf)
            return False, detail
        return True, detail

    if name in STUB_BACKENDS:
        return False, {
            "backend": name,
            "why": (f"{name}.segment_scene returns an empty result without raising, so "
                    "this probe would record passes for calls that computed nothing. A run "
                    "configured onto it produces no detections and no error. Check "
                    "perception.backend in the config the container actually loaded — a2 "
                    "records it as `perception_backend`."),
        }

    # A REAL habitat render at run resolution, shipped beside this file. See its .provenance.
    #
    # This was `np.zeros((64, 64, 3))`. The Modal service takes a different code path for a
    # degenerate input — it returned timing keys ['total','yolo_world'] for the zero frame and
    # ['detector','sam2','total'] for a real one, same endpoint, minutes apart. **So a4 was
    # asserting a contract against an input no run ever sends**, and it refused run 16 for a
    # reason that did not describe the run. A false PASS was equally available and nothing in
    # the probe distinguished them.
    #
    # A missing frame SKIPS. It must never fall back to zeros: that would restore the defect
    # silently, in the probe whose whole purpose is to refuse silent substitutes.
    frame_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "probe_assets", "a4_probe_frame.jpg")
    if frame is None:                       # `frame` is injected only by the host tests
        if not os.path.exists(frame_path):
            return SKIPPED, {"reason": f"a4's probe frame is missing at {frame_path}; refusing "
                                       "to substitute a synthetic one — see its .provenance"}
        import cv2  # container-only; the host tests inject instead
        frame = cv2.imread(frame_path)
        if frame is None:
            return SKIPPED, {"reason": f"a4's probe frame at {frame_path} could not be decoded"}
    attempts = []
    reported = []
    for i in (1, 2, 3):
        t0 = time.time()
        try:
            # GA-468 (2026-09-10). segment_scene, not detect_and_segment. The perception interface
            # was renamed on the abstract base and on every backend, and this gate was left calling
            # the old name -- so a4 reported "the backend answered no call; it is unreachable, not
            # cold" for an AttributeError, which is a different fault with a different fix, and it
            # refused three runs today while the endpoint was healthy.
            #
            # THE SECOND ARGUMENT CHANGED MEANING TOO. It was a label list; segment_scene takes the
            # scene objects the whole-scene VLM proposed. An empty list is the honest probe input:
            # a4 asks whether the backend ANSWERS TWICE, not whether it finds a chair.
            result = backend.segment_scene(frame, [])
            # The contract is (detections, timings). A backend that returns something else is
            # reported as such rather than raised as a TypeError, which would land in the
            # attempt's `error` and read as an unreachable service.
            if isinstance(result, tuple) and len(result) == 2:
                reported.append(sorted(result[1] or {}))
            else:
                reported.append(None)
            attempts.append({"attempt": i, "ok": True, "ms": round((time.time() - t0) * 1000)})
        except Exception as exc:
            attempts.append({"attempt": i, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    # GA-85. The pipeline REFUSES a response with no per-stage timing rather than estimating
    # it from the wall clock — GA-14's remedy, and it is right: a value that is not a
    # measurement must not live where measurements live. But the backend was never made to
    # report what the refusal requires, so on 31 Aug the VLM answered in 16.1 s, well inside
    # budget, and `detection_pipeline.py:93` raised on the FIRST successful detection. Every
    # cloud-backend run dies there.
    #
    # a4 already received that dict on all three calls and threw it away. Asking for it costs
    # nothing and moves the failure from "one cycle into a measured run" to "before the run".
    seen = [r for r in reported if r is not None]
    required, required_from = pipeline_timing_keys()
    missing_all = [k for k in required if all(k not in r for r in seen)]
    if seen and missing_all:
        return False, {
            "backend": name, "attempts": attempts, "reported_timing_keys": reported,
            "probe_frame": os.path.basename(frame_path), "frame_shape": list(frame.shape),
            "missing": missing_all, "required": list(required),
            "required_read_from": required_from or "built-in fallback",
            "why": (f"the backend reported no timing for {missing_all}, and "
                    "detection_pipeline refuses to estimate a stage time from the wall clock. "
                    "The VLM will answer and the pipeline will raise on the FIRST successful "
                    "detection. Compare `reported_timing_keys` against what the pipeline asks "
                    "for: a name mismatch is a port that was never made, not a backend fault, "
                    "and the server may be reporting perfectly under other names. "
                    "Do not 'fix' it by restoring an estimate."),
        }

    first, second, third = (a["ok"] for a in attempts)
    # Recorded on EVERY path, not only on failure. A passing a4 previously said which backend
    # answered and nothing about what it was asked or what came back — so the two facts that
    # resolved the yolo_world scare, the frame and the reported keys, existed only when the
    # probe failed. Same defect a1 had: the verdict recorded, the discriminant discarded.
    detail = {"backend": type(backend).__name__, "attempts": attempts,
              "pattern": "".join("P" if a["ok"] else "F" for a in attempts),
              "probe_frame": os.path.basename(frame_path), "frame_shape": list(frame.shape),
              "reported_timing_keys": reported,
              "required": list(required), "required_read_from": required_from}

    # THREE calls, not two, and the third is why a warm-up must not move into the launcher.
    #
    # A warm-up consumes the FIRST call — and the serve-once fault is DEFINED by the first
    # call succeeding and every later one failing (the CLIP text head was built on request one
    # and never reused). Warm the backend beforehand and that fault presents as F,F: the probe
    # would report "unreachable" for a service that is answering, and it would report it
    # forever, because nothing downstream contradicts a probe that fails for the wrong reason.
    #
    # Three calls separate all four states without a warm-up anywhere:
    #   P P P  healthy
    #   F P P  cold start — the run's own first frames would have absorbed it; PASS
    #   P F F  serve-once, the fault this probe exists for
    #   F F F  unreachable
    if first and second and third:
        return True, detail
    if not first and second and third:
        # The run absorbs a cold start regardless, so refusing it blocks runs that are fine.
        # Passing is safe ONLY because calls two and three are counted: a backend that is
        # merely cold answers them, and one that is broken does not.
        detail["why"] = ("the first call timed out and the next two answered — a cold start. "
                         "The run's own first frames would have absorbed it. Passing on the "
                         "strength of calls two and three, not on the assumption it is warm.")
        return True, detail
    if first and not second:
        # The fault this probe was built for: the CLIP text head was constructed on the first
        # call and never reused, so the service answered once and 500'd afterwards. A one-shot
        # liveness check passes this.
        detail["why"] = ("the backend answered once and then failed. This is the fault a4 "
                         "exists for — a one-shot liveness check would have passed it, and a "
                         "run would have produced detections from the first frame only.")
    elif not first and second and not third:
        # The opposite direction, and it is NOT the same finding. Measured on the first gated
        # run: attempt 1 timed out, attempt 2 returned in 20.5 s against a remote Modal
        # backend. That reads as a cold start rather than a broken service.
        detail["why"] = ("the backend failed the first call and answered the second — a "
                         "warm-up, not the serve-once fault. The run is still refused, "
                         "because a measured run whose first frames time out is not a run. "
                         "WARM THE BACKEND AND RE-RUN. Do not widen the timeout from this "
                         "one sample: a probe that passes on the second attempt is not a "
                         "backend that works, and any new timeout should come from a "
                         "measured distribution.")
    elif not any((first, second, third)):
        detail["why"] = "the backend answered no call; it is unreachable, not cold."
    else:
        detail["why"] = (f"pattern {detail['pattern']} — the backend is answering "
                         "intermittently. That is neither a cold start nor the serve-once "
                         "fault, and a measured run cannot be built on it.")
    return False, detail


# Only these reach the bundle: run_sim.sh copies *.json, *.jsonl and *.log out of the
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
                         "bundle under its source hashes. run_sim.sh renames $OUT_DIR before "
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
# tree at startup, so only it has a freeze point. A live-mounted tree reaches the process through
# hooks.search_paths -> sys.path.insert and /kb through PYTHONPATH: both are live for the
# whole run, so a comparison taken at container start asserts a property that cannot hold
# even when it passes — an edit at T2, mid-run, still changes the code that executes.
FROZEN_ROOTS = {"graph_api": "/graph_api/lost3dsg"}

# The tree that ACTUALLY EXECUTES, and the source it was copied from. `cp -r` runs at
# live_stack_container.sh:10 and colcon builds into /ws/install; the gate runs at :92, AFTER
# both. So the mount above and the launcher's stamp are the SAME host directory read at two
# moments — a7 was comparing the host tree against itself and never touched the copy.
#
# Nothing else in this project measures the copy either. A bundle attests the host tree at two
# moments; it does not attest the code that ran. This closes that.
EXECUTED_TREE = "/ws/install/lost3dsg/lib/lost3dsg"
COPY_SOURCE = "/graph_api/lost3dsg/src/perception_module"
# NO DEFAULT ROOTS. This used to name two mounts by path -- mounts that
# only exists in one deployment, hard-coded into the generic gate. The roots a run must keep
# frozen are the CALLER's to declare: they are whatever that deployment mounts live.
#
# Empty is not "check nothing". a7 with no roots returns SKIPPED, and a skipped probe FAILS
# the gate, so a caller that forgets to declare its live roots is told rather than silently
# passed. That is the difference between an absent check and a check that found nothing.
LIVE_ROOTS = {}


def _found_exercised(flag):
    """-> whether this run executes extension code. "auto" asks the config.

    GA-435 (2026-09-10). The caller used to answer this with "is this a detection run", passing 1
    for everything that was not MAPPING_ONLY. Those are different questions, and on a deployment
    with NO extension the wrong answer is not harmless: a7 fails with live_roots_undeclared, so a
    GRAPH-API-only stack could not pass the gate at all. Found on Gin, where no hooks are
    configured, hooks.filter is empty and no EXT_* variable is set.

    hooks.filter IS the question. It is the single name that routes a run into extension code, and
    a2 already reads it. Empty means nothing extends this run, so there is no live tree to declare
    and none to assert. A run that DOES name a filter and declares no live root still fails, which
    is the case the check exists for.

    "1" and "0" still mean what they meant, so a caller that knows the answer keeps stating it.
    """
    if flag == "auto":
        try:
            import config as cfgmod
            # GA-476 (2026-09-10, found by the ontology lane, whose run this failed). A FILTER IS
            # NOT AUTOMATICALLY EXTENSION CODE. GA-435 replaced "is this a detection run" with
            # "does hooks.filter name something", and wrote that hooks.filter "is the single name
            # that routes a run into extension code". That was true until this repository started
            # shipping a filter of its own, hours later: envelope_size:SizeFilter lives in
            # src/perception_module, needs no extension environment and declares no live root, and
            # a7 failed it with live_roots_undeclared for existing.
            #
            # So the question is WHERE the filter lives, the same correction the launcher's own
            # guard needed. A dotted name is a package path, so it is split before the file test.
            filt = ((cfgmod.CFG.get("hooks") or {}).get("filter") or "").strip()
            if not filt:
                return False
            mod = filt.split(":", 1)[0]
            here = os.path.dirname(os.path.abspath(cfgmod.__file__))
            return not os.path.exists(os.path.join(here, *mod.split(".")) + ".py")
        except Exception:
            # A config that cannot be read is a2's finding, not a7's. Answer the safer way: assert
            # the live roots, because "no extension" is the claim that would let a moving tree pass.
            return True
    return flag == "1"


def a7_source_frozen(expect, executed_tree=None, copy_source=None, live_roots=None,
                     found_exercised=True, teardown=False):
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
    # Does what EXECUTES match what is mounted? Whole-tree digests cannot answer this — colcon
    # rearranges the layout — so compare file by file for the names present in both. A
    # difference means an edit landed between the copy and the gate, or the build produced
    # something other than its source. Either way the run is not the code the bundle describes.
    exec_tree = executed_tree or EXECUTED_TREE
    src_tree = copy_source or COPY_SOURCE
    executed = {"tree": exec_tree, "compared": 0, "differs": [], "only_in_mount": []}
    if os.path.isdir(exec_tree) and os.path.isdir(src_tree):
        for fn in sorted(os.listdir(src_tree)):
            if not fn.endswith(".py"):
                continue
            src_f, run_f = os.path.join(src_tree, fn), os.path.join(exec_tree, fn)
            if not os.path.isfile(run_f):
                executed["only_in_mount"].append(fn)
                continue
            executed["compared"] += 1
            if file_sha16(src_f) != file_sha16(run_f):
                executed["differs"].append(fn)
        # GA-157's new failure mode, and it only exists now that /ws is a NAMED VOLUME.
        #
        # colcon does not remove stale installs. A module DELETED from the source stays in a
        # persistent /ws/install and keeps being importable — code nobody can find by reading the
        # sources, which is worse than a stale interface because there is nothing to notice.
        # While /ws died with `docker run --rm` this could not happen: the install tree could not
        # outlive its source, so checking only mount-files-missing-from-install was symmetric
        # enough. With a build cache it is not.
        for fn in sorted(os.listdir(exec_tree)):
            if fn.endswith(".py") and not os.path.isfile(os.path.join(src_tree, fn)):
                executed.setdefault("only_in_install", []).append(fn)
        if executed["differs"]:
            mismatches["executed_vs_mount"] = executed["differs"]
        if executed.get("only_in_install"):
            mismatches["stale_in_install"] = executed["only_in_install"]
    else:
        executed["note"] = "install tree or mount absent; not running in the container"
    detail_executed = executed

    # GA-373 (2026-09-08). The live roots were SAMPLED and compared with nothing, so a write into
    # a live-mounted tree inside the freeze window (run 131906: launcher 13:19:06 eaed38df, gate 13:20:27
    # 9c49ecb9, same instrument) refused nothing. Now: a differing live root against the launcher's
    # stamp is a MISMATCH — blocking when the run executes extension code, recorded-only under MAPPING_ONLY
    # (no extension code runs there, so the run is not refused for code it never executes). The
    # container runs this probe once more at teardown (a7_teardown.json), so a live tree that moved
    # DURING the run is a bundle finding, not a reader's inference.
    live_mismatches = {}
    # NO ROOTS DECLARED IS A FACT TO RECORD, NOT A REASON TO ABANDON THE PROBE. a7 does two
    # jobs: it compares the EXECUTED tree against the mount, and it samples the live roots.
    # Refusing the whole probe when no roots are declared threw away the first job too -- the
    # one that catches a container running a tree nobody hashed. So the executed-tree half
    # always runs, and the absence of roots is written into the detail where a reader meets it.
    # It becomes a FAILURE only when the caller says this run executes extension code, because
    # then an undeclared live tree is exactly the thing that can change underneath it.
    roots = live_roots if live_roots is not None else LIVE_ROOTS
    if not roots and found_exercised:
        mismatches["live_roots_undeclared"] = (
            "this run executes extension code and declared no live roots; pass "
            "--live-root NAME=PATH for each tree mounted live, or a tree can change "
            "underneath the run with nothing to notice")
    for name, root in roots.items():
        if os.path.isdir(root):
            sha, n = tree_sha(root)
            live[name] = {"sha256_16": sha, "files": n,
                          "note": "live mount, not copied — compared against the launcher's stamp "
                                  "when one was given; sampled once more at teardown "
                                  "(a7_teardown.json)"}
            want = (expect or {}).get(name)
            if want and want != sha:
                live_mismatches[name] = {"launcher": want, "container": sha,
                                         "blocking": bool(found_exercised)}
                if found_exercised:
                    mismatches[name + "_live"] = live_mismatches[name]
    if teardown:
        # AT TEARDOWN THE MOUNT IS EXPECTED TO HAVE MOVED: the vendored tree is released to the
        # lanes at the copied ping, and what executed is the COPY taken at startup, which the
        # startup a7 already matched against the stamp. Comparing the mount again here refused
        # run 20260908_135714 for edits that reached nothing. Only the LIVE roots are asserted at
        # teardown; the mount readings stay in the detail as information.
        detail_informational = {"frozen": frozen, "mount_vs_stamp": mismatches}
        mismatches = {k: v for k, v in mismatches.items() if k.endswith("_live")}
    detail = {"frozen": frozen, "live_sampled": live, "mismatches": mismatches,
              "live_mismatches": live_mismatches, "found_exercised": bool(found_exercised),
              "teardown": bool(teardown),
              "executed": detail_executed,
              "note": ("`frozen` compares the MOUNT against the launcher's stamp — the same host "
                       "directory at two moments. `executed` compares what runs against that "
                       "mount. Neither alone attests the code that produced the numbers.")}
    if teardown:
        detail["informational_at_teardown"] = detail_informational
    if live_mismatches and not mismatches:
        detail["why_recorded"] = ("a live root changed between the launcher's stamp and this sample "
                                  "(a write into a live-mounted tree inside the freeze window); not blocking "
                                  "because this run executes no extension code (MAPPING_ONLY)")
    if mismatches:
        moved = ", ".join(mismatches)
        detail["why"] = (
            f"an edit landed between the provenance stamp and the container start ({moved}). "
            "This run would execute code that its own bundle does not describe. That is not a "
            "fault in the code — it is the freeze rule. Restart the run: it costs minutes, and "
            "the alternative voided every number produced before 26 August.")
    if not frozen:
        # SKIPPED, but carrying what WAS measured. A probe that cannot reach its verdict still
        # observed something, and discarding it makes the skip less informative than it earned.
        return SKIPPED, dict(detail,
                             reason="no frozen root present; not running in the container?")
    return (not mismatches), detail


# The modules the container starts as nodes. Every one has a `__main__` guard, so importing
# them runs their top-level imports and nothing else — which is precisely the failure mode:
# a node that cannot import dies seconds after a passing gate, and the gate says nothing.
STACK_ENTRY_POINTS = ("perception_2", "object_manager_6", "graph_api_bridge",
                      "habitat_feed_node")


INSTALL_TREE = "/ws/install/lost3dsg/lib/lost3dsg"


def a8_stack_imports(modules=None, install=None):
    """Every node the run will start must be importable BEFORE the run starts.

    GA-66. The gate passed 7/7 and the stack then died on `from models import OWLv2, VitSam`
    at perception_2.py:61, because `efficientvit`'s subpackages are absent. Seven probes said
    the wiring was sound and none of them asked the cheapest question there is: **does the code
    the run is about to execute load at all?**

    A passing gate followed by an immediate stack death is worse than a failing gate. It spends
    the bringup, produces a bundle shell, and moves the operator's attention to the wrong layer.
    """
    import importlib

    # Outside the container there is no install tree, so every import would fail for a reason
    # that says nothing about the run. Asserted nothing -> SKIPPED, which is still a failed
    # verdict and still stops a run; it just does not claim the nodes are broken.
    install = install or INSTALL_TREE
    if not os.path.isdir(install):
        return SKIPPED, {"reason": f"{install} does not exist; not running in the container"}
    sys.path.insert(0, install)
    failed, ok = {}, []
    for name in (modules or STACK_ENTRY_POINTS):
        try:
            importlib.import_module(name)
            ok.append(name)
        except BaseException as exc:      # noqa: BLE001 - a node dying on ANY error is the finding
            failed[name] = f"{type(exc).__name__}: {exc}"
    detail = {"importable": ok, "failed": failed}
    # GA-439 (2026-09-10). ONE CALL, NOT A NEW PROBE. check_install_list.py catches the module that
    # is IMPORTED but not INSTALLED -- the same defect shape as GA-128, which it was written for --
    # and its only caller was test_perception_smoke.py:590, a suite rather than a run path. Its own
    # docstring says to let it run in the gate, and `grep -c check_install_list preflight_gate.py`
    # was 0. A8 is where it belongs: a8 asks whether the installed modules import, and this asks
    # whether everything they import got installed. The two halves of one question.
    #
    # NOT BLOCKING, deliberately. a8's own failure is a node that cannot start; this is a node that
    # starts and then cannot import something at runtime, which is a real fault but not one that is
    # certain to fire this run. Recorded where a reader meets it rather than refusing a run the
    # owner authorised on the strength of the rest of the gate.
    # THE PACKAGE ROOT, not the repository root: audit() joins "CMakeLists.txt" and
    # "src/perception_module" onto what it is given, and both live under lost3dsg/. Checked against
    # the real tree rather than assumed -- /graph_api/CMakeLists.txt does not exist and the audit
    # would have reported "cannot be audited" on every run, which reads exactly like a pass.
    src_root = os.environ.get("PREFLIGHT_SRC_ROOT", "/graph_api/lost3dsg")
    if os.path.isfile(os.path.join(src_root, "CMakeLists.txt")):
        try:
            sys.path.insert(0, os.path.join(src_root, "src", "perception_module"))
            from check_install_list import audit
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = audit(src_root)
            detail["install_list"] = {"rc": rc, "clean": rc == 0,
                                      "report": buf.getvalue().strip().split("\n")[-12:],
                                      "root": src_root}
        except Exception as exc:      # noqa: BLE001 - the audit failing is information, not a stop
            detail["install_list"] = {"rc": None, "clean": None,
                                      "error": f"{type(exc).__name__}: {exc}", "root": src_root}
    else:
        detail["install_list"] = {"rc": None, "clean": None, "root": src_root,
                                  "reason": f"no CMakeLists.txt under {src_root}; the source tree is "
                                            "not mounted here, so the install list cannot be audited"}
    if failed:
        detail["why"] = (
            "a node the run is about to start cannot be imported, so it will die within seconds "
            "of this gate passing. The error above is the one the stack would have hit. This is "
            "an INSTALLATION fault, not a wiring fault — the module is missing from the image or "
            "from the mounted tree, and no configuration change fixes it.")
    return (not failed), detail


class Probe:
    """Blueprint for a probe supplied by an extension, mirroring `hooks.py`'s Filter/Refiner/
    Store. GRAPH-API ships the harness and the generic probes; anything that asserts about a
    specific belief layer belongs to the package that implements it.

    An ontology-identity probe is the case that forced this. It imports an extension's modules and
    hardcodes an ontology-specific golden pair, so **outside that deployment it cannot pass** — and a
    probe that cannot pass makes the gate's verdict permanently `fail` for a reason the operator
    cannot fix. Ontology knowledge lives in the extension; GRAPH-API ships the seam.

    Subclasses set `id` and `name` and implement `run()`, returning `(ok, detail)` with the same
    contract as the built-ins: True, False, or SKIPPED — and SKIPPED is not a pass.
    """

    id = "x0"
    name = "unnamed_probe"

    def run(self, args):
        return SKIPPED, {"reason": f"{type(self).__name__} does not implement run()"}


def load_external_probes(cfg):
    """Read `preflight.probes` from the merged config: a list of 'pkg.module:Class' specs.

    Uses hooks.load_hook so there is ONE path resolution in the tree, not two — `search_paths`
    falls back to `hooks.search_paths`, since a package supplying a probe is the same package
    supplying the filter. A spec that will not load RAISES: a probe silently absent is the
    defect this gate exists to catch, one level up.
    """
    pre = (cfg.get("preflight") or {})
    specs = pre.get("probes") or []
    if not specs:
        return {}
    sys.path.insert(0, "/ws/install/lost3dsg/lib/lost3dsg")
    import hooks
    paths = pre.get("search_paths") or (cfg.get("hooks") or {}).get("search_paths") or []
    out = {}
    for spec in specs:
        probe = hooks.load_hook(spec, Probe, paths)
        if probe.id in out:
            raise ValueError(f"two probes claim id {probe.id!r}: {spec}")
        out[probe.id] = (probe.name, probe.run, spec)
    return out


def a9_feed_streaming(log_path="/tmp/feed_node.log", window_s=12.0, min_new=2):
    """Are frames STILL ARRIVING? Sampled twice, seconds apart, after the stack is up.

    THE GAP THIS FILLS, measured on run 20260901_140710: every other probe passed, both nodes
    were alive, the port was open — and the feed node had relayed ONE frame and was spinning
    on a closed socket. `recv()` returns b'' at end of stream WITHOUT raising, so its framing
    loop appended nothing forever: RUNNABLE, burning a core, logging nothing. Six minutes of
    that looked exactly like a slow cold start, and the run produced no detection at all.

    a4 proves the DETECTOR answers twice. Nothing proved the FEED keeps coming, and a stream
    that stops after one frame is not a state any single-sample check can distinguish from a
    stream that has not started.

    Reads the node's own counter rather than subscribing: the probe must not open a second
    connection to a feed host that accepts one client, and must not perturb what it measures.
    """
    import re as _re
    import time as _t

    def _count():
        """The LAST frame count in the log, from either line that carries one.

        The node writes two: an occasional `frames relayed: N` milestone and a periodic
        `feed heartbeat: frames=N reconnects=N`. The first version of this probe read only
        the milestone — which is written rarely — so on run 20260901_144539 it sampled the
        same stale `frames relayed: 1` twice and reported a stall while the heartbeat beside
        it read `frames=35 connected=True`. A FALSE ALARM ON A HEALTHY RUN, and rule 50's
        shape once more: the pattern matched a line that exists but is not the counter that
        moves. Both forms are read now, and the maximum wins.
        """
        try:
            with open(log_path) as f:
                text = f.read()
        except OSError:
            return None
        hits = [int(x) for x in _re.findall(r"frames relayed: (\d+)", text)]
        hits += [int(x) for x in _re.findall(r"feed heartbeat: frames=(\d+)", text)]
        return max(hits) if hits else 0

    first = _count()
    if first is None:
        return False, {"why": f"{log_path} is not readable; the feed node writes it on start",
                       "log": log_path}
    _t.sleep(window_s)
    second = _count()
    delta = (second or 0) - first
    detail = {"log": log_path, "window_s": window_s, "frames_before": first,
              "frames_after": second, "new_frames": delta, "required": min_new}
    if delta >= min_new:
        return True, detail
    detail["why"] = (
        f"the feed relayed {delta} new frame(s) in {window_s:.0f}s (need {min_new}). "
        f"A frozen counter with a live process is the GA-200 shape: the node holds a dead "
        f"socket and spins. Check /tmp/feed_node.log for 'feed connection lost' and the host "
        f"log for 'client disconnected'.")
    return False, detail


def a10_frame_age_rejected_frames(expect_rejected_max_s=None):
    """`perception.max_frame_age_s` must exceed the frame ages the LAST RUN actually rejected.

    A GUARD SHORTER THAN THE AGES FRAMES ARRIVE AT IS A DEADLOCK BY ARITHMETIC: no frame can
    satisfy it, and nothing in the logs names the cause -- the symptom reads as "Cached frame
    too old", which looks like a slow feed. It has happened twice:
      * GA-164, run 20260901_144539: guard 1.0 s, frames a median 7.14 s stale. 1650 rejected,
        ZERO perception cycles in 17 minutes.
      * GA-281, run 20260902_221606: guard 5.0 s. 243 frames discarded across 543 cycle
        attempts; 26 /bbox_3d reached object_manager_6, which waited 204 s and os._exit(1)'d.

    IT COMPARES A MEASUREMENT AGAINST THE SAME MEASUREMENT, and that is the second version.
    The first compared the guard against the previous run's `total_ms`, which is WRONG in
    principle: total_ms sums work that does not gate the loop, so it could fail a healthy run
    or pass a deadlocked one. Run 20260903_110622 showed both quantities at once -- total_ms
    17290.7 ms beside a 15.0 s guard and ZERO frames rejected -- which is only explicable if
    they are different quantities, and they are.

    The ages at which frames were REJECTED are logged verbatim ("Cached frame too old (6.60s)")
    and are exactly what the guard is compared against at runtime. If the last run rejected a
    frame at 8.52 s and the guard is still 5.0 s, it will reject them again.

    PASSES when the last run rejected NOTHING -- there is no evidence of a problem and none is
    invented. Records `asserted` either way, so a bundle says whether a check was made.
    """
    cfg_name = os.environ.get("CFG_NAME", "")
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, cfg_name) if cfg_name else ""
    guard = None
    if cfg_path and os.path.isfile(cfg_path):
        try:
            import yaml
            guard = ((yaml.safe_load(open(cfg_path)) or {}).get("perception") or {}).get(
                "max_frame_age_s")
        except Exception as exc:
            return SKIPPED, {"reason": f"could not read {cfg_name}: {exc}"}
    if guard is None:
        return SKIPPED, {"reason": f"no perception.max_frame_age_s in {cfg_name or '<no CFG_NAME>'}"}

    try:
        worst = float(expect_rejected_max_s) if expect_rejected_max_s not in (None, "", "0") else None
    except (TypeError, ValueError):
        worst = None
    if worst is None or worst <= 0:
        return True, {"max_frame_age_s": float(guard), "last_run_rejected_frames": False,
                      "asserted": False,
                      "note": "the previous run rejected no frames for age (or there was no "
                              "previous run); the guard is recorded and nothing was asserted"}

    ok = float(guard) > worst
    return ok, {
        "max_frame_age_s": float(guard),
        "last_run_worst_rejected_age_s": round(worst, 2),
        "margin_s": round(float(guard) - worst, 2),
        "asserted": True,
        "reason": ("" if ok else
                   f"DEADLOCK BY ARITHMETIC: the last run REJECTED a frame at {worst:.2f} s "
                   f"and the guard is still {guard} s. Frames arriving at that age will be "
                   f"rejected again and the perception loop will starve. Raise "
                   f"perception.max_frame_age_s above {worst:.2f} s, or make frames arrive "
                   f"fresher."),
    }


def a11_tf_buffer_outlasts_frame_window():
    """`tf.buffer_cache_s` must exceed `perception.max_frame_age_s` with margin.

    A FRAME THAT OUTLIVES THE TF BUFFER CANNOT BE PLACED. The stamps are already explicit --
    utils.py:109 looks the transform up at `cached_rgb.header.stamp`, the frame's own
    timestamp, which is exactly right -- but you cannot look up a time that has been EVICTED,
    however precisely you name it. The lookup then fails or extrapolates, and a 3D box built
    on a bad transform lands BESIDE its object. That is visible in the feed overlay as boxes
    offset from the furniture they describe, and it is not a projection bug: the overlay
    projects with the current pose onto the current frame, correctly.

    MEASURED, run 20260903_123748: 7 `habitat_camera_optical->map` extrapolation failures and
    16 agent-pose failures, requesting a median 6.4 s (max 31.3 s) BEFORE the oldest data in
    the buffer. It followed raising max_frame_age_s 5.0 -> 15.0 to break a different deadlock,
    against a buffer left at 30 s -- and config.py's own comment already stated the rule the
    two must obey ("Must match the Buffer(cache_time=...) in perception_2") while nothing
    enforced it. A rule in a comment is not a rule.

    The margin is 2x rather than 1x: equality means the very oldest surviving frame lands on
    the very edge of the buffer, where a moment of TF publisher lag evicts it anyway.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_name = os.environ.get("CFG_NAME", "")
    cfg_path = os.path.join(here, cfg_name) if cfg_name else ""
    merged = {}
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(here), "src", "perception_module"))
        from config import CFG
        merged = CFG
    except Exception as exc:
        return SKIPPED, {"reason": f"could not import config: {exc}"}
    cache = (merged.get("tf") or {}).get("buffer_cache_s")
    age = (merged.get("perception") or {}).get("max_frame_age_s")
    # The yaml overrides the code default; read it the same way the run will.
    if cfg_path and os.path.isfile(cfg_path):
        try:
            import yaml
            y = yaml.safe_load(open(cfg_path)) or {}
            age = ((y.get("perception") or {}).get("max_frame_age_s", age))
            cache = ((y.get("tf") or {}).get("buffer_cache_s", cache))
        except Exception:
            pass
    if cache is None or age is None:
        return SKIPPED, {"reason": f"missing tf.buffer_cache_s ({cache}) or "
                                   f"perception.max_frame_age_s ({age})"}
    ok = float(cache) >= 2.0 * float(age)
    return ok, {
        "tf_buffer_cache_s": float(cache),
        "max_frame_age_s": float(age),
        "ratio": round(float(cache) / max(float(age), 1e-9), 2),
        "required_ratio": 2.0,
        "reason": ("" if ok else
                   f"A FRAME CAN OUTLIVE THE TF BUFFER: max_frame_age_s is {age} s against a "
                   f"{cache} s buffer. A frame accepted at {age} s will be transformed against "
                   f"a buffer that may no longer hold its stamp, and its 3D box will land "
                   f"beside its object. Raise tf.buffer_cache_s to at least {2.0*float(age)} s, "
                   f"or lower max_frame_age_s."),
    }



# GA-355 (owner, 2026-09-07: "a preflight rule that checks that ground-truth does not contaminate
# the inference process"). Design: .handoff/plan/4-preflight-gate/1-a12-gt-isolation-GA-355.md.
GT_TOKENS = r"FEED_GT_|GT_SEMANTIC|gt_semantic|habitat_gt|/gt/semantic_instance|semantic_sensor|gt_codec|_gt_"
# Files that may name a GT token, each with the reason it is allowed. Anything else FAILS.
A12_ALLOWED_FILES = {
    "test/habitat_feed_host.py": "renders the semantic sensor (producer)",
    # ../run_sim.sh, not test/run_sim.sh: the launcher was inlined into the repository root on
    # 2026-09-11 (owner: "live_run will have to be discarded"), which moved it OUT of the tree
    # this probe greps. The scan below reaches it by name for that reason.
    "../run_sim.sh": "exports FEED_GT_SEMANTIC and stamps gt_semantic",
    "test/preflight_gate.py": "this probe names the tokens",
    # Added 2026-09-11. Verified before listing, the same way the two entries below were:
    # nothing in CMakeLists' install list carries it (grep: 0), nothing under src/ or test/ imports
    # it, and it runs AFTER a run from eval.sh -- never in the container and never on the inference
    # path. Its four tokens are TABLE LABELS: "floors in ground truth", "regions in ground truth",
    # "ground-truth room labels". A reporter that may not say "ground truth" cannot label a column.
    "test/eval_report.py": "offline PDF reporter; imported by nothing, installed nowhere",

    "test/test_preflight_gate.py": "the negative test names the tokens",
    "src/perception_module/habitat_feed_node.py": "relays the blob to /gt/semantic_instance (transport)",
    "src/perception_module/gt_codec.py": "the run-length codec",
    "src/perception_module/detection_archive.py": "the archive join: habitat_gt_* row keys, validation only",
    "src/perception_module/perception_2.py": "subscription + cache + hand-off to the archive; functions audited below",
    "src/perception_module/test_perception_smoke.py": "smoke test",
    "src/perception_module/habitat_camera_node.py": "NOT installed (GA-299); host-side node",
    "src/perception_module/habitat_camera_objects_node.py": "NOT installed (GA-299); host-side node",
    # Added 2026-09-10. Both arrived on main with the Habitat scripting work and neither was listed,
    # so a CLEAN CHECKOUT failed this probe and the gate refused every run -- reported by the
    # experiment lane, whose stack came up and could not get past it. Each names `semantic_sensor`
    # once, to CONFIGURE habitat_sim's semantic camera: they are producers and exporters, the same
    # category as the two host-side nodes above. Verified before listing rather than assumed:
    # nothing under src/ or test/ imports either, and neither is in CMakeLists' install list, so
    # neither can be running while inference is.
    "src/perception_module/hm3d_ground_truth_manifest.py":
        "offline GT exporter; imported by nothing, installed nowhere",
    "src/perception_module/scene_script.py":
        "offline Habitat scripting; imported by nothing, installed nowhere",
    # Added 2026-09-10. metrics_eval.py arrived with the metrics tool and IS in CMakeLists' install
    # list, so it ships into the container -- unlike the two above. Listed anyway because a12 asks
    # whether ground truth can reach INFERENCE, and nothing on the runtime path imports it: verified
    # by grep, its only importers are build_hm3d_eval_manifest.py and metrics_eval_visualize.py,
    # both offline. A module that ships but is never imported cannot leak what it can read.
    "src/perception_module/metrics_eval.py":
        "offline HOV-SG metrics; installed, but imported only by two offline tools",
    # Added 2026-09-10. The config DECLARES the `gt_semantic` switch — a config that carries a
    # setting has to name it, exactly as run_sim.sh does when it exports FEED_GT_SEMANTIC. Both
    # entries are a declaration plus a comment; neither reads a ground-truth value. Verified before
    # listing: the only token in either file is the key's own name, once.
    "src/perception_module/config.py": "declares the gt_semantic switch; the key's name, not a read",
    "src/perception_module/config.yaml": "declares the gt_semantic switch; the key's name, not a read",
}
# Inside perception_2.py a GT token may occur only in these functions (AST, not grep).
# _record_cycle_ms carries the gt_semantic_hit latency key: MEASURED by this probe's first run on the
# clean tree, which named it in place of the two names I had guessed (rule 26: run it).
A12_ALLOWED_FUNCTIONS = {"__init__", "_gt_semantic_callback", "_gt_semantic_for",
                         "_archive_detections", "_record_cycle_ms"}


def a12_gt_isolation(root=None):
    """No file outside the allow-list, and no function of perception_2.py outside its allow-list,
    may name a ground-truth token. Static, over the copied tree the gate runs from -- the same
    root a7 hashes. A reader added to the association loop, the labeler or the describer path
    would show up here as a file or a function not on the list, which is the reading this probe
    exists to produce (rule 18: the answer differs when contamination exists).

    What it does NOT assert: the pose channel. habitat_feed_node publishes /odom and the
    odom->base_link TF from the simulator's exact pose, and rtabmap supplies map->odom only, so
    "pose_source is never habitat GT" fails by construction on the odometry axis. Recorded as a
    stamp until the owner rules (design note, question 1).
    """
    import ast
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(root or os.path.dirname(here))
    if not os.path.isdir(os.path.join(root, "src")) or not os.path.exists("/bin/grep"):
        return SKIPPED, {"reason": f"no src/ under {root} or /bin/grep absent"}
    dirs = [d for d in ("src", "test", "launch") if os.path.isdir(os.path.join(root, d))]
    cmd = ["/bin/grep", "-rlE", GT_TOKENS, "--include=*.py", "--include=*.sh", "--include=*.yaml",
           "--exclude-dir=__pycache__", "--exclude-dir=.ruff_cache"] + dirs
    out = subprocess.run(cmd, cwd=root, capture_output=True, text=True).stdout.split()
    files = sorted(f.replace(os.sep, "/") for f in out)
    # THE LAUNCHER IS OUTSIDE THIS ROOT AND IS STILL SCANNED. It was run_sim.sh and
    # was therefore inside the grepped tree; on 2026-09-11 it was inlined into the repository root,
    # which silently took the file that exports FEED_GT_SEMANTIC out of a12's reach. A probe whose
    # coverage shrinks when a file moves, and which keeps passing, is rule 78 exactly: frozen on a
    # pass. Reached by name, and its absence is NOT a pass -- a missing launcher is reported.
    launcher = os.path.join(os.path.dirname(root), "run_sim.sh")
    if not os.path.isfile(launcher):
        return SKIPPED, {"reason": f"the launcher {launcher} is absent, so its GT tokens were "
                                   f"not examined; a12 cannot attest a tree it could not read"}
    if subprocess.run(["/bin/grep", "-qE", GT_TOKENS, launcher]).returncode == 0:
        files = sorted(files + ["../run_sim.sh"])
    not_allowed = [f for f in files if f not in A12_ALLOWED_FILES]
    bad_functions = {}
    p2 = os.path.join(root, "src", "perception_module", "perception_2.py")
    if os.path.isfile(p2):
        import re
        src = open(p2).read()
        tree = ast.parse(src)
        lines = src.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = "\n".join(lines[node.lineno - 1:node.end_lineno])
                if re.search(GT_TOKENS, body) and node.name not in A12_ALLOWED_FUNCTIONS:
                    bad_functions[node.name] = node.lineno
    ok = not not_allowed and not bad_functions
    return ok, {
        "root": root,
        "files_with_gt_tokens": files,
        "not_allowed_files": not_allowed,
        "perception_2_functions_not_allowed": bad_functions,
        "pose_source": "simulator odometry (habitat_feed_node /odom + TF) re-anchored by rtabmap map->odom; NOT asserted (design question 1)",
        "reason": "" if ok else (f"ground-truth tokens outside the allow-list: files {not_allowed}, "
                                 f"perception_2 functions {bad_functions}"),
    }


# GA-359 (owner 2026-09-07: boxes placed with rtabmap's localised pose). Design: plan/14.
def a13_verdict(pose_source, dynamic_pairs, static_pairs, parent="map", child="odom"):
    """EXACTLY ONE authority for map->odom, on the side the stamp names.

    MEASURED, run 20260907_152446: rtabmap published map->odom on /tf (publish_tf_map defaults
    true; the publish_tf:=false both trees passed is not a launch argument) while the feed node
    published a static identity on /tf_static -- and for 108 s rtabmap's correction won in tf2,
    placing 54 of 747 detections 2-4 m from their true pose. Two authorities is the defect; this
    verdict names it whichever side wins.
    """
    pair = (parent, child)
    dyn, sta = pair in set(dynamic_pairs), pair in set(static_pairs)
    want_dyn = pose_source == "rtabmap"
    ok = (dyn != sta) and (dyn == want_dyn)
    return ok, {
        "pose_source": pose_source, "map_odom_in_tf": dyn, "map_odom_in_tf_static": sta,
        "reason": "" if ok else (
            "TWO AUTHORITIES for map->odom (/tf and /tf_static)" if dyn and sta else
            "NO authority for map->odom in the window" if not (dyn or sta) else
            f"map->odom is published on the wrong side for pose_source={pose_source}: "
            f"{'/tf (rtabmap)' if dyn else '/tf_static (feed node identity)'}"),
    }


def a13_pose_authority(pose_source=None, window_s=5.0):
    """POST-START. Sample /tf and /tf_static for `window_s` and apply a13_verdict.

    Container only: rclpy is not importable on the host, and a probe that cannot run is
    SKIPPED (which fails the gate), never a pass (rule 2). /tf_static is latched, so the
    subscription uses transient-local durability or it would miss a transform published
    before the probe started.
    """
    pose_source = pose_source or os.environ.get("FEED_POSE_SOURCE", "simulator")
    try:
        import rclpy
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from tf2_msgs.msg import TFMessage
    except Exception as exc:
        return SKIPPED, {"reason": f"rclpy/tf2_msgs not importable here ({type(exc).__name__}); "
                                   "a13 answers only inside the container"}
    dynamic, static = set(), set()
    rclpy.init(args=[])
    try:
        node = rclpy.create_node("preflight_a13")
        latched = QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(TFMessage, "/tf", lambda m: dynamic.update(
            (t.header.frame_id, t.child_frame_id) for t in m.transforms), 100)
        node.create_subscription(TFMessage, "/tf_static", lambda m: static.update(
            (t.header.frame_id, t.child_frame_id) for t in m.transforms), latched)
        end = time.time() + window_s
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
        node.destroy_node()
    finally:
        rclpy.shutdown()
    ok, detail = a13_verdict(pose_source, dynamic, static)
    detail.update({"window_s": window_s, "tf_pairs": sorted(f"{a}->{b}" for a, b in dynamic),
                   "tf_static_pairs": sorted(f"{a}->{b}" for a, b in static)})
    return ok, detail

PROBES = {
    "a2": ("config_identity", a2_config_identity),
    "a3": ("policy_reached_container", a3_policy_reached_container),
    "a4": ("perception_twice", a4_perception_twice),
    "a5": ("bundle_clean", a5_bundle_clean),
    "a7": ("source_frozen", a7_source_frozen),
    "a8": ("stack_imports", a8_stack_imports),
    "a10": ("frame_age_vs_rejected", a10_frame_age_rejected_frames),
    "a11": ("tf_buffer_outlasts_frames", a11_tf_buffer_outlasts_frame_window),
    "a12": ("gt_isolation", a12_gt_isolation),
}

# PROBES THAT ONLY MAKE SENSE AFTER THE STACK IS UP, kept in a SEPARATE dict on purpose.
#
# a9 observes a node that does not exist when the other eight run. Registering it in PROBES
# put it in the DEFAULT set, where it raised KeyError (no entry in `bound`), was recorded
# SKIPPED, and a skipped probe fails the gate -- so run 20260901_143918 was refused before it
# started. The comment said "not part of the pre-start gate" while the code said otherwise,
# and the gate was right to stop it: a check that could not run has asserted nothing.
#
# These are merged in ONLY when named explicitly with --only, so the default gate is exactly
# the eight probes that can answer before the stack exists.
POST_START_PROBES = {
    # a6 MOVED HERE 2026-09-11, and it is not a demotion. It reads the base_link ->
    # habitat_camera offset, which the feed NODE publishes only once frames arrive. Since
    # b6da753 ("Warm VitSAM before releasing the Habitat feed") the feed HOST blocks on
    # $VITSAM_READY_FILE, which perception_2 writes AFTER its warmup -- and perception_2
    # starts AFTER this gate. So in the default set a6 could not pass by construction: it
    # waited 20 s for a transform that waited for a file written later. MEASURED: three
    # launches on 2026-09-11 reported skipped=['a6'], a SKIP fails the gate, and no run
    # could start. live_stack_container.sh now runs it after the warmup gate opens, and it
    # still BLOCKS: a wrong mount height ends the run, at the first moment it can answer.
    "a6": ("camera_pose_offset", a6_camera_pose_offset),
    "a9": ("feed_streaming", a9_feed_streaming),
    "a13": ("pose_authority", a13_pose_authority),
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
    ap.add_argument("--only", help="comma-separated probe ids, e.g. a2,a3 (post-start probes "
                                   "such as a9 are available ONLY through this flag)")
    ap.add_argument("--feed-log", default="/tmp/feed_node.log",
                    help="a9: the feed node's log, whose frame counter is sampled twice")
    ap.add_argument("--pose-source", default=None,
                    help="a13: the stamped pose source (default: FEED_POSE_SOURCE env)")
    ap.add_argument("--feed-window-s", type=float, default=12.0,
                    help="a9: seconds between the two samples")
    ap.add_argument("--live-root", action="append", default=[], metavar="NAME=PATH",
                    help="a tree this deployment mounts LIVE, which a7 must find unchanged. "
                         "Repeatable. The gate ships no default: what must stay frozen is the "
                         "caller's to declare, and a7 SKIPS (and so fails) when none is given.")
    ap.add_argument("--expect-config-name")
    ap.add_argument("--expect-config-sha", help="sha of the config FILE, from the launcher")
    ap.add_argument("--expect-merged-sha", help="sha of the MERGED cfg, from the launcher")
    ap.add_argument("--expect-src-sha", help="K=V,K=V tree digests the launcher recorded "
                                             "(graph_api=…,found=…; a live root named here is compared)")
    ap.add_argument("--teardown", action="store_true",
                    help="a7 run at container close: assert only the LIVE roots against the stamp; "
                         "the vendored mount is released at the copied ping and is reported, not judged")
    ap.add_argument("--found-exercised", default="auto",
                    help="1 if this run executes an extension package (a live-root mismatch then FAILS a7); "
                         "0 under MAPPING_ONLY, where it is recorded and does not block; "
                         "auto (the default) answers it from the config's hooks.filter")
    # Read on the HOST, where the previous bundle exists; this gate runs in the container.
    ap.add_argument("--expect-cycle-s", default="",
                    help="worst frame age the previous run REJECTED, seconds, for a10")
    ap.add_argument("--install-tree", default=INSTALL_TREE,
                    help="where the container copied the node sources; a8 imports from it")
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
    # OBSERVE MODE. A probe named here still RUNS IN FULL and its verdict and detail are recorded
    # exactly as always — only its authority to block the launch is withdrawn.
    #
    # This is not a skip and must never become one. A skipped probe asserted nothing, which is why
    # SKIPPED fails the gate; an observed probe asserted everything it always does, and we read it.
    # The distinction is the whole reason a4 exists: the exemplar baseline shipped as a KG result
    # because an aligner that "could not run" was treated as an aligner that agreed.
    #
    # Used for MAPPING_ONLY runs, where a4 (perception called twice) and a8's perception imports
    # guard a detector that a mapping run deliberately does not start. Scoping their authority to
    # the runs whose purpose they guard is not the same as not looking.
    ap.add_argument("--observe", default="",
                    help="comma-separated probe ids that run and report but do not block")
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

    # Extension probes, loaded from the merged config. Failures here are fatal by design: a
    # probe named in config and silently absent is exactly the shape the gate exists to stop.
    external = {}
    try:
        sys.path.insert(0, "/ws/install/lost3dsg/lib/lost3dsg")
        import config as _cfgmod
        external = load_external_probes(_cfgmod.CFG)
    except ImportError:
        pass          # not in the container; the built-ins still run

    all_probes = dict(PROBES)
    # Post-start probes join the roster only when asked for by name. Without this guard they
    # would run in the pre-start gate, which is where a9's first version broke a run.
    if args.only:
        all_probes.update(POST_START_PROBES)
    all_probes.update({pid: (name, fn) for pid, (name, fn, _spec) in external.items()})

    wanted = [p.strip() for p in args.only.split(",")] if args.only else list(all_probes)
    bound = {
        "a2": lambda: a2_config_identity(args.expect_config_name, args.expect_config_sha,
                                         args.expect_merged_sha),
        "a3": lambda: a3_policy_reached_container(_kv(args.expect_policy)),
        "a4": a4_perception_twice,
        "a5": lambda: a5_bundle_clean(args.run_dir, args.run_start, args.scratch_dir),
        "a6": lambda: a6_camera_pose_offset(args.camera_height),
        "a7": lambda: a7_source_frozen(_kv(args.expect_src_sha),
                                       found_exercised=_found_exercised(args.found_exercised),
                                       teardown=args.teardown),
        "a8": lambda: a8_stack_imports(install=args.install_tree),
        "a9": lambda: a9_feed_streaming(args.feed_log, args.feed_window_s),
        "a13": lambda: a13_pose_authority(args.pose_source),
        # GA-281. REGISTERING A PROBE TAKES TWO EDITS, and the comment beside
        # POST_START_PROBES already says what happens when only one is made: the id lands in
        # the default set, `bound[pid]` raises KeyError, the probe records SKIPPED, and a
        # skipped probe fails the gate. I made exactly that mistake and it refused run
        # 20260903_105431 before it started -- which is the gate working, not failing.
        "a10": lambda: a10_frame_age_rejected_frames(args.expect_cycle_s),
        "a11": a11_tf_buffer_outlasts_frame_window,
        "a12": a12_gt_isolation,
    }

    bound.update({pid: fn for pid, (_n, fn, _s) in external.items()})

    # REGISTRATION IS TWO EDITS AND THIS ASSERTS BOTH WERE MADE. A probe declared in PROBES
    # but absent from `bound` raises KeyError, records SKIPPED, and a skipped probe FAILS the
    # gate -- so a half-registered probe refuses every run with a message about itself rather
    # than about the system. That has now happened twice (a9, then a10). Failing here instead
    # names the real fault in one line, before a container is built.
    _declared = set(PROBES) | set(POST_START_PROBES) | set(external)
    _unbound = sorted(_declared - set(bound))
    if _unbound:
        raise SystemExit(
            f"preflight_gate is misconfigured: probe(s) {_unbound} are declared in PROBES but "
            f"have no entry in `bound`. Every declared probe needs both. This would otherwise "
            f"surface as 'probe raised: KeyError' and refuse the run.")

    observe = {p.strip() for p in (args.observe or "").split(",") if p.strip()}
    _unknown = observe - set(all_probes)
    if _unknown:
        # A typo here silently grants no exemption and looks like it did. Refuse instead.
        print(f"!! --observe names probes that do not exist: {sorted(_unknown)}")
        return 2
    results, failed, skipped, observed_nonpass = [], [], [], []
    for pid in wanted:
        name = all_probes[pid][0]
        try:
            ok, detail = bound[pid]()
        except Exception as exc:
            ok, detail = SKIPPED, {"reason": f"probe raised: {type(exc).__name__}: {exc}"}
        row = {"id": pid, "name": name, "ok": ok, "detail": detail}
        if pid in external:
            # WHICH implementation answered, recorded beside its verdict — the same question
            # a1 asks about the aligner, asked about the probe itself.
            row["supplied_by"] = external[pid][2]
        results.append(row)
        mark = "PASS" if ok else ("SKIP" if ok is SKIPPED else "FAIL")
        print(f"  [{mark}] {pid} {name}: {json.dumps(detail, default=str)[:200]}")
        if pid in observe:
            # Recorded on the row, so a mapping bundle's gate record cannot be read as a
            # detection run's. A reader seeing verdict "pass" must be able to see which probes
            # were not permitted to say otherwise.
            row["observed"] = True
            if ok is not True:
                observed_nonpass.append(pid)
        elif ok is False:
            failed.append(pid)
        elif ok is SKIPPED:
            skipped.append(pid)

    verdict = "pass" if not failed and (args.allow_skip or not skipped) else "fail"
    report = {"verdict": verdict, "failed": failed, "skipped": skipped,
              "observed": sorted(observe), "observed_nonpass": observed_nonpass,
              "observed_note": "these probes RAN IN FULL and their verdicts are recorded above; "
                               "they were not permitted to block this launch. A non-empty "
                               "observed list means this verdict is narrower than a normal one.",
              "probes": results}
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
    # Register under the import name BEFORE running. As a script this module is `__main__`,
    # so an extension that does `sys.modules["preflight_gate"]` finds nothing, falls through
    # to loading the file, and gets a SECOND module object — whose `Probe` is a different
    # class from this one. `hooks.load_hook`'s isinstance check then fails, the gate aborts
    # on a correct probe, and the obvious reading is that the probe is broken. It is not:
    # only module identity is wrong, and the next person edits the wrong file.
    #
    # A seam is tested from one end and used from the other, so it must not depend on every
    # extension author getting module identity right. This is the gate's half of that.
    sys.modules.setdefault("preflight_gate", sys.modules["__main__"])
    sys.exit(main())
