#!/usr/bin/env bash
# Runnable check for the environment stamping added to run.sh.
#   ./test_env_stamp.sh
#
# Reads the REAL run.sh rather than a copy — a test that restates the logic drifts
# from it silently, which is how the CFG_NAME bug survived. Everything here is extracted
# from the live file at run time.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/../../run.sh"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

# 1. The whole script still parses.
bash -n "$SRC" || fail "run.sh does not parse"

# 2. _enc_rev resolves BOTH cache layouts. A host cache can hold MiniLM flat AND
#    under hub/ at the same revision today; which one loads depends on the env var the
#    process sets, so a resolver that only knew one layout would return "unknown" for a
#    model that is present — a silent gap in the provenance record, not a loud one.
HF_CACHE="$TMP/cache"
mkdir -p "$HF_CACHE/hub/models--org--hubmodel/refs" "$HF_CACHE/models--org--flatmodel/refs"
echo -n "aaaa1111" > "$HF_CACHE/hub/models--org--hubmodel/refs/main"
echo -n "bbbb2222" > "$HF_CACHE/models--org--flatmodel/refs/main"

# Extract between explicit markers in run.sh. Two earlier versions guessed the boundary
# with a sed pattern and both were wrong — the second ran to end-of-file because the range's
# start line also matched its own terminator. A test that infers where its subject ends drifts
# from it silently, which is the reason this file reads the real script instead of a copy.
eval "$(sed -n '/# >>> TEST-EXTRACT _enc_rev/,/# <<< TEST-EXTRACT _enc_rev/p' "$SRC" | grep -v TEST-EXTRACT)"
[ -n "$(type -t _enc_rev)" ] || fail "_enc_rev not extracted from run.sh"

[ "$(_enc_rev org--hubmodel)"  = "aaaa1111" ] || fail "hub/ layout not resolved"
[ "$(_enc_rev org--flatmodel)" = "bbbb2222" ] || fail "flat layout not resolved"
[ "$(_enc_rev org--absent)"    = "unknown"  ] || fail "missing model must read 'unknown', not empty"

# 3. ORDERING. Every variable the metadata heredoc interpolates must be assigned EARLIER in the
#    script than the heredoc itself. This is the defect check 4 below cannot see: check 4 runs
#    the extracted heredoc with every variable supplied, so it validates the TEMPLATE and is
#    blind to the SEQUENCE. The policy exports sat 78 lines below the heredoc, interpolated
#    empty, and wrote `"enforce": ,` — invalid JSON on every run, on both configs. A test that
#    can never fail for the defect it is nearest to is the shape this whole gate exists for.
python3 - "$SRC" <<'ORDER' || fail "a variable is interpolated into run_metadata.json before it is assigned"
import re, sys
lines = open(sys.argv[1]).read().split("\n")
start = next(i for i, l in enumerate(lines)
             if l.startswith('cat <<EOF > "$RUN_DIR/run_metadata.json"'))
end = next(i for i, l in enumerate(lines) if i > start and l == "EOF")
bad = []
for line in lines[start:end]:
    for m in re.finditer(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)(:-[^}]*)?\}?', line):
        name, has_default = m.group(1), m.group(2)
        if has_default:
            continue
        # GUARDED ON THE SAME LINE. `$([ -n "${X:-}" ] && echo "\"$X\"" || echo null)` reaches the
        # bare $X only when the guard already found it set, so an unassigned X yields null and not
        # a torn JSON value. Without this the check flagged house_id and localize_db, which have
        # been written that way since they were added, and it has been RED at HEAD because of it.
        if ("${%s:-" % name) in line:
            continue
        # re.search, not re.match: `SEED_SOURCE=pinned; SCENE_SOURCE=pinned` assigns two names on
        # one line and the anchored pattern saw only the first. That is the third false positive.
        assigned = any(re.search(rf'(^|[;&|])\s*(export\s+)?{name}=', l) or re.search(rf'read -r [\w ]*\b{name}\b', l)
                       for l in lines[:start])
        if not assigned:
            bad.append(name)
if bad:
    print("interpolated but never assigned above the heredoc: " + ", ".join(sorted(set(bad))))
    sys.exit(1)
ORDER

# 4. run_metadata.json is still valid JSON once the new environment block is in it. The
#    block interpolates eight shell values into a heredoc; one stray quote makes every
#    downstream analysis tool fail on a bundle that otherwise looks complete.
RUN_ID=t SCENE_ARG=s CFG_NAME=c.yaml CFG_SHA=0f9e8d7c6b5a4938 MERGED_SHA=44c1d0aa9b3e2f57 \
FEED_SEED=7 RUN_DIR=/x HERE=/here \
SRC_SHA=aaaa1111 SRC_N=122 EXT_SRC_SHAS=ext=bbbb2222 \
IMAGE_TAG=img IMAGE_DIGEST=sha256:dead ENC_E5=e5 ENC_MINILM=mini \
GT_PATH=/gt/hm3d_00861.json GT_SHA=beef1234 GT_N=870 \
EXT_POLICY_JSON='"enforce": 1, "hold_band": 0.05,' \
FEED_FPS=3 FEED_OVERLAY=1 FEED_SHOW=1 \
FEED_SCHEDULE=/sched/hm3d_00861.schedule.json FEED_EXPLORATION_LAPS=3 FEED_MOVE_FN=navigate \
ROOM_FRAME_MAX=5 ROOM_FRAME_STRIDE_M=1.5 FEED_POSE_SOURCE=simulator \
FEED_CAMERA_PITCH_DEG=-30 SEED_SOURCE=pinned SCENE_SOURCE=pinned \
  bash -c "$(sed -n '/^cat <<EOF > "\$RUN_DIR\/run_metadata.json"/,/^EOF$/p' "$SRC" \
             | sed 's|> "\$RUN_DIR/run_metadata.json"||')" > "$TMP/meta.json"
python3 -m json.tool "$TMP/meta.json" > /dev/null || { cat "$TMP/meta.json"; fail "run_metadata.json is not valid JSON"; }

# 5. The five original keys keep their bytes. Rule 6: add keys, never change or remove them —
#    three consumers read this file by key and some do arithmetic on the values.
for k in run_id scene start_time config_name output_dir; do
  grep -q "\"$k\":" "$TMP/meta.json" || fail "original key '$k' was dropped from run_metadata.json"
done

# 6. The new provenance keys are present and carry the values they were given.
grep -q '"graph_api_src_sha256_16": "aaaa1111"' "$TMP/meta.json" || fail "graph-api tree digest not stamped"
grep -q '"kb_src_sha256_16": null'              "$TMP/meta.json" || fail "kb_src_sha256_16 must be an explicit null since GA-306 — the key stays so a reader can tell 'not applicable' from 'never stamped'"
grep -q '"graph_api_files": 122'                "$TMP/meta.json" || fail "file count not stamped; a count is what makes an empty root visible"
grep -q '"image_digest": "sha256:dead"'         "$TMP/meta.json" || fail "image digest not stamped"
grep -q '"intfloat/e5-small-v2": "e5"'          "$TMP/meta.json" || fail "encoder revision not stamped"
grep -q '"n_objects": "870"'                    "$TMP/meta.json" || fail "ground-truth object count not stamped"
grep -q '"sha256_16": "beef1234"'               "$TMP/meta.json" || fail "ground-truth hash not stamped"
grep -q '"merged_sha256_16": "44c1d0aa9b3e2f57"' "$TMP/meta.json" || fail "merged-config sha not stamped; a file hash alone misses a _DEFAULTS change"

# 6b. The policy block records the human LABEL "default"; the process receives the empty string,
#     because an extension read that variable as a PATH when non-empty and raised if
#     that path is absent. Exporting the label as the value made a1 fail every gated run with
#     `... set to default, which does not exist`. The two must not be re-merged.
# The ontology-extension default and the aligner thresholds are an EXTENSION's to declare and to
# test. They used to be asserted here, against variables this launcher exported -- so this file
# tested a policy vocabulary the repository no longer carries. What is asserted instead is the
# SEAM: that the launcher forwards whatever the extension declared, and stamps it.
grep -q 'for _v in ${EXT_ENV_PASS:-}; do EXT_E_ARGS=' "$SRC" \
  || fail "the launcher must forward every variable the extension declared in EXT_ENV_PASS"
grep -q 'EXT_MOUNT_POINT="${EXT_MOUNT_POINT:-/ext}"' "$SRC" \
  || fail "the extension mount point must be a defaulted variable, never a written path"
grep -q '"policy": {${EXT_POLICY_JSON:-}' "$SRC" \
  || fail "the extension's policy keys must be interpolated INSIDE policy, so the bundle shape is unchanged"

# 7. The two loaders are recorded separately. One field cannot describe two processes, and a
#    single config_name is what let a run report a configuration only half of it used.
grep -q '"config_resolved"'   "$TMP/meta.json" || fail "the feed host's resolved config is not recorded"
grep -q '"provenance_intent"' "$TMP/meta.json" || fail "the pre-run stamp must be labelled as intent, not as confirmation"
grep -q '"live_roots"'        "$TMP/meta.json" || fail "found and kb are live mounts and must be marked sampled, not frozen"

# 8. How the agent moved is IN the bundle. Before 2026-08-31 only the seed was recorded, so two
#    runs with different motion produced byte-identical metadata and the difference between two
#    bundle FAMILIES lived only in whichever message announced it.
#
#    THE SAMPLING POLICY IS REMOVED (owner 2026-09-11), so what has to be stamped is the schedule:
#    which file, how many laps, and how the agent travelled between two stops. walk_frames,
#    dwell_frames, dwell_mode, mapping_seconds and mapping_only are gone from the artefact with
#    the policy they described.
grep -q '"motion_policy": "schedule"' "$TMP/meta.json" || fail "motion_policy not stamped; a bundle must say which policy drove it, because the archive holds both"
grep -q '"schedule": "/sched/hm3d_00861.schedule.json"' "$TMP/meta.json" || fail "the schedule file is not stamped, so nothing says which roadmap this run drove"
grep -q '"exploration_laps": 3' "$TMP/meta.json" || fail "exploration_laps not stamped"
grep -q '"navigation_mode": "navigate"' "$TMP/meta.json" || fail "navigation_mode not stamped; teleport and navigate produce different frame counts for the same route"
for k in walk_frames dwell_frames dwell_mode mapping_seconds mapping_only; do
  grep -q "\"$k\":" "$TMP/meta.json" \
    && fail "$k is still stamped; it belongs to the sampling policy, which was removed on 2026-09-11"
done
grep -q 'export OUT_DIR=' "$SRC" || fail "GA-99: OUT_DIR must be EXPORTED or the feed host never sees it and writes its stats outside the bundle"
# ASSIGNMENTS ONLY, not the word. The launcher keeps a comment naming these variables to say why
# they are gone; a grep for the bare name would fail on the explanation itself.
grep -qE '^[[:space:]]*(export[[:space:]]+)?(FEED_WALK|FEED_DWELL[A-Z_]*|FEED_TEST_TOUR[A-Z_]*|FEED_MAPPING_SECONDS)=' "$SRC" \
  && fail "the launcher still assigns a walk/dwell/mapping name; the feed host refuses those names now"

# 9. No comment sits between two continued lines. `A=1 \` followed by `# ...` does NOT comment
#    the line — the # swallows the continuation and A is SILENTLY DROPPED, with `bash -n` clean.
#    Measured on 2026-08-31: I wrote one above the feed-host invocation and it would have thrown
#    away HABITAT_SCENE and HABITAT_DATASET while the bundle still stamped the requested scene.
awk 'prev ~ /\\$/ && $0 ~ /^[[:space:]]*#/ { print FILENAME ":" NR ": comment after a line continuation"; bad=1 } { prev=$0 } END { exit bad }' \
  "$SRC" || fail "a comment follows a line continuation; the preceding assignments are silently dropped"

# 10. EVERY VARIABLE THE CONTAINER READS MUST REACH IT. GA-105, found by the testing lane before
#     a launch rather than after: RTABMAP_LOCALIZE_DB was read at live_stack_container.sh:129 and
#     absent from `docker run -e`, so GA-97's localization branch was correct and UNREACHABLE.
#     Checking systematically rather than fixing the one instance found TWO more — MAPPING_ONLY
#     and FEED_MAPPING_SECONDS — so tonight's mapping run would have silently started a detector,
#     spent cloud money and published no map, while reporting itself a mapping run.
#
#     The failure shape is why this is a test and not a fix: an unset variable takes the default
#     branch SILENTLY, so "the feature is off" and "the feature never arrived" are the same
#     observation. Same class as the /tmp fallback that hid GA-99.
_missing=""
for _v in $(grep -o '\${[A-Z_][A-Z0-9_]*[:-]*[^}]*}' "$HERE/live_stack_container.sh" \
            | sed 's/\${\([A-Z_][A-Z0-9_]*\).*/\1/' | sort -u); do
  case "$_v" in _|RTABMAP_PID|OM6_PID|WALLS_PID|PERCEPTION_PID|LOG_DIR) continue;; esac
  # set inside the container is fine; otherwise it must be on the docker run -e list
  grep -qE "^ *(export )?${_v}=" "$HERE/live_stack_container.sh" && continue
  grep -q -- "-e ${_v}\b" "$SRC" || _missing="$_missing $_v"
done
[ -z "$_missing" ] || fail "read inside the container but never passed by docker run -e:$_missing"
# GA-33: the same rule for CONTAINER-SIDE PYTHON (os.environ reads), which the shell grep above
#        cannot see. check_env_passthrough.py existed and nothing ran it; now this does.
python3 "$HERE/check_env_passthrough.py" "$HERE/.." >/dev/null \
  || fail "check_env_passthrough.py: a container-side python os.environ read is not on the docker run -e list"

# Stamped: a test result is true at a time, not simply true.
# 8. The run's live output path. results/, never /tmp — owner ruling, relayed.
#
#    THIS CHECK HAS BEEN VACUOUS. Its sed looked for `export OUT_DIR=${OUT_DIR:-$WORKSPACE_ROOT/
#    results...}`, a form the launcher has not used since OUT_DIR became `"$RUN_DIR"`; the sed
#    matched nothing, `_out` was empty, and the comparison was against `/tmp/fake_found/...` while
#    the run supplied `WORKSPACE_ROOT=/tmp/fake_ws` — a literal left over from the workspace layout
#    this repository used before the consolidation. It could not pass for any launcher, correct or not.
#
#    It now evaluates the three lines that actually build the path, which is what the check was
#    always about: RUN_ID from the timestamp and the scene, RESULTS_DIR under the repo, RUN_DIR
#    under that, OUT_DIR equal to it.
_out=$(RUN_TIMESTAMP=20260831_140000 SCENE_ARG=hm3d_00861 REPO=/tmp/fake_ws bash -c \
       'eval "$(sed -n -e "/^RUN_ID=/p" -e "/^RESULTS_DIR=/p" -e "/^RUN_DIR=/p" -e "/^export OUT_DIR=/p" '"$SRC"')"; echo "$OUT_DIR"')
[ "$_out" = "/tmp/fake_ws/results/20260831_140000_hm3d_00861" ] \
  || fail "OUT_DIR default is '$_out', expected <repo>/results/<timestamp>_<scene>"

#    and an explicit OUT_DIR must still win, because the scratch-rename guard exists for the
#    operator who reuses one. The default got safer; the hazard did not go away.
_ovr=$(RUN_TIMESTAMP=t SCENE_ARG=s OUT_DIR=/tmp/explicit bash -c \
       'eval "$(sed -n "/^export OUT_DIR=\${OUT_DIR:-\/tmp\/fake_ws\/results/p" '"$SRC"')"; echo "$OUT_DIR"')
[ "$_ovr" = "/tmp/explicit" ] || fail "an explicit OUT_DIR must override the default, got '$_ovr'"

echo "test_env_stamp.sh: OK (parse, cache layouts, ORDERING, JSON validity, keys kept, provenance) | $(date -Iseconds)"

# THE BOUNDARY, CHECKED. A ruling with no check is a preference: the 2026-08-28 ruling that this
# launcher names nothing of an extension's survived eleven days and reversed itself into 112
# references, which then reached this repository through an ordinary merge.
python3 "$HERE/check_no_extension_refs.py" \
  || fail "this repository names a package that extends it; move it behind the EXT_* seam"
