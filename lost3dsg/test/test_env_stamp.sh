#!/usr/bin/env bash
# Runnable check for the environment stamping added to live_run.sh.
#   ./test_env_stamp.sh
#
# Reads the REAL live_run.sh rather than a copy — a test that restates the logic drifts
# from it silently, which is how the CFG_NAME bug survived. Everything here is extracted
# from the live file at run time.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC="$HERE/live_run.sh"
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

# 1. The whole script still parses.
bash -n "$SRC" || fail "live_run.sh does not parse"

# 2. _enc_rev resolves BOTH cache layouts. /DATA/FOUND/.hf_cache holds MiniLM flat AND
#    under hub/ at the same revision today; which one loads depends on the env var the
#    process sets, so a resolver that only knew one layout would return "unknown" for a
#    model that is present — a silent gap in the provenance record, not a loud one.
HF_CACHE="$TMP/cache"
mkdir -p "$HF_CACHE/hub/models--org--hubmodel/refs" "$HF_CACHE/models--org--flatmodel/refs"
echo -n "aaaa1111" > "$HF_CACHE/hub/models--org--hubmodel/refs/main"
echo -n "bbbb2222" > "$HF_CACHE/models--org--flatmodel/refs/main"

# Extract between explicit markers in live_run.sh. Two earlier versions guessed the boundary
# with a sed pattern and both were wrong — the second ran to end-of-file because the range's
# start line also matched its own terminator. A test that infers where its subject ends drifts
# from it silently, which is the reason this file reads the real script instead of a copy.
eval "$(sed -n '/# >>> TEST-EXTRACT _enc_rev/,/# <<< TEST-EXTRACT _enc_rev/p' "$SRC" | grep -v TEST-EXTRACT)"
[ -n "$(type -t _enc_rev)" ] || fail "_enc_rev not extracted from live_run.sh"

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
        assigned = any(re.match(rf'\s*(export\s+)?{name}=', l) or re.search(rf'read -r [\w ]*\b{name}\b', l)
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
SRC_SHA=aaaa1111 SRC_N=122 FOUND_SHA=bbbb2222 FOUND_N=40 \
IMAGE_TAG=img IMAGE_DIGEST=sha256:dead ENC_E5=e5 ENC_MINILM=mini \
GT_PATH=/gt/hm3d_00861.json GT_SHA=beef1234 GT_N=870 \
FOUND_ENFORCE=1 FOUND_HOLD_BAND=0.05 FOUND_MIN_SUPPORT=30 FOUND_ROOM_ENFORCE=0 \
FOUND_ALIGNER=kg FOUND_ONTOLOGY_EXT=default \
FEED_WALK=6 FEED_DWELL=0 FEED_FPS=3 FEED_MAPPING_SECONDS=150 FEED_OVERLAY=1 FEED_SHOW=1 \
MAPPING_ONLY=0 \
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
#     because found/kg_align.py reads FOUND_ONTOLOGY_EXT as a PATH when non-empty and raises if
#     that path is absent. Exporting the label as the value made a1 fail every gated run with
#     `FOUND_ONTOLOGY_EXT set to default, which does not exist`. The two must not be re-merged.
FOUND_ONTOLOGY_EXT="" bash -c 'v="${FOUND_ONTOLOGY_EXT:-default}"; [ "$v" = "default" ]' \
  || fail "an empty FOUND_ONTOLOGY_EXT must still record the label 'default' in the bundle"
grep -q 'export FOUND_ONTOLOGY_EXT="\${FOUND_ONTOLOGY_EXT:-}"' "$SRC" \
  || fail "FOUND_ONTOLOGY_EXT must default to EMPTY, not to the literal string 'default'"

# 7. The two loaders are recorded separately. One field cannot describe two processes, and a
#    single config_name is what let a run report a configuration only half of it used.
grep -q '"config_resolved"'   "$TMP/meta.json" || fail "the feed host's resolved config is not recorded"
grep -q '"provenance_intent"' "$TMP/meta.json" || fail "the pre-run stamp must be labelled as intent, not as confirmation"
grep -q '"live_roots"'        "$TMP/meta.json" || fail "found and kb are live mounts and must be marked sampled, not frozen"

# 8. How the agent moved is IN the bundle. Before 2026-08-31 only the seed was recorded, so a
#    dwell=0 run and a dwell=60 run produced byte-identical metadata and the difference between
#    two bundle FAMILIES lived only in whichever message announced it. 0 is a legal value and
#    must survive as 0 — a `${VAR:-60}` anywhere on this path turns the new default back into
#    the old one and stamps the lie in the artefact.
grep -q '"dwell_frames": 0' "$TMP/meta.json" || fail "dwell_frames not stamped, or a :- default rewrote the 0"
grep -q '"walk_frames": 6'  "$TMP/meta.json" || fail "walk_frames not stamped"
grep -q '"mapping_seconds": 150' "$TMP/meta.json" || fail "mapping_seconds not stamped; the first 150s of every run ignore walk/dwell entirely"
# A mapping run's bundle must say so. Without this a MAPPING_ONLY bundle with zero detections and
# a detection run that found nothing are the same artefact -- the indistinguishability that cost
# run 19 its merge question.
grep -q '"mapping_only": false' "$TMP/meta.json" || fail "mapping_only not stamped for a normal run"
# GA-33 residual: the container reads FOUND_KG_TOP / FOUND_KG_Z (found/kg_align.py); the bundle must say what they were.
grep -q '"kg_top": 0.87' "$TMP/meta.json" || fail "kg_top not stamped (code default 0.87 when FOUND_KG_TOP is empty)"
grep -q '"kg_z": 3.0'    "$TMP/meta.json" || fail "kg_z not stamped (code default 3.0 when FOUND_KG_Z is empty)"
grep -q 'export OUT_DIR=' "$SRC" || fail "GA-99: OUT_DIR must be EXPORTED or the feed host never sees it and writes its stats outside the bundle"
grep -q 'export FEED_DWELL="\${FEED_DWELL:-0}"' "$SRC" \
  || fail "FEED_DWELL must default to 0 (owner ruling 2026-08-31). A 60 here silently re-bases the family."
grep -q 'FEED_DWELL=\${FEED_DWELL:-60}' "$SRC" \
  && fail "a second FEED_DWELL default survives on the launch line; one name, one default"

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
# 8. The run's live output path. RESULTS/, never /tmp — owner ruling, relayed. The path needs
#    RUN_TIMESTAMP and SCENE_ARG, both defined 99 lines below where OUT_DIR used to sit, so the
#    assignment moved rather than the value changing. Evaluated here rather than eyeballed.
#    FOUND_ROOT is now DERIVED from the script's location so a clone anywhere can run, so this
#    supplies one rather than expecting the machine this was written on.
_out=$(RUN_TIMESTAMP=20260831_140000 SCENE_ARG=hm3d_00861 FOUND_ROOT=/tmp/fake_found bash -c \
       'eval "$(sed -n "/^export OUT_DIR=\${OUT_DIR:-\$FOUND_ROOT\/results/p" '"$SRC"')"; echo "$OUT_DIR"')
[ "$_out" = "/tmp/fake_found/results/20260831_140000_hm3d_00861" ] \
  || fail "OUT_DIR default is '$_out', expected \$FOUND_ROOT/results/<timestamp>_<scene>"

#    and an explicit OUT_DIR must still win, because the scratch-rename guard exists for the
#    operator who reuses one. The default got safer; the hazard did not go away.
_ovr=$(RUN_TIMESTAMP=t SCENE_ARG=s OUT_DIR=/tmp/explicit bash -c \
       'eval "$(sed -n "/^export OUT_DIR=\${OUT_DIR:-\/DATA\/FOUND\/results/p" '"$SRC"')"; echo "$OUT_DIR"')
[ "$_ovr" = "/tmp/explicit" ] || fail "an explicit OUT_DIR must override the default, got '$_ovr'"

echo "test_env_stamp.sh: OK (parse, cache layouts, ORDERING, JSON validity, keys kept, provenance) | $(date -Iseconds)"
