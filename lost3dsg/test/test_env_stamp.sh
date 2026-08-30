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
SRC_SHA=aaaa1111 SRC_N=122 FOUND_SHA=bbbb2222 FOUND_N=40 KB_SHA=cccc3333 KB_N=17 \
KB_SRC=/DATA/ASPIRE/knowledge_bridge \
IMAGE_TAG=img IMAGE_DIGEST=sha256:dead ENC_E5=e5 ENC_MINILM=mini \
GT_PATH=/gt/hm3d_00861.json GT_SHA=beef1234 GT_N=870 \
FOUND_ENFORCE=1 FOUND_HOLD_BAND=0.05 FOUND_MIN_SUPPORT=30 FOUND_ROOM_ENFORCE=0 \
FOUND_ALIGNER=kg FOUND_ONTOLOGY_EXT=default \
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
grep -q '"kb_src_sha256_16": "cccc3333"'        "$TMP/meta.json" || fail "knowledge_bridge digest not stamped — it is on PYTHONPATH and was unhashed"
grep -q '"graph_api_files": 122'                "$TMP/meta.json" || fail "file count not stamped; a count is what makes an empty root visible"
grep -q '"image_digest": "sha256:dead"'         "$TMP/meta.json" || fail "image digest not stamped"
grep -q '"intfloat/e5-small-v2": "e5"'          "$TMP/meta.json" || fail "encoder revision not stamped"
grep -q '"n_objects": "870"'                    "$TMP/meta.json" || fail "ground-truth object count not stamped"
grep -q '"sha256_16": "beef1234"'               "$TMP/meta.json" || fail "ground-truth hash not stamped"
grep -q '"merged_sha256_16": "44c1d0aa9b3e2f57"' "$TMP/meta.json" || fail "merged-config sha not stamped; a file hash alone misses a _DEFAULTS change"

# 7. The two loaders are recorded separately. One field cannot describe two processes, and a
#    single config_name is what let a run report a configuration only half of it used.
grep -q '"config_resolved"'   "$TMP/meta.json" || fail "the feed host's resolved config is not recorded"
grep -q '"provenance_intent"' "$TMP/meta.json" || fail "the pre-run stamp must be labelled as intent, not as confirmation"
grep -q '"live_roots"'        "$TMP/meta.json" || fail "found and kb are live mounts and must be marked sampled, not frozen"

echo "test_env_stamp.sh: OK (parse, cache layouts, ORDERING, JSON validity, keys kept, provenance)"
