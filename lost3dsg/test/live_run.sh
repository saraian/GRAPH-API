#!/usr/bin/env bash
# Live demo on this machine: habitat renders on the host (conda habitat_env),
# the ROS 2 stack runs in the graphapi-run:humble container over a TCP feed.
# Watch: web viewer at http://localhost:8081 and snapshots in $OUT_DIR.
#   ./live_run.sh [scene]  # foreground; ctrl-C stops everything
# scene: hm3d_00861 (default) | hm3d_00337 | hm3d_00770 | mp3d_17DRP
# HABITAT_SCENE/HABITAT_DATASET env vars still override everything.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT_DIR=${OUT_DIR:-/tmp/graphapi_live}
mkdir -p "$OUT_DIR"

MON_PID=""
FEED_PID=""
cleanup() {
  # PID-scoped: only kill what THIS instance started. A pattern-based
  # pkill here would let two concurrent live_run.sh instances destroy
  # each other's feed host (seen live 2026-08-25).
  [ -n "$MON_PID" ] && kill -9 "$MON_PID" 2>/dev/null || true
  [ -n "$FEED_PID" ] && kill -9 "$FEED_PID" 2>/dev/null || true
  # Archive the HOST-written artefacts here rather than only after the container exits. The
  # per-frame viewpoint series and the feed host's own log are written on this side, and the
  # post-run copy below never runs when the script is stopped with Ctrl-C — which is the
  # documented way to stop it. feed_host.log reached 1 of 21 shipped bundles for this reason.
  if [ -n "${RUN_DIR:-}" ] && [ -d "$RUN_DIR" ]; then
    cp "$OUT_DIR"/*.json "$OUT_DIR"/*.jsonl "$RUN_DIR/" 2>/dev/null || true
    cp "$OUT_DIR"/*.log "$RUN_DIR/logs/" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Refuse to start over a live stack instead of killing it (fixed container
# name and fixed ports mean two instances can never coexist).
if docker ps --filter name=graphapi_live --format '{{.Names}}' | grep -q .; then
  echo "!! container graphapi_live is already running — aborting (stop it first: docker rm -f graphapi_live)"; exit 1
fi
if ss -tln 2>/dev/null | grep -q ':7799 '; then
  echo "!! port 7799 busy — a feed host is already running — aborting"; exit 1
fi

echo ">>> host habitat feed (scene renders on the host GPU)"
# Matterport scenes on this machine: three HM3D v0.2 examples (semantic-annotated)
# plus the MP3D example. Pick with the first argument, override with env vars.
HM3D_ROOT=${HM3D_ROOT:-/DATA/habitat_matterport/hm3d_example}
MP3D_ROOT=${MP3D_ROOT:-/DATA/habitat_matterport/versioned_data/mp3d_example_scene_1.1}
case "${1:-hm3d_00861}" in
  hm3d_00861) DEF_SCENE=$HM3D_ROOT/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb
              DEF_DATASET=$HM3D_ROOT/hm3d_annotated_basis.scene_dataset_config.json ;;
  hm3d_00337) DEF_SCENE=$HM3D_ROOT/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb
              DEF_DATASET=$HM3D_ROOT/hm3d_annotated_basis.scene_dataset_config.json ;;
  hm3d_00770) DEF_SCENE=$HM3D_ROOT/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb
              DEF_DATASET=$HM3D_ROOT/hm3d_annotated_basis.scene_dataset_config.json ;;
  mp3d_17DRP) DEF_SCENE=$MP3D_ROOT/17DRP5sb8fy/17DRP5sb8fy.glb
              DEF_DATASET=$MP3D_ROOT/mp3d.scene_dataset_config.json ;;
  *) echo "unknown scene '$1' (hm3d_00861|hm3d_00337|hm3d_00770|mp3d_17DRP)"; exit 1 ;;
esac
# VLM config: real regolo endpoint when a key is present, offline smoke fallback
if [ -n "${REGOLO_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY:-$REGOLO_API_KEY}"
fi
if [ -n "${OPENAI_API_KEY:-}" ]; then
  CFG_NAME=${CFG_NAME:-regolo_config.yaml}
else
  CFG_NAME=${CFG_NAME:-smoke_config.yaml}
fi
# EXPORTED, because `docker run -e CFG_NAME` copies the parent process environment and a
# shell variable that was only assigned is not in it. Without this the echo below prints
# regolo while the container falls back to smoke_config.yaml — the operator reads one
# config and the run loads another, and every bundle records the name that was printed.
# The two files are not interchangeable: smoke omits vlm.base_url and vlm.model, so both
# fall to the defaults (localhost:11434, gemma4:e2b) and an API key aimed at regolo
# reaches a local socket instead.
export CFG_NAME
echo "    config: $CFG_NAME"

# Setup persistent FOUND run bundle (never overwritten across runs)
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCENE_ARG=${1:-hm3d_00861}
# Stamped BEFORE the bundle directory is created, so every artefact the run legitimately
# writes is newer than it. The gate's a5 probe fails on anything older — a scratch directory
# left dirty by the previous run, or a file copied in by hand.
RUN_START_EPOCH=$(date +%s)
export RUN_START_EPOCH

# The scratch directory is never cleared between runs, and its *.json and *.jsonl are copied
# into the bundle below. A run that dies early therefore archives its PREDECESSOR's files
# under its own source hashes. Rename rather than delete: if the previous run's archiving
# failed, this is the only copy of its artefacts, and an inode costs nothing against that.
if [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  mv "$OUT_DIR" "${OUT_DIR}.prev-${RUN_START_EPOCH}"
  mkdir -p "$OUT_DIR"
  echo "    scratch: previous contents moved to ${OUT_DIR}.prev-${RUN_START_EPOCH}"
fi

RUN_ID="${RUN_TIMESTAMP}_${SCENE_ARG}"
FOUND_RUNS_DIR=${FOUND_RUNS_DIR:-/DATA/FOUND/runs}
RUN_DIR="$FOUND_RUNS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/crops" "$RUN_DIR/snapshots"
echo "    run bundle: $RUN_DIR (symlinked as $FOUND_RUNS_DIR/latest)"

# Snapshot calibration, config, and run metadata
cat <<'EOF' > "$RUN_DIR/calibration.json"
{
  "camera_name": "habitat_camera_optical",
  "resolution": {"width": 640, "height": 480},
  "hfov_deg": 90.0,
  "intrinsics": {
    "fx": 320.0,
    "fy": 320.0,
    "cx": 320.0,
    "cy": 240.0
  },
  "distortion_model": "plumb_bob",
  "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0]
}
EOF
# The config the FEED HOST will load. It resolves $HERE/$CFG_NAME through the same
# config.py::_load as the container, and _load returns the DEFAULTS when the file is absent —
# silently, with hooks.filter empty. Two processes, one name, and nothing recording the split.
[ -f "$HERE/$CFG_NAME" ] || { echo "!! $HERE/$CFG_NAME does not exist — the feed host would run on config.py defaults; aborting"; exit 1; }
cp "$HERE/$CFG_NAME" "$RUN_DIR/config.yaml"

# These sit ABOVE run_metadata.json rather than beside `docker run`, because the metadata
# heredoc interpolates all six. They were 78 lines below it and interpolated empty, so the
# policy block was written as `"enforce": ,` and the validator aborted every run. A shell
# variable's VALUE depends on where it is read, and a heredoc is read where it is written.

# Every knob the admission policy reads, EXPORTED with its default rather than forwarded bare.
# `docker run -e VAR` sends nothing when VAR is unset in the parent environment, so a bare
# forward silently ships the container's own idea of the default while the bundle records the
# launcher's. Exporting here makes the two the same value, and it is what lets the pre-flight
# gate compare an intention against what actually arrived instead of echoing what it finds.
export FOUND_ENFORCE="${FOUND_ENFORCE:-0}"
export FOUND_HOLD_BAND="${FOUND_HOLD_BAND:-0.05}"
export FOUND_MIN_SUPPORT="${FOUND_MIN_SUPPORT:-30}"
export FOUND_ROOM_ENFORCE="${FOUND_ROOM_ENFORCE:-0}"
export FOUND_ALIGNER="${FOUND_ALIGNER:-kg}"
export FOUND_ONTOLOGY_EXT="${FOUND_ONTOLOGY_EXT:-default}"
export FOUND_STORE_PATH="${FOUND_STORE_PATH:-/ws/output/knowledge_graph.ttl}"
export FOUND_SCENE="${FOUND_SCENE:-$SCENE_ARG}"

# What the launcher INTENDS the policy to be. The gate compares this against the environment
# actually present inside the container, rather than echoing whatever it finds there — an echo
# is what let an "enforcing" run be a pass-through for weeks.
PREFLIGHT_EXPECT_POLICY="FOUND_ENFORCE=$FOUND_ENFORCE,FOUND_HOLD_BAND=$FOUND_HOLD_BAND"
PREFLIGHT_EXPECT_POLICY="$PREFLIGHT_EXPECT_POLICY,FOUND_MIN_SUPPORT=$FOUND_MIN_SUPPORT"
PREFLIGHT_EXPECT_POLICY="$PREFLIGHT_EXPECT_POLICY,FOUND_ROOM_ENFORCE=$FOUND_ROOM_ENFORCE"
PREFLIGHT_EXPECT_POLICY="$PREFLIGHT_EXPECT_POLICY,FOUND_ALIGNER=$FOUND_ALIGNER"
export PREFLIGHT_EXPECT_POLICY
echo "    policy: enforce=$FOUND_ENFORCE hold_band=$FOUND_HOLD_BAND \
min_support=$FOUND_MIN_SUPPORT rooms_enforced=$FOUND_ROOM_ENFORCE \
aligner=$FOUND_ALIGNER ontology_ext=$FOUND_ONTOLOGY_EXT"

# ---- provenance ---------------------------------------------------------------------------
# WHICH CODE produced this bundle. The digests come from preflight_gate.py rather than from a
# `find | xargs cat` here, so the launcher and the gate cannot drift apart — and so a root
# matching no files ABORTS instead of yielding e3b0c442..., the sha256 of nothing, which is a
# plausible sixteen-hex provenance stamp for a hash that covered zero files.
#
# The roots are typed. Only $REPO/lost3dsg is copied into the container at startup, so only it
# has a freeze point; /DATA/FOUND/found and knowledge_bridge are live on the path for the whole
# run and are SAMPLED, never asserted frozen.
_tree_sha() {
  local out
  out=$(python3 "$HERE/preflight_gate.py" --print-tree-sha "$1")     || { echo "!! cannot hash $1 — aborting rather than stamping an unrecorded run"; exit 1; }
  echo "$out"
}
read -r SRC_SHA SRC_N   <<<"$(_tree_sha "$REPO/lost3dsg")"
read -r FOUND_SHA FOUND_N <<<"$(_tree_sha /DATA/FOUND/found)"
KB_SRC=${KB_SRC:-/DATA/ASPIRE/knowledge_bridge}
read -r KB_SHA KB_N     <<<"$(_tree_sha "$KB_SRC")"
CFG_SHA=$(sha256sum "$HERE/$CFG_NAME" | cut -c1-16)
MERGED_SHA=$(GRAPH_API_CONFIG="$HERE/$CFG_NAME" python3 "$HERE/preflight_gate.py" --print-merged-sha)   || { echo "!! cannot compute the merged-config sha — aborting rather than passing an empty expectation"; exit 1; }
echo "    sources: graph-api $SRC_SHA ($SRC_N)  found $FOUND_SHA ($FOUND_N)  kb $KB_SHA ($KB_N)"
echo "    config:  file $CFG_SHA  merged $MERGED_SHA"

# Handed to the gate, which recomputes them INSIDE the container after the source copy. A
# difference means an edit landed in the window and the run is not the code stamped here.
export PREFLIGHT_EXPECT_SRC_SHA="graph_api=$SRC_SHA"
export PREFLIGHT_EXPECT_CFG_SHA="$CFG_SHA"
export PREFLIGHT_EXPECT_MERGED_SHA="$MERGED_SHA"

# WHICH ENVIRONMENT answered. The encoders are NOT in the image: live_stack_container.sh
# exports HF_HOME=/found/.hf_cache, so weights come from a host cache at runtime and two runs
# on one image digest can load different weights. Resolved through refs/main, a floating tag —
# recording them pins going forward and proves nothing about any earlier run.
IMAGE_TAG=${IMAGE_TAG:-graphapi-run:humble}
IMAGE_DIGEST=$(docker image inspect -f '{{.Id}}' "$IMAGE_TAG" 2>/dev/null || echo "unknown")
HF_CACHE=${HF_CACHE:-/DATA/FOUND/.hf_cache}
# >>> TEST-EXTRACT _enc_rev  (test_env_stamp.sh sources the block between these markers.
# It guessed the boundary with a sed pattern twice and was wrong twice: /^$/ swallowed the
# call sites below, and /; }$/ ran to end-of-file because this definition is a single line,
# so the range's START line also matched its terminator. The boundary is stated now, not
# inferred. Do not remove the markers.)
_enc_rev() { cat "$HF_CACHE/hub/models--$1/refs/main" 2>/dev/null              || cat "$HF_CACHE/models--$1/refs/main" 2>/dev/null || echo "unknown"; }
# <<< TEST-EXTRACT _enc_rev
ENC_E5=$(_enc_rev intfloat--e5-small-v2)
ENC_MINILM=$(_enc_rev sentence-transformers--all-MiniLM-L6-v2)

# WHICH GROUND TRUTH the numbers will be scored against. found/gt.py reads FOUND_SCENE_INSTANCE
# and has no default, so an unset or stale value silently rescopes every recall figure while the
# bundle still looks complete.
GT_PATH=${FOUND_SCENE_INSTANCE:-}
GT_SHA="unset"; GT_N="unset"
if [ -n "$GT_PATH" ] && [ -f "$GT_PATH" ]; then
  GT_SHA=$(sha256sum "$GT_PATH" | cut -c1-16)
  GT_N=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1])).get('objects') or []))" "$GT_PATH" 2>/dev/null || echo unknown)
  [ "$GT_N" = "0" ] && echo "!! WARNING: ground truth $GT_PATH contains 0 objects — every recall figure will be vacuous"
elif [ -n "$GT_PATH" ]; then
  echo "!! FOUND_SCENE_INSTANCE=$GT_PATH does not exist — aborting"; exit 1
fi
echo "    ground truth: ${GT_PATH:-<unset>} sha=$GT_SHA objects=$GT_N"
echo "    image: ${IMAGE_DIGEST:0:19}  encoders: ${ENC_E5:0:8} ${ENC_MINILM:0:8}"
# REFUSE rather than default. Every name below is interpolated into the JSON that follows, and
# four of them sit in NUMERIC positions where an empty expansion produces `"enforce": ,` — a
# syntactically broken file rather than a wrong value. That happened: the policy exports were
# 78 lines BELOW this heredoc and every run aborted at the validator.
#
# A default here would be worse than an abort. A bundle stamped with a policy it did not run
# under is the artefact nobody can detect later, and stamping the policy at all exists for the
# ablation case — where a default would silently record neither arm. Same standard as
# live_stack_container.sh refusing a guessed config: name what is missing and stop.
: "${FOUND_ENFORCE:?not set at run_metadata.json — the policy exports must precede this heredoc}"
: "${FOUND_HOLD_BAND:?not set at run_metadata.json}"
: "${FOUND_MIN_SUPPORT:?not set at run_metadata.json}"
: "${FOUND_ROOM_ENFORCE:?not set at run_metadata.json}"
: "${FOUND_ALIGNER:?not set at run_metadata.json}"
: "${FOUND_ONTOLOGY_EXT:?not set at run_metadata.json}"
: "${SRC_SHA:?not set at run_metadata.json — the provenance block must precede this heredoc}"
: "${SRC_N:?not set at run_metadata.json}"
: "${FOUND_SHA:?not set at run_metadata.json}"
: "${FOUND_N:?not set at run_metadata.json}"
: "${KB_SHA:?not set at run_metadata.json}"
: "${KB_N:?not set at run_metadata.json}"
: "${CFG_SHA:?not set at run_metadata.json}"
: "${MERGED_SHA:?not set at run_metadata.json}"

# Keys are ADDED, never changed: three consumers read this file by key and some do arithmetic
# on the values. The five original keys keep their bytes. "config_name" keeps its meaning too,
# and that meaning is now stated: it is the name the LAUNCHER INTENDED. What the two processes
# actually loaded is recorded separately — the container's in preflight.json (a2 reads
# config.CFG_PATH, the file config.py actually read), the feed host's below.
cat <<EOF > "$RUN_DIR/run_metadata.json"
{
  "run_id": "$RUN_ID",
  "scene": "$SCENE_ARG",
  "start_time": "$(date -Iseconds)",
  "config_name": "$CFG_NAME",
  "output_dir": "$RUN_DIR",
  "config_name_note": "the name the launcher intended; see config_resolved and preflight.json for what each process loaded",
  "config_resolved": {
    "feed_host_path": "$HERE/$CFG_NAME",
    "file_sha256_16": "$CFG_SHA",
    "merged_sha256_16": "$MERGED_SHA",
    "container_path": "recorded by the gate in preflight.json (a2.loaded_path)"
  },
  "seed": ${FEED_SEED:-7},
  "policy": {"enforce": $FOUND_ENFORCE, "hold_band": $FOUND_HOLD_BAND,
             "min_support": $FOUND_MIN_SUPPORT, "rooms_enforced": $FOUND_ROOM_ENFORCE,
             "aligner": "$FOUND_ALIGNER", "ontology_ext": "$FOUND_ONTOLOGY_EXT"},
  "provenance_intent": {
    "note": "host-side, taken BEFORE docker run. provenance_confirmed in preflight.json is taken after the container copies its sources, and is the authoritative record of what executed.",
    "graph_api_src_sha256_16": "$SRC_SHA", "graph_api_files": $SRC_N,
    "found_src_sha256_16": "$FOUND_SHA", "found_files": $FOUND_N,
    "kb_src_sha256_16": "$KB_SHA", "kb_files": $KB_N,
    "kb_root": "$KB_SRC",
    "frozen_roots": ["graph_api"],
    "live_roots": ["found", "kb"],
    "live_root_note": "not copied into the container; on sys.path for the whole run, so sampled rather than asserted frozen"
  },
  "environment": {
    "image_tag": "$IMAGE_TAG",
    "image_digest": "$IMAGE_DIGEST",
    "encoders": {"intfloat/e5-small-v2": "$ENC_E5",
                 "sentence-transformers/all-MiniLM-L6-v2": "$ENC_MINILM"},
    "encoder_pin_source": "refs/main (floating) — recorded, not asserted"
  },
  "ground_truth": {
    "path": "${GT_PATH:-unset}",
    "sha256_16": "$GT_SHA",
    "n_objects": "$GT_N",
    "note": "run-time INTENT. analyse_run.py opens this file offline and is the authoritative record of what was actually scored against."
  }
}
EOF
python3 -m json.tool "$RUN_DIR/run_metadata.json" > /dev/null   || { echo "!! run_metadata.json is not valid JSON — aborting rather than shipping an unreadable bundle"; exit 1; }

# `latest` is repointed only AFTER the bundle validates. It used to move before the checks, so
# an abort left the archive's most-followed path aimed at a directory holding an unreadable
# run_metadata.json, no logs and no preflight.json. A reader following `latest` cannot tell a
# truncated run from a complete one, and three shipped bundles already have that shape.
ln -sfn "$RUN_DIR" "$FOUND_RUNS_DIR/latest"
echo "    run bundle: $RUN_DIR (symlinked as $FOUND_RUNS_DIR/latest)"

HABITAT_SCENE=${HABITAT_SCENE:-$DEF_SCENE} \
HABITAT_DATASET=${HABITAT_DATASET:-$DEF_DATASET} \
FEED_SEED=${FEED_SEED:-7} FEED_FPS=${FEED_FPS:-3} FEED_WALK=${FEED_WALK:-6} FEED_DWELL=${FEED_DWELL:-60} \
FEED_MAPPING_SECONDS=${FEED_MAPPING_SECONDS:-150} FEED_OVERLAY=${FEED_OVERLAY:-1} \
FEED_SHOW=${FEED_SHOW:-1} DISPLAY="${DISPLAY:-:1}" PYTHONUNBUFFERED=1 \
GRAPH_API_CONFIG="${GRAPH_API_CONFIG:-$HERE/$CFG_NAME}" \
  nohup "$HOME/miniconda3/envs/habitat_env/bin/python" "$HERE/habitat_feed_host.py" \
  > "$OUT_DIR/feed_host.log" 2>&1 &
FEED_PID=$!
# habitat import + scene load can take >2 min on cold caches
for i in $(seq 1 90); do grep -q "listening" "$OUT_DIR/feed_host.log" 2>/dev/null && break; sleep 2; done
grep -q "listening" "$OUT_DIR/feed_host.log" || { echo "feed host failed:"; tail -20 "$OUT_DIR/feed_host.log"; exit 1; }
echo "    feed host up"

# Asynchronous health & memory monitor
(
  while true; do
    echo "=== $(date) ===" >> "$OUT_DIR/system_health.log"
    free -m >> "$OUT_DIR/system_health.log"
    docker stats --no-stream graphapi_live >> "$OUT_DIR/system_health.log" 2>/dev/null || true
    # per-process VRAM/CPU + per-model location inventory (JSON snapshot)
    python3 "$HERE/resource_monitor.py" --out "$OUT_DIR/model_resources.json" 2>/dev/null || true
    sleep 30
  done
) &
MON_PID=$!

echo ">>> ROS stack in container (web viewer -> http://localhost:8081)"
docker run --name graphapi_live --rm --entrypoint bash --gpus all --network=host \
  -e OPENAI_API_KEY -e CFG_NAME -e MODAL_PERCEPTION_URL \
  -e FOUND_ENFORCE -e FOUND_HOLD_BAND -e FOUND_MIN_SUPPORT -e FOUND_ROOM_ENFORCE \
  -e FOUND_ALIGNER -e FOUND_ONTOLOGY_EXT -e FOUND_STORE_PATH -e FOUND_SCENE \
  -e RUN_START_EPOCH -e PREFLIGHT_EXPECT_POLICY -e PREFLIGHT_SKIP \
  -e PREFLIGHT_EXPECT_CFG_SHA -e PREFLIGHT_EXPECT_MERGED_SHA -e PREFLIGHT_EXPECT_SRC_SHA \
  -v "$REPO":/graph_api:ro \
  -v /DATA/FOUND:/found \
  -v "${KB_SRC:-/DATA/ASPIRE/knowledge_bridge}":/kb:ro \
  -v "$RUN_DIR":/ws/output \
  -v /DATA/models/efficientvit_sam:/models/vitsam:ro \
  -v /DATA/huggingface_cache:/models/hf \
  -v "$OUT_DIR":/out \
  graphapi-run:humble /graph_api/lost3dsg/test/live_stack_container.sh

# Post-run archive
cp "$OUT_DIR"/*.log "$RUN_DIR/logs/" 2>/dev/null || true
cp "$OUT_DIR"/*.json "$RUN_DIR/" 2>/dev/null || true
cp "$OUT_DIR"/*.jsonl "$RUN_DIR/" 2>/dev/null || true
echo ">>> Run complete. All artifacts safely archived in $RUN_DIR"
