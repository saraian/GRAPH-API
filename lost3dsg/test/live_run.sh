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
echo "    config: $CFG_NAME"

# Setup persistent FOUND run bundle (never overwritten across runs)
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCENE_ARG=${1:-hm3d_00861}
RUN_ID="${RUN_TIMESTAMP}_${SCENE_ARG}"
FOUND_RUNS_DIR=${FOUND_RUNS_DIR:-/DATA/FOUND/runs}
RUN_DIR="$FOUND_RUNS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/crops" "$RUN_DIR/snapshots"
ln -sfn "$RUN_DIR" "$FOUND_RUNS_DIR/latest"
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
cp "$HERE/$CFG_NAME" "$RUN_DIR/config.yaml" 2>/dev/null || true
cat <<EOF > "$RUN_DIR/run_metadata.json"
{
  "run_id": "$RUN_ID",
  "scene": "$SCENE_ARG",
  "start_time": "$(date -Iseconds)",
  "config_name": "$CFG_NAME",
  "output_dir": "$RUN_DIR"
}
EOF

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
export FOUND_ENFORCE="${FOUND_ENFORCE:-0}"
export FOUND_ROOM_ENFORCE="${FOUND_ROOM_ENFORCE:-0}"
export FOUND_STORE_PATH="${FOUND_STORE_PATH:-/ws/output/knowledge_graph.ttl}"
export FOUND_SCENE="${FOUND_SCENE:-$SCENE_ARG}"

docker run --name graphapi_live --rm --entrypoint bash --gpus all --network=host \
  -e OPENAI_API_KEY -e CFG_NAME -e MODAL_PERCEPTION_URL \
  -e FOUND_ENFORCE -e FOUND_ROOM_ENFORCE -e FOUND_STORE_PATH -e FOUND_SCENE \
  -v "$REPO":/graph_api:ro \
  -v /DATA/FOUND:/found \
  -v "$RUN_DIR":/ws/output \
  -v /DATA/models/efficientvit_sam:/models/vitsam:ro \
  -v /DATA/huggingface_cache:/models/hf \
  -v "$OUT_DIR":/out \
  graphapi-run:humble /graph_api/lost3dsg/test/live_stack_container.sh

# Post-run archive
cp "$OUT_DIR"/*.log "$RUN_DIR/logs/" 2>/dev/null || true
cp "$OUT_DIR"/*.json "$RUN_DIR/" 2>/dev/null || true
echo ">>> Run complete. All artifacts safely archived in $RUN_DIR"
