#!/usr/bin/env bash
# Live demo on this machine: habitat renders on the host (conda habitat_env),
# the ROS 2 stack runs in the graphapi-run:humble container over a TCP feed.
# Watch: web viewer at http://localhost:8080 and snapshots in $OUT_DIR.
#   ./live_run.sh          # foreground; ctrl-C stops everything
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT_DIR=${OUT_DIR:-/tmp/graphapi_live}
mkdir -p "$OUT_DIR"

cleanup() {
  # bracket trick: the pattern must not match this script's own command line
  pkill -f "habitat_feed_[h]ost" 2>/dev/null || true
  docker rm -f graphapi_live 2>/dev/null || true
}
trap cleanup EXIT
cleanup

echo ">>> host habitat feed (scene renders on the host GPU)"
# Matterport (HM3D v0.2 example scene, semantic-annotated). The other two public
# HM3D examples are 00337-CFVBbU9Rsyb and 00770-NBg5UqG3di3 in the same root;
# the MP3D example 17DRP5sb8fy lives in /DATA/habitat_matterport/versioned_data.
HM3D_ROOT=${HM3D_ROOT:-/DATA/habitat_matterport/hm3d_example}
HABITAT_SCENE=${HABITAT_SCENE:-$HM3D_ROOT/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb} \
HABITAT_DATASET=${HABITAT_DATASET:-$HM3D_ROOT/hm3d_annotated_basis.scene_dataset_config.json} \
FEED_SEED=${FEED_SEED:-7} FEED_FPS=${FEED_FPS:-3} FEED_WALK=${FEED_WALK:-6} FEED_DWELL=${FEED_DWELL:-60} \
FEED_MAPPING_SECONDS=${FEED_MAPPING_SECONDS:-150} FEED_OVERLAY=${FEED_OVERLAY:-1} \
FEED_SHOW=${FEED_SHOW:-1} DISPLAY="${DISPLAY:-:1}" PYTHONUNBUFFERED=1 \
GRAPH_API_CONFIG="${GRAPH_API_CONFIG:-$HERE/smoke_config.yaml}" \
  nohup "$HOME/miniconda3/envs/habitat_env/bin/python" "$HERE/habitat_feed_host.py" \
  > "$OUT_DIR/feed_host.log" 2>&1 &
for i in $(seq 1 30); do grep -q "listening" "$OUT_DIR/feed_host.log" 2>/dev/null && break; sleep 2; done
grep -q "listening" "$OUT_DIR/feed_host.log" || { echo "feed host failed:"; tail -20 "$OUT_DIR/feed_host.log"; exit 1; }
echo "    feed host up"

echo ">>> ROS stack in container (web viewer -> http://localhost:8080)"
docker run --name graphapi_live --rm --entrypoint bash --gpus all --network=host \
  -v "$REPO":/graph_api:ro \
  -v graphapi_ws:/ws \
  -v /DATA/models/efficientvit_sam:/models/vitsam:ro \
  -v /DATA/huggingface_cache:/models/hf \
  -v "$OUT_DIR":/out \
  graphapi-run:humble /graph_api/lost3dsg/test/live_stack_container.sh
