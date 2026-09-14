#!/usr/bin/env bash
# Small focused merge-debug run on THIS host (owner 2026-09-14: "small focused runs to debug and
# improve the algorithm", "try running it yourself, here using modal"). One capped storey of
# hm3d_00824 through the GA-493 debug entry point, Modal perception, no GT, replay capture on.
#
#   bash lost3dsg/test/debug_configs/launch_merge_debug.sh evidence   # the ruling's engine
#   bash lost3dsg/test/debug_configs/launch_merge_debug.sh legacy     # the ablation arm
#
# Everything below is a HOST path (the feed host renders on the host); the container copies the
# sources from this checkout (rule 9), so launch from the checkout whose code you mean to run.
set -euo pipefail
ENGINE="${1:?evidence|legacy}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
# DETACH=1: run under nohup/setsid with the log in the results dir and return at once, so a
# caller with a short command timeout does not kill a 10-15 minute run.
if [ "${DETACH:-0}" = "1" ]; then
  LOGDIR="${RESULTS_DIR:-${WORKSPACE_ROOT:-/home/xps/graphapi_ws}/results}"
  mkdir -p "$LOGDIR"
  LOG="$LOGDIR/launch_${ENGINE}_$(date +%Y%m%d_%H%M%S).log"
  DETACH=0 setsid nohup bash "${BASH_SOURCE[0]}" "$ENGINE" > "$LOG" 2>&1 < /dev/null &
  echo "detached pid $! log $LOG"
  exit 0
fi
# VARIANT=ms1 selects the min_consecutive-1 + min_sightings-2 arm (see make_merge_debug_config.py).
CFG="$HERE/merge_debug_00824_${ENGINE}${VARIANT:+_$VARIANT}.yaml"
[ -f "$CFG" ] || python3 "$HERE/make_merge_debug_config.py" --engine "$ENGINE" ${VARIANT:+--variant "$VARIANT"} >/dev/null
[ -f "$REPO/lost3dsg/test/env.local.sh" ] || { echo "!! no env.local.sh (MODAL_PERCEPTION_URL, REGOLO_API_KEY)" >&2; exit 2; }
set -a
# shellcheck disable=SC1091
. "$REPO/lost3dsg/test/env.local.sh"
set +a
[ -n "${MODAL_PERCEPTION_URL:-}" ] || { echo "!! MODAL_PERCEPTION_URL is empty" >&2; exit 2; }

DATASET_ROOT="${DATASET_ROOT:-/DATA/GRAPH-API/lost3dsg/FOUND-Dataset}"
export HM3D_ROOT="$DATASET_ROOT/habitat/hm3d-val-habitat-v0.2"
export HABITAT_DATASET="$DATASET_ROOT/habitat/hm3d-val-semantic-configs-v0.2/hm3d_annotated_basis.scene_dataset_config.json"
export FEED_SCHEDULE="${FEED_SCHEDULE:-$DATASET_ROOT/schedules/00824-Dd4bFSTQ8gi.schedule.json}"
for f in "$HM3D_ROOT/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb" "$HABITAT_DATASET" "$FEED_SCHEDULE"; do
  [ -f "$f" ] || { echo "!! missing: $f" >&2; exit 2; }
done
export GRAPH_API_CONFIG="$CFG"
export GRAPH_API_PARALLEL_FUSION=1
export RVIZ=0
export IMAGE_TAG="${IMAGE_TAG:-graphapi-run:humble-ga290}"
# run_sim.sh wants a WORKSPACE holding maps/, runs/, results/ (it derives one two levels above
# the checkout otherwise, which is wrong for a worktree). /DATA is 98% full, so it lives on /.
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/home/xps/graphapi_ws}"
export RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE_ROOT/results}"
export GA493_REPLAY_CAPTURE_MAX_BYTES="${GA493_REPLAY_CAPTURE_MAX_BYTES:-1073741824}"
export GA493_REPLAY_CAPTURE_MAX_CYCLES="${GA493_REPLAY_CAPTURE_MAX_CYCLES:-6}"
mkdir -p "$WORKSPACE_ROOT/maps" "$WORKSPACE_ROOT/runs" "$WORKSPACE_ROOT/schedules" "$RESULTS_DIR"
echo ">>> merge debug run: engine=$ENGINE config=$CFG image=$IMAGE_TAG results=$RESULTS_DIR"
echo "    HEAD $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --short | wc -l)"
cd "$REPO"
exec bash "$REPO/lost3dsg/test/ga493_debug_run.sh" hm3d_00824 --one-storey --config "$CFG"
