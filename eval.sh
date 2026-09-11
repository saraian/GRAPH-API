#!/usr/bin/env bash
# EVALUATE A RUN BUNDLE. Four steps, in order, from the colleague's own metrics scripts:
#
#   1. hm3d_ground_truth_manifest.py   the scene's ground truth      (needs habitat-sim)
#   2. build_hm3d_eval_manifest.py     joins ground truth to the run
#   3. metrics_eval.py                 the metrics
#   4. metrics_eval_visualize.py       an HTML view of the boxes
#
#   ./eval.sh                    the newest run in the workspace
#   ./eval.sh <bundle-dir>       that bundle
#   ./eval.sh --force            rebuild the ground-truth manifest even if it exists
#
# EVERYTHING IS WRITTEN INTO THE BUNDLE, under <bundle>/eval/, which is a host directory. Owner
# instruction 2026-09-10: input and output data live outside the container, mounted as volumes.
#
# THE SCENE COMES FROM THE BUNDLE, NOT FROM THIS FILE. The recipe this follows named
# /root/exchange/lost3dsg/habitat/... , which is one machine's path. The bundle carries the
# RESOLVED config it actually ran (`config.yaml`), so the ground truth is built for the scene the
# run really used. A hardcoded scene path is how an evaluation comes to describe a different house.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM="$HERE/lost3dsg/src/perception_module"
FORCE=0
BUNDLE=""
for a in "$@"; do
  case "$a" in
    --force)   FORCE=1 ;;
    -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)        echo "!! unknown option: $a" >&2; exit 2 ;;
    *)         BUNDLE="$a" ;;
  esac
done

# shellcheck disable=SC1090
[ -f "$HERE/lost3dsg/test/env.local.sh" ] && . "$HERE/lost3dsg/test/env.local.sh" 2>/dev/null || true
if [ -z "$BUNDLE" ]; then
  [ -n "${WORKSPACE_ROOT:-}" ] || { echo "!! no bundle given and WORKSPACE_ROOT is not set. Run ./install.sh, or pass a bundle directory." >&2; exit 2; }
  BUNDLE="$WORKSPACE_ROOT/runs/latest"
fi
BUNDLE="$(cd "$BUNDLE" 2>/dev/null && pwd)" || { echo "!! no such bundle: $BUNDLE" >&2; exit 2; }
[ -f "$BUNDLE/run_metadata.json" ] || { echo "!! $BUNDLE has no run_metadata.json — that is not a run bundle." >&2; exit 2; }
OUT="$BUNDLE/eval"; mkdir -p "$OUT"
echo "bundle: $BUNDLE"
echo "output: $OUT"

# THE SCENE COMES FROM THE BUNDLE, resolved by lost3dsg/test/eval_scene_of_bundle.py, which has its
# own self-check. A scene resolved wrongly evaluates a DIFFERENT HOUSE and nobody can spot that
# afterwards, so it is a file with tests rather than a line in this script.
#
# NOTE THAT A BUNDLE RECORDS A SCENE *NAME*, NOT A FILE. run_metadata.json says
# `scene: hm3d_00861`, and live_run.sh:469-476 turns it into a path with HM3D_ROOT on whichever
# machine ran it. So an evaluation is NOT reproducible from a bundle alone. Filed as GA-464. The
# resolver prints WHICH source answered, so a reader can see when a name was resolved rather than
# read, and this script prints that line.
_res="$(python3 "$HERE/lost3dsg/test/eval_scene_of_bundle.py" "$BUNDLE")" || {
  echo "!! $_res" >&2
  echo "   Pass EVAL_SCENE=/path/scene.basis.glb and EVAL_DATASET_CONFIG=/path/cfg.json to override." >&2
  [ -n "${EVAL_SCENE:-}" ] || exit 3
}
SCENE="${EVAL_SCENE:-$(echo "$_res" | cut -d" " -f1)}"
DSCFG="${EVAL_DATASET_CONFIG:-$(echo "$_res" | cut -d" " -f2)}"
echo "scene:  $SCENE"
echo "        source: $(echo "$_res" | cut -d" " -f3-)"
[ -f "$SCENE" ] || { echo "!! that scene file is not on this machine: $SCENE" >&2; exit 3; }
[ -f "$DSCFG" ] || { echo "!! the dataset config is not on this machine: $DSCFG" >&2; exit 3; }

# ---- 1. ground truth. NEEDS habitat-sim, so it runs in the conda environment, not this shell.
GT="$OUT/manifest_gt.json"
if [ -s "$GT" ] && [ "$FORCE" = "0" ]; then
  echo "== 1/4  ground truth: $GT exists, reusing it (--force to rebuild)"
else
  # The recipe used an env named habitat310; ours is habitat_env. Find one that can import
  # habitat_sim rather than assuming a name -- a wrong env name fails three steps later as a
  # missing manifest, which reads like a different fault.
  PY_HAB=""
  for c in "${EVAL_CONDA_PY:-}" "$HOME/miniconda3/envs/${EVAL_CONDA_ENV:-habitat310}/bin/python" \
           "$HOME/miniconda3/envs/habitat_env/bin/python" "$HOME/anaconda3/envs/habitat310/bin/python"; do
    [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import habitat_sim' 2>/dev/null && { PY_HAB="$c"; break; }
  done
  [ -n "$PY_HAB" ] || { echo "!! no python that can import habitat_sim. Set EVAL_CONDA_PY to one." >&2; exit 4; }
  echo "== 1/4  ground truth  ($PY_HAB)"
  "$PY_HAB" "$PM/hm3d_ground_truth_manifest.py" "$SCENE" --dataset-config "$DSCFG" --output "$GT" \
    || { echo "!! step 1 failed; the output above says why" >&2; exit 4; }
fi

# metrics_eval.py IS REQUIRED BY THREE OF THE FOUR STEPS, so it is checked here, before step 2,
# rather than before step 3 where I first put it. MEASURED: build_hm3d_eval_manifest.py:13 does
# `from metrics_eval import assignment`, and metrics_eval_visualize.py:19 imports assignment,
# geometry_iou and load from it. So a missing file stops 2, 3 and 4 — only the ground truth runs.
# It is in nobody's checkout here: not in this repository, not on this machine, not on the lab
# machine. The colleague runs all four, so it exists on hers and has never been committed.
if [ ! -f "$PM/metrics_eval.py" ]; then
  echo "!! metrics_eval.py is not in this repository, and steps 2, 3 and 4 all import it." >&2
  echo "   Expected at: $PM/metrics_eval.py" >&2
   echo "   build_hm3d_eval_manifest.py:13 needs \`assignment\`; metrics_eval_visualize.py:19 needs" >&2
  echo "   assignment, geometry_iou and load. Ask the colleague to commit it." >&2
  echo "   Step 1 is done and reusable: $GT" >&2
  exit 6
fi

# ---- 2. join the ground truth to what the run recorded.
EV="$OUT/manifest_eval.json"
echo "== 2/4  join to the run"
python3 "$PM/build_hm3d_eval_manifest.py" --ground-truth "$GT" --run-dir "$BUNDLE" --output "$EV" \
  || { echo "!! step 2 failed" >&2; exit 5; }

# ---- 3. the metrics.
echo "== 3/4  metrics"
python3 "$PM/metrics_eval.py" "$EV" --output "$OUT/metrics.json" \
  || { echo "!! step 3 failed" >&2; exit 6; }

# ---- 4. the HTML view.
echo "== 4/4  boxes"
python3 "$PM/metrics_eval_visualize.py" "$EV" --output "$OUT/boxes.html" \
  || { echo "!! step 4 failed" >&2; exit 7; }

echo
echo "DONE. In $OUT:"
ls -1 "$OUT"
