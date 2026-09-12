#!/usr/bin/env bash
# EVALUATE A RUN BUNDLE. Four steps, in order, from the colleague's own metrics scripts:
#
#   1. hm3d_ground_truth_manifest.py   the scene's ground truth      (needs habitat-sim)
#   2. build_hm3d_eval_manifest.py     joins ground truth to the run
#   3. metrics_eval.py                 the metrics
#   4. metrics_eval_visualize.py       an HTML view of the boxes, into metrics_eval_visualizer/
#   5. time_metrics.py                  latency per stage, run/exploration/movement/online time
#   6. eval_report.py                  Comparison_<stamp>.pdf with every statistic
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

# HM3D_ROOT, THE SAME WAY install.sh FINDS IT. The scene of record in a bundle is a NAME
# (run_metadata.json's `scene`), and resolving it to a path needs the scene library. Without this
# the resolver answers "the name 'hm3d_00861', but HM3D_ROOT is not set" and the evaluation stops
# on a machine that has the library sitting in its usual place. Set only when the caller has not.
if [ -z "${HM3D_ROOT:-}" ]; then
  for _d in "$HOME/Musumeci/habitat_matterport/hm3d_example" /DATA/habitat_matterport/hm3d_example; do
    [ -d "$_d" ] && { export HM3D_ROOT="$_d"; break; }
  done
  [ -n "${HM3D_ROOT:-}" ] && echo ">>> HM3D_ROOT=$HM3D_ROOT (discovered; export it to override)"
fi

# THE SCENE COMES FROM THE BUNDLE, resolved by lost3dsg/test/eval_scene_of_bundle.py, which has its
# own self-check. A scene resolved wrongly evaluates a DIFFERENT HOUSE and nobody can spot that
# afterwards, so it is a file with tests rather than a line in this script.
#
# NOTE THAT A BUNDLE RECORDS A SCENE *NAME*, NOT A FILE. run_metadata.json says
# `scene: hm3d_00861`, and run_sim.sh-476 turns it into a path with HM3D_ROOT on whichever
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
PY_HAB=""
GT="$OUT/manifest_gt.json"
if [ -s "$GT" ] && [ "$FORCE" = "0" ]; then
  echo "== 1/6  ground truth: $GT exists, reusing it (--force to rebuild)"
else
  # The recipe used an env named habitat310; ours is habitat_env. Find one that can import
  # habitat_sim rather than assuming a name -- a wrong env name fails three steps later as a
  # missing manifest, which reads like a different fault.
  PY_HAB="${PY_HAB:-}"
  for c in "${EVAL_CONDA_PY:-}" "$HOME/miniconda3/envs/${EVAL_CONDA_ENV:-habitat310}/bin/python" \
           "$HOME/miniconda3/envs/habitat_env/bin/python" "$HOME/anaconda3/envs/habitat310/bin/python"; do
    [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import habitat_sim' 2>/dev/null && { PY_HAB="$c"; break; }
  done
  [ -n "$PY_HAB" ] || { echo "!! no python that can import habitat_sim. Set EVAL_CONDA_PY to one." >&2; exit 4; }
  echo "== 1/6  ground truth  ($PY_HAB)"
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

# THE METRICS RUN ON A PYTHON THAT HAS cv2 AND numpy, not on whatever `python3` is. Measured
# 2026-09-10: the ambient python3 has no cv2, and metrics_eval falls back to a polygon IoU that
# needs it -- `RuntimeError: opencv-python è necessario per l'IoU dei poligoni` from three frames
# down, which reads like a broken manifest rather than a missing package. The habitat environment
# has cv2 4.9, so steps 2 to 4 use the same interpreter step 1 does.
PY_EVAL=""
for c in "${EVAL_PY:-}" "$PY_HAB" "$HOME/miniconda3/envs/${EVAL_CONDA_ENV:-habitat_env}/bin/python" python3; do
  [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 || [ -x "$c" ] || continue
  "$c" -c 'import cv2, numpy' 2>/dev/null && { PY_EVAL="$c"; break; }
done
[ -n "$PY_EVAL" ] || { echo "!! no python with cv2 and numpy. Set EVAL_PY to one." >&2; exit 5; }
echo "python: $PY_EVAL (has cv2 and numpy)"

# ---- 2. join the ground truth to what the run recorded.
EV="$OUT/manifest_eval.json"
echo "== 2/6  join to the run"
"$PY_EVAL" "$PM/build_hm3d_eval_manifest.py" --ground-truth "$GT" --run-dir "$BUNDLE" --output "$EV" \
  || { echo "!! step 2 failed" >&2; exit 5; }

# ---- 3. the metrics.
echo "== 3/6  metrics"
"$PY_EVAL" "$PM/metrics_eval.py" "$EV" --output "$OUT/metrics.json" \
  || { echo "!! step 3 failed" >&2; exit 6; }

# ---- 4. the HTML view.
# THE VISUALISATIONS GET THEIR OWN DIRECTORY, named for the tool that writes them, so a reader
# opening a bundle finds them without knowing which script produced which file.
echo "== 4/6  boxes"
VIZ="$OUT/metrics_eval_visualizer"; mkdir -p "$VIZ"
"$PY_EVAL" "$PM/metrics_eval_visualize.py" "$EV" --output "$VIZ/boxes.html" \
  || { echo "!! step 4 failed" >&2; exit 7; }

# ---- 5. where the time went. Six measurements nothing else computes; see time_metrics.py for
# what each is derived from. Runs BEFORE the report so the PDF can carry them.
echo "== 5/6  time"
"$PY_EVAL" "$HERE/lost3dsg/test/time_metrics.py" "$BUNDLE" \
  || echo "   note: time metrics could not be computed; the report will say so"

# ---- 6. the PDF report. Every statistic, with the provenance that makes it attributable.
echo "== 6/6  report"
# A DIFFERENT INTERPRETER AGAIN, and for the mirror of step 2's reason. The metrics need cv2,
# which only the habitat environment has; the PDF needs reportlab, which only the system python
# has. Measured 2026-09-10: reusing $PY_EVAL here failed with ModuleNotFoundError: reportlab.
# Each step asks for the interpreter that can answer it rather than one that can answer most.
PY_REPORT=""
for c in "${EVAL_REPORT_PY:-}" python3 "$PY_EVAL" "$HOME/miniconda3/envs/${EVAL_CONDA_ENV:-habitat_env}/bin/python"; do
  [ -n "$c" ] || continue
  "$c" -c 'import reportlab' 2>/dev/null && { PY_REPORT="$c"; break; }
done
if [ -z "$PY_REPORT" ]; then
  echo "!! no python with reportlab, so no PDF was written. pip install reportlab, or set" >&2
  echo "   EVAL_REPORT_PY to an interpreter that has it. Steps 1-4 are done: $OUT" >&2
  exit 8
fi
"$PY_REPORT" "$HERE/lost3dsg/test/eval_report.py" "$BUNDLE" \
  || { echo "!! step 5 failed: no PDF written" >&2; exit 8; }

echo
echo "DONE. In $OUT:"
ls -1 "$OUT"
