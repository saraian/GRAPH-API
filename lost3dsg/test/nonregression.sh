#!/usr/bin/env bash
# Non-regression run for the lost3dsg stack on an HM3D (Matterport) scene.
#
# Launches the full stack (habitat camera node + perception_2 + object_manager_6
# via habitat_launch.py), lets it run RUN_SECONDS, then compares the resulting
# output/persistent_perception.json against a committed baseline with
# regression_check.py (object count, distinct labels, per-axis span, ±TOL).
#
#   ./nonregression.sh                      # run + check
#   UPDATE_BASELINE=1 ./nonregression.sh    # run + record new baseline
#
# Scene selection lives in src/perception_module/config.yaml (habitat: section)
# or a file pointed at by GRAPH_API_CONFIG. Baselines are per scene+duration:
# record one per configuration you care about, after a run you trust.
#
# ponytail: motion is whatever the stack does by default for RUN_SECONDS —
# deterministic scripted tours (publishing /habitat/action on a seed) can be
# added when run-to-run variance proves too large for the tolerance.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
PKG_ROOT=$(cd "$HERE/.." && pwd)
RUN_SECONDS=${RUN_SECONDS:-180}
TOL=${TOL:-0.15}

SCENE_TAG=$(python3 -c "
import sys; sys.path.insert(0, '$PKG_ROOT/src/perception_module')
from config import CFG; import os
print(os.path.basename(CFG['habitat']['scene']).split('.')[0])")
BASELINE=$HERE/baseline_${SCENE_TAG}_${RUN_SECONDS}s.json
BELIEF=$PKG_ROOT/output/persistent_perception.json

rm -f "$BELIEF"
ros2 launch lost3dsg habitat_launch.py use_rviz:=false &
LAUNCH_PID=$!
trap 'kill $LAUNCH_PID 2>/dev/null || true' EXIT
sleep "$RUN_SECONDS"
kill "$LAUNCH_PID" 2>/dev/null || true
wait "$LAUNCH_PID" 2>/dev/null || true

if [ ! -f "$BELIEF" ]; then
  echo "no belief written at $BELIEF — stack did not run correctly" >&2
  exit 2
fi

if [ "${UPDATE_BASELINE:-0}" = "1" ]; then
  python3 "$HERE/regression_check.py" "$BELIEF" "$BASELINE" --update-baseline
else
  python3 "$HERE/regression_check.py" "$BELIEF" "$BASELINE" --tol "$TOL"
fi
