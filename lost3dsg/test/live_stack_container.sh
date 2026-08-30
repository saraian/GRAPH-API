#!/usr/bin/env bash
# Runs INSIDE the container: build, then start the full stack against the host
# habitat feed. Started by live_run.sh — not meant to be run directly.
set -e
source /opt/ros/humble/setup.bash

echo ">>> build"
mkdir -p /ws/src
rm -rf /ws/src/lost3dsg
cp -r /graph_api/lost3dsg /ws/src/lost3dsg
cd /ws
colcon build --packages-select lost3dsg --cmake-args -DCMAKE_BUILD_TYPE=Release >/tmp/build.log 2>&1 \
  || { tail -30 /tmp/build.log; exit 1; }
source /ws/install/setup.bash

python3 - <<'PY'
import numpy as np
from gensim.models import KeyedVectors
words = ("chair table bed cabinet sofa lamp tv door sink toilet shelf desk "
         "monitor plant curtain unknown white black red green blue brown gray "
         "wood metal plastic fabric glass ceramic small large room wall floor").split()
kv = KeyedVectors(vector_size=32)
kv.add_vectors(words, np.random.default_rng(0).normal(size=(len(words), 32)).astype(np.float32))
kv.save_word2vec_format("/tmp/smoke_w2v.bin", binary=True)
PY

# The KG aligner's bridge, and the model cache. found/kg_align.py inserts KG_BRIDGE_SRC
# (default /DATA/ASPIRE/knowledge_bridge, a HOST path that does not exist in here) and then
# imports knowledge_bridge.alignment.embedder. Without /kb on the path that raises, and the
# perception node dies when it loads the hook. Without HF_HOME the in-container default points
# at a host path that does not exist here, so MiniLM is re-fetched from the hub every run.
#
# This hunk was marked "port" in the lane's own plan and was not ported. The first gated run
# found it: a1 could not import `found` and reported it as a probe fault.
export PYTHONPATH=/kb:${PYTHONPATH}
export HF_HOME=/found/.hf_cache

# CFG_NAME comes from live_run.sh (regolo_config.yaml when an API key is set).
# No default. This line used to read ${CFG_NAME:-smoke_config.yaml}, and because
# live_run.sh assigned CFG_NAME without exporting it, `docker run -e CFG_NAME`
# passed nothing and every live run silently used the smoke config while the bundle
# recorded regolo. Guessing here is what made that invisible.
: "${CFG_NAME:?CFG_NAME not set — live_run.sh must export it; refusing to guess a config}"
export GRAPH_API_CONFIG=/graph_api/lost3dsg/test/${CFG_NAME}
# The config must EXIST. config.py::_load returns the defaults when it does not, silently —
# so a mistyped or unported config name yields a run with hooks.filter empty, FOUND out of
# the loop, and a bundle that looks complete. Fail here instead.
[ -f "$GRAPH_API_CONFIG" ] || { echo "!! GRAPH_API_CONFIG=$GRAPH_API_CONFIG does not exist — refusing to run on defaults"; exit 1; }
export GRAPH_API_OUTPUT_DIR=/ws/output
LOG_DIR=/ws/output/logs
mkdir -p "$LOG_DIR" /ws/output/crops /ws/output/snapshots /out

# Ensure logs stream directly to the persistent host-mounted volume
touch "$LOG_DIR/feed_node.log" "$LOG_DIR/rtabmap.log" "$LOG_DIR/om6.log" "$LOG_DIR/bridge.log" "$LOG_DIR/perception.log" "$LOG_DIR/saver.log"
ln -sfn "$LOG_DIR/feed_node.log" /tmp/feed_node.log
ln -sfn "$LOG_DIR/rtabmap.log" /tmp/rtabmap.log
ln -sfn "$LOG_DIR/om6.log" /tmp/om6.log
ln -sfn "$LOG_DIR/bridge.log" /tmp/bridge.log
ln -sfn "$LOG_DIR/perception.log" /tmp/perception.log
ln -sfn "$LOG_DIR/saver.log" /tmp/saver.log

container_exit_cleanup() {
  cp /tmp/*.log "$LOG_DIR/" 2>/dev/null || true
}
trap container_exit_cleanup EXIT

# Occupancy-grid hygiene. rtabmap's defaults put floor, ceiling and far depth
# into the grid (whole rooms painted as obstacles). Heights are relative to
# base_link, which sits on the floor; the camera is 1.5 m up, ceilings ~2.7 m.
RTABMAP_GRID_ARGS=${RTABMAP_GRID_ARGS:-"--Grid/NormalsSegmentation false --Grid/MaxGroundHeight 0.25 --Grid/MaxObstacleHeight 1.8 --Grid/RangeMax 4.0 --Grid/RayTracing true --Grid/NoiseFilteringRadius 0.1 --Grid/NoiseFilteringMinNeighbors 5 --Grid/CellSize 0.05"}

echo ">>> starting stack (feed -> rtabmap -> perception_2 -> object_manager_6 -> web viewer :8081)"
ros2 run lost3dsg habitat_feed_node.py > /tmp/feed_node.log 2>&1 &

# ---- Class A pre-flight gate -------------------------------------------------------------
# Placed here on purpose: the feed node is up, so the TF probe has base_link -> habitat_camera
# to look at, and nothing that writes a measured artefact has started yet — so a failure costs
# a restart rather than a bundle. It is also the only correct moment for a7: the source copy
# above has happened, so this is the freeze point.
#
# The gate rides the bringup the run ALREADY pays for. It adds its own probe time (~10 s) and
# no second startup. A gate with its own two-minute bringup is a gate that gets skipped under
# pressure, and a skipped gate is worse than no gate: it manufactures confidence nobody checked.
#
# PREFLIGHT_SKIP=1 exists for stack development and is deliberately NOT silent — the bundle
# records verdict "skipped" and no digests, so no bundle can ever imply a gate that never ran.
if [ "${PREFLIGHT_SKIP:-0}" = "1" ]; then
  echo "!! PRE-FLIGHT SKIPPED (PREFLIGHT_SKIP=1) — this bundle is NOT gated"
  echo '{"verdict": "skipped", "reason": "PREFLIGHT_SKIP=1"}' > /ws/output/preflight.json
else
  echo ">>> pre-flight gate (Class A)"
  python3 /graph_api/lost3dsg/test/preflight_gate.py \
      --out /ws/output/preflight.json \
      --run-dir /ws/output \
      --scratch-dir /out \
      --run-start "${RUN_START_EPOCH:-0}" \
      --expect-config-name "$CFG_NAME" \
      --expect-config-sha "${PREFLIGHT_EXPECT_CFG_SHA:-}" \
      --expect-merged-sha "${PREFLIGHT_EXPECT_MERGED_SHA:-}" \
      --expect-src-sha "${PREFLIGHT_EXPECT_SRC_SHA:-}" \
      --expect-policy "${PREFLIGHT_EXPECT_POLICY:-}" \
    || { echo "!! PRE-FLIGHT FAILED — no measured run produced. See /ws/output/preflight.json"; exit 1; }
fi
# -------------------------------------------------------------------------------------------

# same rtabmap arguments as launch/habitat_launch.py (odometry from /odom, no TF publish)
ros2 launch rtabmap_launch rtabmap.launch.py visual_odometry:=false odom_topic:=/odom \
  rgb_topic:=/camera/rgb depth_topic:=/camera/depth camera_info_topic:=/camera/camera_info \
  approx_sync:=true rtabmap_viz:=false publish_tf:=false database_path:=/tmp/rtabmap.db \
  rtabmap_args:="--delete_db_on_start --RGBD/NeighborLinkRefining false $RTABMAP_GRID_ARGS" \
  > /tmp/rtabmap.log 2>&1 &
ros2 run lost3dsg object_manager_6.py > /tmp/om6.log 2>&1 &
python3 /ws/install/lost3dsg/lib/lost3dsg/graph_api_bridge.py > /tmp/bridge.log 2>&1 &
ros2 run lost3dsg perception_2.py > /tmp/perception.log 2>&1 &

# periodic snapshots of the annotated detection image for the host
ros2 run image_view image_saver --ros-args -r image:=/image_with_bb \
  -p filename_format:="/out/detection_%04d.png" -p sec_per_frame:=5.0 \
  > /tmp/saver.log 2>&1 &

echo ">>> stack up. logs in /tmp/*.log — tailing perception:"
touch /tmp/perception.log /tmp/om6.log
tail -f /tmp/perception.log /tmp/om6.log
