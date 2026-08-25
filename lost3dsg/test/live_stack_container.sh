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

# CFG_NAME comes from live_run.sh (regolo_config.yaml when an API key is set)
export GRAPH_API_CONFIG=/graph_api/lost3dsg/test/${CFG_NAME:-smoke_config.yaml}
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
