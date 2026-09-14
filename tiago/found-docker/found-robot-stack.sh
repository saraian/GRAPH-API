#!/bin/bash
# Start the Graph API / LOST-3DSG physical stack against a running TIAGo (no Gazebo).
# Robot drivers stay on the robot; this container runs RTAB-Map, perception, and Graph API.
set -euo pipefail
# shellcheck disable=SC1091
. /etc/profile.d/99-found.sh

SESSION=${FOUND_TMUX_SESSION:-found_tiago}
ACTION=${1:-start}
BAG_PATH=${TIAGO_BAG_PATH:-}
BAG_MODE=${TIAGO_BAG_MODE:-0}
[ -n "$BAG_PATH" ] && BAG_MODE=1
OUTPUT=${GRAPH_API_OUTPUT_DIR:-/ws/output}
RGB=${TIAGO_RGB_TOPIC:-/head_front_camera/rgb/image_raw}
DEPTH=${TIAGO_DEPTH_TOPIC:-/head_front_camera/depth/image_raw}
INFO=${TIAGO_CAMERA_INFO_TOPIC:-/head_front_camera/rgb/camera_info}
ODOM=${TIAGO_ODOM_TOPIC:-/mobile_base_controller/odom}
CAMERA_FRAME=${TIAGO_CAMERA_FRAME:-head_front_camera_color_optical_frame}
BRIDGE_PORT=${TIAGO_BRIDGE_PORT:-8082}
RTABMAP_DB=${TIAGO_RTABMAP_DB:-$OUTPUT/rtabmap.db}
START_PERCEPTION=${FOUND_START_PERCEPTION:-1}
START_RVIZ=${FOUND_START_RVIZ:-0}
if [ "$BAG_MODE" = 1 ]; then
  START_RTABMAP=${FOUND_START_RTABMAP:-0}
  USE_SIM_TIME=${FOUND_USE_SIM_TIME:-true}
else
  START_RTABMAP=${FOUND_START_RTABMAP:-1}
  USE_SIM_TIME=${FOUND_USE_SIM_TIME:-false}
fi
VITSAM_REQUIRE_CUDA=${VITSAM_REQUIRE_CUDA:-1}
VITSAM_CUDNN_CONV_ALGO_SEARCH=${VITSAM_CUDNN_CONV_ALGO_SEARCH:-HEURISTIC}
BAG_RATE=${TIAGO_BAG_RATE:-1.0}
BAG_LOOP=${TIAGO_BAG_LOOP:-0}
BAG_TF_DROP_FRAMES=${TIAGO_BAG_TF_DROP_FRAMES:-map}
BAG_FILTER_CONFLICTING=${TIAGO_BAG_FILTER_CONFLICTING:-}

if [ "$BAG_MODE" = 1 ]; then
  if [ -n "${GRAPH_API_CONFIG:-}" ]; then
    CONFIG=$GRAPH_API_CONFIG
  elif [ "$START_RTABMAP" = 1 ]; then
    CONFIG=/etc/found/tiago_bag_rtabmap.yaml
  else
    CONFIG=/etc/found/tiago_bag.yaml
  fi
  if [ -z "$BAG_FILTER_CONFLICTING" ]; then
    if [ "$START_RTABMAP" = 1 ]; then
      BAG_FILTER_CONFLICTING=1
    else
      BAG_FILTER_CONFLICTING=0
    fi
  fi
else
  CONFIG=${GRAPH_API_CONFIG:-/etc/found/tiago_robot.yaml}
fi

case "$START_RVIZ" in
  0|1) ;;
  *) echo "FOUND_START_RVIZ must be 0 or 1: $START_RVIZ" >&2; exit 2 ;;
esac
if [ -n "${TIAGO_RVIZ_CONFIG:-}" ]; then
  RVIZ_CONFIG=$TIAGO_RVIZ_CONFIG
elif [ "$BAG_MODE" = 1 ] && [ "$START_RTABMAP" = 1 ]; then
  RVIZ_CONFIG=/etc/found/tiago_bag_rtabmap.rviz
elif [ "$BAG_MODE" = 1 ]; then
  RVIZ_CONFIG=/etc/found/tiago_bag_recorded.rviz
else
  RVIZ_CONFIG=/etc/found/tiago_physical.rviz
fi

BAG_TOPICS=(
  /head_front_camera/rgb/image_raw
  /head_front_camera/depth/image_raw
  /head_front_camera/rgb/camera_info
  /head_front_camera/depth/camera_info
  /tf
  /tf_static
  /mobile_base_controller/odom
  /map
  /scan
  /joint_states
)

q() { printf '%q' "$1"; }

send() {
  local name=$1 cmd=$2
  local logfile=$name
  [ "$name" != object_manager ] || logfile=om6
  tmux pipe-pane -o -t "$SESSION:$name" "cat >> $(q "$OUTPUT/$logfile.log")"
  tmux send-keys -t "$SESSION:$name" "$cmd" C-m
}

new_window() {
  local name=$1 cmd=$2
  tmux new-window -t "$SESSION" -n "$name"
  tmux set-window-option -t "$SESSION:$name" remain-on-exit on
  send "$name" "$cmd"
}

ensure_ws() {
  local src="${GRAPH_API_SRC:-/graph_api/lost3dsg}"
  local stamp=/ws/install/lost3dsg/.found_build_stamp
  local legacy_marker=/ws/install/lost3dsg/share/lost3dsg/package.xml
  local rebuild=0
  if [ ! -f "$stamp" ] && [ ! -f "$legacy_marker" ]; then
    rebuild=1
    echo "lost3dsg is not installed in /ws; building it now"
  elif [ ! -f "$stamp" ]; then
    rebuild=1
    echo "lost3dsg build stamp is missing; rebuilding it now"
  elif find "$src" -type f \
      ! -path '*/__pycache__/*' \
      ! -path '*/.git/*' \
      -newer "$stamp" -print -quit | grep -q .; then
    rebuild=1
    echo "lost3dsg source is newer than /ws/install; rebuilding it now"
  fi
  if [ "$rebuild" = 1 ]; then
    found-build-ws
    # shellcheck disable=SC1091
    # Colcon and ROS setup scripts read several optional trace variables
    # directly. They are valid when unset, but this launcher uses `set -u`.
    # Keep nounset disabled only for the generated environment chain, then
    # restore the launcher's strict shell settings for the rest of the stack.
    set +u
    . /ws/install/setup.bash
    set -u
  fi
  mkdir -p "$OUTPUT" /ws
  if [ ! -f /ws/smoke_w2v.bin ]; then
    python3 - <<'PY'
import numpy as np
from gensim.models import KeyedVectors
words = "chair table sofa couch desk bed lamp door window bottle monitor robot person cabinet shelf plant book cup kitchen office room floor wall unknown".split()
kv = KeyedVectors(vector_size=32)
kv.add_vectors(words, np.random.default_rng(0).normal(size=(len(words), 32)).astype(np.float32))
kv.save_word2vec_format("/ws/smoke_w2v.bin", binary=True)
print("created /ws/smoke_w2v.bin")
PY
  fi
}

start_stack() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    echo "attach with: tmux attach -t $SESSION"
    return 0
  fi
  if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 1
  fi
  if [ "$BAG_MODE" = 1 ]; then
    if [ -z "$BAG_PATH" ] || [ ! -d "$BAG_PATH" ] || [ ! -f "$BAG_PATH/metadata.yaml" ]; then
      echo "bag mode requires a mounted rosbag directory with metadata.yaml: $BAG_PATH" >&2
      echo "run ./run_tiago.sh bag BAG_NAME with TIAGO_BAG_DIR set to the host bag directory" >&2
      exit 1
    fi
    case "$BAG_FILTER_CONFLICTING" in
      0|1) ;;
      *) echo "TIAGO_BAG_FILTER_CONFLICTING must be 0 or 1: $BAG_FILTER_CONFLICTING" >&2; exit 2 ;;
    esac
    if [ "$START_RTABMAP" = 1 ] && [ "$BAG_FILTER_CONFLICTING" != 1 ]; then
      echo "FOUND_START_RTABMAP=1 requires TIAGO_BAG_FILTER_CONFLICTING=1" >&2
      echo "the bag contains a recorded map -> odom transform" >&2
      exit 2
    fi
    if [ "$START_RTABMAP" != 1 ] && [ "$BAG_FILTER_CONFLICTING" = 1 ]; then
      echo "TIAGO_BAG_FILTER_CONFLICTING=1 requires FOUND_START_RTABMAP=1" >&2
      echo "otherwise the replay would have no global map owner" >&2
      exit 2
    fi
    if [ "$BAG_FILTER_CONFLICTING" = 1 ]; then
      echo "bag SLAM mode: /map will be omitted and TF frames touching" \
        "$(printf '%s' "$BAG_TF_DROP_FRAMES" | tr ',' ' ') will be filtered"
    fi
    if ! awk "BEGIN { exit !($BAG_RATE > 0) }" 2>/dev/null; then
      echo "TIAGO_BAG_RATE must be a positive number: $BAG_RATE" >&2
      exit 2
    fi
    case "$BAG_LOOP" in
      0|1) ;;
      *) echo "TIAGO_BAG_LOOP must be 0 or 1: $BAG_LOOP" >&2; exit 2 ;;
    esac
  fi
  if [ "$BAG_MODE" = 0 ] && [ "${ROS_LOCALHOST_ONLY:-0}" = "1" ]; then
    echo "warning: ROS_LOCALHOST_ONLY=1; this container will not see the robot's topics" >&2
    echo "         recreate with --robot (or export ROS_LOCALHOST_ONLY=0)" >&2
  fi
  if [ "$BAG_MODE" = 0 ] && [ -n "${TIAGO_ROBOT_IP:-}" ] && [ -f /usr/local/bin/found-cyclone-connect ]; then
    # shellcheck disable=SC1091
    . /usr/local/bin/found-cyclone-connect
    ros2 daemon stop >/dev/null 2>&1 || true
  fi
  ensure_ws
  local map_frame
  map_frame=$(python3 -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["tf"]["world_frame"])' "$CONFIG")
  if [ "$START_RTABMAP" = 1 ] && [ "$map_frame" = odom ]; then
    echo "Mapping world_frame must differ from odom (e.g. found_map)" >&2
    exit 1
  fi
  if [ "$START_RVIZ" = 1 ] && [ ! -f "$RVIZ_CONFIG" ]; then
    echo "RViz config not found: $RVIZ_CONFIG" >&2
    echo "set TIAGO_RVIZ_CONFIG to a config file inside the container" >&2
    exit 1
  fi

  if [ "$START_RTABMAP" = "1" ] && [ "$BAG_MODE" = 0 ]; then
    # PAL AMCL / SLAM toolbox can also parent odom, even with a different map name.
    python3 "$GRAPH_API_SRC/src/perception_module/check_mapping_tf.py"
  elif [ "$START_RTABMAP" = "1" ]; then
    echo "bag mapping: live TF ownership preflight skipped; the bag TF relay" \
      "removes the recorded global-map transforms"
  fi

  if [ "$START_PERCEPTION" = "1" ]; then
    # Fail before creating a degraded tmux session.  In particular, an NVIDIA CDI
    # mismatch can leave ONNX Runtime advertising CUDA while the real sessions fall
    # back or fail with CUDA error 999; found-check instantiates both sessions.
    echo "preflighting VitSAM CUDA sessions"
    GRAPH_API_CONFIG="$CONFIG" GRAPH_API_OUTPUT_DIR="$OUTPUT" \
      GRAPH_API_SRC="${GRAPH_API_SRC:-/graph_api/lost3dsg}" \
      VITSAM_REQUIRE_CUDA="$VITSAM_REQUIRE_CUDA" \
      VITSAM_CUDNN_CONV_ALGO_SEARCH="$VITSAM_CUDNN_CONV_ALGO_SEARCH" \
      FOUND_CHECK_REQUIRE_VLM=1 \
      /usr/local/bin/found-check
  fi

  local transport
  if [ "$BAG_MODE" = 1 ]; then
    transport="export ROS_LOCALHOST_ONLY=1"
    local ros_domain_id=${TIAGO_BAG_DOMAIN_ID:-72}
    local cyclone_uri=""
  else
    transport="unset ROS_LOCALHOST_ONLY"
    local ros_domain_id=${ROS_DOMAIN_ID:-1}
    local cyclone_uri=${CYCLONEDDS_URI:-}
  fi
  local common
  common="$transport; export ROS_DOMAIN_ID=$(q "$ros_domain_id") RMW_IMPLEMENTATION=$(q "${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}") CYCLONEDDS_URI=$(q "$cyclone_uri") PAL_ROBOT_CONNECTED=$(q "${PAL_ROBOT_CONNECTED:-0}") DISPLAY=$(q "${DISPLAY:-}") XAUTHORITY=$(q "${XAUTHORITY:-}") GRAPH_API_CONFIG=$(q "$CONFIG") GRAPH_API_OUTPUT_DIR=$(q "$OUTPUT") GRAPH_API_AUTOSTART=0 TIAGO_RTABMAP_DB=$(q "$RTABMAP_DB") BRIDGE_PORT=$(q "$BRIDGE_PORT") GRAPH_API_BASE_URL=$(q "http://127.0.0.1:$BRIDGE_PORT") VITSAM_REQUIRE_CUDA=$(q "$VITSAM_REQUIRE_CUDA") VITSAM_CUDNN_CONV_ALGO_SEARCH=$(q "$VITSAM_CUDNN_CONV_ALGO_SEARCH") TIAGO_BAG_FILTER_CONFLICTING=$(q "$BAG_FILTER_CONFLICTING") TIAGO_BAG_TF_DROP_FRAMES=$(q "$BAG_TF_DROP_FRAMES") TIAGO_RVIZ_CONFIG=$(q "$RVIZ_CONFIG") PYTHONUNBUFFERED=1; mkdir -p $(q "$OUTPUT"); cd $(q "${GRAPH_API_SRC}/src/perception_module")"

  tmux new-session -d -s "$SESSION" -n bridge
  tmux set-window-option -t "$SESSION:bridge" remain-on-exit on
  send bridge "$common; exec python3 graph_api_bridge.py --ros-args -p use_sim_time:=$(q "$USE_SIM_TIME") -r /camera/rgb:=$RGB"

  if [ "$START_PERCEPTION" = "1" ]; then
  new_window object_manager "$common; exec python3 object_manager_6.py --ros-args -p use_sim_time:=$(q "$USE_SIM_TIME")"

  new_window perception "$common; exec python3 perception_2.py --ros-args -p use_sim_time:=$(q "$USE_SIM_TIME") -r /camera/rgb:=$RGB -r /camera/depth:=$DEPTH -r /camera/camera_info:=$INFO"

  fi

  if [ "$BAG_MODE" = 1 ] && [ "$BAG_FILTER_CONFLICTING" = 1 ]; then
    new_window tf_filter "$common; exec python3 /usr/local/bin/found-tf-filter-relay.py"
  fi

  if [ "$START_RTABMAP" = "1" ]; then
    new_window rtabmap "$common; ros2 launch rtabmap_launch rtabmap.launch.py use_sim_time:=$(q "$USE_SIM_TIME") visual_odometry:=false frame_id:=base_footprint odom_topic:=$ODOM rgb_topic:=$RGB depth_topic:=$DEPTH camera_info_topic:=$INFO approx_sync:=true qos:=2 rtabmap_viz:=false rviz:=false publish_tf_map:=true map_frame_id:=$(q "$map_frame") localization:=false rtabmap_args:='--Grid/FromDepth true' database_path:=$(q "$RTABMAP_DB")"
  fi

  if [ "$START_RVIZ" = 1 ]; then
    # The Habitat-style config keeps generic /camera/* names so it can also be
    # opened manually.  Resolve them to the physical/bag camera topics here;
    # the MarkerArray topics remain global and are not remapped.
    new_window rviz "$common; exec rviz2 -d $(q "$RVIZ_CONFIG") --ros-args -p use_sim_time:=$(q "$USE_SIM_TIME") -r /camera/rgb:=$(q "$RGB") -r /camera/depth:=$(q "$DEPTH") -r /camera/camera_info:=$(q "$INFO")"
  fi

  if [ "$BAG_MODE" = 1 ]; then
    local bag_cmd
    bag_cmd="ros2 bag play $(q "$BAG_PATH") --clock --rate $(q "$BAG_RATE")"
    if [ "$BAG_LOOP" = 1 ]; then
      bag_cmd="$bag_cmd --loop"
    fi
    bag_cmd="$bag_cmd --topics"
    local topic
    for topic in "${BAG_TOPICS[@]}"; do
      if [ "$BAG_FILTER_CONFLICTING" = 1 ] && [ "$topic" = /map ]; then
        continue
      fi
      bag_cmd="$bag_cmd $(q "$topic")"
    done
    if [ "$BAG_FILTER_CONFLICTING" = 1 ]; then
      # Keep the original bag topics available to the relay while preventing
      # the recorded TF publishers from competing with RTAB-Map on /tf.
      bag_cmd="$bag_cmd --remap /tf:=/bag/tf /tf_static:=/bag/tf_static"
    fi
    new_window bag "$common; sleep 2; exec $bag_cmd"
  fi

  if [ "$BAG_MODE" = 1 ]; then
    new_window monitor "$common; echo 'Offline TIAGO rosbag replay'; echo \"  bag=$BAG_PATH\"; echo \"  rate=$BAG_RATE loop=$BAG_LOOP\"; echo \"  rgb=$RGB\"; echo \"  depth=$DEPTH\"; echo \"  info=$INFO\"; echo \"  camera frame=$CAMERA_FRAME\"; echo \"  rviz=$START_RVIZ config=$RVIZ_CONFIG\"; if [ \"$BAG_FILTER_CONFLICTING\" = 1 ]; then echo '  mode=fresh RTAB-Map SLAM (/map omitted; TF filtered)'; else echo '  mode=recorded map (/map replayed)'; fi; echo; echo 'Checks:'; echo '  ros2 topic list | grep head_front_camera'; echo '  ros2 topic echo /bbox_3d'; echo \"  curl -s http://127.0.0.1:$BRIDGE_PORT/graph_data\"; exec bash"
  else
    new_window monitor "$common; echo 'Topics (override with TIAGO_RGB_TOPIC / TIAGO_DEPTH_TOPIC / TIAGO_CAMERA_INFO_TOPIC):'; echo \"  rgb=$RGB\"; echo \"  depth=$DEPTH\"; echo \"  info=$INFO\"; echo \"  camera frame=$CAMERA_FRAME\"; echo \"  rviz=$START_RVIZ config=$RVIZ_CONFIG\"; echo; echo 'Checks:'; echo '  ros2 topic list | grep head_front_camera'; echo '  ros2 topic echo /bbox_3d'; echo \"  curl -s http://127.0.0.1:$BRIDGE_PORT/graph_data\"; exec bash"
  fi

  echo "started tmux session: $SESSION"
  echo "attach: tmux attach -t $SESSION"
  echo "viewer: http://localhost:$BRIDGE_PORT"
  echo "config: $CONFIG"
  echo "camera: $RGB  $DEPTH  ($CAMERA_FRAME)"
  if [ "$START_RVIZ" = 1 ]; then
    echo "rviz: enabled ($RVIZ_CONFIG)"
  fi
  if [ "$BAG_MODE" = 1 ]; then
    if [ "$START_RTABMAP" = 1 ]; then
      echo "bag: $BAG_PATH (rate=$BAG_RATE loop=$BAG_LOOP; fresh RTAB-Map SLAM; TF filter=$BAG_TF_DROP_FRAMES)"
    else
      echo "bag: $BAG_PATH (rate=$BAG_RATE loop=$BAG_LOOP; recorded map; RTAB-Map disabled)"
    fi
  fi
}

stop_stack() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    # Let ros2 launch send SIGINT to RTAB-Map and close SQLite before removing panes.
    for window in $(tmux list-windows -t "$SESSION" -F '#W'); do
      tmux send-keys -t "$SESSION:$window" C-c
    done
    for attempt in {1..30}; do
      busy=0
      while read -r command; do
        case "$command" in bash|sh|zsh|"") ;; *) busy=1 ;; esac
      done < <(tmux list-panes -s -t "$SESSION" -F '#{?pane_dead,,#{pane_current_command}}')
      [ "$busy" = 1 ] || break
      sleep 1
    done
    if [ "$busy" = 1 ]; then
      echo "Stack still shutting down; leaving panes intact. Inspect RTAB-Map before retrying stop." >&2
      return 1
    fi
    tmux kill-session -t "$SESSION"
    echo "stopped $SESSION"
  else
    echo "session not running: $SESSION"
  fi
}

archive_output() {
  local archive_root="${FOUND_RUN_ARCHIVE_DIR:-/ws/runs}"
  local stamp archive
  stamp=$(date -u +%Y%m%d_%H%M%S)
  archive="$archive_root/tiago_$stamp"
  [ ! -e "$archive" ] || archive="${archive}_$$"
  mkdir -p "$archive_root"
  if [ -e "$OUTPUT" ]; then
    mv -- "$OUTPUT" "$archive"
    echo "archived previous run: $archive"
  else
    echo "no previous run output at $OUTPUT"
  fi
  mkdir -p "$OUTPUT"
}

new_run() {
  stop_stack
  archive_output
  start_stack
}

case "$ACTION" in
  start) start_stack ;;
  stop) stop_stack ;;
  new) new_run ;;
  attach)
    tmux attach-session -t "$SESSION"
    ;;
  *)
    echo "usage: found-robot-stack {start|new|stop|attach}" >&2
    exit 2
    ;;
esac
