#!/usr/bin/env bash
# Run FOUND / LOST-3DSG with a physical TIAGo or a recorded TIAGo RGB-D bag.
#
# This is the public entry point for the TIAGo mode.  The runtime helpers and
# RViz layouts are checked into this repository under tiago/.  The PAL image
# builder, ISO and credentials stay private and are expected under TIAGO_ISO/.
# The script never derives paths from the caller's working directory.
set -euo pipefail

REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Private inputs.  TIAGO_ISO_DIR is overridable for machines that keep the
# private PAL bundle elsewhere, while the default is deliberately clone-local.
TIAGO_ISO_DIR=${TIAGO_ISO_DIR:-$REPO_DIR/TIAGO_ISO}
TIAGO_BUILDER=${TIAGO_BUILDER:-$TIAGO_ISO_DIR/uni-sap-rome-build-docker.sh}

# Public inputs.  TIAGO_HELPER_DIR is an escape hatch for a locally patched
# helper bundle, but a normal clone needs no files outside this repository and
# TIAGO_ISO.
TIAGO_HELPER_DIR=${TIAGO_HELPER_DIR:-$REPO_DIR/tiago/found-docker}
TIAGO_HOST_DDS=${TIAGO_HOST_DDS:-$REPO_DIR/tiago/tiago-host-dds.sh}

# Bags are part of the checkout's local runtime layout.  On a development
# machine this directory may be a symlink to a larger storage volume; callers
# can still override it with TIAGO_BAG_DIR when needed.
TIAGO_BAG_DIR_EXPLICIT=0
if [ -n "${TIAGO_BAG_DIR+x}" ]; then
  TIAGO_BAG_DIR_EXPLICIT=1
fi
TIAGO_BAG_ROOT=${TIAGO_BAG_ROOT:-$REPO_DIR/bags}
TIAGO_BAG_DIR=${TIAGO_BAG_DIR:-$TIAGO_BAG_ROOT}
TIAGO_BAG_DIR=${TIAGO_BAG_DIR%/}

# The private builder was written for an outer FOUND checkout and looks for its
# Docker build context next to itself.  A generated staging directory lets us
# use the public context without copying public files into the private bundle.
TIAGO_BUILDER_STAGE=${TIAGO_BUILDER_STAGE:-$REPO_DIR/.tiago-runtime}

TIAGO_DOCKER_TARGET=${TIAGO_DOCKER_TARGET:-tiago-127}

usage() {
  cat <<EOF
Usage: $(basename "$0") physical [--resume] [options]
       $(basename "$0") bag <bag-name-or-path> [--resume] [options]
       $(basename "$0") {check|build|firewall|start|bag|new|stop|attach|shell} [container]

The command is independent of the current working directory.  Public TIAGo
runtime files live in:
  $REPO_DIR/tiago/

Private PAL inputs are expected in:
  $TIAGO_ISO_DIR/
  uni-sap-rome-build-docker.sh, ROS2_Alum.iso, keys/ (and any PAL files it needs)

Actions:
  physical
             start a new physical TIAGo run; use --resume to reuse the current run
  bag BAG [--rtabmap]
             start a new RGB-D rosbag run; use --resume to reuse the current run.
             A relative BAG is resolved under $REPO_DIR/bags; an absolute CLI
             path is also accepted
  check      verify the Python/VitSAM runtime in an existing container
  build      build the lost3dsg ROS package into the /ws volume
  firewall   allow host DDS traffic from the physical robot (needs sudo)
  start      resume the current run (compatibility form; canonical forms use --resume)
  new        archive the current /ws/output and start a new run explicitly
  stop       stop the tmux stack cleanly
  attach     attach to the stack's tmux session
  shell      open a login shell in the container

Physical mode defaults:
  TIAGO_ROBOT_IP=10.68.0.1
  TIAGO_ROS_DOMAIN_ID=1
  TIAGO_NET_IFACE is detected from the route to TIAGO_ROBOT_IP
  --rviz/--no-rviz and --no-perception/--perception control optional windows.

Bag mode:
  ./run_tiago.sh physical
  ./run_tiago.sh physical --resume --rviz
  ./run_tiago.sh bag BAG_NAME
  ./run_tiago.sh bag BAG_NAME --rtabmap
  ./run_tiago.sh bag bags/BAG_NAME --rtabmap --rviz --resume
  TIAGO_BAG_PATH=/path/to/bag ./run_tiago.sh new
  TIAGO_BAG_PATH=/path/to/bag ./run_tiago.sh start  # compatibility resume form
  TIAGO_BAG_PATH may be a host path under TIAGO_BAG_DIR or a container path
  under /bags.  TIAGO_BAG_DIR defaults to $REPO_DIR/bags.
  TIAGO_BAG_RATE=1.0 and TIAGO_BAG_LOOP=1 control playback.
  --rtabmap selects fresh RTAB-Map SLAM; --recorded-map selects the bag map.
  --rviz/--no-rviz and --no-perception/--perception control optional windows.
  FOUND_START_RTABMAP=1 starts fresh RTAB-Map SLAM, omits /map and filters
  recorded transforms touching map.  TIAGO_BAG_FILTER_CONFLICTING is selected
  automatically, and TIAGO_BAG_TF_DROP_FRAMES customizes the frame list.

RViz and startup:
  FOUND_START_RVIZ=1 enables RViz2 in its own tmux window in physical and bag modes.
  TIAGO_RVIZ_CONFIG overrides the mode-specific config inside the container.
  FOUND_START_PERCEPTION=0 starts mapping/dashboard without VLM perception.
  REGOLO_API_KEY may be exported; otherwise physical/bag/start/new prompt for it
  when perception is enabled.

Container bootstrap:
  A missing canonical container is created automatically by the private builder
  on physical/bag/start/build/new/shell.  Set TIAGO_AUTO_CREATE=0 to require
  manual creation.
  TIAGO_DOCKER_UPDATE=1 passes --update to the private builder.
  TIAGO_DOCKER_TARGET=tiago-127 selects the PAL target (tiago-127 or tiago-130).
  TIAGO_FOUND_ROOT may point to an outer FOUND checkout.  It is detected when
  this clone is at <FOUND>/vendor/graph-api; standalone clones use a generated
  lightweight mount root automatically.

The old VITSAM_WARMUP and VITSAM_REQUIRE_WARMUP variables are obsolete.  The
current startup check instantiates the real CUDA encoder and decoder sessions.
EOF
  exit "${1:-1}"
}

ACTION=${1:-}
[ -n "$ACTION" ] || usage
ORIGINAL_ACTION=$ACTION
case "$ACTION" in
  -h|--help) usage 0 ;;
esac

# The concise physical/bag forms are parsed here so run_tiago.sh remains the
# only public entry point. The old action names stay available too.
CLI_BAG_PATH=""
CLI_CONTAINER=""
CLI_RTABMAP=""
CLI_RVIZ=""
CLI_PERCEPTION=""
CLI_RATE=""
CLI_LOOP=""
CLI_RESUME=""

case "$ACTION" in
  physical)
    # Concise modes are intentionally fresh by default.  --resume changes the
    # lifecycle action back to start after all options have been parsed.
    ACTION=new
    shift
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --resume)
          CLI_RESUME=1
          shift
          ;;
        --rtabmap)
          CLI_RTABMAP=1
          shift
          ;;
        --recorded-map|--no-rtabmap)
          CLI_RTABMAP=0
          shift
          ;;
        --rviz)
          CLI_RVIZ=1
          shift
          ;;
        --no-rviz)
          CLI_RVIZ=0
          shift
          ;;
        --no-perception)
          CLI_PERCEPTION=0
          shift
          ;;
        --perception)
          CLI_PERCEPTION=1
          shift
          ;;
        --container)
          [ "$#" -ge 2 ] || { echo "--container requires a name" >&2; usage 2; }
          [ -z "$CLI_CONTAINER" ] || { echo "container specified twice" >&2; usage 2; }
          CLI_CONTAINER=$2
          shift 2
          ;;
        --container=*)
          [ -z "$CLI_CONTAINER" ] || { echo "container specified twice" >&2; usage 2; }
          CLI_CONTAINER=${1#--container=}
          [ -n "$CLI_CONTAINER" ] || { echo "--container requires a name" >&2; usage 2; }
          shift
          ;;
        -h|--help)
          usage 0
          ;;
        -*)
          echo "unknown physical option: $1" >&2
          usage 2
          ;;
        *)
          [ -z "$CLI_CONTAINER" ] || { echo "physical accepts one container name" >&2; usage 2; }
          CLI_CONTAINER=$1
          shift
          ;;
      esac
    done
    ;;
  bag)
    # See the physical mode above: the public bag form starts a fresh run by
    # default, while --resume is the explicit opt-in to reuse the old one.
    ACTION=new
    shift
    # `bag CONTAINER` was the old environment-driven spelling. Keep it
    # working when TIAGO_BAG_PATH/TIAGO_ROSBAG is already set; otherwise the
    # first positional argument is the new repository-relative bag path.
    if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
      if [ -n "${TIAGO_BAG_PATH:-${TIAGO_ROSBAG:-}}" ]; then
        CLI_CONTAINER=$1
      else
        CLI_BAG_PATH=$1
      fi
      shift
    fi
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --resume)
          CLI_RESUME=1
          shift
          ;;
        --rtabmap)
          CLI_RTABMAP=1
          shift
          ;;
        --recorded-map|--no-rtabmap)
          CLI_RTABMAP=0
          shift
          ;;
        --rviz)
          CLI_RVIZ=1
          shift
          ;;
        --no-rviz)
          CLI_RVIZ=0
          shift
          ;;
        --no-perception)
          CLI_PERCEPTION=0
          shift
          ;;
        --perception)
          CLI_PERCEPTION=1
          shift
          ;;
        --rate)
          [ "$#" -ge 2 ] || { echo "--rate requires a value" >&2; usage 2; }
          CLI_RATE=$2
          shift 2
          ;;
        --rate=*)
          CLI_RATE=${1#--rate=}
          [ -n "$CLI_RATE" ] || { echo "--rate requires a value" >&2; usage 2; }
          shift
          ;;
        --loop)
          CLI_LOOP=1
          shift
          ;;
        --no-loop)
          CLI_LOOP=0
          shift
          ;;
        --container)
          [ "$#" -ge 2 ] || { echo "--container requires a name" >&2; usage 2; }
          [ -z "$CLI_CONTAINER" ] || { echo "container specified twice" >&2; usage 2; }
          CLI_CONTAINER=$2
          shift 2
          ;;
        --container=*)
          [ -z "$CLI_CONTAINER" ] || { echo "container specified twice" >&2; usage 2; }
          CLI_CONTAINER=${1#--container=}
          [ -n "$CLI_CONTAINER" ] || { echo "--container requires a name" >&2; usage 2; }
          shift
          ;;
        --bag)
          [ "$#" -ge 2 ] || { echo "--bag requires a name or path" >&2; usage 2; }
          [ -z "$CLI_BAG_PATH" ] || { echo "bag specified twice" >&2; usage 2; }
          CLI_BAG_PATH=$2
          shift 2
          ;;
        --bag=*)
          [ -z "$CLI_BAG_PATH" ] || { echo "bag specified twice" >&2; usage 2; }
          CLI_BAG_PATH=${1#--bag=}
          [ -n "$CLI_BAG_PATH" ] || { echo "--bag requires a name or path" >&2; usage 2; }
          shift
          ;;
        --)
          shift
          [ "$#" -eq 1 ] || { echo "only one container name may follow --" >&2; usage 2; }
          [ -z "$CLI_CONTAINER" ] || { echo "container specified twice" >&2; usage 2; }
          CLI_CONTAINER=$1
          shift
          ;;
        -h|--help)
          usage 0
          ;;
        -*)
          echo "unknown bag option: $1" >&2
          usage 2
          ;;
        *)
          if [ -z "$CLI_BAG_PATH" ] && [ -z "${TIAGO_BAG_PATH:-${TIAGO_ROSBAG:-}}" ]; then
            CLI_BAG_PATH=$1
          else
            [ -z "$CLI_CONTAINER" ] || { echo "container specified twice" >&2; usage 2; }
            CLI_CONTAINER=$1
          fi
          shift
          ;;
      esac
    done
    ;;
  check|build|firewall|start|new|stop|attach|shell)
    ;;
  *)
    usage
    ;;
esac

if [ -n "$CLI_RESUME" ]; then
  ACTION=start
fi

# `physical` and `bag` have their own option parser, so their container is
# always the optional parsed value.  The compatibility actions keep the old
# positional [container] contract.
if [ "$ORIGINAL_ACTION" = physical ] || [ "$ORIGINAL_ACTION" = bag ]; then
  CONTAINER=${CLI_CONTAINER:-$TIAGO_DOCKER_TARGET-dev}
else
  CONTAINER=${2:-$TIAGO_DOCKER_TARGET-dev}
fi

case "$TIAGO_DOCKER_TARGET" in
  tiago-127|tiago-130) ;;
  *)
    echo "TIAGO_DOCKER_TARGET must be tiago-127 or tiago-130: $TIAGO_DOCKER_TARGET" >&2
    exit 2
    ;;
esac

TIAGO_BAG_PATH=${TIAGO_BAG_PATH:-${TIAGO_ROSBAG:-}}
if [ -n "$CLI_BAG_PATH" ]; then
  case "$CLI_BAG_PATH" in
    bags/*)
      TIAGO_BAG_PATH="$REPO_DIR/$CLI_BAG_PATH"
      ;;
    ./bags/*)
      TIAGO_BAG_PATH="$REPO_DIR/${CLI_BAG_PATH#./}"
      ;;
    /bags/*|/*)
      TIAGO_BAG_PATH="$CLI_BAG_PATH"
      if [ "$TIAGO_BAG_DIR_EXPLICIT" = 0 ] && [[ "$CLI_BAG_PATH" != /bags/* ]]; then
        # Keep absolute CLI paths useful without changing the default: the
        # directory containing this bag becomes the bind-mounted bag root.
        TIAGO_BAG_DIR=$(dirname -- "$CLI_BAG_PATH")
        TIAGO_BAG_DIR=${TIAGO_BAG_DIR%/}
      fi
      ;;
    *)
      TIAGO_BAG_PATH="$TIAGO_BAG_DIR/$CLI_BAG_PATH"
      ;;
  esac
fi
# Keep the environment-driven form consistent with the CLI form: an absolute
# host path can be used directly, and its parent becomes the bind-mounted bag
# root unless the caller explicitly selected TIAGO_BAG_DIR.  /bags/... is
# already a container path and therefore must keep the configured host root.
if [ -z "$CLI_BAG_PATH" ] && [ "$TIAGO_BAG_DIR_EXPLICIT" = 0 ]; then
  case "$TIAGO_BAG_PATH" in
    /bags/*|"") ;;
    /*)
      TIAGO_BAG_DIR=$(dirname -- "${TIAGO_BAG_PATH%/}")
      TIAGO_BAG_DIR=${TIAGO_BAG_DIR%/}
      ;;
  esac
fi
if [ -n "$CLI_RTABMAP" ]; then
  FOUND_START_RTABMAP=$CLI_RTABMAP
fi
if [ -n "$CLI_RVIZ" ]; then
  FOUND_START_RVIZ=$CLI_RVIZ
fi
if [ -n "$CLI_PERCEPTION" ]; then
  FOUND_START_PERCEPTION=$CLI_PERCEPTION
fi
if [ -n "$CLI_RATE" ]; then
  TIAGO_BAG_RATE=$CLI_RATE
fi
if [ -n "$CLI_LOOP" ]; then
  TIAGO_BAG_LOOP=$CLI_LOOP
fi
BAG_MODE=0
if [ "$ORIGINAL_ACTION" = bag ] || [ -n "$TIAGO_BAG_PATH" ]; then
  BAG_MODE=1
fi

if [ "$BAG_MODE" = 1 ]; then
  if [ -z "$TIAGO_BAG_PATH" ]; then
    echo "bag mode requires BAG_NAME or TIAGO_BAG_PATH=/path/to/rosbag-directory" >&2
    echo "example: ./run_tiago.sh bag rosbag2_..." >&2
    exit 2
  fi
  case "$TIAGO_BAG_PATH" in
    /bags/*)
      bag_host_path="$TIAGO_BAG_DIR/${TIAGO_BAG_PATH#/bags/}"
      bag_container_path="$TIAGO_BAG_PATH"
      ;;
    "$TIAGO_BAG_DIR"/*)
      bag_host_path="$TIAGO_BAG_PATH"
      bag_container_path="/bags/${TIAGO_BAG_PATH#"$TIAGO_BAG_DIR"/}"
      ;;
    /*)
      echo "TIAGO_BAG_PATH must be under $TIAGO_BAG_DIR or already use /bags/..." >&2
      echo "set TIAGO_BAG_DIR to the host directory containing the bag" >&2
      exit 2
      ;;
    *)
      # Keep the environment form convenient as well: a relative bag name is
      # interpreted relative to the repository-local bag root.
      bag_host_path="$TIAGO_BAG_DIR/$TIAGO_BAG_PATH"
      bag_container_path="/bags/$TIAGO_BAG_PATH"
      ;;
  esac
  if [ ! -d "$bag_host_path" ] || [ ! -f "$bag_host_path/metadata.yaml" ]; then
    echo "rosbag directory or metadata.yaml not found: $bag_host_path" >&2
    exit 1
  fi
  TIAGO_BAG_PATH="$bag_container_path"
  TIAGO_ROBOT_IP=""
  TIAGO_BAG_DOMAIN_ID=${TIAGO_BAG_DOMAIN_ID:-72}
  TIAGO_ROS_DOMAIN_ID=$TIAGO_BAG_DOMAIN_ID
  TIAGO_NET_IFACE=""
  TIAGO_BAG_MODE=1
  FOUND_START_RTABMAP=${FOUND_START_RTABMAP:-0}
  if [ "$FOUND_START_RTABMAP" = 1 ]; then
    TIAGO_BAG_FILTER_CONFLICTING=${TIAGO_BAG_FILTER_CONFLICTING:-1}
  else
    TIAGO_BAG_FILTER_CONFLICTING=${TIAGO_BAG_FILTER_CONFLICTING:-0}
  fi
  FOUND_USE_SIM_TIME=${FOUND_USE_SIM_TIME:-true}
  if [ -z "${GRAPH_API_CONFIG:-}" ]; then
    if [ "$FOUND_START_RTABMAP" = 1 ]; then
      GRAPH_API_CONFIG=/etc/found/tiago_bag_rtabmap.yaml
    else
      GRAPH_API_CONFIG=/etc/found/tiago_bag.yaml
    fi
  fi
  TIAGO_BAG_RATE=${TIAGO_BAG_RATE:-1.0}
  TIAGO_BAG_LOOP=${TIAGO_BAG_LOOP:-0}
else
  TIAGO_ROBOT_IP=${TIAGO_ROBOT_IP:-10.68.0.1}
  TIAGO_ROS_DOMAIN_ID=${TIAGO_ROS_DOMAIN_ID:-1}
  TIAGO_NET_IFACE=${TIAGO_NET_IFACE:-}
fi

# Use the bind-mounted checkout as the only source of code.  The historical
# /graph_api_live copy is intentionally not accepted because it can be stale.
case "${GRAPH_API_SRC:-}" in
  /graph_api_live|/graph_api_live/lost3dsg)
    echo "warning: GRAPH_API_SRC=$GRAPH_API_SRC is obsolete; using /graph_api/lost3dsg" >&2
    GRAPH_API_SRC=/graph_api/lost3dsg
    ;;
  "") GRAPH_API_SRC=/graph_api/lost3dsg ;;
esac

export TIAGO_BAG_PATH TIAGO_BAG_MODE TIAGO_BAG_DOMAIN_ID FOUND_START_RTABMAP FOUND_USE_SIM_TIME GRAPH_API_CONFIG GRAPH_API_SRC

detect_iface() {
  ip -o route get "$TIAGO_ROBOT_IP" 2>/dev/null | awk '{
    for (i = 1; i <= NF; i++) if ($i == "dev") { print $(i + 1); exit }
  }'
}

if [ "$BAG_MODE" = 0 ] && [ -z "$TIAGO_NET_IFACE" ]; then
  TIAGO_NET_IFACE=$(detect_iface || true)
fi

require_public_assets() {
  local missing=0 rel
  for rel in \
    found-profile.sh \
    found-cyclone-connect.sh \
    found-build-ws.sh \
    found-check.sh \
    found-robot-stack.sh \
    tf_filter_relay.py \
    tiago_robot.yaml \
    tiago_bag.yaml \
    tiago_bag_rtabmap.yaml \
    tiago_live.rviz; do
    if [ ! -f "$TIAGO_HELPER_DIR/$rel" ]; then
      echo "missing TIAGo runtime asset: $TIAGO_HELPER_DIR/$rel" >&2
      missing=1
    fi
  done
  [ "$missing" = 0 ] || exit 1
  [ -f "$TIAGO_HOST_DDS" ] || {
    echo "missing host DDS helper: $TIAGO_HOST_DDS" >&2
    exit 1
  }
}

require_docker() {
  command -v docker >/dev/null 2>&1 || {
    echo "Docker is required for TIAGo actions; install Docker and retry" >&2
    exit 1
  }
}

host_dds() {
  if [ "$BAG_MODE" = 1 ]; then
    echo "skipping host DDS/firewall setup in bag mode"
    return 0
  fi
  if [ "${SKIP_TIAGO_HOST_DDS:-0}" = 1 ]; then
    echo "skipping host DDS firewall (SKIP_TIAGO_HOST_DDS=1)"
    return 0
  fi
  [ -x "$(command -v bash)" ] || {
    echo "bash is required for the host DDS helper" >&2
    exit 1
  }
  local args=(apply --robot-ip "$TIAGO_ROBOT_IP")
  if [ -n "$TIAGO_NET_IFACE" ]; then
    args+=(--interface "$TIAGO_NET_IFACE")
  fi
  bash "$TIAGO_HOST_DDS" "${args[@]}"
}

prompt_regolo_api_key() {
  if [ "${FOUND_START_PERCEPTION:-1}" = 0 ] || [ -n "${REGOLO_API_KEY:-}" ]; then
    return 0
  fi

  local key=""
  if command -v zenity >/dev/null 2>&1 && {
    [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ];
  }; then
    key=$(zenity --password --title="Regolo API key" \
      --text="Enter the Regolo API key for this Graph API run:" 2>/dev/null) || true
  elif command -v kdialog >/dev/null 2>&1 && {
    [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ];
  }; then
    key=$(kdialog --title "Regolo API key" \
      --password "Enter the Regolo API key for this Graph API run:") || true
  elif [ -t 0 ]; then
    printf 'Regolo API key: ' >&2
    read -r -s key
    printf '\n' >&2
  else
    echo "REGOLO_API_KEY is unset and no GUI/TTY prompt is available" >&2
    echo "export REGOLO_API_KEY='...' and retry" >&2
    return 1
  fi

  if [ -z "$key" ]; then
    echo "No Regolo API key was supplied; refusing to start perception" >&2
    return 1
  fi
  export REGOLO_API_KEY="$key"
  unset key
}

sync_container_helpers() {
  require_public_assets
  docker cp "$TIAGO_HELPER_DIR/found-profile.sh" \
    "$CONTAINER:/etc/profile.d/99-found.sh"
  docker cp "$TIAGO_HELPER_DIR/found-cyclone-connect.sh" \
    "$CONTAINER:/usr/local/bin/found-cyclone-connect"
  docker cp "$TIAGO_HELPER_DIR/found-build-ws.sh" \
    "$CONTAINER:/usr/local/bin/found-build-ws"
  docker cp "$TIAGO_HELPER_DIR/found-check.sh" \
    "$CONTAINER:/usr/local/bin/found-check"
  docker cp "$TIAGO_HELPER_DIR/found-robot-stack.sh" \
    "$CONTAINER:/usr/local/bin/found-robot-stack"
  docker cp "$TIAGO_HELPER_DIR/tf_filter_relay.py" \
    "$CONTAINER:/usr/local/bin/found-tf-filter-relay.py"
  docker cp "$TIAGO_HELPER_DIR/tiago_robot.yaml" \
    "$CONTAINER:/etc/found/tiago_robot.yaml"
  docker cp "$TIAGO_HELPER_DIR/tiago_bag.yaml" \
    "$CONTAINER:/etc/found/tiago_bag.yaml"
  docker cp "$TIAGO_HELPER_DIR/tiago_bag_rtabmap.yaml" \
    "$CONTAINER:/etc/found/tiago_bag_rtabmap.yaml"
  docker cp "$TIAGO_HELPER_DIR/tiago_live.rviz" \
    "$CONTAINER:/etc/found/tiago_live.rviz"

  # The checked-in RViz layout is shared by all modes.  Only the fixed frame
  # and recorded-vs-fresh map topic differ between the generated defaults.
  docker exec -u root "$CONTAINER" sh -eu -c '
    source_config=/etc/found/tiago_live.rviz
    test -r "$source_config"
    cp "$source_config" /etc/found/tiago_physical.rviz
    cp "$source_config" /etc/found/tiago_bag_rtabmap.rviz
    cp "$source_config" /etc/found/tiago_bag_recorded.rviz
    sed -i "s/^    Fixed Frame: map$/    Fixed Frame: found_map/" \
      /etc/found/tiago_physical.rviz
    sed -i \
      -e "s|Value: /rtabmap/map$|Value: /map|" \
      -e "s|Value: /rtabmap/map_updates$|Value: /map_updates|" \
      /etc/found/tiago_bag_recorded.rviz
    chmod 644 /etc/found/tiago_physical.rviz \
      /etc/found/tiago_bag_rtabmap.rviz /etc/found/tiago_bag_recorded.rviz
  '
  docker exec -u root "$CONTAINER" chmod 755 \
    /usr/local/bin/found-cyclone-connect \
    /usr/local/bin/found-build-ws \
    /usr/local/bin/found-check \
    /usr/local/bin/found-robot-stack \
    /usr/local/bin/found-tf-filter-relay.py
  docker exec -u root "$CONTAINER" chmod 644 \
    /etc/profile.d/99-found.sh \
    /etc/found/tiago_robot.yaml \
    /etc/found/tiago_bag.yaml \
    /etc/found/tiago_bag_rtabmap.yaml \
    /etc/found/tiago_live.rviz
}

resolve_found_root() {
  local explicit=${TIAGO_FOUND_ROOT:-${FOUND_ROOT:-}}
  local repo_real candidate_real
  repo_real=$(readlink -f "$REPO_DIR")

  if [ -n "$explicit" ]; then
    [ -d "$explicit" ] || {
      echo "TIAGO_FOUND_ROOT does not exist: $explicit" >&2
      exit 1
    }
    if [ ! -d "$explicit/vendor/graph-api" ] || \
       [ "$(readlink -f "$explicit/vendor/graph-api")" != "$repo_real" ]; then
      echo "TIAGO_FOUND_ROOT must contain this checkout at vendor/graph-api:" >&2
      echo "  $explicit/vendor/graph-api" >&2
      echo "or omit TIAGO_FOUND_ROOT for standalone-clone mode" >&2
      exit 1
    fi
    printf '%s\n' "$(readlink -f "$explicit")"
    return 0
  fi

  candidate_real=$(readlink -f "$REPO_DIR/../.." 2>/dev/null || true)
  if [ -d "$candidate_real/vendor/graph-api" ] && \
     [ "$(readlink -f "$candidate_real/vendor/graph-api")" = "$repo_real" ]; then
    printf '%s\n' "$candidate_real"
    return 0
  fi

  # The private builder always mounts /found and only adds /graph_api when it
  # sees vendor/graph-api below that root.  This shim preserves the builder's
  # contract for people who clone graph-api by itself.
  local shim="$REPO_DIR/.tiago-found-root"
  local link="$shim/vendor/graph-api"
  mkdir -p "$shim/vendor"
  if [ -e "$link" ] || [ -L "$link" ]; then
    if [ "$(readlink -f "$link")" != "$repo_real" ]; then
      rm -f -- "$link"
      ln -s -- "$repo_real" "$link"
    fi
  else
    ln -s -- "$repo_real" "$link"
  fi
  printf '%s\n' "$shim"
}

private_apt_key() {
  local candidate
  for candidate in \
    "$TIAGO_ISO_DIR/found-docker/pal-apt-keys.deb" \
    "$TIAGO_ISO_DIR/pal-apt-keys.deb" \
    "$TIAGO_ISO_DIR/keys/pal-apt-keys.deb"; do
    if [ -f "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  if [ -d "$TIAGO_ISO_DIR/keys" ]; then
    candidate=$(find "$TIAGO_ISO_DIR/keys" -type f -name pal-apt-keys.deb -print -quit)
    if [ -n "$candidate" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  return 1
}

stage_private_builder() {
  require_public_assets
  [ -f "$TIAGO_BUILDER" ] || {
    echo "missing private TIAGo Docker builder: $TIAGO_BUILDER" >&2
    echo "put uni-sap-rome-build-docker.sh under $TIAGO_ISO_DIR" >&2
    exit 1
  }
  mkdir -p "$TIAGO_BUILDER_STAGE/found-docker"
  # cp is runtime staging, not a second source tree: the checked-in helpers
  # remain authoritative and are synced again on every run.
  cp -a "$TIAGO_HELPER_DIR/." "$TIAGO_BUILDER_STAGE/found-docker/"
  local key
  if key=$(private_apt_key); then
    cp -- "$key" "$TIAGO_BUILDER_STAGE/found-docker/pal-apt-keys.deb"
  elif [ "${TIAGO_DOCKER_UPDATE:-0}" = 1 ] || \
       ! docker image inspect "development-$TIAGO_DOCKER_TARGET:alum-25.01" >/dev/null 2>&1; then
    echo "the private builder needs pal-apt-keys.deb to build/update the TIAGo image" >&2
    echo "place it under $TIAGO_ISO_DIR/found-docker/, $TIAGO_ISO_DIR/, or $TIAGO_ISO_DIR/keys/" >&2
    exit 1
  fi
  local staged_builder="$TIAGO_BUILDER_STAGE/uni-sap-rome-build-docker.sh"
  if [ -e "$staged_builder" ] || [ -L "$staged_builder" ]; then
    rm -f -- "$staged_builder"
  fi
  ln -s -- "$TIAGO_BUILDER" "$staged_builder"
  printf '%s\n' "$staged_builder"
}

create_container() {
  local builder_script found_root suffix
  builder_script=$(stage_private_builder)
  found_root=$(resolve_found_root)

  # The private builder names containers as <target>-<suffix>.  Keep the old
  # [container] argument useful while avoiding an unexpected second container.
  case "$CONTAINER" in
    "$TIAGO_DOCKER_TARGET"-*) suffix="${CONTAINER#"$TIAGO_DOCKER_TARGET"-}" ;;
    *)
      echo "cannot auto-create arbitrary container name '$CONTAINER'" >&2
      echo "use a name of the form $TIAGO_DOCKER_TARGET-<suffix>, or create it manually" >&2
      exit 1
      ;;
  esac
  [ -n "$suffix" ] || suffix=dev

  local args=(--create --robot --name "$suffix" --found-root "$found_root")
  if [ -d "$TIAGO_BAG_DIR" ]; then
    args+=(--bag-dir "$TIAGO_BAG_DIR")
  fi
  if [ "${TIAGO_DOCKER_UPDATE:-0}" = 1 ]; then
    args+=(--update)
  fi
  args+=("$TIAGO_DOCKER_TARGET")

  echo "creating $CONTAINER with the private TIAGo Docker builder"
  bash "$builder_script" "${args[@]}"
}

ensure_container() {
  require_docker
  if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    case "$ACTION" in
      start|bag|build|new|shell)
        if [ "${TIAGO_AUTO_CREATE:-1}" = 1 ]; then
          create_container
        else
          echo "container '$CONTAINER' does not exist" >&2
          echo "retry with TIAGO_AUTO_CREATE=1 or create it with the private builder" >&2
          exit 1
        fi
        ;;
      *)
        echo "container '$CONTAINER' does not exist" >&2
        echo "run ./run_tiago.sh physical to bootstrap the default container" >&2
        exit 1
        ;;
    esac
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]; then
    echo "starting $CONTAINER"
    docker start "$CONTAINER" >/dev/null
  fi
}

exec_user() {
  if [ -t 0 ] && [ -t 1 ]; then
    docker exec -u user -it "$@"
  else
    docker exec -u user "$@"
  fi
}

make_robot_env() {
  if [ "$BAG_MODE" = 1 ]; then
    robot_env=(
      -e "TIAGO_ROBOT_IP="
      -e "TIAGO_ROS_DOMAIN_ID=$TIAGO_ROS_DOMAIN_ID"
      -e "TIAGO_BAG_DOMAIN_ID=${TIAGO_BAG_DOMAIN_ID:-}"
      -e "ROS_DOMAIN_ID=$TIAGO_ROS_DOMAIN_ID"
      -e "ROS_LOCALHOST_ONLY=1"
      -e "TIAGO_BAG_MODE=1"
      -e "TIAGO_BAG_PATH=$TIAGO_BAG_PATH"
      -e "GRAPH_API_CONFIG=${GRAPH_API_CONFIG:-}"
      -e "GRAPH_API_OUTPUT_DIR=${GRAPH_API_OUTPUT_DIR:-/ws/output}"
    )
  else
    robot_env=(
      -e "TIAGO_ROBOT_IP=$TIAGO_ROBOT_IP"
      -e "TIAGO_ROS_DOMAIN_ID=$TIAGO_ROS_DOMAIN_ID"
      -e "ROS_DOMAIN_ID=$TIAGO_ROS_DOMAIN_ID"
      -e "ROS_LOCALHOST_ONLY="
      -e "GRAPH_API_OUTPUT_DIR=${GRAPH_API_OUTPUT_DIR:-/ws/output}"
    )
  fi

  for key in \
    FOUND_TMUX_SESSION \
    FOUND_START_PERCEPTION \
    FOUND_START_RTABMAP \
    FOUND_START_RVIZ \
    FOUND_USE_SIM_TIME \
    GRAPH_API_CONFIG \
    GRAPH_API_OUTPUT_DIR \
    GRAPH_API_SRC \
    TIAGO_RTABMAP_DB \
    TIAGO_BRIDGE_PORT \
    TIAGO_MERGE_INTERVAL_S \
    TIAGO_BAG_RATE \
    TIAGO_BAG_LOOP \
    TIAGO_BAG_DOMAIN_ID \
    TIAGO_BAG_FILTER_CONFLICTING \
    TIAGO_BAG_TF_DROP_FRAMES \
    TIAGO_RVIZ_CONFIG \
    TIAGO_RGB_TOPIC \
    TIAGO_DEPTH_TOPIC \
    TIAGO_CAMERA_INFO_TOPIC \
    TIAGO_ODOM_TOPIC \
    TIAGO_CAMERA_FRAME \
    VITSAM_REQUIRE_CUDA \
    VITSAM_CUDNN_CONV_ALGO_SEARCH \
    REGOLO_API_KEY \
    OPENAI_API_KEY; do
    if [ -n "${!key:-}" ]; then
      robot_env+=(-e "$key=${!key}")
    fi
  done

  if [ -n "${DISPLAY:-}" ]; then
    robot_env+=(-e "DISPLAY=$DISPLAY")
  fi
  if [ -n "${XAUTHORITY:-}" ] && [ -f "$XAUTHORITY" ]; then
    local exec_xauthority=$XAUTHORITY
    if [ -n "${XDG_RUNTIME_DIR:-}" ]; then
      case "$exec_xauthority" in
        "$XDG_RUNTIME_DIR"/*)
          exec_xauthority="/tmp/local-xdg-runtime/${exec_xauthority#"$XDG_RUNTIME_DIR"/}"
          ;;
      esac
    fi
    robot_env+=(-e "XAUTHORITY=$exec_xauthority")
  fi
  if [ "$BAG_MODE" = 1 ]; then
    robot_env+=(
      -e "TIAGO_BAG_MODE=1" -e "TIAGO_BAG_PATH=$TIAGO_BAG_PATH"
    )
  fi
  if [ -n "$TIAGO_NET_IFACE" ]; then
    robot_env+=(-e "TIAGO_NET_IFACE=$TIAGO_NET_IFACE")
  fi
}

make_robot_env

case "$ACTION" in
  firewall)
    host_dds
    ;;
  check)
    ensure_container
    sync_container_helpers
    exec_user "$CONTAINER" bash -lc found-check
    ;;
  build)
    ensure_container
    sync_container_helpers
    exec_user "${robot_env[@]}" "$CONTAINER" bash -lc found-build-ws
    ;;
  start|bag)
    prompt_regolo_api_key
    make_robot_env
    host_dds
    ensure_container
    sync_container_helpers
    if [ "$BAG_MODE" = 1 ]; then
      exec_user "${robot_env[@]}" "$CONTAINER" bash -lc 'found-robot-stack start'
    else
      exec_user "${robot_env[@]}" "$CONTAINER" bash -lc 'unset ROS_LOCALHOST_ONLY; found-robot-stack start'
    fi
    ;;
  new)
    prompt_regolo_api_key
    make_robot_env
    host_dds
    ensure_container
    sync_container_helpers
    if [ "$BAG_MODE" = 1 ]; then
      exec_user "${robot_env[@]}" "$CONTAINER" bash -lc 'found-robot-stack new'
    else
      exec_user "${robot_env[@]}" "$CONTAINER" bash -lc 'unset ROS_LOCALHOST_ONLY; found-robot-stack new'
    fi
    ;;
  stop)
    ensure_container
    if [ -n "${FOUND_TMUX_SESSION:-}" ]; then
      exec_user -e "FOUND_TMUX_SESSION=$FOUND_TMUX_SESSION" "$CONTAINER" bash -lc 'found-robot-stack stop'
    else
      exec_user "$CONTAINER" bash -lc 'found-robot-stack stop'
    fi
    ;;
  attach)
    ensure_container
    if [ -n "${FOUND_TMUX_SESSION:-}" ]; then
      exec_user -e "FOUND_TMUX_SESSION=$FOUND_TMUX_SESSION" "$CONTAINER" bash -lc 'found-robot-stack attach'
    else
      exec_user "$CONTAINER" bash -lc 'found-robot-stack attach'
    fi
    ;;
  shell)
    ensure_container
    exec_user "${robot_env[@]}" "$CONTAINER" bash -l
    ;;
esac
