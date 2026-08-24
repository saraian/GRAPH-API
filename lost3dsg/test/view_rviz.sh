#!/usr/bin/env bash
# Open rviz2 on the running live stack. Runs in a sibling container that shares
# the host network (so DDS discovery sees the stack's topics) and the X11 socket.
# Output is mirrored to $LOG so a crash leaves a trace.
#   ./view_rviz.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
LOG=${LOG:-/tmp/graphapi_live/rviz.log}
mkdir -p "$(dirname "$LOG")"
xhost +local:docker >/dev/null
docker rm -f graphapi_rviz >/dev/null 2>&1 || true   # a stale name kills the new one instantly

DRI=()
[ -d /dev/dri ] && DRI=(--device /dev/dri)            # hardware GL via Mesa instead of llvmpipe

echo ">>> rviz log: $LOG"
docker run --rm --name graphapi_rviz --network=host \
  --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all "${DRI[@]}" \
  -e DISPLAY="${DISPLAY:-:1}" -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$REPO":/graph_api:ro \
  --entrypoint bash graphapi-run:humble -c \
  'source /opt/ros/humble/setup.bash && rviz2 -d /graph_api/lost3dsg/test/live.rviz' 2>&1 | tee "$LOG"
echo ">>> rviz exited (rc=${PIPESTATUS[0]}) — last lines of $LOG:"; tail -5 "$LOG"
