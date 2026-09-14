#!/bin/bash
# Build the Graph API lost3dsg ROS 2 package into /ws.
set -euo pipefail
# shellcheck disable=SC1091
. /etc/profile.d/99-found.sh

src=${GRAPH_API_SRC:-/graph_api/lost3dsg}
if [ ! -f "$src/package.xml" ]; then
  echo "lost3dsg sources not found at $src" >&2
  echo "mount the Graph API checkout at /graph_api/lost3dsg or set GRAPH_API_SRC" >&2
  exit 1
fi

mkdir -p /ws/src /ws/output
rm -rf /ws/src/lost3dsg
cp -a "$src" /ws/src/lost3dsg
cd /ws
colcon build --packages-select lost3dsg --cmake-args -DCMAKE_BUILD_TYPE=Release
touch /ws/install/lost3dsg/.found_build_stamp
echo "built lost3dsg into /ws/install; open a new shell or: . /ws/install/setup.bash"
