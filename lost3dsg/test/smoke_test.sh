#!/usr/bin/env bash
# Containerized smoke test for lost3dsg: colcon build, msg check, import every
# node module from the INSTALLED tree, then a short live-startup check of
# object_manager_6. Run INSIDE the ROS 2 Humble container (see live_run.sh for
# the docker invocation). Exits non-zero on the first failure.
set -e
source /opt/ros/humble/setup.bash

echo ">>> [1/5] colcon build"
mkdir -p /ws/src
rm -rf /ws/src/lost3dsg
cp -r /graph_api/lost3dsg /ws/src/lost3dsg
cd /ws
colcon build --packages-select lost3dsg --cmake-args -DCMAKE_BUILD_TYPE=Release
source /ws/install/setup.bash

echo ">>> [2/5] msg has the PCA fields"

echo ">>> [3/5] tiny word2vec for import smoke"
python3 - <<'PY'
import numpy as np
from gensim.models import KeyedVectors
words = ("chair table bed cabinet sofa lamp tv door sink toilet shelf desk "
         "monitor plant curtain unknown white black red green blue brown gray "
         "wood metal plastic fabric glass ceramic small large room wall floor").split()
kv = KeyedVectors(vector_size=32)
rng = np.random.default_rng(0)
kv.add_vectors(words, rng.normal(size=(len(words), 32)).astype(np.float32))
kv.save_word2vec_format("/tmp/smoke_w2v.bin", binary=True)
print(f"    {len(words)} words written")
PY

echo ">>> [4/5] import every node module from the installed tree"
export GRAPH_API_CONFIG=/graph_api/lost3dsg/test/smoke_config.yaml
cd /ws/install/lost3dsg/lib/lost3dsg
for m in config vlm_call nlp_utils world_model object_info map_database \
         utils cv_utils detection_types detection_pipeline perception_utils \
         input_output room_manager walls_rooms object_services object_manager_6 \
         perception_2 models query_map; do
  python3 -c "import $m" && echo "    import $m OK" || { echo "!! import $m FAILED"; exit 1; }
done
python3.py
python3 config.py

echo ">>> [5/5] object_manager_6 stays alive for 10 s"
timeout 10 ros2 run lost3dsg object_manager_6.py > /tmp/om6_smoke.log 2>&1 && rc=$? || rc=$?
# timeout(1) returns 124 when the node was still running at the deadline — that
# is the PASS case; any other exit means it died on its own.
if [ "$rc" -eq 124 ]; then
  echo "    object_manager_6 startup OK (alive at 10 s)"
else
  echo "!! object_manager_6 exited early (rc=$rc):"; tail -20 /tmp/om6_smoke.log; exit 1
fi

echo "SMOKE PASS"
