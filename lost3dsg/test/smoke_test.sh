#!/usr/bin/env bash
# Containerized smoke test for lost3dsg: colcon build, msg check, import every
# node module from the INSTALLED tree, then a short live-startup check of
# object_manager_6. Run INSIDE the ROS 2 Humble container (see run_sim.sh for
# the docker invocation). Exits non-zero on the first failure.
set -e
source /opt/ros/humble/setup.bash

echo ">>> [1/5] colcon build"
mkdir -p /ws/src
rm -rf /ws/src/lost3dsg
cp -r /graph_api/lost3dsg /ws/src/lost3dsg
cd /ws
# GA-300. DELETE THE INSTALLED NODE SCRIPTS BEFORE BUILDING. colcon copies files in and never
# takes them out, and /ws is a persistent named volume (GA-157), so a module DELETED from
# CMakeLists stays in /ws/install and keeps being importable and runnable. Measured 4 Sep:
# after habitat_camera_node.py was removed from the install list, a fresh build still left it
# in the tree, and lib/lost3dsg held 37 entries for 31 installed modules. A stale module is
# worse than a missing one: `ros2 run` would launch it and the bundle would record a source
# hash that does not describe what executed. Only the copied .py files go; the generated
# interfaces live under local/ and share/ and are what the build key exists to preserve.
rm -rf /ws/install/lost3dsg/lib/lost3dsg
colcon build --packages-select lost3dsg --cmake-args -DCMAKE_BUILD_TYPE=Release
source /ws/install/setup.bash

echo ">>> [2/5] PCA fields ride the bbox dicts (detection_pipeline self-check)"
# Bbox3d.msg stays AABB-only by design — yaw/oriented_* travel in the bbox dicts;
# the pipeline's __main__ asserts them (and isotropic -> omitted).
export GRAPH_API_CONFIG=/graph_api/lost3dsg/test/smoke_config.yaml
python3 /ws/install/lost3dsg/lib/lost3dsg/detection_pipeline.py

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
cd /ws/install/lost3dsg/lib/lost3dsg

# GA-84. DERIVED FROM THE TREE, not hand-maintained.
#
# This was a literal list, and it drifted: `walls_rooms` was deleted under an owner ruling and
# the list kept importing it, so the smoke test exited 1 for a module that was correctly gone.
# The lane that deleted it was right not to edit this file to cover its own deletion — the
# defect is that an inventory of what exists was written by hand instead of read.
#
# Same class as CMakeLists installing two files nothing launched. A hand-kept list is correct
# only until the tree moves, and it fails in the direction that looks like a real breakage.
# Excluded BY RULE, not by name: test modules are not node modules, and `rosstub` exists to
# stand in for ROS on a host that has none — importing it here would prove nothing about the
# installed tree. Any other exclusion would be a hand-list creeping back in.
_mods=$(find . -maxdepth 1 -name '*.py' -printf '%f\n' \
        | sed 's/\.py$//' | grep -v '^test_' | grep -vx 'rosstub' | sort)
[ -n "$_mods" ] || { echo "!! no modules found in $(pwd) — the install tree is empty"; exit 1; }
echo "    $(echo "$_mods" | wc -l) modules found in the install tree"
for m in $_mods; do
  python3 -c "import $m" && echo "    import $m OK" || { echo "!! import $m FAILED"; exit 1; }
done

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
