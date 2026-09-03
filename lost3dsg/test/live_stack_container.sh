#!/usr/bin/env bash
# Runs INSIDE the container: build, then start the full stack against the host
# habitat feed. Started by live_run.sh — not meant to be run directly.
set -e
source /opt/ros/humble/setup.bash

# GA-157. /ws IS A NAMED VOLUME NOW, so the build tree survives `docker run --rm`.
#
# It never was: `docker inspect` showed six binds and the only one under /ws was /ws/output, so
# /ws/build and /ws/install lived in the container's writable layer and died with it. The sources
# have been mounted at /graph_api all along and the build could not use any of it, because there
# was nothing left to reuse. Tonight's four builds: 31.9s, 49.0s, 1min14s, 1min54s.
#
# THE COPY STAYS. It is what makes "the tree that executed is the tree a7 stamped" true, and it
# costs milliseconds. Symlinking the install tree to the sources (--symlink-install) would save
# nothing that matters and would make a7's executed-vs-mount check TAUTOLOGICAL — run_f becomes a
# symlink to src_f, both hashes are read through the same file, and the probe can never fail.
# That is the defect a7 was repaired for earlier tonight, re-entering through a different door.
echo ">>> build"
mkdir -p /ws/src
rm -rf /ws/src/lost3dsg
cp -r /graph_api/lost3dsg /ws/src/lost3dsg
cd /ws

# THE CACHE KEY covers only what genuinely compiles: the 13 interface files, CMakeLists and
# package.xml. A pure-Python change leaves it unchanged and colcon reinstalls the modules
# incrementally; an interface change forces a CLEAN rebuild, because a regenerated header against
# a stale build directory is the one failure that must be impossible.
# GA-199. LC_ALL=C and a SORTED FILE LIST, because the previous form hashed a glob whose
# ORDER is locale-dependent: under C `Bbox3d.msg` precedes `Bbox3dArray.msg`, under en_US.UTF-8
# it does not, and the same tree produced e173fff37f0b0617 or b9f7eda717059968 depending on
# which shell asked. Identical bytes, different order, different hash. The failure was in the
# safe direction -- a spurious clean rebuild, never a stale-header build -- which is why it
# survived, but it made the key non-comparable across machines and any hand-computed value
# untrustworthy.
#
# `sha256sum` PER FILE rather than `cat` of the contents, so a RENAME also changes the key.
# Concatenated bytes cannot see a rename: swapping two interface filenames would leave the key
# unchanged while the generated code differs, which is the one mismatch this cache must never
# permit.
BUILD_KEY=$( { find /ws/src/lost3dsg/msg /ws/src/lost3dsg/srv -type f \( -name '*.msg' -o -name '*.srv' \) 2>/dev/null; \
               ls /ws/src/lost3dsg/CMakeLists.txt /ws/src/lost3dsg/package.xml 2>/dev/null; } \
             | LC_ALL=C sort | xargs -r sha256sum | LC_ALL=C sort \
             | sha256sum | cut -c1-16)
BUILD_MODE=incremental
if [ ! -f /ws/.build_key ] || [ "$(cat /ws/.build_key)" != "$BUILD_KEY" ]; then
  BUILD_MODE=clean
  echo "    interface key changed ($(cat /ws/.build_key 2>/dev/null || echo none) -> $BUILD_KEY); clean rebuild"
  rm -rf /ws/build /ws/install /ws/log
else
  echo "    interface key unchanged ($BUILD_KEY); reusing the generated interfaces"
fi
_build_t0=$(date +%s)
colcon build --packages-select lost3dsg --cmake-args -DCMAKE_BUILD_TYPE=Release >/tmp/build.log 2>&1 \
  || { tail -30 /tmp/build.log; exit 1; }
_build_t1=$(date +%s)
printf '%s' "$BUILD_KEY" > /ws/.build_key
# In the bundle, so "it reused" is a recorded fact and not an inference from a fast run.
printf '{"build_key": "%s", "mode": "%s", "seconds": %d}\n' \
       "$BUILD_KEY" "$BUILD_MODE" "$((_build_t1 - _build_t0))" > /ws/output/build_cache.json
echo "    build $BUILD_MODE in $((_build_t1 - _build_t0))s (key $BUILD_KEY)"
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

# GA-264. A real triple store, installed from the vendored wheel so this needs no network.
# found/store.py falls back to the in-memory rdflib graph if the import fails, and SAYS so --
# a silent fallback would report a performance fix as landed when it is not.
if ! python3 -c "import pyoxigraph" 2>/dev/null; then
  pip install --quiet --no-index --find-links=/found/vendor/wheels pyoxigraph 2>&1 | tail -1 ||     echo "!! pyoxigraph install failed; the triple store will use the slow in-memory path"
fi
python3 -c "import pyoxigraph as _o; print('    triple store: pyoxigraph', _o.__version__)" 2>/dev/null ||   echo "    triple store: rdflib in-memory (pyoxigraph unavailable)"
LOG_DIR=/ws/output/logs
mkdir -p "$LOG_DIR" /ws/output/crops /ws/output/snapshots /out

# Ensure logs stream directly to the persistent host-mounted volume
touch "$LOG_DIR/feed_node.log" "$LOG_DIR/rtabmap.log" "$LOG_DIR/om6.log" "$LOG_DIR/bridge.log" "$LOG_DIR/perception.log" "$LOG_DIR/saver.log" "$LOG_DIR/walls.log"
ln -sfn "$LOG_DIR/feed_node.log" /tmp/feed_node.log
ln -sfn "$LOG_DIR/rtabmap.log" /tmp/rtabmap.log
ln -sfn "$LOG_DIR/om6.log" /tmp/om6.log
ln -sfn "$LOG_DIR/bridge.log" /tmp/bridge.log
ln -sfn "$LOG_DIR/perception.log" /tmp/perception.log
ln -sfn "$LOG_DIR/saver.log" /tmp/saver.log
ln -sfn "$LOG_DIR/walls.log" /tmp/walls.log

_close_map_and_check() {
  # GA-106. THIS RUNS FROM THE EXIT TRAP, and that is the whole point.
#
# It used to sit in the main flow after the node wait. A capped run ends with `docker stop`,
# which SIGTERMs bash: the EXIT trap fires (so logs were copied and the bundle LOOKED complete)
# and the main flow never advances. So the close never ran, no SIGINT reached rtabmap, no
# integrity check, no marker — and the publish step correctly refused a map it was never told
# was good. Measured on 20260831_224508_mp3d_17DRP: no rtabmap.db.params-sha, no INTEGRITY_OK,
# and rtabmap still logging node 400 at its normal cadence at the cut.
#
# And a MAPPING_ONLY run can ONLY end by the cap. With no detector the post-mapping phase
# produces nothing forever, so the sole ending is the one that skipped the close. Two correct
# pieces deadlocked; more cap time cannot fix it.
#
# Found by the testing lane, which tested its own root cause before sending it and discarded
# the wrong one: a trap on EXIT alone DOES fire under SIGTERM, and `trap ... EXIT TERM` fires it
# TWICE. So EXIT alone, and this guard for anything else that might reach it.
  [ -n "${_MAP_CLOSE_DONE:-}" ] && return 0
  _MAP_CLOSE_DONE=1

# GA-96. Let rtabmap CLOSE its database before the container goes.
#
# Measured on run A, 2026-08-31, by the testing lane against its own method: `docker cp` of the
# live db gave 84 MB with hundreds of invalid page numbers on PRAGMA integrity_check — a torn
# b-tree, rtabmap mid-write. The SQLite backup API is NOT the safe alternative: it holds a shared
# lock for the whole copy (8.1 s on 99 MB), rtabmap tried to commit a statistics row inside that
# window, got SQLITE_BUSY, and rtabmap DOES NOT RETRY — DBDriverSqlite3 asserts and throws.
# SIGABRT, exit -6. That killed run A nine minutes in.
#
# A torn copy costs a file. A lock costs the whole run. So: DO NOT SNAPSHOT A LIVE RTABMAP
# DATABASE BY ANY METHOD — not docker cp, not the backup API, not a filesystem copy. The only
# safe map is one rtabmap has finished with.
if [ -n "${RTABMAP_PID:-}" ] && kill -0 "$RTABMAP_PID" 2>/dev/null; then
  # GA-104. THE SIGINT NEVER REACHED THE NODE ON RUN B, and the log proves it rather than
  # suggesting it: rtabmap logged node 1096 at its normal 2.5 s cadence THREE SECONDS BEFORE the
  # FATAL. It was not slowly closing a 1.2 GB database — it was still mapping, undisturbed.
  #
  # Cause is my own earlier finding: $RTABMAP_PID is the `ros2 launch` process, not the rtabmap
  # node. Signalling the launcher is not signalling the node. So signal the NODE directly, by the
  # binary path that appears in ros2 launch's own death message, and keep signalling the launcher
  # too so the rest of the launch tree comes down in order.
  # THE CALLER MUST GIVE THIS TIME. Now that the close runs from the EXIT trap, a capped run
# reaches it via `docker stop`, which SIGKILLs after its -t grace period. `docker stop -t 30`
# leaves 30 s for a close that may need 120, and the map dies mid-write with the script looking
# correct. So the budget is explicit and the requirement is printed: CAP WITH
# `docker stop -t $((RTABMAP_CLOSE_TIMEOUT + 30))` OR LONGER.
_CLOSE_T=${RTABMAP_CLOSE_TIMEOUT:-120}
echo ">>> asking rtabmap to close its database (SIGINT to the node, up to ${_CLOSE_T}s)"
echo "    NOTE: if this run is being capped, `docker stop` must allow at least $((_CLOSE_T + 30))s"
echo "    (docker stop -t $((_CLOSE_T + 30))). A shorter grace SIGKILLs the close mid-write."
  kill -INT "$RTABMAP_PID" 2>/dev/null || true
  pkill -INT -f 'rtabmap_slam/rtabmap' 2>/dev/null || true
  # 120 s, not 30. A 1.2 GB database is not closed instantly, and the cost of waiting too long is
  # a slow shutdown while the cost of waiting too little is the map. Poll the NODE, not the
  # launcher: the launcher outlives the node, which is how the 30 s expired on run B.
  for _ in $(seq 1 "$_CLOSE_T"); do pgrep -f 'rtabmap_slam/rtabmap' >/dev/null 2>&1 || break; sleep 1; done
  # Re-point the aliveness test at the node too, so the branch below judges the right process.
  pgrep -f 'rtabmap_slam/rtabmap' >/dev/null 2>&1 || RTABMAP_PID=""
  if kill -0 "$RTABMAP_PID" 2>/dev/null; then
    # GA-98, found by the experiment lane. The old code warned here and then FELL THROUGH to the
    # integrity check below — which opens the db and holds a read lock for seconds on a ~100 MB
    # file, against the live rtabmap this file's own comment establishes does NOT retry on
    # SQLITE_BUSY. On the one branch where the writer is known to still be writing, it took the
    # exact lock that killed run A, and the bundle then reported "integrity_check failed".
    #
    # THE CHECK WOULD HAVE CAUSED THE CORRUPTION IT REPORTED. It is skipped deliberately, and the
    # skip is stated: an UNCHECKED map and a TORN map are different states, and only one of them
    # is recoverable. The absence of a verdict is information and must not be replaced by a
    # verdict obtained unsafely.
    echo "!! rtabmap did not close after ${_CLOSE_T}s."
    echo "[map] NOT CHECKED — rtabmap did not close. The map's state is UNKNOWN, not bad."
    _SKIP_MAP_CHECK=1
  fi
fi
# There is no sqlite3 binary on this host or in this image — every check goes through python3.
[ -n "${_SKIP_MAP_CHECK:-}" ] || python3 - <<'PYCHK' || true
import sqlite3, pathlib
db = pathlib.Path("/ws/output/rtabmap.db")
if not db.exists():
    print("[map] no rtabmap.db — nothing to check"); raise SystemExit
try:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ok = c.execute("PRAGMA integrity_check").fetchone()[0]
    n = c.execute("SELECT count(*) FROM Node").fetchone()[0]
    print(f"[map] rtabmap.db integrity={ok} nodes={n} bytes={db.stat().st_size}")
    if ok == "ok":
        # A MARKER, not just a printed verdict. The host publishes on this file and nothing else,
        # so an unchecked map and a failed map both stay unpublished without the host having to
        # parse a log to tell them apart.
        pathlib.Path("/ws/output/rtabmap.db.INTEGRITY_OK").write_text(f"integrity={ok} nodes={n}\n")
    else:
        print("[map] REFUSING to mark this map usable: it does not pass integrity_check. "
              "A torn db looks like a complete 80-100 MB map to anyone who does not run one.")
except Exception as exc:
    print(f"[map] rtabmap.db WILL NOT OPEN: {exc}")
PYCHK
# Record the parameters this map was built under, so a later localization run can refuse a
# mismatch instead of guessing. Only for a map this run MADE — never overwrite a localize copy's.
if [ -z "${RTABMAP_LOCALIZE_DB:-}" ] && [ -f /ws/output/rtabmap.db ]; then
  printf '%s' "$RTABMAP_GRID_ARGS" | sha256sum | cut -c1-16 > /ws/output/rtabmap.db.params-sha
  echo "[map] params-sha $(cat /ws/output/rtabmap.db.params-sha) recorded beside the map"
fi
}

container_exit_cleanup() {
  # The close runs FIRST: it can still write into $LOG_DIR, and the log copy below should carry
  # whatever it says. Log-copy-then-close would ship a bundle whose logs predate its own verdict.
  _close_map_and_check
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
  # A mapping run starts no detector, so a4 (perception called twice) and a8's perception imports
  # guard something that is deliberately absent. They still RUN and are still recorded — only
  # their authority to block is withdrawn, and preflight.json says so on every affected row.
  # Every other probe blocks as usual: a mapping run that mapped the wrong scene with the wrong
  # camera is worse than no map, and a2/a5/a6/a7 are what catch that.
  PREFLIGHT_OBSERVE=""
  [ "${MAPPING_ONLY:-0}" = "1" ] && PREFLIGHT_OBSERVE="--observe a4,a8"
  echo ">>> pre-flight gate (Class A)${PREFLIGHT_OBSERVE:+ — OBSERVING a4,a8 for a mapping run}"
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
      --expect-cycle-s "${PREFLIGHT_EXPECT_CYCLE_S:-}" \
      --install-tree /ws/install/lost3dsg/lib/lost3dsg \
      ${PREFLIGHT_OBSERVE:-} \
    || { echo "!! PRE-FLIGHT FAILED — no measured run produced. See /ws/output/preflight.json"; exit 1; }
fi
# -------------------------------------------------------------------------------------------

# GA-97. LOCALIZATION MODE — LANDED OFF BY DEFAULT. Unset RTABMAP_LOCALIZE_DB and everything
# below is byte-for-byte today's behaviour.
#
# Set it to a map path and the stack localizes against a COPY of that map instead of mapping from
# scratch. The copy is not politeness: localizing against the original would let this run write
# into the canonical map, and the next run would inherit changes nobody recorded.
#
# THE SIDECAR CHECK IS THE POINT, and it is rule 14's shape. A map built under different grid
# parameters is not a map of the same world — cell size, ray tracing and the height bands all
# change what is occupied. Localizing against a stale one produces poses that look fine and are
# wrong, in a bundle that looks complete. So a mismatch REFUSES TO START rather than warning.
_RT_DB_ARGS="--delete_db_on_start"
_RT_DB_PATH="/ws/output/rtabmap.db"
if [ -z "${RTABMAP_LOCALIZE_DB:-}" ]; then
  # SAY SO WHEN THE BRANCH IS NOT TAKEN. An unset variable took this path silently, and the
  # testing lane found the passthrough missing only because it went looking BEFORE launching
  # rather than after. Silence is how "the feature is off" and "the feature never arrived" became
  # the same observation.
  echo ">>> localization OFF — mapping from scratch (RTABMAP_LOCALIZE_DB is unset)"
fi
if [ -n "${RTABMAP_LOCALIZE_DB:-}" ]; then
  [ -f "$RTABMAP_LOCALIZE_DB" ] || { echo "!! RTABMAP_LOCALIZE_DB=$RTABMAP_LOCALIZE_DB does not exist"; exit 1; }
  _want=$(printf '%s' "$RTABMAP_GRID_ARGS" | sha256sum | cut -c1-16)
  _sidecar="${RTABMAP_LOCALIZE_DB}.params-sha"
  [ -f "$_sidecar" ] || { echo "!! $_sidecar is missing. A map with no recorded parameters cannot be shown to match this run's; refusing to localize against it."; exit 1; }
  _have=$(cat "$_sidecar")
  [ "$_have" = "$_want" ] || {
    echo "!! MAP PARAMETER MISMATCH — refusing to start."
    echo "   map was built under params-sha $_have"
    echo "   this run's RTABMAP_GRID_ARGS hash is $_want"
    echo "   A map built under different grid parameters is not a map of the same world. Rebuild"
    echo "   the map or clear RTABMAP_LOCALIZE_DB; do NOT localize against it."
    exit 1; }
  # GA-158. THE MAP IS MOUNTED READ-ONLY, NOT COPIED.
  #
  # The copy existed so the run could not write into the canonical map. It cost about FOUR MINUTES
  # for the 1.2 GB database and left a second 1.2 GB copy in every bundle as rtabmap_localize.db —
  # 3.6 GB across three runs. Measured by the testing lane on run C, with the sharp consequence:
  # with FEED_MAPPING_SECONDS=0 a localization run spent ~240 s copying to save ~150 s of mapping,
  # so localizing was WORSE on the clock than mapping while being right on the substance.
  #
  # A read-only bind mount gives the same guarantee and gives it harder. The copy HOPES the run
  # will not write to the original; the filesystem MAKES it so. That is the same argument as
  # single_floor being a publish precondition rather than a sidecar field — a guarantee that
  # depends on nobody doing the wrong thing versus one that cannot be violated.
  #
  # IF RTABMAP REFUSES A READ-ONLY DATABASE the run fails here with sqlite's own error, which is
  # the honest outcome: it is a fact about rtabmap worth discovering explicitly rather than one
  # papered over by a copy nobody had costed. Mem/IncrementalMemory false is set below.
  _RT_DB_PATH="$RTABMAP_LOCALIZE_DB"
  if [ -w "$_RT_DB_PATH" ]; then
    echo "!! WARNING: $_RT_DB_PATH is WRITABLE inside the container. Localization will not write"
    echo "   to it, but nothing is enforcing that. Mount the map read-only (-v <host>:<path>:ro)"
    echo "   so the canonical map cannot be modified by a run that is only reading it."
  fi
  _RT_DB_ARGS="--Mem/IncrementalMemory false"
  echo ">>> LOCALIZATION MODE against a copy of $RTABMAP_LOCALIZE_DB (params-sha $_have)"
  echo "    mapping is OFF; the driver must also set FEED_MAPPING_SECONDS=0"
fi

# same rtabmap arguments as launch/habitat_launch.py (odometry from /odom, no TF publish)
ros2 launch rtabmap_launch rtabmap.launch.py visual_odometry:=false odom_topic:=/odom \
  rgb_topic:=/camera/rgb depth_topic:=/camera/depth camera_info_topic:=/camera/camera_info \
  approx_sync:=true rtabmap_viz:=false publish_tf:=false database_path:="$_RT_DB_PATH" \
  rtabmap_args:="$_RT_DB_ARGS --RGBD/NeighborLinkRefining false $RTABMAP_GRID_ARGS" \
  > /tmp/rtabmap.log 2>&1 &
RTABMAP_PID=$!
ros2 run lost3dsg object_manager_6.py > /tmp/om6.log 2>&1 &
OM6_PID=$!
python3 /ws/install/lost3dsg/lib/lost3dsg/graph_api_bridge.py > /tmp/bridge.log 2>&1 &
# MAPPING_ONLY: the detector is NOT STARTED. Not idled, not stubbed -- absent. A mapping run
# needs no detections and every Modal call it makes is money spent on an image nobody reads.
if [ "${MAPPING_ONLY:-0}" = "1" ]; then
  echo ">>> MAPPING_ONLY=1: perception_2 and the cloud path are OFF. No detector, no Modal calls."
  PERCEPTION_PID=""
else
  echo ">>> MAPPING_ONLY is ${MAPPING_ONLY:-unset} — normal run, detector ON."
  ros2 run lost3dsg perception_2.py > /tmp/perception.log 2>&1 &
  PERCEPTION_PID=$!
fi

# GA-29 / wall_detector's FIRST LAUNCH. Four reasons it never produced a wall, and the first was
# that nothing ever started it — this line. It is in CMakeLists.txt:66 so `ros2 run` resolves it;
# it subscribes /camera/depth and /camera/camera_info, the topics rtabmap already takes above, and
# publishes /detected_wall_segments in the schema object_manager_6.walls_callback actually reads.
# GA-206. OPT-OUT, default OFF for this run. wall_detector.py measured at 4.4 CORES on
# 2026-09-01 while rtabmap -- the ONLY source of the map->odom transform perception waits on --
# was taking 2.1-2.4 s per iteration against its own 1.0 s rate limit. Every frame then aged out
# with "Synced data not ready, missing: transform", and six launch attempts produced zero
# detection cycles.
#
# The layer it feeds is separately known to produce nothing: ridge segmentation yields ZERO
# critical points on ~87% of sweeps (GA-195), so no doorway is ever cut and no room is split.
# Spending 4.4 cores on it while starving the transform chain buys a room layer that does not
# work at the cost of the detections that do.
#
# WALL_DETECTOR=1 restores it. This is a resource decision for a contended machine, NOT a
# claim that wall detection is wrong.
if [ "${WALL_DETECTOR:-0}" = "1" ]; then
  ros2 run lost3dsg wall_detector.py > /tmp/walls.log 2>&1 &
else
  echo ">>> wall_detector DISABLED (WALL_DETECTOR=1 to enable) — 4.4 cores returned to rtabmap"
  : > /tmp/walls.log
fi
WALLS_PID=$!

# periodic snapshots of the annotated detection image for the host
ros2 run image_view image_saver --ros-args -r image:=/image_with_bb \
  -p filename_format:="/out/detection_%04d.png" -p sec_per_frame:=5.0 \
  > /tmp/saver.log 2>&1 &

# GA-200 / probe a9. THE FEED MUST STILL BE ARRIVING, and only a node that exists can be
# asked. The eight-probe gate runs BEFORE these nodes start, so it cannot see this: run
# 20260901_140710 passed 8/8, brought up every node, relayed ONE frame and spun on a dead
# socket for six minutes while every liveness signal said healthy.
#
# Non-fatal on purpose. A slow first frame is normal (a cold start has taken ~40 s), so this
# WARNS rather than aborting -- the point is that a frozen feed becomes visible in the first
# minute instead of being discovered when the bundle turns out empty.
(
  sleep 45
  if python3 /graph_api/lost3dsg/test/preflight_gate.py --only a9 \
       --feed-log /tmp/feed_node.log --feed-window-s 12 --out /tmp/preflight_a9.json \
       >> /tmp/a9.log 2>&1; then
    echo ">>> a9 feed_streaming: frames are arriving"
  else
    echo "!! a9 feed_streaming FAILED — the feed is not delivering. $(tail -2 /tmp/a9.log | tr '\n' ' ')"
    echo "   the run will continue, but it is very likely to produce nothing. See GA-200."
  fi
) &

echo ">>> stack up. logs in /tmp/*.log — tailing perception:"
touch /tmp/perception.log /tmp/om6.log
tail -f /tmp/perception.log /tmp/om6.log &
TAIL_PID=$!

# GA-83. WAIT ON THE DETECTOR, not on a tail that never ends.
#
# The container used to end in `tail -f`, which outlives every node. On 31 Aug the detector died
# at 12:04:41 with a backend timeout — the handler removal working, cause named in one line —
# and the stack ran on for 55 MINUTES at 322% CPU with nothing to process. A lane sat waiting on
# a row count that had stopped growing 55 minutes earlier, and NOTHING in the bundle said the
# detector was gone: it was visible only in the container's process table.
#
# The handler removal made the node die honestly. This propagates that death to the run, so
# `docker run` returns, the launcher's EXIT trap archives, and `latest` decides on a bundle that
# has stopped changing. A run that ends early with a traceback is the intended outcome; a run
# that continues without its detector is not a run.
# GA-94. GA-83 above propagated the DETECTOR's death to the run and stopped there. Run A
# (20260831_174209) showed the other half: rtabmap aborted at 17:51:39 with a SIGABRT, and
# perception_2 stayed alive, so this `wait` did not return and THE CONTAINER RAN 38 MORE MINUTES
# producing nothing. Every liveness check said healthy because the container was Up.
#
# A CONTAINER THAT IS UP IS NOT A RUN THAT IS RUNNING, AND A LOG THAT IS GROWING IS NOT A RUN THAT
# IS PRODUCING — perception_2 wrote 3.4 MB after the run stopped producing anything.
#
# So wait on the GROUP: whichever of the four load-bearing nodes exits first ends the run, and the
# message names which one. rtabmap is in the list because without it there is no map frame, and
# perception's TF lookups then fail with an extrapolation hole that grows for as long as nobody
# looks. The image saver and the tail are NOT in the list — neither produces measurement.
#
# NOT A SUBSTITUTE FOR WATCHING om6. Run A announced its own death in the bundle at 17:53:35
# ("[INPUT] no /bbox_3d for 104s ... The producer may have stopped") and nothing read it. This
# catches a node that EXITS; a producer that is alive but starved still needs that line read.
# NOT in a function called from $( ), which is how I first wrote it. A command substitution runs
# in a SUBSHELL, and a subshell cannot `wait` on the parent's children — it returned rc=-1 for a
# node that had exited 7, so the status this run reports would have been meaningless. Measured on
# a standalone harness before landing; the loop below returns 7 for an exit 7 and 0 for a clean 0.
if [ "${MAPPING_ONLY:-0}" = "1" ]; then
  _WATCH_NODES="RTABMAP OM6"
  _MAP_DEADLINE=$(( $(date +%s) + ${FEED_MAPPING_SECONDS%.*} + 60 ))
else
  _WATCH_NODES="PERCEPTION RTABMAP OM6 WALLS"
  _MAP_DEADLINE=0
fi
_dead_node=""; _dead_rc=0
while [ -z "$_dead_node" ]; do
  if [ "$_MAP_DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$_MAP_DEADLINE" ]; then
    _dead_node="MAPPING_TIME"; _dead_rc=0; break
  fi
  # With the detector off, PERCEPTION and OM6 are not in the list -- waiting on a node that was
  # never started ends the run instantly. A mapping run watches the map and the feed, and ends on
  # TIME rather than on a death.
  for _n in $_WATCH_NODES; do
    _pid_var="${_n}_PID"; _pid="${!_pid_var:-}"
    [ -n "$_pid" ] || continue
    if ! kill -0 "$_pid" 2>/dev/null; then
      _dead_rc=0; wait "$_pid" || _dead_rc=$?
      _dead_node="$_n"; break
    fi
  done
  # THE PID WATCH ABOVE WOULD NOT HAVE CAUGHT RUN A, and saying so is the point of this block.
  # RTABMAP_PID is the `ros2 launch` process, not the rtabmap node. In run A the node aborted
  # ("[ERROR] [rtabmap-1]: process has died [pid 1458, exit code -6]") and the LAUNCH SURVIVED it,
  # so `kill -0 $RTABMAP_PID` stayed true for the 38 minutes that followed. A watch on the wrong
  # process is not a watch. ros2 launch reports the death in its own log and that is the only
  # place it is visible, so read it there.
  if [ -z "$_dead_node" ] && grep -q "process has died" /tmp/rtabmap.log 2>/dev/null; then
    _dead_node="RTABMAP(node)"
    _dead_rc=$(sed -n 's/.*process has died.*exit code \(-\?[0-9]*\).*/\1/p' /tmp/rtabmap.log | head -1)
    _dead_rc=${_dead_rc:-1}
  fi
  # GA-283. THE NODE'S CODE BEATS THE LAUNCHER'S, and the block above only ran when the
  # launcher SURVIVED. On run 20260903_110622 the launcher did NOT survive: rtabmap the node
  # aborted (SIGABRT, exit code -6) and `ros2 launch` then exited 0, so the pid watch above
  # fired first, recorded 0, and printed "!! RTABMAP exited with status 0 -- ending the run."
  # A SIGABRT REPORTED AS A CLEAN EXIT is why that crash read as benign to two people for an
  # hour. `ros2 launch` returning 0 after its child aborts is not information about the child.
  if [ "$_dead_node" = "RTABMAP" ] && grep -q "process has died" /tmp/rtabmap.log 2>/dev/null; then
    _node_rc=$(sed -n 's/.*process has died.*exit code \(-\?[0-9]*\).*/\1/p' /tmp/rtabmap.log | head -1)
    if [ -n "$_node_rc" ]; then
      _dead_node="RTABMAP(node, launcher exited ${_dead_rc})"
      _dead_rc="$_node_rc"
    fi
  fi
  # And say what it was, since a negative code is a SIGNAL and reads as an error code.
  case "$_dead_rc" in
    -6)  _dead_why=" (SIGABRT — an assertion or uncaught exception; see rtabmap.log for the FATAL)" ;;
    -9)  _dead_why=" (SIGKILL — killed from outside, commonly OOM; check the kernel log)" ;;
    -11) _dead_why=" (SIGSEGV)" ;;
    *)   _dead_why="" ;;
  esac
  [ -z "$_dead_node" ] && sleep 2
done
kill "$TAIL_PID" 2>/dev/null || true
echo "!! $_dead_node exited with status ${_dead_rc}${_dead_why:-} — ending the run."
echo "   The stack is not left running: a run missing any of these nodes measures nothing further."
tail -5 /tmp/perception.log

# An EMPTY variable here is "exit: : numeric argument required", which is what run
# 20260903_110622 printed instead of an exit status. Default it, and prefer the code of the
# node that actually died when perception was not the one that died.
_rc="${_perception_rc:-}"
[ -z "$_rc" ] && _rc="${_dead_rc:-1}"
case "$_rc" in ''|*[!0-9-]*) _rc=1 ;; esac
[ "$_rc" -lt 0 ] 2>/dev/null && _rc=$(( 128 - _rc ))   # a signal, as a shell exit status
exit "$_rc"
