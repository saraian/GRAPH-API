#!/usr/bin/env bash
# Runs INSIDE the container: build, then start the full stack against the host
# habitat feed. Started by live_run.sh — not meant to be run directly.
set -e
# Console into the bundle: the "!! <node> exited" verdicts went to the terminal only, so three of
# the 3 Sep runs have no recorded cause. /tmp/*.log is copied to $LOG_DIR at exit.
exec > >(tee -a /tmp/stack.log) 2>&1
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
# GA-300. DELETE THE INSTALLED NODE SCRIPTS BEFORE BUILDING. colcon copies files in and never
# takes them out, and /ws is a persistent named volume (GA-157), so a module DELETED from
# CMakeLists stays in /ws/install and keeps being importable and runnable. Measured 4 Sep:
# after habitat_camera_node.py was removed from the install list, a fresh build still left it
# in the tree, and lib/lost3dsg held 37 entries for 31 installed modules. A stale module is
# worse than a missing one: `ros2 run` would launch it and the bundle would record a source
# hash that does not describe what executed. Only the copied .py files go; the generated
# interfaces live under local/ and share/ and are what the build key exists to preserve.
rm -rf /ws/install/lost3dsg/lib/lost3dsg
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

# The model cache. Without HF_HOME the in-container default points at a host path that does not
# exist here, so the encoder is re-fetched from the hub every run.
#
# This line used to be preceded by `export PYTHONPATH=/kb:...`, which put ASPIRE's knowledge_bridge
# on the path because kg_align.py imported ConceptEmbedder from it. GA-306 vendored that class into
# found/concept_embedder.py, so there is nothing to mount and nothing to add to the path. `found`
# itself never came from PYTHONPATH -- the hook config carries its path.
EXT_MOUNT_POINT="${EXT_MOUNT_POINT:-/ext}"
# GA-437 (2026-09-10). HF_HOME POINTED AT A PATH NOTHING MOUNTS, on every machine, in every run.
#
# It read "$EXT_MOUNT_POINT/.hf_cache", i.e. /ext/.hf_cache. Checked against the launcher's actual
# mount list: the run container gets $WORKSPACE_ROOT/maps at $EXT_MOUNT_POINT/maps and NOTHING at
# $EXT_MOUNT_POINT itself, and the extension mounts its own tree at /found:ro (tools/ext/env.sh),
# not at /ext. So /ext/.hf_cache never existed inside a run; huggingface created it in the
# container's writable layer, downloaded the weights into it, and --rm threw them away.
#
# CONSEQUENCE, and it is a measurement error rather than a slow start: the download lands in the
# FIRST perception cycle, which is the cycle anyone quotes as cold-start latency, on a machine whose
# cache is warm. a4 recorded owlv2_weights_cached false and was right. Warming a cache is not the
# fix; the path is.
#
# /models/hf IS WHERE THE CACHE IS: the launcher mounts $HF_SHARED_CACHE (default
# /DATA/huggingface_cache) there, and that directory has the hub/ layout HF_HOME expects --
# hub/models--google--owlv2-base-patch16-ensemble is present today. It is also the fallback that
# nlp_utils.py:22 and preflight_gate.py already use when HF_HOME is unset, so this line was
# overriding a correct default with a path that does not exist.
export HF_HOME="${HF_HOME:-/models/hf}"
# GA-438 (2026-09-10). OFFLINE MAKES A MISSING MODEL FAIL AT LOAD instead of being paid silently in
# the first perception cycle, which is the owner's policy ("models should be already downloaded and
# cached beforehand"). The experiment lane proved all five models load with the hub switched off --
# but they set it in Gin's own environment, and docker passes only what the -e list names, so that
# enforcement never left that machine.
#
# DEFAULT ON. Owner ruling 2026-09-10, in their words: "Everything should be prepared and cached.
# No in-run fetches." Offline is what makes that a fact rather than an intention -- a missing model
# then fails at load with a named cause instead of being paid silently inside the first perception
# cycle, which is the cycle a cold-start number is read from.
#
# THE CACHE WAS WARMED FIRST AND THE LOADS WERE MEASURED, not assumed. facebook/dinov2-small and
# facebook/dinov2-base were cached NOWHERE on this host until 2026-09-10 -- not in
# /DATA/huggingface_cache, not in any extension's own cache, not in ~/.cache/huggingface -- and
# visual_reid.py loads one of them on EVERY run whatever the backend. All four models were then
# loaded inside this image with HF_HUB_OFFLINE=1: dinov2-small, dinov2-base,
# all-MiniLM-L6-v2 and owlv2-base-patch16-ensemble. a4 asserts the same four, so a cold cache is
# refused at the gate rather than discovered at load. FEED_HF_OFFLINE=0 is the escape.
if [ "${FEED_HF_OFFLINE:-1}" = "1" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  echo ">>> HF OFFLINE: a model missing from $HF_HOME will fail at load, not download"
else
  echo "!! HF ONLINE (FEED_HF_OFFLINE=0): a model missing from $HF_HOME will DOWNLOAD inside the"
  echo "   run, in the first perception cycle. Owner policy 2026-09-10 is no in-run fetches."
fi

# CFG_NAME comes from live_run.sh (regolo_config.yaml when an API key is set).
# No default. This line used to read ${CFG_NAME:-smoke_config.yaml}, and because
# live_run.sh assigned CFG_NAME without exporting it, `docker run -e CFG_NAME`
# passed nothing and every live run silently used the smoke config while the bundle
# recorded regolo. Guessing here is what made that invisible.
: "${CFG_NAME:?CFG_NAME not set — live_run.sh must export it; refusing to guess a config}"
export GRAPH_API_CONFIG=/graph_api/lost3dsg/test/${CFG_NAME}
# The config must EXIST. config.py::_load returns the defaults when it does not, silently —
# so a mistyped or unported config name yields a run with hooks.filter empty, the extension out of
# the loop, and a bundle that looks complete. Fail here instead.
[ -f "$GRAPH_API_CONFIG" ] || { echo "!! GRAPH_API_CONFIG=$GRAPH_API_CONFIG does not exist — refusing to run on defaults"; exit 1; }
export GRAPH_API_OUTPUT_DIR=/ws/output

# GA-264. A real triple store, installed from the vendored wheel so this needs no network.
# found/store.py falls back to the in-memory rdflib graph if the import fails, and SAYS so --
# a silent fallback would report a performance fix as landed when it is not.
if ! python3 -c "import pyoxigraph" 2>/dev/null; then
  pip install --quiet --no-index --find-links="$EXT_MOUNT_POINT"/vendor/wheels pyoxigraph 2>&1 | tail -1 ||     echo "!! pyoxigraph install failed; the triple store will use the slow in-memory path"
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
echo "    NOTE: if this run is being capped, \`docker stop\` must allow at least $((_CLOSE_T + 30))s"
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
  # GA-373. a7 ONCE MORE AT TEARDOWN, before anything else in the close: the extension is a live mount, so
  # a live tree that moved during the run is what the stack executed, and only a second sample can
  # say so. Verdict lands in the bundle as a7_teardown.json; the launcher's latest-pointer refuses a
  # failing one. Same expectations and the same found-exercised rule as the startup gate.
  python3 /graph_api/lost3dsg/test/preflight_gate.py --only a7 --teardown --out /ws/output/a7_teardown.json \
      --expect-src-sha "${PREFLIGHT_EXPECT_SRC_SHA:-}" \
      --found-exercised "$([ "${MAPPING_ONLY:-0}" = "1" ] && echo 0 || echo auto)" \
      --install-tree /ws/install/lost3dsg/lib/lost3dsg > /tmp/a7_teardown.log 2>&1 \
    && echo ">>> a7 at teardown: PASS — $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));a=[p for p in d['probes'] if p['id']=='a7'][0]['detail'];m=a.get('live_mismatches') or {};print('live roots unchanged since the launch stamp' if not m else 'live root(s) MOVED during the run, recorded non-blocking: '+', '.join(f'{k} {v[\"launcher\"]}->{v[\"container\"]}' for k,v in m.items()))" /ws/output/a7_teardown.json 2>/dev/null || echo 'a7_teardown.json unreadable')" \
    || echo "!! a7 at teardown FAILED — a source root moved during the run; see a7_teardown.json (latest will not move)"
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

echo ">>> starting stack (feed -> rtabmap -> perception_2 -> object_manager_6 -> web viewer :${BRIDGE_PORT:-8081})"
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
      --found-exercised "$([ "${MAPPING_ONLY:-0}" = "1" ] && echo 0 || echo auto)" \
      --expect-policy "${PREFLIGHT_EXPECT_POLICY:-}" \
      --expect-cycle-s "${PREFLIGHT_EXPECT_CYCLE_S:-}" \
      --install-tree /ws/install/lost3dsg/lib/lost3dsg \
      ${PREFLIGHT_OBSERVE:-} \
    || { echo "!! PRE-FLIGHT FAILED — no measured run produced. See /ws/output/preflight.json"; exit 1; }
fi
# -------------------------------------------------------------------------------------------

# GA-97 / GA-433. LOCALIZATION MODE, RETIRED. Localizing against a published map used to happen
# here: a params-sha sidecar check refused a map built under different grid parameters, and the run
# read a scratch copy so it could not write into the canonical map. habitat_launch.py owns rtabmap
# now and hardcodes its database, so neither is reachable. Both branches below only report that.
if [ -z "${RTABMAP_LOCALIZE_DB:-}" ]; then
  # SAY SO WHEN THE BRANCH IS NOT TAKEN. An unset variable took this path silently, and the
  # testing lane found the passthrough missing only because it went looking BEFORE launching
  # rather than after. Silence is how "the feature is off" and "the feature never arrived" became
  # the same observation.
  echo ">>> localization OFF — mapping from scratch (RTABMAP_LOCALIZE_DB is unset)"
fi
if [ -n "${RTABMAP_LOCALIZE_DB:-}" ]; then
  # GA-433 (2026-09-10). LOCALIZING AGAINST A PUBLISHED MAP IS NOT REACHABLE ANY MORE, so this
  # refuses instead of accepting the variable and ignoring it.
  #
  # habitat_launch.py owns rtabmap now, and it hardcodes database_path /root/.ros/rtabmap.db with
  # --delete_db_on_start. Nothing here can hand it another database. The apparatus that used to do
  # that — the params-sha sidecar check, the read-only canonical mount (GA-158), the writable
  # scratch copy (GA-336), the per-floor publish refusal (GA-380) — has no caller under this
  # configuration and is preserved only in git history at a4c5957^ and in PLAN_1.3 §54-66.
  #
  # A variable that is set, printed and then dropped is the failure this whole file argues against
  # (rule 68: a setting is not an outcome). Until the owner rules on the localization regime, the
  # honest behaviour is to stop.
  echo "!! RTABMAP_LOCALIZE_DB=$RTABMAP_LOCALIZE_DB is set, and this stack CANNOT honour it."
  echo "   habitat_launch.py hardcodes database_path /root/.ros/rtabmap.db --delete_db_on_start,"
  echo "   so every launch MAPS FRESH and never localizes against a published map."
  echo "   Clear RTABMAP_LOCALIZE_DB, or restore a launch path that accepts a database."
  exit 1
fi

# same rtabmap arguments as launch/habitat_launch.py (odometry from /odom, no TF publish)
# GA-359 (2026-09-07). `publish_tf:=false`, passed here and in upstream habitat_launch.py since the
# first run, IS NOT A LAUNCH ARGUMENT of rtabmap.launch.py: the declared names are publish_tf_map
# (default TRUE) and publish_tf_odom, and ros2 launch accepts an undeclared name silently. So rtabmap
# published map->odom on /tf in every run while habitat_feed_node published a static identity
# map->odom on /tf_static -- two authorities. MEASURED on 20260907_152446: 54 of 747 detection rows
# (18 of 139 frames, T+121..229 s, right after rtabmap's "Localization mode" line) carried a camera
# pose 2-4 m from the true one; the rest matched to 3 cm. The flag is now EXPLICIT and follows the
# pose source the launcher stamps: simulator arm -> false (the feed's identity is the only authority),
# rtabmap arm -> true (and the feed node must then NOT publish the identity: perception's half).
case "${FEED_POSE_SOURCE:-simulator}" in
  rtabmap)   _RT_PUBLISH_TF_MAP=true ;;
  simulator) _RT_PUBLISH_TF_MAP=false ;;
  *) echo "!! FEED_POSE_SOURCE=${FEED_POSE_SOURCE} is neither simulator nor rtabmap"; exit 1 ;;
esac
echo ">>> pose source: ${FEED_POSE_SOURCE:-simulator} (rtabmap publish_tf_map:=$_RT_PUBLISH_TF_MAP)"
# The args are the first line of rtabmap.log so the bundle records them; before this they were
# visible only in ros2 launch's death message, i.e. only when the node died.
# GA-359 (2026-09-07 21:10). THE NODE IS LAUNCHED DIRECTLY, not through rtabmap.launch.py, for one
# parameter the launch file cannot set: pub_loc_pose_only_when_localizing. Perception gates every
# cycle under FEED_POSE_SOURCE=rtabmap on the AGE of /rtabmap/localization_pose; with the launch
# file's default (false, echoed in every bundle's rtabmap.log) rtabmap publishes that pose on every
# frame, localised or not, and the gate can never trip (rule 18). Passing the parameter inside
# rtabmap_args was tried and measured: still false. Every other parameter below reproduces what
# rtabmap.launch.py handed the node in run 20260907_152446, verified by diffing rtabmap's own
# startup echo (36 lines; two differences: this parameter, and ground_truth_base_frame_id which the
# node defaults to "base_link" where the launch pinned "" -- inert while ground_truth_frame_id is "",
# and rcl refuses an empty -p value, so the node defaults stand: odom_frame_id "" from the /odom topic). Namespace /rtabmap and node name rtabmap are kept, so /rtabmap/map and
# /rtabmap/cloud_map (read by object_manager_6) do not move. The library args stay positional and
# override any node parameter, as before ("Update ... from arguments" in the log).
# OUR DIRECT rtabmap INVOCATION IS RETIRED (owner 2026-09-10). It stood here and started a SECOND
# rtabmap beside the one habitat_launch.py starts — same namespace, same node name, both
# subscribing the same topics and both publishing map->odom. That double start was introduced
# when the launch route was added and never exercised through this script, only through a
# standalone harness; removing the flag is what made it visible. Her launch file owns rtabmap
# now, including its parameters, and `localization_mode` there decides publish_tf_map by
# construction, which is the single-authority property our publish_tf juggling was reaching for.
# GA-359 (C): RECORD /rtabmap/localization_pose FOR THE WHOLE RUN, so perception's covariance gate
# gets a MEASURED threshold from the first rtabmap-mode bundle instead of an invented one. CSV, one
# line per message, no header (ros2 topic echo --csv): header.stamp.sec, header.stamp.nanosec,
# header.frame_id, pose.pose.position x y z, pose.pose.orientation x y z w, then the 36 covariance
# values (row-major x y z roll pitch yaw). Named .log so the exit copy carries it into logs/.
# With pub_loc_pose_only_when_localizing=true the file's GAPS are the not-localised intervals.
# The TYPE IS GIVEN: without it `ros2 topic echo` exits at once when the topic is not yet advertised
# ("Could not determine the type", run 20260907_170421, 2 lines, nothing recorded) -- rtabmap starts
# ~120 s after this line. With the type it subscribes now and waits.
ros2 topic echo --csv --full-length /rtabmap/localization_pose geometry_msgs/msg/PoseWithCovarianceStamped > /tmp/localization_pose.log 2>&1 &
# ===== HER PROCEDURE (owner 2026-09-09): the stack comes up through the ROS launch file =====
# "From now on we have to use habitat launch ros2 launch file and then run the feed separately."
# The feed host (host side) and habitat_feed_node (above) stay separate, as they already were;
# what changes is that rtabmap, perception_2, object_manager_6, graph_api_bridge, rviz2 and the
# wall detector are started by lost3dsg/launch/habitat_launch.py instead of one by one here.
# The launch file is Sara's, taken from GRAPH-API main: it carries the rtabmap settings her working
# localisation uses (exact sync, empty odom frame, the publish_tf split) and it is the only version
# that declares use_wall_detector. EXERCISED end to end on 2026-09-09 before this was written:
# rtabmap, perception_2, object_manager_6, rviz2 and wall_detector all up, first cycle 5 detections.
#
# LEGACY=1 restores the one-by-one starts below. Kept for one cycle because the close path's
# RTABMAP_PID becomes the `ros2 launch` process rather than the node — harmless, since the close
# also signals by pattern (`pkill -INT -f rtabmap_slam/rtabmap`, the case this file was originally
# written for), but not yet exercised through a full close.
# THE ONE START PATH (owner 2026-09-10). Her configuration IS the configuration: our direct
# rtabmap invocation is retired, not kept as a second route. The LEGACY_NODE_STARTS escape that
# stood here for one cycle is REMOVED — a retired configuration kept behind a flag is a
# configuration somebody will set, and then two machines run different stacks and nothing says so.
_wall_arg=$([ "${WALL_DETECTOR:-0}" = "1" ] && echo true || echo false)
_loc_arg=$([ "${FEED_POSE_SOURCE:-simulator}" = "rtabmap" ] && echo rtabmap || echo ground_truth)
echo ">>> stack via habitat_launch.py (use_wall_detector:=$_wall_arg localization_mode:=$_loc_arg)"
ros2 launch lost3dsg habitat_launch.py \
    use_wall_detector:="$_wall_arg" localization_mode:="$_loc_arg" \
    > /tmp/launch.log 2>&1 &
LAUNCH_PID=$!
# The close path signals rtabmap by PATTERN as well as by pid, so a launcher pid here is safe.
RTABMAP_PID=$LAUNCH_PID; OM6_PID=""; PERCEPTION_PID=""; WALLS_PID=""

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

# GA-359 / probe a13. EXACTLY ONE AUTHORITY FOR map->odom. Measured on 20260907_152446: with
# rtabmap publishing map->odom (publish_tf_map defaulted true; publish_tf was never a launch
# argument) AND the feed node's static identity, tf2 served rtabmap's correction for 108 s and
# 54 of 747 detections landed 2-4 m from their true pose. BLOCKING: a run with two authorities
# measures that defect, so it ends here. Foreground; samples 5 s ONLY AFTER RTABMAP IS UP: rtabmap
# starts last and loads the 1.2 GB map first -- "Localization mode" came 119 s after the feed node's
# first line in run 20260907_170421, and a13 sampled at 15 s, saw no map->odom from anyone, and ended
# a healthy run (my probe's false positive; rule 24's asymmetry, paid once). So: wait for rtabmap's
# own readiness line (localization or mapping), up to 300 s, then sample. No line by then is itself
# a finding, and a13 then reports whatever the TF tree holds.
_a13_deadline=$(( $(date +%s) + 300 ))
until grep -q "Localization mode\|Mapping mode\|rtabmap: subscribe_odom" /tmp/rtabmap.log 2>/dev/null; do
  [ "$(date +%s)" -ge "$_a13_deadline" ] && { echo "!! a13: rtabmap printed no readiness line in 300 s; sampling anyway"; break; }
  sleep 3
done
sleep 10
if python3 /graph_api/lost3dsg/test/preflight_gate.py --only a13 \
     --pose-source "${FEED_POSE_SOURCE:-simulator}" --out /tmp/preflight_a13.json \
     >> /tmp/a13.log 2>&1; then
  echo ">>> a13 pose_authority: one authority for map->odom (${FEED_POSE_SOURCE:-simulator})"
else
  echo "!! a13 pose_authority FAILED — ending the run. $(tail -3 /tmp/a13.log | tr '\n' ' ')"
  exit 3
fi

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
  # GA-434 / RULE 73. THE FEED ENDS THE RUN, because with no cap nothing else does.
  #
  # The owner removed the cap for base runs ("no caps this time"), and the tour used to turn in
  # place forever once its waypoints ran out. habitat_feed_host.py now writes feed_ended.json when
  # the house tour is complete and its settle period has passed. That file is the ONLY channel
  # between the two: the feed host runs on the host, this script runs in the container, and
  # /ws/output is the directory they share.
  #
  # It is a normal end, not a death, and terminating_node.json already says so in its own note --
  # a status of 0 with a node name is how MAPPING_TIME ends a mapping run.
  if [ -f /ws/output/feed_ended.json ]; then
    _dead_node="FEED_ENDED"; _dead_rc=0
    _dead_why=" (the feed host completed the house tour)"
    break
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
  # UNDER THE LAUNCH FILE THE PID WATCH SEES ONE PROCESS, NOT SIX. `ros2 launch` keeps running when
  # a node under it dies — measured 2026-09-09: perception_2 died at startup and rtabmap carried on
  # for minutes — so without this the run would continue with the detector gone. The launch output
  # names the node and its exit code, so that line ends the run and supplies GA-430's field. It is
  # prose rather than an interface, which is exactly what GA-430 was written to stop depending on;
  # under this route it is the only source there is, and the field is parsed from it ONCE here
  # rather than grepped by every reader afterwards.
  if [ -z "$_dead_node" ] && [ -n "${LAUNCH_PID:-}" ] && [ -f /tmp/launch.log ]; then
    _died=$(grep -m1 -oE "\[[a-zA-Z0-9_.-]+\]: process has died \[pid [0-9]+, exit code -?[0-9]+" /tmp/launch.log || true)
    if [ -n "$_died" ]; then
      _dead_node=$(printf '%s' "$_died" | sed -E 's/^\[([a-zA-Z0-9_.-]+)\].*/\1/')
      _dead_rc=$(printf '%s' "$_died" | sed -E 's/.*exit code (-?[0-9]+)/\1/')
      break
    fi
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      _dead_rc=0; wait "$LAUNCH_PID" || _dead_rc=$?
      _dead_node="ROS2_LAUNCH"; break
    fi
  fi
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
# GA-430. THE TERMINATING NODE, WRITTEN AS A FIELD AND NOT LEFT TO A GREP. The testing lane's crash
# condition greps the announce line below out of the stack log, and said so inside its own verdict:
# the same producer records status-ZERO deaths too, so a non-zero status does not discriminate and
# the line does. A line is not an interface — it is prose that a future edit here would silently
# break. This writes what the loop above already knows, at the moment it knows it, into the bundle;
# the launcher folds it into run_metadata beside the cap block, same producer and same moment.
python3 - "$_dead_node" "$_dead_rc" "${_dead_why:-}" > /ws/output/terminating_node.json <<'PY'   || echo "!! could not write terminating_node.json — the bundle cannot name what ended the run"
import json, sys
node, rc, why = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    rc_i = int(rc)
except ValueError:
    rc_i = None
print(json.dumps({
    "node": node,
    "exit_status": rc_i,
    "exit_status_raw": rc,
    "reason": why.strip(" ()") or None,
    "note": "the node whose exit ENDED the run, recorded by the container that watched it. A "
            "status of 0 is still a death: MAPPING_TIME ends a mapping run normally, and a watched "
            "node exiting 0 unexpectedly ends a run too, so read `node` and not the status alone.",
}, indent=2))
PY
if [ "$_dead_node" = "FEED_ENDED" ]; then
  echo ">>> FEED ENDED${_dead_why:-} — closing the stack. This is the normal end of a base run."
else
  echo "!! $_dead_node exited with status ${_dead_rc}${_dead_why:-} — ending the run."
  echo "   The stack is not left running: a run missing any of these nodes measures nothing further."
fi
tail -5 /tmp/perception.log

# An EMPTY variable here is "exit: : numeric argument required", which is what run
# 20260903_110622 printed instead of an exit status. Default it, and prefer the code of the
# node that actually died when perception was not the one that died.
_rc="${_perception_rc:-}"
[ -z "$_rc" ] && _rc="${_dead_rc:-1}"
case "$_rc" in ''|*[!0-9-]*) _rc=1 ;; esac
[ "$_rc" -lt 0 ] 2>/dev/null && _rc=$(( 128 - _rc ))   # a signal, as a shell exit status
exit "$_rc"
