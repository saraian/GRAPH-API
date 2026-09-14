#!/usr/bin/env bash
# THE ONLY SCRIPT A PERSON RUNS. There is nothing behind it.
#
# Owner, 2026-09-10: "ONE config file, ONE install script and ONE launch script."
# Owner, 2026-09-11: "We're not using live_run.sh anymore. The storey schedule should come before
# as a multi-storey run is actually multiple runs. live_run will have to be discarded and use the
# official config of run and run_sim_headless.sh". So this file holds BOTH halves:
#   the storeys are resolved FIRST, then one run is performed per storey.
# lost3dsg/test/live_run.sh and lost3dsg/test/run_house.sh are DELETED. If a step anywhere names
# either of them, the step is wrong.
#
# WHAT A BASE RUN IS (rule 73, owner 2026-09-10): the whole house, every storey, NO CAP, one
# mapping session per storey. It produces ONE BUNDLE PER STOREY, not one per run. Nothing spans
# the house: each storey's map has its own SLAM origin, so coverage, an object seen on two
# storeys and the duplicate rate are all post-hoc joins across the bundles the manifest names.
#
#   ./run_sim.sh                       every storey of the scene in the config
#   ./run_sim.sh hm3d_00861            that scene, this run only
#   ./run_sim.sh --one-storey          a single storey
#   ./run_sim.sh --config <file>       a specific run configuration
#   ./run_sim.sh --schedule <file>     a list of runs, each with its own configuration
#   ./run_sim_headless.sh ...          the same, with no rviz and no preview window
#
# SETTINGS LIVE IN THE CONFIG FILE, not in flags here. A setting you cannot find in the config is
# a bug in the config, not a missing flag. The environment still overrides for one run:
#
#   WHERE THINGS GO
#     WORKSPACE_ROOT  root holding maps/ and results/ (default: this checkout)
#     RESULTS_DIR     run bundles; the bundle IS the live output (default $WORKSPACE_ROOT/results)
#     SCHEDULE_DIR    cached exploration schedules (default $WORKSPACE_ROOT/schedules)
#   THE HOUSE
#     HOUSE_FLOORS    the storeys to tour, e.g. "-1.59 +1.21". Read from the published maps
#                     when unset, so no discovery launch is needed.
#     FEED_SPAWN_FLOOR  with --one-storey, which storey. REQUIRED when the scene has per-floor
#                     maps and you are pinning by hand.
#   EXPLORATION
#     FEED_SCHEDULE          a schedule file; built and cached automatically when unset.
#                            MANDATORY: it is the only motion policy, so a run refuses without one
#     FEED_EXPLORATION_LAPS  complete passes of the storey (config habitat.exploration_laps)
#     FEED_NAVIGATION_MODE   navigate (drive it) or teleport (set the pose)
#   FEED (each also readable from the config's habitat.* section)
#     FEED_FPS  FEED_WIDTH  FEED_HEIGHT  FEED_MOVE_FN  FEED_CAMERA_PITCH_DEG
#     FEED_GT_SEMANTIC  FEED_SHOW  FEED_OVERLAY
#   STACK
#     RVIZ            0 to run headless; also skipped automatically with no X socket
#     WALL_DETECTOR   1 starts the wall detector (default 0)
#     FEED_HF_OFFLINE 1 refuses to download models mid-run (default 1)
#     IMAGE_TAG       container image
#   EXTENSION
#     EXT_ENV_FILE    a shell file the extension ships; REQUIRED when the config names hooks.filter
#     EXT_MOUNTS EXT_ENV_PASS EXT_TREES EXT_POLICY_JSON EXT_STORE_REPAIR EXT_POST_RUN
#
# HOW IT RUNS N STOREYS FROM ONE FILE: it re-invokes itself once per storey, as a child process
# with GRAPH_API_STOREY_CHILD=1. One file, N processes. A shell function would have shared this
# process's traps, background jobs and exported variables across storeys, and storey 2 would
# inherit storey 1's feed host and container names.

# ======================================================================================
# ONE STOREY, ONE RUN. Everything from here to the parent section is the engine: the gate, the
# host feed, the container, the archive. It is entered only as a child of the parent below.
# ======================================================================================
if [ "${GRAPH_API_STOREY_CHILD:-0}" = "1" ]; then
# Live demo on this machine: habitat renders on the host (conda habitat_env),
# the ROS 2 stack runs in the graphapi-run:humble-ga290 container (patched rtabmap, GA-290)
# over a TCP feed.
# Watch: web viewer at http://localhost:${BRIDGE_PORT:-8081} and snapshots in $OUT_DIR.
#   ./live_run.sh [scene]  # foreground; ctrl-C stops everything
# scene: hm3d_00861 (default) | hm3d_00337 | hm3d_00770 | mp3d_17DRP
# HABITAT_SCENE/HABITAT_DATASET env vars still override everything.
# THE ENGINE RUNS UNDER `set -e` ALONE, deliberately. It was written that way over three
# weeks and reads unset variables in a hundred places with a bare $VAR; the parent above runs
# under `set -euo pipefail`, and inheriting -u here would abort the first such read. The modes
# are set per section rather than once at the top so neither half constrains the other.
set -e
set +u
set +o pipefail

# NUMBERS ARE FORMATTED IN THE C LOCALE, NOT THE MACHINE'S. Measured on Gin 2026-09-10, whose
# LANG is it_IT.UTF-8: `printf "floor_%+.2f" 1.21` failed outright with "1.21: numero non valido",
# and on a locale that ACCEPTS the comma it would have produced `floor_+1,21` -- a map directory
# name that matches nothing, so the run would have looked for a published map, not found it, and
# mapped from scratch while reporting the floor it was asked for. An error is the lucky outcome
# here. Every float this script formats or compares goes through the same locale, so it is set
# once, at the top, rather than guarded per call site.
export LC_ALL=C
# HERE IS lost3dsg/test, AND IT IS NO LONGER DERIVED FROM $0. This engine used to be
# lost3dsg/test/live_run.sh, so `dirname $0` was that directory and ~200 paths below are
# written as "$HERE/<something in lost3dsg/test>". The script now lives at the repository
# root (owner 2026-09-11: "live_run will have to be discarded"), so $0 is the root and HERE
# is pinned instead. Deriving it would silently move every one of those paths.
REPO=$(cd "$(dirname "$0")" && pwd)
HERE="$REPO/lost3dsg/test"

# GA-319. The Modal endpoint URL is a CREDENTIAL -- the deployed app exposes fastapi_endpoint with
# no proxy auth, so the URL alone buys GPU time on this account. It used to live in config.yaml and
# was therefore committed. It now lives in an untracked, gitignored file beside this script, and is
# forwarded into the container by the existing `-e MODAL_PERCEPTION_URL`. Sourced, not required: a
# local-backend run needs none of this, and client.py already fails loudly and by name when the
# backend is "modal" and neither the config nor the environment supplies an endpoint.
[ -f "$HERE/env.local.sh" ] && . "$HERE/env.local.sh"
# Local setup is REQUIRED for a modal-backend run: the default configs ship with an empty
# modal_endpoint on purpose, so without env.local.sh there is no endpoint, the perception node
# would take zero detections and die on GA-94b ~4 min in, after the simulator was spent. Refuse
# here instead, and say what to do.
if [ -z "${MODAL_PERCEPTION_URL:-}" ] && [ "$(python3 -c "import sys,yaml;c=yaml.safe_load(open(sys.argv[1])) or {};print((c.get('perception') or {}).get('backend','local'))" "$HERE/${CFG_NAME:-regolo_config.yaml}" 2>/dev/null)" = "modal" ]; then
  echo "!! perception.backend is 'modal' but MODAL_PERCEPTION_URL is unset."
  echo "!! Local setup required: cp $HERE/env.local.sh.example $HERE/env.local.sh && chmod 600 $HERE/env.local.sh, then fill in the URL."
  echo "!! (env.local.sh is gitignored on purpose -- the URL is a credential, GA-319.)"
  exit 1
fi
# THE WORKSPACE. Maps, results and run bundles live here. Derived from this script's own
# location, not hardcoded, so a clone anywhere works: two levels above $REPO is the directory the
# checkout sits in.
#
# IT REQUIRES NO PARTICULAR PACKAGE. This used to insist on an extension's package directory being
# present and EXIT when it was absent, so a clone of this stack alone could not launch at all --
# a dependency pointing the wrong way, from the generic stack onto the thing that extends it.
# Owner ruling 2026-09-09. An extension supplies itself through EXT_ENV_FILE and EXT_MOUNTS.
# THE CHECKOUT IS THE WORKSPACE. Owner instruction 2026-09-10: results live in <repo>/results and
# there is no separate workspace directory. So the default is the repository itself, and the
# "two levels above the checkout" derivation is GONE -- from /DATA/GRAPH-API it computed to "/",
# the -d test passed, and a run would have written its bundle to /runs and published maps to
# /maps. A default that cannot be wrong beats a check that catches it being wrong.
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$REPO}
[ -d "$WORKSPACE_ROOT" ] || { echo "!! WORKSPACE_ROOT=$WORKSPACE_ROOT does not exist"; exit 1; }
# GA-434 (2026-09-10). EXISTING IS NOT ENOUGH, and "/" exists.
#
# The derivation above is "two levels above the checkout", which was right while this repo was
# vendored inside the workspace. Consolidated to its own directory, two levels above it is the
# filesystem root: WORKSPACE_ROOT became "/", the -d test passed, and a run would have written its
# bundle to /runs, published maps to /maps, and found no published map at /maps -- mapping from
# scratch and reporting it, with every path in the bundle pointing somewhere nobody looks.
#
# So the root must LOOK like a workspace: one of the three directories a run reads or writes. That
# is a fact about the directory rather than a fact about this file's location, which is what went
# stale. Set WORKSPACE_ROOT in env.local.sh to point somewhere else.
if [ ! -d "$WORKSPACE_ROOT/maps" ] && [ ! -d "$WORKSPACE_ROOT/runs" ] && [ ! -d "$WORKSPACE_ROOT/results" ]; then
  echo "!! WORKSPACE_ROOT=$WORKSPACE_ROOT holds no maps/, runs/ or results/ directory."
  echo "   That is where bundles, maps and scratch go, so this is not a workspace. It is derived"
  echo "   as two levels above $REPO, which is wrong whenever the checkout is not inside the"
  echo "   workspace. Set WORKSPACE_ROOT explicitly (env.local.sh) and re-run."
  exit 1
fi

MON_PID=""
FEED_PID=""
# `latest` means THE LAST BUNDLE WORTH READING. Two conditions, and it runs from the EXIT trap
# so it fires HOWEVER the run ends.
#
#   1. the gate passed  -- a pre-flight verdict: the run was allowed to start
#   2. the bundle holds a measured artefact -- it actually produced something
#
# It lived after `docker run` and never ran when the script was stopped with Ctrl-C, which is
# the documented way to stop it. `20260831_024019` -- gate 8/8, 952 decisions, the largest
# bundle this project has produced -- did not claim `latest`, while an aborted run from 20:31
# with no decisions still held it. Moving the check EARLY let bad runs claim the pointer;
# moving it LATE stopped good ones claiming it. The trap is where it belongs.
#
# CEILING, stated: this separates measured-nothing from measured-something. It CANNOT separate
# truncated from complete -- that is the flow question and it belongs to a post-run check.
_publish_map_if_earned() {
  # The map library lives on the HOST at $WORKSPACE_ROOT/maps/<scene>/, not inside a bundle — a
  # library that lives in one run's output directory is not a library. The container cannot write
  # there, so it leaves a marker and the host copies.
  #
  # The marker is written ONLY by a passing integrity_check. No marker means either the check
  # failed or it never ran (GA-98: rtabmap did not close). BOTH mean do not publish, and the
  # message says which is which, because unchecked and torn are different states.
  [ "${MAPPING_ONLY:-0}" = "1" ] || return 0
  local db="$RUN_DIR/rtabmap.db" mark="$RUN_DIR/rtabmap.db.INTEGRITY_OK"
  if [ ! -f "$mark" ]; then
    echo "!! map NOT published: no INTEGRITY_OK marker — the map either failed integrity_check or was never checked."
    return 0
  fi
  # GA-119. STAMP THE FLOOR BEFORE PUBLISHING, and publish INTO that floor's directory.
  #
  # hm3d_00861 has four floors and only one has ever been mapped. Until tonight the map did not
  # say which, so a robot loading it could not tell whether it was the right storey and every
  # coverage claim citing it was silently a claim about one floor.
  #
  # The height is MEASURED from the map's own Node poses, never from the floor the run was asked
  # for. A run that requested 0.43 and spawned on 1.35 publishes to floor_+1.35, and the sidecar
  # says 1.35. The artefact reports what happened, not what was intended.
  #
  # The scene's floor set comes from bev_data.json, which the feed host already writes into
  # OUT_DIR — the same navmesh clustering the BEV renders (GA-93). Not scraped from a log.
  local floors="[]"
  [ -f "$OUT_DIR/bev_data.json" ] && floors=$(python3 -c \
    "import json,sys;print(json.dumps(json.load(open(sys.argv[1])).get('floors') or []))" \
    "$OUT_DIR/bev_data.json" 2>/dev/null || echo "[]")
  python3 "$HERE/stamp_floor.py" "$db" "$floors" || {
    echo "!! map NOT published: the floor could not be measured from its own node poses."
    echo "   An unstamped map is honest; a map published without knowing its storey is not."
    return 0; }
  # Directory name from the NEAREST SCENE FLOOR when the clustering supplies one, so two runs of
  # the same storey land in the same directory rather than floor_+1.207 and floor_+1.209. Falls
  # back to the measured median when the scene floors are unknown.
  # OWNER RULING 25, 2026-09-01, their words: "a map spanning two storeys will flatten and become
  # a wrong map." So single_floor is a PUBLISH PRECONDITION, not a sidecar field to read later.
  #
  # A 2D occupancy grid has one cell per (x, y). Two storeys stacked into it put the upstairs
  # furniture into the downstairs grid as obstacles that are not there. The map is not merely
  # incomplete — it is WRONG in a way a robot cannot detect at navigation time.
  #
  # Measured on the -1.59 run, which is why this exists: 251 nodes on -1.59, 99 on +1.35, 18 on
  # +0.43. A third of the map was a different floor. It published, because the gate only asked
  # whether the database opened. That map is now relabelled out of the canonical namespace.
  # COVERAGE, the sibling precondition. A map can be one storey, open cleanly, and be a map of
  # nowhere: floor_+0.43 published 471 nodes of SEVENTEEN distinct viewpoints in a 1.04 x 0.42 m
  # box and passed every check there was. Node count does not separate that from a real map;
  # distinct poses does.
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('covers_a_storey', True) else 1)" \
      "${db}.floor.json" || {
    echo "!! map NOT published: it COVERS ALMOST NOTHING."
    python3 -c "import json,sys
d=json.load(open(sys.argv[1]))
print('   %d nodes but only %d distinct poses, footprint %s m (threshold %d distinct)'
      % (sum(d['nodes_per_nearest_floor'].values()), d['distinct_poses'],
         d['footprint_m'], d['min_distinct_poses']))" "${db}.floor.json"
    echo "   Either the tour could not move, or the target is not a storey. Check the feed log's"
    echo "   floor_detail: a cluster below min_floor_share is a landing or a gallery, not a floor."
    return 0; }
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d['single_floor'] else 1)" \
      "${db}.floor.json" || {
    echo "!! map NOT published: it SPANS MORE THAN ONE STOREY and a 2D grid cannot represent that."
    python3 -c "import json,sys
d=json.load(open(sys.argv[1]))
print('   nodes per storey:', d['nodes_per_nearest_floor'])
print('   spread: %.2f m, basis: %s' % (d['node_z_spread_m'], d['single_floor_basis']))" "${db}.floor.json"
    echo "   The bundle keeps the map; the library does not take it. Fix the tour's floor"
    echo "   confinement (habitat.floor_confinement) and re-run this floor."
    return 0; }
  local fl; fl=$(python3 -c "import json,sys
d=json.load(open(sys.argv[1]))
z=d.get('nearest_scene_floor'); z=d['floor_height_m'] if z is None else z
print(f'floor_{round(z,2)+0.0:+.2f}')" "${db}.floor.json")   # +0.0 turns -0.0 into +0.00: the lookup side prints floor_+0.00 for FEED_SPAWN_FLOOR=0.00 (run 135714 published floor_-0.00)
  local dest="$WORKSPACE_ROOT/maps/${SCENE_ARG}/${fl}"
  mkdir -p "$dest"
  # NEVER OVERWRITE A PUBLISHED MAP. The copy in the library is the only copy (GA-295), and the cp
  # below would replace it silently. A re-map of a floor moves the previous map aside under its
  # own publish date, sidecars with it, the way the 640x480 map was kept by hand on 7 Sep.
  # ponytail: named by mtime, not resolution; the provenance sidecar (camera_db) says the rest.
  if [ -f "$dest/rtabmap.db" ]; then
    local old; old="rtabmap_superseded_$(date -r "$dest/rtabmap.db" +%Y%m%d_%H%M)"
    local s; for s in "" .floor.json .params-sha .provenance.json .INTEGRITY_OK; do
      [ -e "$dest/rtabmap.db$s" ] && mv -n "$dest/rtabmap.db$s" "$dest/$old.db$s"
    done
    # mv -n refuses when $old already exists (two publishes in one minute); then the cp below
    # would overwrite after all. Refuse to publish instead of pretending the move happened.
    [ ! -e "$dest/rtabmap.db" ] || { echo "!! map NOT published: previous map could not be moved aside ($old.db exists)"; return 1; }
    echo "    previous map moved aside: $dest/$old.db"
  fi
  # COPY, NOT HARD LINK. Owner ruling GA-295(c), 4 Sep: the library is an INDEPENDENT copy. The
  # link era was a disk-space decision when /DATA was 99% full (it is ~72% now); its cost was
  # measured on hm3d_00861: the canonical map, its source bundle and a localize db were ONE
  # INODE UNDER THREE NAMES, so when the write path was live (before the :ro mount, 546fd17)
  # every surviving copy drifted together by 24 MB and no copy could say which table grew. A
  # copy decouples the library from the bundle it came from. A publish that cannot afford the
  # copy FAILS LOUDLY — no fallback to the link: a silent fallback would resurrect the
  # single-inode library exactly when the disk is tight again.
  cp "$db" "$dest/rtabmap.db"
  cp "${db}.floor.json" "$dest/rtabmap.db.floor.json"
  cp "$RUN_DIR/rtabmap.db.params-sha" "$dest/rtabmap.db.params-sha" 2>/dev/null || true
  # PROVENANCE AT PUBLISH TIME (GA-295: mp3d_17DRP was published with none and its drift became
  # uncheckable). bytes + sha256 of the copy, the bundle's own integrity marker, source run —
  # the drift check travels with the map. Failure here is LOUD by design: this is the last call
  # in the EXIT trap, so nothing after it is lost.
  python3 "$HERE/stamp_map_provenance.py" "$dest/rtabmap.db" "$SCENE_ARG" "$RUN_ID" "$mark" \
      "$dest/rtabmap.db.params-sha"
  echo "    map published: $dest/rtabmap.db ($(cat "$mark"))"
}

_point_latest_if_earned() {
  local verdict measured f
  verdict=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('verdict','absent'))" \
              "$RUN_DIR/preflight.json" 2>/dev/null || echo absent)
  measured=""
  for f in hook_decisions.jsonl actual_perceptions.json perception_latencies.jsonl; do
    [ -s "$RUN_DIR/$f" ] && { measured="$f"; break; }
  done
  # GA-373. A live-mounted tree that moved DURING the run (a7 re-run at teardown) makes
  # the bundle describe code it did not execute; such a bundle is never `latest`.
  local teardown; teardown=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('verdict','absent'))" \
              "$RUN_DIR/a7_teardown.json" 2>/dev/null || echo absent)
  if [ "$teardown" = "fail" ]; then
    echo ">>> latest NOT moved: a7 at teardown reads '$teardown' — a live root changed during the run."
    echo "    it still points at $(readlink "$RUNS_DIR/latest" 2>/dev/null || echo '<unset>')"
  elif [ "$verdict" = "pass" ] && [ -n "$measured" ]; then
    ln -sfn "$RUN_DIR" "$RUNS_DIR/latest"
    echo ">>> gate passed, measured output present ($measured) -- $RUN_DIR is now latest"
  else
    echo ">>> latest NOT moved: preflight '$verdict', measured artefact '${measured:-none}'."
    echo "    it still points at $(readlink "$RUNS_DIR/latest" 2>/dev/null || echo '<unset>')"
  fi
}

cleanup() {
  # PID-scoped: only kill what THIS instance started. A pattern-based
  # pkill here would let two concurrent live_run.sh instances destroy
  # each other's feed host (seen live 2026-08-25).
  [ -n "$MON_PID" ] && kill -9 "$MON_PID" 2>/dev/null || true
  # GA-464: this lane no longer starts a sibling RViz -- the launch file's is the only one. The
  # removal stays so a stale graphapi_rviz from an older run cannot sit on the network holding
  # subscriptions and a window that shows the WRONG run's topics.
  docker rm -f graphapi_rviz >/dev/null 2>&1 || true
  [ -n "$FEED_PID" ] && kill -9 "$FEED_PID" 2>/dev/null || true
  # Archive the HOST-written artefacts here rather than only after the container exits. The
  # per-frame viewpoint series and the feed host's own log are written on this side, and the
  # post-run copy below never runs when the script is stopped with Ctrl-C — which is the
  # documented way to stop it. feed_host.log reached 1 of 21 shipped bundles for this reason.
  if [ -n "${RUN_DIR:-}" ] && [ -d "$RUN_DIR" ]; then
    # NOTHING TO COPY: $OUT_DIR IS $RUN_DIR since 2026-09-10. These two lines copied the scratch
    # directory into the bundle; with one directory per run they would copy files onto themselves,
    # and `cp` reporting "are the same file" into /dev/null is how a no-op looks like a step.
    # GA-336. Delete the scratch localization copy (~1.2 GB). Here, not earlier: the container
    # holds it open until rtabmap closes, and this trap runs after `docker run` has returned.
    # Its identity is preserved in run_metadata.json (localize_db_source + sha), so deleting the
    # bytes loses no evidence. Left behind, every run would cost 1.2 GB of results/ for nothing.
    [ -n "${LOCALIZE_DB_COPY:-}" ] && rm -f "$LOCALIZE_DB_COPY" 2>/dev/null || true
    # GA-238. NO COPY IS NEEDED, and the first version of this block wrongly added one.
    # `-v "$RUN_DIR":/ws/output` (below) means the container's output directory IS the bundle,
    # and live_stack_container.sh exports GRAPH_API_OUTPUT_DIR=/ws/output. So once
    # input_output.prepare_crops honours that variable -- which was the actual fix -- the
    # crops are written straight into $RUN_DIR/cropped_images and there is nothing to move.
    #
    # $OUT_DIR was a DIFFERENT mount (/out) until 2026-09-10; it is now the bundle itself, so
    # would print "no crops to harvest" over a bundle that has them, which is worse than
    # silence: it would have sent the next reader looking for a perception failure that did
    # not happen. Counted and reported instead, so the run states what it retained.
    _ncrops=$(ls -1 "$RUN_DIR/cropped_images"/*.jpg 2>/dev/null | wc -l)
    echo "    crops retained: $_ncrops in $RUN_DIR/cropped_images"

    # Per-class entry counts against the scene annotation, written INTO the bundle so a run
    # states this about itself. 20260901_174810_hm3d_00861 left 112 "dining chair" entries in
    # a scene annotated with sixteen chairs, and nobody noticed until it was reconstructed by
    # hand days later. It reports class counts and NOT a clustering, because clustering needs
    # a radius and the radius moves the answer by more than 2x.
    #
    # HOST SIDE, AFTER THE CONTAINER HAS EXITED. Ground truth must never be readable from the
    # runtime path; this runs here for the same reason analyse_run.py does.
    # GA-437 (2026-09-10). tools/ IS THE EXTENSION'S, NOT THIS STACK'S, so its absence is normal.
    # A GRAPH-API-only checkout has no tools/ and every run on Gin printed
    # "ModuleNotFoundError: No module named 'tools'" -- non-fatal, but it is an extension-only tool
    # that survived the 2026-09-09 removal ("this repository names none of them"), and the boundary
    # checker cannot see it because the name contains no "found". Skipped with a REASON rather than
    # a stack trace; its proper home is EXT_POST_RUN, which the extension already supplies.
    if [ -f "$WORKSPACE_ROOT/tools/class_counts.py" ]; then
      ( cd "$WORKSPACE_ROOT" && python3 -m tools.class_counts "$RUN_DIR" ) 2>&1 \
        | sed 's/^/    /' || echo "    class_counts failed (non-fatal)"
    else
      echo "    class counts SKIPPED: $WORKSPACE_ROOT/tools/class_counts.py is absent (no extension)"
    fi

    # GA-395. THE CAP IS NOT IN THE BUNDLE, and the envelope's cap condition therefore reads
    # UNCHECKABLE on every bundle in the archive. run_capped.sh records it in the TESTING lane's
    # directory (last_cap.json), which is not the bundle and does not travel with it. These three
    # values are what a reader needs to say whether a run was cut short, and this script holds all
    # three by the time the trap runs.
    #
    # `capped` is READ FROM THE DRIVER'S MARKER, NOT INFERRED. Inferring it from the elapsed time or
    # from a non-zero container status would call a manual `docker stop` a cap and would call a cap
    # that fired one second before a clean finish a clean finish (rule 5: a value that is not a
    # measurement must not sit where measurements sit). The marker is checked for FRESHNESS against
    # this run's own start: run_capped.sh removes a stale one at launch, but a run started WITHOUT
    # the driver would otherwise inherit the previous run's marker and report itself capped.
    _cap_marker="${CAP_FIRED_MARKER:-/DATA/GRAPH-API/.handoff/lanes/testing/.cap_fired}"
    # The absent-marker case has TWO meanings and the note must not assert the wrong one. When a cap
    # was armed and simply did not fire — the commonest healthy outcome — run_capped.sh never writes
    # the marker, and the old single sentence ("no cap was armed or this run was not launched through
    # run_capped.sh") was FALSE on both counts for exactly that run. The numbers were right and the
    # sentence beside them was not, which is the shape that gets believed because its neighbours are
    # trustworthy. Found by the testing lane reading the implementation rather than waiting for a bundle.
    _capped=false; _cap_fired_at=null
    if [ -n "${CAP_MIN:-}" ]; then
      _cap_note="a cap was armed at ${CAP_MIN} min and did NOT fire: the run ended on its own"
    else
      _cap_note="no cap was armed (CAP_MIN unset), so nothing could fire"
    fi
    if [ -f "$_cap_marker" ]; then
      _mt=$(stat -c %Y "$_cap_marker" 2>/dev/null || echo 0)
      if [ "$_mt" -ge "${RUN_START_EPOCH:-0}" ]; then
        _capped=true; _cap_fired_at=$(cat "$_cap_marker" 2>/dev/null || echo null)
        _cap_note="the driver stopped the container at the cap"
      else
        _cap_note="a cap marker exists but PREDATES this run's start, so it belongs to an earlier run and was ignored"
      fi
    fi
    # GA-430. The container names what ended the run in terminating_node.json; fold it into the
    # metadata beside the cap block so a reader has a FIELD instead of grepping the stack log for a
    # sentence. Absent means the container never reached its own end — killed, or it died before the
    # watch loop — and that is itself worth recording rather than defaulting to "unknown".
    python3 - "$RUN_DIR/run_metadata.json" "${CAP_MIN:-}" "$(( $(date +%s) - ${RUN_START_EPOCH:-0} ))" \
             "$_capped" "$_cap_fired_at" "$_cap_note" "${START_AFTER_STACK:-0}" <<'PY' \
      || echo "!! could not stamp the cap block into run_metadata.json — the bundle cannot state whether it was cut short"
# >>> TEST-EXTRACT stamp_block  (test_terminating_node.py runs the block between these
# markers against synthetic bundles. GA-430's absent-branch is the one no run exercises:
# a container killed before its watch loop leaves no terminating_node.json, and 'absent'
# must not read as 'unknown'.)
import json, os, sys
p, cap, elapsed, capped, fired, note, anchor = sys.argv[1:8]
d = json.load(open(p))
term = os.path.join(os.path.dirname(p), "terminating_node.json")   # GA-430
try:
    d["terminating_node"] = json.load(open(term))
except (OSError, ValueError) as exc:
    d["terminating_node"] = {"node": None, "note": f"no terminating_node.json in the bundle ({exc.__class__.__name__}): "
                                                   "the container did not reach its own end — killed from outside, or "
                                                   "it died before the watch loop. Absent is not 'unknown'."}
# GA-430, second half. A CLOSED VOCABULARY FOR HOW THE LAUNCH ENDED, so a reader tests a value
# instead of matching a node name it has to know. The testing lane's early-death condition reads a
# LOG LINE today; a line is prose that a future edit breaks silently, and with no cap the ending is
# the fact their whole eligibility test turns on.
#   tour_complete  the feed wrote feed_ended.json with reason house_tour_complete (rule 73's normal end)
#   operator_abort feed_ended.json existed WITHOUT that reason -- somebody ended the launch by hand
#                  through the archive path. A finished tour and a hand-stopped one must never be
#                  the same fact: "the tour completed" is the claim a baseline turns on.
#   mapping_time   a mapping run reached its own deadline, which is also a normal end
#   node_death     a watched node exited, whatever its status -- 0 included, which is why the NODE
#                  and not the status is what discriminates
#   unrecorded     no terminating_node.json: killed from outside, or dead before the watch loop.
#                  NOT "unknown": it says the container never reached its own end.
_tn = d["terminating_node"].get("node")
d["terminating_node"]["ended"] = ({"FEED_ENDED": "tour_complete",
                                   "FEED_ABORTED": "operator_abort",
                                   "MAPPING_TIME": "mapping_time"}.get(_tn, "node_death")
                                  if _tn else "unrecorded")
d["cap"] = {                                            # GA-395, keys ADDED (rule 6)
    "cap_minutes": int(cap) if cap.strip().isdigit() else None,
    "cap_anchor": "stack_up" if anchor == "1" else "launch",
    "elapsed_seconds": int(elapsed),
    "elapsed_note": "measured by live_run.sh from its own start to its exit trap; run_capped.sh's "
                    "last_cap.json starts a few seconds earlier, so the two differ by the driver's startup",
    "capped": capped == "true",
    "cap_fired_at": None if fired in ("null", "") else int(fired),
    "cap_note": note,
}
json.dump(d, open(p, "w"), indent=2)
# <<< TEST-EXTRACT stamp_block
PY
    # GA-293 + GA-401 (written by the ontology lane for this file, revision 3; applied here after
    # two interactions with this trap that neither of us could see from one side alone). A killed
    # container leaves two faults in one bundle: a Turtle short of its last records, and no
    # prov:generated edges — both are written at close(), which a SIGKILL never reaches. One call
    # repairs both, in the image, because neither the host NOR the image has pyoxigraph until the
    # stack's own startup installs it from the vendored wheel (live_stack_container.sh:112), so this
    # repeats that install rather than assuming it.
    #
    # GATED ON THE RUN'S CONFIGURATION, NOT ON THE STORE BEING ABSENT. MAPPING_ONLY runs no detector
    # and no extension, so there is nothing to repair. Skipping on a MISSING STORE would collapse
    # two states that mean opposite things: nothing was supposed to produce one, versus it ran and
    # its store is gone — and the second is exactly what rc 2 exists to catch.
    #
    # NOTHING HERE EXITS. This runs inside the EXIT trap, and an exit would skip the two calls below
    # it, so a bookkeeping failure would throw away a fifteen-minute map. Inside a trap, `exit` is
    # not failing loudly: it is silently skipping whatever the trap had left to do.
    if [ "${MAPPING_ONLY:-0}" = "1" ]; then
      echo "GA-293 store repair: skipped, MAPPING_ONLY run has no extension store" >> "$RUN_DIR/logs/store_repair.log"
    elif [ -z "${EXT_STORE_REPAIR:-}" ]; then
      # NO HOOK IS A SKIP, NOT A SUCCESS. The command below read `${EXT_STORE_REPAIR:-true}`, so
      # with no hook configured the container ran `true <path>`: nothing done, exit 0, and the
      # guard -- which tests rc -ne 0 -- recorded a SUCCESSFUL repair. A default that cannot fail
      # is not a default, it is a silent pass. Absent-hook and failed-repair are different states.
      echo "GA-293 store repair: skipped, no extension repair hook configured (EXT_STORE_REPAIR unset)" \
        >> "$RUN_DIR/logs/store_repair.log"
    else
      _kg="$RUN_DIR/knowledge_graph.ttl"
      _kg_before=$( [ -f "$_kg" ] && wc -l < "$_kg" || echo 0 )
      # NON-ROOT (revision 4). A root container rewrites knowledge_graph.ttl — the bundle's main
      # artefact — as root, in a directory the host's own tooling then has to manage; I hit the same
      # wall deleting a root-owned store from my scratch. EXERCISED here, which the author could not
      # do: --user with PYTHONUSERBASE installs the wheel with pip --user and the artefacts come out
      # owned by the invoking user. The one warning it prints is pip's cache being unwritable.
      # THE TWO EXT_ VARIABLES MUST BE PASSED WITH -e. They are HOST variables; inside the
      # single-quoted block below they expand in the CONTAINER, where they are unset.
      # GA-437 (2026-09-10). WORKSPACE_ROOT IS THE DATA ROOT, NOT THE EXTENSION TREE, and mounting
      # it here made one variable mean both. It held while the extension and the data lived in one
      # directory; the owner's move to a neutral /DATA/workspace separates them, and this mount then
      # provides an /ext with no vendor/wheels and no repair script. The repair would run against
      # nothing and the guard below would read whatever rc that produced.
      #
      # REFUSE RATHER THAN REPAIR NOTHING. The extension declares its own tree through EXT_MOUNTS
      # (owner ruling 2026-09-09), which is passed through below; what this block still needs from
      # the extension is a mount point that holds vendor/wheels. Checked on the HOST, where the
      # directory is, so the failure names its cause instead of appearing as a pip error.
      if [ ! -d "$WORKSPACE_ROOT/vendor/wheels" ] && [ -z "${EXT_MOUNTS:-}" ]; then
        echo "GA-293 store repair: SKIPPED. EXT_STORE_REPAIR=$EXT_STORE_REPAIR is set, but nothing" \
          >> "$RUN_DIR/logs/store_repair.log"
        echo "  declares an extension tree: EXT_MOUNTS is empty and $WORKSPACE_ROOT holds no" \
          >> "$RUN_DIR/logs/store_repair.log"
        echo "  vendor/wheels. The repair needs the extension's tree, not the data root." \
          >> "$RUN_DIR/logs/store_repair.log"
        _repair_rc=4
      else
      docker run --rm --entrypoint bash --user "$(id -u):$(id -g)" -e PYTHONUSERBASE=/tmp/pyuser \
        -e EXT_MOUNT_POINT -e EXT_STORE_REPAIR ${EXT_MOUNTS:-} \
        -v "$WORKSPACE_ROOT":"$EXT_MOUNT_POINT":ro -v "$RUN_DIR":/ws/output "$IMAGE_TAG" -lc '
          python3 -c "import pyoxigraph" 2>/dev/null ||
            pip install --user --quiet --no-index --find-links="$EXT_MOUNT_POINT"/vendor/wheels pyoxigraph
          cd "$EXT_MOUNT_POINT" && $EXT_STORE_REPAIR /ws/output/knowledge_graph.ttl
        ' >> "$RUN_DIR/logs/store_repair.log" 2>&1
      _repair_rc=$?
      fi
      _kg_after=$( [ -f "$_kg" ] && wc -l < "$_kg" || echo 0 )
      echo "GA-293 store repair: rc=$_repair_rc, $_kg_before -> $_kg_after lines" >> "$RUN_DIR/logs/store_repair.log"
      # rc 0 repaired-or-nothing-to-repair; 2 no store; 3 empty store; 4 the repair could not run.
      # A SHRINKING file is never acceptable at any rc, so it folds into the same condition.
      if [ "$_repair_rc" -ne 0 ] || [ "$_kg_after" -lt "$_kg_before" ]; then
        echo "!! GA-293: store repair FAILED (rc=$_repair_rc, $_kg_before -> $_kg_after lines); see logs/store_repair.log" >&2
        printf '{"rc": %s, "lines_before": %s, "lines_after": %s, "note": "the knowledge store was not repaired; prov:generated and the final dump may be short"}\n' \
          "$_repair_rc" "$_kg_before" "$_kg_after" > "$RUN_DIR/store_repair_failed.json"
      fi
    fi
    _point_latest_if_earned
    _publish_map_if_earned
  fi
}
trap cleanup EXIT

# Refuse to start over a live stack instead of killing it (fixed container
# name and fixed ports mean two instances can never coexist).
if docker ps --filter name=graphapi_live --format '{{.Names}}' | grep -q .; then
  echo "!! container graphapi_live is already running — aborting (stop it first: docker rm -f graphapi_live)"; exit 1
fi
if ss -tln 2>/dev/null | grep -q ':7799 '; then
  echo "!! port 7799 busy — a feed host is already running — aborting"; exit 1
fi

echo ">>> host habitat feed (scene renders on the host GPU)"
# Matterport scenes on this machine: three HM3D v0.2 examples (semantic-annotated)
# plus the MP3D example. Pick with the first argument, override with env vars.
HM3D_ROOT=${HM3D_ROOT:-/DATA/habitat_matterport/hm3d_example}
MP3D_ROOT=${MP3D_ROOT:-/DATA/habitat_matterport/versioned_data/mp3d_example_scene_1.1}
# RANDOMISATION (owner ruling 2026-09-07 17:35, "random seed + scene per corpus run, fixed seed for A/B
# arms"; this lane implements, after the mapping runs, which are done). OPT-IN via MAP_DRAW=1: an omitted
# scene argument already MEANS hm3d_00861 in every existing recipe, so drawing on absence would silently
# randomise every A/B arm that omits it.
#
# A (SCENE, FLOOR) PAIR IS DRAWN, NOT A BARE SCENE. Maps are published per floor, and a scene that has
# per-floor maps with no FEED_SPAWN_FLOOR is REFUSED below, so drawing a scene name alone would refuse the
# very run it had just chosen. The pool is read off the PUBLISHED maps, so a draw can only name something
# that can actually be localised against. Each pin is honoured independently; the floor rides with the
# scene, because pinning one and drawing the other yields a pair nobody published.
SEED_SOURCE=pinned; SCENE_SOURCE=pinned
if [ "${MAP_DRAW:-0}" = "1" ]; then
  _pairs=$(ls -d "$WORKSPACE_ROOT"/maps/*/floor_* 2>/dev/null | sed "s|.*/maps/||")
  [ -n "$_pairs" ] || { echo "!! MAP_DRAW=1 but no published per-floor map under $WORKSPACE_ROOT/maps — nothing to draw from"; exit 1; }
  if [ "$#" -ge 1 ]; then
    _pairs=$(printf "%s\n" "$_pairs" | grep "^$1/") \
      || { echo "!! MAP_DRAW=1 with scene '$1' pinned, but it has no published per-floor map"; exit 1; }
  fi
  _pick=$(printf "%s\n" "$_pairs" | shuf -n 1)
  _scene=${_pick%%/*}; _floor=${_pick#*/floor_}
  [ "$#" -ge 1 ] || { set -- "$_scene"; SCENE_SOURCE=drawn; }
  [ -n "${FEED_SPAWN_FLOOR:-}" ] || FEED_SPAWN_FLOOR="$_floor"
  [ -n "${FEED_SEED:-}" ] || { FEED_SEED=$(shuf -i 1-2147483647 -n 1); SEED_SOURCE=drawn; }
  echo "    draw: scene $1 floor $FEED_SPAWN_FLOOR seed $FEED_SEED (scene_source=$SCENE_SOURCE seed_source=$SEED_SOURCE, $(printf "%s\n" "$_pairs" | wc -l) mapped pairs)"
fi
case "${1:-hm3d_00861}" in
  hm3d_00861) DEF_SCENE=$HM3D_ROOT/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb
              DEF_DATASET=$HM3D_ROOT/hm3d_annotated_basis.scene_dataset_config.json ;;
  hm3d_00337) DEF_SCENE=$HM3D_ROOT/00337-CFVBbU9Rsyb/CFVBbU9Rsyb.basis.glb
              DEF_DATASET=$HM3D_ROOT/hm3d_annotated_basis.scene_dataset_config.json ;;
  hm3d_00770) DEF_SCENE=$HM3D_ROOT/00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb
              DEF_DATASET=$HM3D_ROOT/hm3d_annotated_basis.scene_dataset_config.json ;;
  mp3d_17DRP) DEF_SCENE=$MP3D_ROOT/17DRP5sb8fy/17DRP5sb8fy.glb
              DEF_DATASET=$MP3D_ROOT/mp3d.scene_dataset_config.json ;;
  *) echo "unknown scene '$1' (hm3d_00861|hm3d_00337|hm3d_00770|mp3d_17DRP)"; exit 1 ;;
esac
# VLM config: real regolo endpoint when a key is present, offline smoke fallback
if [ -n "${REGOLO_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${OPENAI_API_KEY:-$REGOLO_API_KEY}"
fi
# THE CONFIG COMES FROM THE PARENT, AND ONLY FROM THE PARENT. Owner 2026-09-11: "use the
# official config of run and run_sim_headless.sh". This used to default CFG_NAME to
# regolo_config.yaml when a key was present and smoke_config.yaml when it was not, so a run
# could load a config nobody chose -- and regolo_config.yaml is UNTRACKED, so the fallback
# named a file that does not exist in a fresh clone. A missing config is now a refusal.
# ASSIGNED HERE, not merely inherited. test_env_stamp.sh requires every variable interpolated
# into run_metadata.json to be ASSIGNED above the heredoc, and the reason is a measured one: a
# variable that is only inherited is one some other caller can leave unset, and the stamp then
# records "" while the run loads a config nobody named. The :? form is both the assignment and
# the refusal, so there is one place to read rather than two.
CFG_NAME="${CFG_NAME:?is not set. This section is not an entry point: run ./run_sim.sh or ./run_sim_headless.sh, which read the config and set it. Pass --config <file> to choose one.}"
# EXPORTED, because `docker run -e CFG_NAME` copies the parent process environment and a
# shell variable that was only assigned is not in it. Without this the echo below prints
# regolo while the container falls back to smoke_config.yaml — the operator reads one
# config and the run loads another, and every bundle records the name that was printed.
# The two files are not interchangeable: smoke omits vlm.base_url and vlm.model, so both
# fall to the defaults (localhost:11434, gemma4:e2b) and an API key aimed at regolo
# reaches a local socket instead.
export CFG_NAME
echo "    config: $CFG_NAME"

# Setup the persistent run bundle (never overwritten across runs)
# GA-434. OVERRIDABLE, so a caller that launches the stack ONCE PER STOREY knows each bundle's path
# without scraping it from a log line. run_house.sh sets a distinct stamp per storey. Unset, this is
# what it always was. The bundle directory is refused below if it already exists, so a stale export
# of this variable cannot make two runs share one bundle.
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
SCENE_ARG=${1:-hm3d_00861}

# Live run output. A TIMESTAMPED DIRECTORY under $WORKSPACE_ROOT/results/, never /tmp.
# Owner ruling, relayed to this lane rather than given to it directly (rule 8's second half):
#   "no output should go to the temp directory, always in a timestamped experiment results dir
#    inside the results/ dir inside the $WORKSPACE_ROOT directory."
#
# Assigned HERE and not at the top of the file because the name needs RUN_TIMESTAMP and
# SCENE_ARG. The only earlier references are inside cleanup(), a function body evaluated when
# the trap fires — long after this line — so moving the assignment down is safe. That was
# checked rather than assumed.
#
# results/ is outside every hashed root, so run output moves no digest. A test asserts it.
# runs/<id> stays the curated archive that every tool and criterion reads; results/ is the
# live working directory. Two things at once is the anti-pattern of the week.
# GA-99. EXPORTED, and the export is the whole fix.
#
# This was a bare assignment from f3fc88a (2026-08-24) until 2026-08-31, and HARMLESS for six days
# because its default was /tmp/graphapi_live -- the same directory habitat_feed_host.py fell back
# to. The two agreed by COINCIDENCE, not by wiring. Moving this default to results/ (same day)
# made the missing export real, and the feed host's fallback chain then hid it: every run wrote
# feed_stats.json and bev_data.json outside its own bundle, which is why NO BUNDLE HAS EVER
# CONTAINED THEM -- not run 19, not run A, not run B.
#
# Found by the warning that replaced the fallback chain, on the first run after it landed.
# The class is "two paths that agree by accident until one of them moves". Nothing else in this
# tree reports it, so that warning is permanent.
# ONE DIRECTORY PER RUN, IN THE REPOSITORY. Owner instruction 2026-09-10: "results should be
# stored in /DATA/GRAPH-API/results. The live output should be part of the run results in the
# results/ folder (so no /DATA/workspace folder and no duplicate directories to hold the output)."
#
# WHAT THIS REPLACES. There were TWO directories per run: $WORKSPACE_ROOT/runs/<id> (the bundle,
# mounted at /ws/output) and $WORKSPACE_ROOT/results/<id> (the "live output", mounted at /out),
# whose *.json, *.jsonl and *.log were COPIED into the bundle at the end. Measured on
# 20260910_141811: every file in results/ also existed in the bundle, so the second directory was
# 7 MB of duplicate per run and 37 of the 93 results directories had no bundle at all.
#
# OUT_DIR IS NOW THE BUNDLE. /out and /ws/output become two mount points onto ONE host directory,
# so nothing needs copying and nothing can be left behind in a scratch directory. The a5 probe is
# unaffected: it asserts that no artefact predates run start in each directory it is given, and it
# does not compare the two against each other.
RUN_ID="${RUN_TIMESTAMP}_${SCENE_ARG}"
RESULTS_DIR=${RESULTS_DIR:-$REPO/results}
RUN_DIR="$RESULTS_DIR/$RUN_ID"
# RUNS_DIR IS THE SAME DIRECTORY NOW, kept as a name because nine places use it: the `latest`
# symlink, the cycle-budget reader, and run_house.sh's per-storey bookkeeping. MEASURED, and it is
# why this line exists: when RESULTS_DIR replaced RUNS_DIR and the uses were left behind, the
# launcher printed "symlinked as /latest" and would have written that symlink at the FILESYSTEM
# ROOT, while `last_frame_age_rejected.py` was handed an empty argument. Replacing a definition is
# not replacing its uses.
RUNS_DIR="$RESULTS_DIR"
# Stamped BEFORE the directory is created, so every artefact the run legitimately writes is newer
# than it. The gate's a5 probe fails on anything older -- a directory left dirty by a previous run,
# or a file copied in by hand.
RUN_START_EPOCH=$(date +%s)
export RUN_START_EPOCH
# A bundle directory that already holds a run is never reused: two runs in one directory produce a
# bundle whose files come from both and whose metadata describes one.
if [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
  echo "!! $RUN_DIR already exists and is not empty. RUN_TIMESTAMP=$RUN_TIMESTAMP is already taken."
  exit 1
fi
# CHECKED BEFORE THE DIRECTORY IS CREATED. When OUT_DIR and RUN_DIR became one directory the
# `mkdir -p "$RUN_DIR/logs"` below moved above this test, so logs/ existed by the time it ran and
# every run refused itself. Order is the whole content of this check.
export OUT_DIR="$RUN_DIR"
mkdir -p "$RUN_DIR/logs"
echo "    run results: $RUN_DIR (live output and bundle are the same directory)"

# GA-258b. EXPORTED, because the FEED HOST needs it. The host process reads
# merge_pending.json to decide how long to dwell, and that file is written by the container
# into /ws/output -- which is bind-mounted to $RUN_DIR, not to $OUT_DIR (/out). The feed host
# was building the path from GRAPH_API_OUTPUT_DIR, which only exists INSIDE the container, so
# on the host it resolved to a bare relative filename and never opened. Measured on run
# 20260902_125130: every dwell line read "? merges pending (sweep None)" and every waypoint
# ran to the 90-frame cap.
export RUN_DIR
# GA-463. rtabmap's database is written to /root/.ros, which habitat_launch.py hardcodes, and until
# now that was the container's own writable layer -- outside every mount and discarded by --rm.
# MEASURED: no bundle since the launch-file route holds an rtabmap.db, and the integrity check reads
# /ws/output/rtabmap.db, a path nothing writes, so it printed "no rtabmap.db — nothing to check" on
# every run. That reads like a benign skip and is a missing map. Mounting the storey's own directory
# there writes the database outside docker AND into the bundle, and a later launch cannot destroy it
# with --delete_db_on_start because each storey has its own.
mkdir -p "$RUN_DIR/ros"
# NO "crops" DIRECTORY. One name for the crops, and it is `cropped_images` -- the name
# input_output.py writes and graph_api_bridge._crop_dirs reads. This line used to create an
# EMPTY `crops/` beside it, and that empty directory was not merely untidy: the dashboard
# resolves `crops` BEFORE `cropped_images` (dashboard/server.py:86, `next(d for d in ... if
# d.is_dir())`), so it found the empty one and served nothing. MEASURED 2026-09-11: 64
# bundles on disk carry a crops/ directory and NOT ONE of them has a file in it, while
# cropped_images/ holds 1012 files and 37 MB for a single run.
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/snapshots"
# GA-381. THE DIRECTORY IS CREATED BEFORE THE CHECKS RUN, so every early refusal (a10, the mapping
# cap check, the missing-map refusal, a bad config) leaves a directory that a sweep cannot tell from
# a genuine early run — and 67 bundles exist, the oldest of which predate the gate and have no
# preflight.json either, so "no preflight.json" does not separate them. A marker written here and
# removed at the last moment before `docker run` does separate them: anything still carrying it was
# refused before the stack ever started. Cheap by choice; not creating the directory before the
# checks is the expensive fix and would move every path that writes into it.
printf "%s\n" \
  "This run was REFUSED before the container started, or died before it. Not a run." \
  "Written when the bundle directory was created; removed immediately before docker run." \
  "run_id: $RUN_ID" "created: $(date -Is)" > "$RUN_DIR/NOT_STARTED"
echo "    run bundle: $RUN_DIR (symlinked as $RUNS_DIR/latest)"

# Snapshot calibration, config, and run metadata
# GA-233. DERIVED, not a literal. This was a heredoc stating 640x480 with fx=320 while the
# run used 1280x960 -- right only because the sensor kept a 90 deg hfov and a 4:3 aspect, so
# the RATIO the frustum tools actually read came out the same by luck. The day FEED_HFOV or
# the aspect changes, tools/visible_gt.py and tools/frustum_gt.py would score against the
# wrong cone and say nothing about it.
#
# Written from the resolved sensor configuration instead, with fx from the pinhole relation
# fx = (w/2) / tan(hfov/2). The values are ALSO recorded as `source: derived` so a reader can
# tell a computed calibration from a copied constant.
_CAL_W="${FEED_WIDTH:-1280}"; _CAL_H="${FEED_HEIGHT:-960}"; _CAL_HFOV="${FEED_HFOV:-90}"
python3 - "$RUN_DIR/calibration.json" "$_CAL_W" "$_CAL_H" "$_CAL_HFOV" <<'PYCAL'
import json, math, sys
path, w, h, hfov = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
fx = (w / 2.0) / math.tan(math.radians(hfov) / 2.0)
json.dump({
    "camera_name": "habitat_camera_optical",
    "resolution": {"width": w, "height": h},
    "hfov_deg": hfov,
    "intrinsics": {"fx": round(fx, 4), "fy": round(fx, 4),
                   "cx": w / 2.0, "cy": h / 2.0},
    "distortion_model": "plumb_bob",
    "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
    "source": "derived",
    "source_note": ("GA-233. fx = (w/2)/tan(hfov/2) from the resolved sensor settings of THIS "
                    "run. Was a hardcoded 640x480/fx=320 heredoc that stayed correct only "
                    "while the hfov and aspect happened not to change."),
}, open(path, "w"), indent=2)
print(f"    calibration: {w}x{h} hfov {hfov} -> fx {fx:.1f} (derived)")
PYCAL
# The config the FEED HOST will load. It resolves $HERE/$CFG_NAME through the same
# config.py::_load as the container, and _load returns the DEFAULTS when the file is absent —
# silently, with hooks.filter empty. Two processes, one name, and nothing recording the split.
[ -f "$HERE/$CFG_NAME" ] || { echo "!! $HERE/$CFG_NAME does not exist — the feed host would run on config.py defaults; aborting"; exit 1; }
cp "$HERE/$CFG_NAME" "$RUN_DIR/config.yaml"

# These sit ABOVE run_metadata.json rather than beside `docker run`, because the metadata
# heredoc interpolates all six. They were 78 lines below it and interpolated empty, so the
# policy block was written as `"enforce": ,` and the validator aborted every run. A shell
# variable's VALUE depends on where it is read, and a heredoc is read where it is written.

# Room VIEW frames (perception's half of GA-350, vendor 4c0e0dc): object_manager_6 saves a room view
# on room entry and every ROOM_FRAME_STRIDE_M metres of travel, at most ROOM_FRAME_MAX per room, and
# the typer reads those. Defaults here equal object_manager_6.py:85-86 so the stamp says what ran.
# WHERE AN EXTENSION IS MOUNTED INSIDE THE CONTAINER. One name, defaulted, so this repository
# never writes another deployment's path. It was one hard-coded path in eleven places.
EXT_MOUNT_POINT="${EXT_MOUNT_POINT:-/ext}"

# ---- EXTENSION ENVIRONMENT ---------------------------------------------------------------
# An extension package (an admission layer, a policy layer) has knobs of its own. They used to be
# exported HERE, by name, so this launcher carried one deployment's policy vocabulary and could not
# run without it. Owner ruling 2026-09-09: this repository names none of them.
#
# EXT_ENV_FILE is a shell file the extension ships. It exports whatever it needs and sets
# EXT_ENV_PASS to the names that must cross into the container -- `docker run -e VAR` sends nothing
# when VAR is unset in the parent, so a bare forward silently ships the container's own default
# while the bundle records the launcher's. Exporting in that file makes the two the same value.
#
# EXT_MOUNTS holds any extra `-v` arguments the extension needs (its own tree, read-only).
# EXT_POST_RUN is a command run once the bundle is closed.
#
# ABSENT IS NORMAL AND SILENT: a run with no extension is a valid run of this stack alone. A named
# file that does not EXIST is an error, because somebody meant to load something and it is not there.
if [ -n "${EXT_ENV_FILE:-}" ]; then
  [ -r "$EXT_ENV_FILE" ] || { echo "!! EXT_ENV_FILE=$EXT_ENV_FILE is not readable"; exit 1; }
  # shellcheck disable=SC1090
  . "$EXT_ENV_FILE"
  echo "    extension env: $EXT_ENV_FILE ($(echo ${EXT_ENV_PASS:-} | wc -w) variable(s) forwarded)"
fi
EXT_E_ARGS=""
for _v in ${EXT_ENV_PASS:-}; do EXT_E_ARGS="$EXT_E_ARGS -e $_v"; done

# A CONFIG THAT NAMES AN EXTENSION FILTER NEEDS THAT EXTENSION'S ENVIRONMENT. Without it the
# object manager dies at the first proposal carrying a room, mid-run, twenty minutes in --
# measured 2026-09-09 by the simulator lane running the launch path by hand. A launch-time
# refusal is the same information, an hour earlier and with the map still unbuilt.
# GA-475 (2026-09-10, found by the ontology lane, whose run this refused). NOT EVERY FILTER IS AN
# EXTENSION FILTER. The test was "the config names a filter", which is one step wider than the
# reason above: a filter that ships INSIDE this repository needs no extension environment, and
# there is nothing for EXT_ENV_FILE to point at. The message even offered "or clear hooks.filter",
# which for an in-repo filter means "or stop using the feature".
#
# So ask WHERE the filter lives: take the module before the colon and look for it beside the
# perception code. Verified against all four configs before landing -- size_gate_config names
# envelope_size, whose module is present, and is allowed; regolo_config names found.filter, whose
# module is not, and is refused exactly as before; smoke_config and graphapi_only_config name no
# filter and are untouched.
# THE COMMENT IS STRIPPED FIRST, AND THE MATCH IS ANCHORED. The previous form was
# `sed -E 's/.*"([^":]+):.*/\1/'`, and `.*"` is GREEDY: on the tracked config the line reads
#     filter: "envelope_size:SizeFilter"   # "pkg.module:ClassName", subclass of hooks.Filter
# so it matched the last quote before a colon -- the one in the COMMENT -- and extracted
# `pkg.module`. The launcher then refused the DEFAULT config for wiring a filter that does not
# exist. Measured on Gin 2026-09-10, on the first run with the default config. Anchoring at the
# start of the line and dropping the comment reads the value rather than the nearest quoted text.
_filter_module=$(sed -E 's/#.*//' "$HERE/$CFG_NAME" 2>/dev/null \
                 | grep -E '^[[:space:]]*filter:[[:space:]]*"[^"]+"' \
                 | sed -E 's/^[[:space:]]*filter:[[:space:]]*"([^":]+):.*/\1/')
# A dotted name is a package path, so it becomes a directory path before the file test.
_filter_path="$HERE/../src/perception_module/$(printf '%s' "${_filter_module:-}" | tr '.' '/').py"
if [ -z "${EXT_ENV_FILE:-}" ] && [ -n "$_filter_module" ] && [ ! -f "$_filter_path" ]; then
  echo "!! $CFG_NAME wires the filter '$_filter_module', which is NOT in this repository"
  echo "   ($_filter_path does not exist), and EXT_ENV_FILE is unset."
  echo "   The extension's variables would never reach the container and the run would die"
  echo "   at the first proposal that needs one. Set EXT_ENV_FILE, or name a filter that ships here."
  exit 1
fi

# THE HABITAT KEYS THIS SCRIPT DEFAULTS, READ FROM THE CONFIG FILE ONCE. Owner instruction
# 2026-09-10: "we need to read config.yaml ALWAYS". Three settings in this file were exported with
# HARDCODED defaults, and because the feed reads the environment FIRST, the config's values were
# dead: `camera_pitch_deg` could not leave the file at all, and `mapping_seconds` was overridden by
# whichever branch below ran. Measured on Gin: a config saying `mapping_seconds: 0` and
# `camera_pitch_deg: -30.0` produced a feed reporting `mapping_seconds=150.0 camera_pitch_deg=0.0`.
# An exported default is not a default -- it is an override nobody asked for.
_cfg_hab() {   # $1 = key, $2 = fallback when the file is silent
  python3 -c "
import sys, yaml
try:
    c = yaml.safe_load(open(sys.argv[1])) or {}
    v = (c.get('habitat') or {}).get(sys.argv[2])
    print(sys.argv[3] if v is None else v)
except Exception:
    print(sys.argv[3])" "${GRAPH_API_CONFIG:-$HERE/$CFG_NAME}" "$1" "$2" 2>/dev/null || echo "$2"
}

# CAMERA PITCH, in degrees, negative looks DOWN. It reaches the feed host and the bundle: a
# setting that changes what the camera SEES and is not recorded is the shape that made six days
# of tour runs unreadable.
export FEED_CAMERA_PITCH_DEG="${FEED_CAMERA_PITCH_DEG:-$(_cfg_hab camera_pitch_deg 0)}"
export ROOM_FRAME_MAX="${ROOM_FRAME_MAX:-5}"
export ROOM_FRAME_STRIDE_M="${ROOM_FRAME_STRIDE_M:-1.5}"
# GA-359 (owner 2026-09-07 ~18:20 "switch to rtabmap localised poses"; design plan/14). The pose
# source the boxes are placed in. simulator = the feed node's static identity map->odom is the only
# authority and rtabmap does not publish TF (today's behaviour, made explicit); rtabmap = rtabmap
# publishes map->odom and the feed node must not (perception's half, not landed yet: do NOT set
# rtabmap before it lands, or two authorities publish again). Read by live_stack_container.sh and
# by habitat_feed_node.py (once perception lands its half); stamped as pose_source.
# THE DEFAULT COMES FROM THE CONFIG FILE, not from this line. Owner instruction 2026-09-10: "we
# need to read config.yaml ALWAYS". `habitat.localization_mode` (rtabmap | ground_truth) is the
# switch her launch file already uses, and NOTHING read it: this defaulted to `simulator`,
# live_stack_container.sh:460 derived `localization_mode:=ground_truth` from that, and a run whose
# config said `rtabmap` did ground-truth pose while the bundle recorded the value it was given.
# Measured on Gin 2026-09-10 on the first baseline run. An explicit FEED_POSE_SOURCE still wins,
# so a one-off arm needs no config edit.
#
# The warning that used to sit here -- "do NOT set rtabmap before perception lands its half, or two
# authorities publish again" -- is DISCHARGED: habitat_feed_node.py:197-201 reads FEED_POSE_SOURCE
# and publishes map->odom only under `simulator` (GA-359). One authority either way.
_cfg_loc=$(python3 -c "
import sys, yaml
try:
    c = yaml.safe_load(open(sys.argv[1])) or {}
    print(((c.get('habitat') or {}).get('localization_mode') or '').strip().lower())
except Exception:
    print('')" "${GRAPH_API_CONFIG:-$HERE/$CFG_NAME}" 2>/dev/null || echo "")
case "$_cfg_loc" in
  rtabmap)      _pose_default=rtabmap ;;
  ground_truth) _pose_default=simulator ;;
  "")           _pose_default=simulator ;;
  *)            echo "!! habitat.localization_mode=$_cfg_loc is neither rtabmap nor ground_truth"; exit 1 ;;
esac
export FEED_POSE_SOURCE="${FEED_POSE_SOURCE:-$_pose_default}"
[ -n "$_cfg_loc" ] && echo "    pose source: $FEED_POSE_SOURCE (config localization_mode: $_cfg_loc)"
case "$FEED_POSE_SOURCE" in simulator|rtabmap) ;; *) echo "!! FEED_POSE_SOURCE=$FEED_POSE_SOURCE is neither simulator nor rtabmap"; exit 1 ;; esac

# Feed geometry. These were interpolated ONLY into the launch line 160 lines below and appeared
# NOWHERE in the bundle — a run recorded its seed and nothing else about how the agent moved.
#
# THE WALK/DWELL FAMILY IS GONE (owner 2026-09-11, "completely remove the old sampling policy").
# FEED_WALK, FEED_DWELL, FEED_DWELL_MODE, FEED_DWELL_MIN, FEED_DWELL_MAX and
# FEED_DWELL_SIGNAL_MAX_AGE_S set a burst cycle that no longer exists: the schedule states its own
# stops and its own scan at each, so there is nothing left for a dwell to hold. habitat_feed_host.py
# REFUSES any of these names rather than ignoring it, and this launcher no longer exports them.
#
# WHY THE NOTES BELOW MATTERED, kept in one sentence because the bundles still exist: dwell 60,
# dwell 0 (from 2026-08-31) and adaptive (from 2026-09-07) are three different bundle families and
# none of them is comparable with a scheduled run. The full record is in git history at commit
# eae203e and in the run_metadata.json of every bundle made before today.
export FEED_SEED="${FEED_SEED:-7}"
export FEED_FPS="${FEED_FPS:-3}"
# THE SCHEDULE'S OWN SETTINGS, exported here so run_metadata.json can record them and so the
# `:?` guards below have something to check. Config first, environment second, the same precedence
# every other feed setting uses. Three laps is the owner's default (2026-09-10): one lap cannot
# tell a change in the world from a change in the route, and the laps are identical by design.
export FEED_EXPLORATION_LAPS="${FEED_EXPLORATION_LAPS:-$(_cfg_hab exploration_laps 3)}"
export FEED_MOVE_FN="${FEED_MOVE_FN:-$(_cfg_hab navigation_mode navigate)}"
# GA-330. Ground truth ON by default. The scene ships its semantic mesh, the feed host renders
# it, the feed node publishes /gt/semantic_instance and the archive joins it per detection --
# and the switch below was 0 in every one of the first 12 bundles, so not one row was ever
# labelled ("no semantic frame" on 100% of rows). The cost is a third render per frame on the
# host; the archive refuses on any shape mismatch rather than guessing. Set 0 to opt out.
# THE WALL DETECTOR, FROM THE CONFIG. live_stack_container.sh reads ${WALL_DETECTOR:-0} and
# nothing exported it, so `run.wall_detector` in the config was INERT -- a key a reader would
# take for the setting while the stack always ran with the detector off. That is how a whole
# storey stayed one room: no walls, so no doorway candidate had support, so no cut was proposed.
# An explicit WALL_DETECTOR in the environment still wins, as with every other knob here.
_cfg_run() {   # $1 = key under `run:`, $2 = fallback
  python3 -c "
import sys, yaml
try:
    c = yaml.safe_load(open(sys.argv[1])) or {}
    v = (c.get('run') or {}).get(sys.argv[2])
    print(sys.argv[3] if v is None else ('1' if v is True else ('0' if v is False else v)))
except Exception:
    print(sys.argv[3])" "${GRAPH_API_CONFIG:-$HERE/$CFG_NAME}" "$1" "$2" 2>/dev/null || echo "$2"
}
export WALL_DETECTOR="${WALL_DETECTOR:-$(_cfg_run wall_detector 0)}"
echo "    wall detector: $WALL_DETECTOR (config run.wall_detector)"

export FEED_GT_SEMANTIC="${FEED_GT_SEMANTIC:-1}"
# MAPPING_ONLY builds a localization map and runs no detector. 900 s is a STARTING POINT AND
# NOT A MEASUREMENT: the only dwell=0 coverage figure that exists is run A's 7.5 m in 636 s, and
# run A did not achieve full coverage -- it is the run that died. hm3d_00861's navmesh has FOUR
# floor levels (measured on run B: [-1.59, 0.43, 1.35, 2.21]), so five minutes was never
# plausible. REPLACE FROM THE FIRST MAPPING RUN'S OWN COVERAGE STATS.
export MAPPING_ONLY="${MAPPING_ONLY:-0}"
export FEED_SPAWN_FLOOR="${FEED_SPAWN_FLOOR:-}"

# MAPPING IS THE EXCEPTION NOW, NOT THE DEFAULT. A detection run used to rebuild an occupancy map
# from scratch for 150 s unless somebody remembered FEED_MAPPING_SECONDS=0 — and forgetting cost a
# run tonight. If a published map exists for this scene and floor, localize against it and map for
# zero seconds. If none exists, map, and SAY SO rather than doing it silently.
#
# The published map is chosen by the requested spawn floor when there is one, because a map of a
# different storey is not a map of this run's world. With no spawn floor, only a scene-level map
# is eligible: guessing a floor here would localize against the wrong one and every pose would be
# confidently wrong.
# RTABMAP_SLAM=1 IS REFUSED, HARD — NO FALLBACK. Owner ruling, 4 Sep ~15:55: "we will not use
# slam" (GA-290 register, SLAM ban). This supersedes 2f0a57f's notice block, which let a run map
# from scratch and leave the published map unused: SLAM grows the graph it is optimising and
# starves the perception loop (run 20260904_082146: ~0.14 Hz, 14 frames rejected, worst 24.30 s
# against the 15.0 s guard), so its readings are mode artifacts that a10 would then refuse every
# localization launch for, and poses from a SLAM run are NOT comparable to the published map's
# frame. The route away from the localization-mode abort is the rtabmap patch (PLAN_1.3 §31),
# not SLAM.
if [ "${RTABMAP_SLAM:-0}" = "1" ]; then
  echo "!! RTABMAP_SLAM=1 is banned by owner ruling 2026-09-04 (GA-290 register, SLAM ban)."
  echo "   Localization against the published map is the only mode; see PLAN_1.3 §31."
  exit 1
fi
if [ "${MAPPING_ONLY:-0}" != "1" ] && [ "${RTABMAP_SLAM:-0}" != "1" ] && [ -z "${RTABMAP_LOCALIZE_DB:-}" ]; then
  if [ -n "$FEED_SPAWN_FLOOR" ]; then
    _mapdir=$(printf "$WORKSPACE_ROOT/maps/%s/floor_%+.2f" "$SCENE_ARG" "$FEED_SPAWN_FLOOR")
  else
    _mapdir="$WORKSPACE_ROOT/maps/$SCENE_ARG"
  fi
  # A CHECKED FALLBACK, not a guess. The canonical hm3d map lives at the SCENE level rather than
  # under floor_<z>, because it was published before per-floor publication existed. Falling back to
  # it blindly would localize a floor-1.35 run against whatever that file happens to be. So the
  # fallback is allowed only when the map's OWN STAMP says it covers the requested floor — the
  # nearest_scene_floor that stamp_floor.py measured from its node poses.
  if [ ! -f "$_mapdir/rtabmap.db" ] && [ -n "$FEED_SPAWN_FLOOR" ] \
     && [ -f "$WORKSPACE_ROOT/maps/$SCENE_ARG/rtabmap.db.floor.json" ]; then
    if python3 -c "import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if abs(float(d.get('nearest_scene_floor') or 1e9) - float(sys.argv[2])) < 1e-6 else 1)" \
        "$WORKSPACE_ROOT/maps/$SCENE_ARG/rtabmap.db.floor.json" "$FEED_SPAWN_FLOOR" 2>/dev/null; then
      _mapdir="$WORKSPACE_ROOT/maps/$SCENE_ARG"
      echo "    scene-level map stamped for floor $FEED_SPAWN_FLOOR — using it"
    fi
  fi
if [ -f "$_mapdir/rtabmap.db" ]; then
    # GA-433 (2026-09-10). THE PUBLISHED MAP IS NOT USED, AND THIS SAYS SO INSTEAD OF PRETENDING.
    #
    # Until today this branch copied the map to scratch (~1.2 GB, 12 s), exported
    # RTABMAP_LOCALIZE_DB and printed "localizing against a scratch COPY". habitat_launch.py owns
    # rtabmap now and hardcodes database_path /root/.ros/rtabmap.db with --delete_db_on_start, so
    # the copy was never opened: the run mapped fresh while its log and its bundle said localized.
    # That is the failure GA-380 refuses in the other direction, so the copy and the claim are gone
    # and the regime is named in run_metadata.json instead (localization_regime).
    #
    # THE MAP LIBRARY IS DEAD CODE UNDER THIS CONFIGURATION — the params-sha sidecar, the :ro
    # canonical mount (GA-295/158), the scratch copy (GA-336) and the per-floor publish refusal
    # (GA-380). It is left standing, unused, pending the owner's ruling on the localization regime.
    LOCALIZE_DB_SOURCE=""
    LOCALIZE_DB_SHA=""
    echo "    a published map exists at $_mapdir/rtabmap.db and THIS RUN WILL NOT USE IT."
    echo "    habitat_launch.py maps fresh every launch (--delete_db_on_start); see PLAN_1.3 §67."
  elif ls -d "$WORKSPACE_ROOT/maps/$SCENE_ARG"/floor_* >/dev/null 2>&1; then
    # GA-380 (2026-09-08). Maps are published PER FLOOR now and the scene-level rtabmap.db of
    # hm3d_00861 was moved aside on 7 Sep, so an unpinned localisation run would have fallen
    # through to "map from scratch" with a note nobody reads — a verification run that was meant
    # to localise would have mapped for 150 s and measured a different regime (rule 14: the
    # fallback is the defect). Refuse and name the floors that exist.
    # GA-433 (2026-09-10): no run localises any more, so the refusal no longer protects a regime.
    # It is kept because it still forces the spawn floor to be chosen deliberately rather than
    # inherited from seed-7's unconstrained spawn, which lands on whichever storey it lands on.
    echo "!! NO map at $_mapdir, but this scene has per-floor maps: $(ls -d "$WORKSPACE_ROOT/maps/$SCENE_ARG"/floor_* | xargs -n1 basename | tr '\n' ' ')"
    echo "   Pin FEED_SPAWN_FLOOR=<z> to localise against one of them (or set RTABMAP_LOCALIZE_DB). Refusing to map from scratch by accident."
    exit 1
  else
    echo "    NO published map at $_mapdir. This run maps from scratch — as every run does now."
    echo "    Publishing a map will NOT change that: habitat_launch.py passes --delete_db_on_start."
  fi
fi
# MAPPING_ONLY selected a run that was nothing but the mapping phase, and the mapping phase went
# with the sampling policy (owner 2026-09-11). A scheduled run maps while it drives the roadmap, so
# "map first, detect later" is not a shape this launcher can produce any more. It REFUSES rather
# than starting an ordinary run under a name that promises something else.
if [ "${MAPPING_ONLY:-0}" = "1" ]; then
  echo "!! MAPPING_ONLY=1: the mapping phase is removed with the sampling policy (owner 2026-09-11)."
  echo "   A scheduled run builds the map while it drives the roadmap; there is no separate phase"
  echo "   to run on its own. Drop MAPPING_ONLY, or check out a commit before eae203e."
  exit 1
fi
if [ -n "${FEED_MAPPING_SECONDS:-}" ]; then
  echo "!! FEED_MAPPING_SECONDS=$FEED_MAPPING_SECONDS: retired with the sampling policy it selected."
  echo "   It was never a duration — it chose between the coverage tour and the walk/dwell bursts,"
  echo "   and both are gone. Clear it."
  exit 1
fi
export FEED_OVERLAY="${FEED_OVERLAY:-1}"
export FEED_SHOW="${FEED_SHOW:-1}"

# What the launcher INTENDS the policy to be. The gate compares this against the environment
# actually present inside the container, rather than echoing whatever it finds there — an echo
# is what let an "enforcing" run be a pass-through for weeks.
# The gate compares an INTENTION against what arrived. Which keys matter is the extension's
# declaration (EXT_ENV_PASS), not a list this launcher carries -- it used to name one deployment's
# five policy variables, so a different policy layer was silently unchecked.
PREFLIGHT_EXPECT_POLICY=""
for _v in ${EXT_ENV_PASS:-}; do
  eval "_val=\${$_v-}"
  PREFLIGHT_EXPECT_POLICY="${PREFLIGHT_EXPECT_POLICY:+$PREFLIGHT_EXPECT_POLICY,}$_v=$_val"
done
[ -n "$PREFLIGHT_EXPECT_POLICY" ] && echo "    policy: $PREFLIGHT_EXPECT_POLICY"

# ---- provenance ---------------------------------------------------------------------------
# WHICH CODE produced this bundle. The digests come from preflight_gate.py rather than from a
# `find | xargs cat` here, so the launcher and the gate cannot drift apart — and so a root
# matching no files ABORTS instead of yielding e3b0c442..., the sha256 of nothing, which is a
# plausible sixteen-hex provenance stamp for a hash that covered zero files.
#
# The roots are typed. Only $REPO/lost3dsg is copied into the container at startup, so only it
# has a freeze point; an extension's tree is live on the path for the whole run and is SAMPLED,
# never asserted frozen. knowledge_bridge was a third root until GA-306 vendored the one class
# the extension used; it is no longer read, mounted or sampled.
_tree_sha() {
  local out
  out=$(python3 "$HERE/preflight_gate.py" --print-tree-sha "$1")     || { echo "!! cannot hash $1 — aborting rather than stamping an unrecorded run"; exit 1; }
  echo "$out"
}
read -r SRC_SHA SRC_N   <<<"$(_tree_sha "$REPO/lost3dsg")"
# EXT_TREES is "name=path" pairs the extension asks to be hashed and stamped. This used to hash
# one package by name, so a deployment without it stamped a digest of nothing.
EXT_SRC_SHAS=""
for _pair in ${EXT_TREES:-}; do
  _n="${_pair%%=*}"; _p="${_pair#*=}"
  [ -d "$_p" ] || { echo "!! EXT_TREES names $_n=$_p, which is not a directory"; exit 1; }
  read -r _sha _cnt <<<"$(_tree_sha "$_p")"
  EXT_SRC_SHAS="${EXT_SRC_SHAS:+$EXT_SRC_SHAS,}$_n=$_sha"
  echo "    sources: $_n $_sha ($_cnt)"
done
CFG_SHA=$(sha256sum "$HERE/$CFG_NAME" | cut -c1-16)
# GA-283. The worst frame age the PREVIOUS RUN REJECTED, read HOST-SIDE: preflight_gate.py
# runs INSIDE the container, where /ws/output is the current bundle and previous ones are not
# mounted. Feeds probe a10, which refuses a run whose max_frame_age_s guard sits below an age
# frames were already seen arriving at -- a deadlock by arithmetic that has killed two runs.
#
# NOT the cycle time, which the first version used and which was wrong in principle: total_ms
# sums async work that never gates the loop. Run 20260903_110622 showed a 17.3 s total_ms
# beside a 15 s guard and ZERO rejections, which is only possible if they are different
# quantities. The rejected AGES are what the guard is actually compared against at runtime.
#
# Empty when the last run rejected nothing; a10 then passes and records that it asserted
# nothing.
PREFLIGHT_EXPECT_CYCLE_S=$(python3 "$HERE/last_frame_age_rejected.py" "$RUNS_DIR" 2>/dev/null || echo "")
export PREFLIGHT_EXPECT_CYCLE_S
[ -n "$PREFLIGHT_EXPECT_CYCLE_S" ] && \
  echo "    last run REJECTED a frame at ${PREFLIGHT_EXPECT_CYCLE_S}s (a10 checks max_frame_age_s against it)"

# GA-36. GRAPH_API_CONFIG used to be set only as a one-off prefix on three commands and never
# exported, so the health-monitor subshell below inherited nothing, resource_monitor.build_inventory
# read no config, and every bundle recorded "no endpoint configured" for runs that used a real
# endpoint and model. Exported once here; the prefixes below stay as harmless restatements.
export GRAPH_API_CONFIG="${GRAPH_API_CONFIG:-$HERE/$CFG_NAME}"
MERGED_SHA=$(GRAPH_API_CONFIG="$HERE/$CFG_NAME" python3 "$HERE/preflight_gate.py" --print-merged-sha)   || { echo "!! cannot compute the merged-config sha — aborting rather than passing an empty expectation"; exit 1; }
echo "    sources: graph-api $SRC_SHA ($SRC_N)"
echo "    config:  file $CFG_SHA  merged $MERGED_SHA"

# Handed to the gate, which recomputes them INSIDE the container after the source copy. A
# difference means an edit landed in the window and the run is not the code stamped here.
export PREFLIGHT_EXPECT_SRC_SHA="graph_api=$SRC_SHA${EXT_SRC_SHAS:+,$EXT_SRC_SHAS}"   # GA-373: live roots compared too
export PREFLIGHT_EXPECT_CFG_SHA="$CFG_SHA"
export PREFLIGHT_EXPECT_MERGED_SHA="$MERGED_SHA"

# WHICH ENVIRONMENT answered. The encoders are NOT in the image: live_stack_container.sh
# exports HF_HOME under the extension mount, so weights come from a host cache and two runs
# on one image digest can load different weights. Resolved through refs/main, a floating tag —
# recording them pins going forward and proves nothing about any earlier run.
# DEFAULT IS THE PATCHED IMAGE, graphapi-run:humble-ga290: the rtabmap.cpp:4090 guard (GA-290,
# owner ruling "patch locally", 4 Sep; patch + build provenance in
# lost3dsg/test/patches/rtabmap-0.23.7-ga290-guard.patch). The pristine apt-built
# graphapi-run:humble stays on the machine for comparison; runs must NOT launch on it.
# The docker run line at the bottom now uses "$IMAGE_TAG" — until this change it hardcoded
# graphapi-run:humble, so IMAGE_TAG only ever stamped metadata and an override would have
# launched the pristine image while recording itself as the patched one.
# THE IMAGE. Owner ruling 2026-09-11: the upstream default `hrai/sim:saved` replaces
# graphapi-run:humble-ga290. It is NOT on this machine and is not in any registry reachable
# from here, so it has to be built or loaded before a run works. Checked below rather than
# discovered at `docker run`, and NOT silently replaced by whatever image happens to be
# present: a run on a different image is a different experiment.
IMAGE_TAG=${IMAGE_TAG:-hrai/sim:saved}
# THE IMAGE MUST EXIST BEFORE ANYTHING ELSE IS SPENT. Checked HERE, where IMAGE_TAG is settled,
# rather than at `docker run` several hundred lines later: otherwise the gate, the build and the
# feed host are all paid for first and the failure arrives minutes later as a docker error.
#
# It also keeps run_metadata.json valid. MEASURED 2026-09-11: with the image absent,
# `docker image inspect` printed an EMPTY LINE to stdout and exited non-zero, so the `|| echo`
# below produced "\nunknown" -- a raw newline inside a JSON string. The bundle then failed to
# parse at "line 82 column 22" and the run died writing its own header.
if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  echo "!! the container image '$IMAGE_TAG' is not on this machine, and no run can start without it."
  echo "   Build or load it, or name one that is present:"
  echo "     IMAGE_TAG=<image> ./run_sim_headless.sh ..."
  echo "   Present now:"
  docker images --format '     {{.Repository}}:{{.Tag}}' | grep -iE 'graphapi|sim' || echo "     (no image here looks like a run image)"
  exit 1
fi
# `tr -d` as well as the guard above: a digest is a single token, and anything that puts a newline
# in it corrupts the bundle's header rather than merely reading oddly.
IMAGE_DIGEST=$(docker image inspect -f '{{.Id}}' "$IMAGE_TAG" 2>/dev/null | tr -d '\r\n' || true)
IMAGE_DIGEST=${IMAGE_DIGEST:-unknown}
# GA-437 (2026-09-10). THE STAMP MUST READ THE CACHE THE RUN USES. This was
# $WORKSPACE_ROOT/.hf_cache, which made WORKSPACE_ROOT do double duty as the data root AND the model
# cache; after the workspace moved to a neutral directory it names nothing, and _enc_rev below would
# have stamped every encoder revision as unknown. The container loads its weights from the mount at
# /models/hf, whose host side is HF_SHARED_CACHE, so that is the directory whose refs describe the
# run. Same value the container now sets HF_HOME to (live_stack_container.sh).
HF_CACHE=${HF_CACHE:-${HF_SHARED_CACHE:-/DATA/huggingface_cache}}

# GA-463 (2026-09-10). THE BUILD TREE IS AN EXTERNAL DIRECTORY, NOT A DOCKER-MANAGED VOLUME.
# Owner instruction: "both input dataset and output data is stored in directories external to the
# docker mounted as volumes." `-v graphapi_ws:/ws` put build, install, src and log inside
# /DATA/docker/volumes/graphapi_ws/_data, where nothing lists them and nobody checks them. That is
# also where the sixteen-day-old belief files hid underneath the /ws/output bind mount: a stale
# world model that any launch omitting that mount would have read.
WS_DIR=${WS_DIR:-$WORKSPACE_ROOT/ws}
mkdir -p "$WS_DIR"
echo "    build tree: $WS_DIR (external, was the docker volume graphapi_ws)"
# >>> TEST-EXTRACT _enc_rev  (test_env_stamp.sh sources the block between these markers.
# It guessed the boundary with a sed pattern twice and was wrong twice: /^$/ swallowed the
# call sites below, and /; }$/ ran to end-of-file because this definition is a single line,
# so the range's START line also matched its terminator. The boundary is stated now, not
# inferred. Do not remove the markers.)
_enc_rev() { cat "$HF_CACHE/hub/models--$1/refs/main" 2>/dev/null              || cat "$HF_CACHE/models--$1/refs/main" 2>/dev/null || echo "unknown"; }
# <<< TEST-EXTRACT _enc_rev
ENC_E5=$(_enc_rev intfloat--e5-small-v2)
ENC_MINILM=$(_enc_rev sentence-transformers--all-MiniLM-L6-v2)

# WHICH GROUND TRUTH the numbers will be scored against. The scorer reads GT_SCENE_INSTANCE
# and has no default, so an unset or stale value silently rescopes every recall figure while the
# bundle still looks complete.
GT_PATH=${GT_SCENE_INSTANCE:-}
GT_SHA="unset"; GT_N="unset"
if [ -n "$GT_PATH" ] && [ -f "$GT_PATH" ]; then
  GT_SHA=$(sha256sum "$GT_PATH" | cut -c1-16)
  GT_N=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1])).get('objects') or []))" "$GT_PATH" 2>/dev/null || echo unknown)
  [ "$GT_N" = "0" ] && echo "!! WARNING: ground truth $GT_PATH contains 0 objects — every recall figure will be vacuous"
elif [ -n "$GT_PATH" ]; then
  echo "!! GT_SCENE_INSTANCE=$GT_PATH does not exist — aborting"; exit 1
fi
echo "    ground truth: ${GT_PATH:-<unset>} sha=$GT_SHA objects=$GT_N"
echo "    image: ${IMAGE_DIGEST:0:19}  encoders: ${ENC_E5:0:8} ${ENC_MINILM:0:8}"
# REFUSE rather than default. Every name below is interpolated into the JSON that follows, and
# four of them sit in NUMERIC positions where an empty expansion produces `"enforce": ,` — a
# syntactically broken file rather than a wrong value. That happened: the policy exports were
# 78 lines BELOW this heredoc and every run aborted at the validator.
#
# A default here would be worse than an abort. A bundle stamped with a policy it did not run
# under is the artefact nobody can detect later, and stamping the policy at all exists for the
# ablation case — where a default would silently record neither arm. Same standard as
# live_stack_container.sh refusing a guessed config: name what is missing and stop.
# ORDER IS LOAD-BEARING TWICE OVER. It must run before the feed host starts -- it sat 120 lines
# below the launch once, and that run exported FEED_SCHEDULE into a process that had already
# started, so the feed used no schedule while the launcher printed a cache hit. It must ALSO run
# before run_metadata.json is written, or the bundle records an empty schedule for a run that had
# one. And because it now REFUSES instead of falling back, running it early means a run that
# cannot move is stopped before the bundle directory is populated.
# GA-465 (owner 2026-09-10). THE EXPLORATION SCHEDULE IS CACHED PER SCENE AND BUILT WHEN ABSENT.
# A schedule is one scene's roadmap and the order to walk it: waypoints on the generalized Voronoi
# diagram of the navmesh -- the line equidistant from two or more walls, so it runs down the middle
# of corridors -- visited depth-first from the busiest junction, with a 360 degree scan at each.
#
#   A) CACHED when $SCHEDULE_DIR holds a file for this scene whose recorded settings match this run.
#   B) BUILT when it is missing, when the settings differ, or when habitat.regenerate_schedule is
#      true in config.yaml. THE DIGEST DECIDES, NOT THE FILE NAME: a schedule built at
#      merge_radius 0.75 is not the schedule for 1.5, and reusing it because a file happens to exist
#      would run one geometry while the config describes another.
#
# EXTERNAL, per the owner's storage instruction: $WORKSPACE_ROOT/schedules, beside maps and runs.
# The navmesh sits next to the scene mesh. A scene shipped without one cannot get a schedule, and
# since 2026-09-11 that is a REFUSAL rather than a fallback: there is no second policy to fall to.
SCHEDULE_DIR=${SCHEDULE_DIR:-$WORKSPACE_ROOT/schedules}
# THE SCHEDULE IS MANDATORY (owner 2026-09-11). The sampling policy that used to stand behind
# every "NONE" branch here is removed, so a run without a schedule has NO MOTION AT ALL -- it would
# publish frames from a robot standing still, spend the whole cap, and produce a bundle that looks
# complete. Every branch below therefore either produces a schedule or exits non-zero.
#
# GA-476's three-state FEED_SCHEDULE is gone with the policy it selected. Set-and-empty meant "use
# the sampling policy" and there is no such policy to ask for; it now refuses rather than being
# read as "unset", because a lane that deliberately typed `FEED_SCHEDULE=` asked for something
# specific and must be told it no longer exists.
if [ -n "${FEED_SCHEDULE+x}" ] && [ -z "$FEED_SCHEDULE" ]; then
  echo "!! FEED_SCHEDULE is set and empty. That used to mean \"use the sampling policy\", which is"
  echo "   removed (owner 2026-09-11). Unset FEED_SCHEDULE to build or reuse this scene's schedule,"
  echo "   or point it at a schedule file."
  exit 1
elif [ -z "${FEED_SCHEDULE:-}" ]; then
  _scene_glb="${HABITAT_SCENE:-$DEF_SCENE}"
  _navmesh="${_scene_glb%.glb}.navmesh"
  _regen=$(python3 -c "
import sys, yaml
try:
    c = yaml.safe_load(open(sys.argv[1])) or {}
    print('1' if (c.get('habitat') or {}).get('regenerate_schedule') else '0')
except Exception:
    print('0')" "$HERE/$CFG_NAME" 2>/dev/null || echo 0)
  if [ ! -f "$_navmesh" ]; then
    echo "!! no navmesh at $_navmesh, so this scene's schedule cannot be built."
    echo "   A schedule is the only motion policy, so the run would not move at all. Refusing."
    echo "   Generate the navmesh beside the scene mesh, or name another scene."
    exit 1
  fi
  # THE MULTIPLE-STOP TOUR, from the config so a run can ask for it without a flag. The route
  # re-enters a parent waypoint every time it leaves a branch and used to walk through without
  # turning; this is the angle it turns on re-entry. 0 is the single-stop tour. The value is part
  # of the settings digest, so changing it rebuilds the schedule rather than reusing a stale one.
  _revisit=$(python3 -c "
import sys, yaml
try:
    c = yaml.safe_load(open(sys.argv[1])) or {}
    print(float((c.get('habitat') or {}).get('revisit_scan_deg', 0) or 0))
except Exception:
    print(0.0)" "$HERE/$CFG_NAME" 2>/dev/null || echo 0.0)
  # How far from the waypoint a re-observation stands. Same objects, different parallax.
  _revisit_off=$(python3 -c "
import sys, yaml
try:
    c = yaml.safe_load(open(sys.argv[1])) or {}
    print(float((c.get('habitat') or {}).get('revisit_offset_m', 0) or 0))
except Exception:
    print(0.0)" "$HERE/$CFG_NAME" 2>/dev/null || echo 0.0)
  _sched_out=$("${SCHEDULE_PY:-$HOME/miniconda3/envs/habitat_env/bin/python}" \
    "$HERE/schedule_batch.py" --navmesh "$_navmesh" --scene-id "$SCENE_ARG" \
    --ensure --out-dir "$SCHEDULE_DIR" --revisit-scan-deg "$_revisit" \
    --revisit-offset-m "$_revisit_off" \
    $([ "$_regen" = "1" ] && echo --regenerate) 2>&1) || {
      echo "!! schedule generation FAILED for $SCENE_ARG:"; echo "$_sched_out" | tail -15
      echo "   A schedule is the only motion policy, so there is nothing to fall back to."
      exit 1; }
  echo "$_sched_out" | grep -E "^\[schedule\]|^  y=" | sed 's/^/    /'
  FEED_SCHEDULE=$(echo "$_sched_out" | sed -n 's/^SCHEDULE_FILE=//p' | tail -1)
  export FEED_SCHEDULE
  if [ -z "$FEED_SCHEDULE" ] || [ ! -f "$FEED_SCHEDULE" ]; then
    echo "!! schedule_batch.py reported success but named no readable file for $SCENE_ARG."
    echo "   Printed SCHEDULE_FILE=${FEED_SCHEDULE:-<nothing>}. Refusing to start a run with no motion."
    exit 1
  fi
  echo "    schedule: $FEED_SCHEDULE"
else
  if [ ! -f "$FEED_SCHEDULE" ]; then
    echo "!! FEED_SCHEDULE=$FEED_SCHEDULE does not exist. Refusing to start a run with no motion."
    exit 1
  fi
  echo "    schedule: $FEED_SCHEDULE (given, not generated)"
fi

# NUMERIC positions in the feed block below. An empty expansion yields `"fps": ,` and the
# validator kills the run — which is the correct outcome, but these refuse first and say which
# name is missing.
: "${FEED_SEED:?not set at run_metadata.json — the feed exports must precede this heredoc}"
: "${FEED_FPS:?not set at run_metadata.json}"
: "${FEED_EXPLORATION_LAPS:?not set at run_metadata.json}"
: "${ROOM_FRAME_MAX:?not set at run_metadata.json}"   # GA-350
: "${ROOM_FRAME_STRIDE_M:?not set at run_metadata.json}"
: "${FEED_POSE_SOURCE:?not set at run_metadata.json}"   # GA-359
: "${FEED_SPAWN_FLOOR?not set at run_metadata.json}"   # no colon: empty means "no floor requested"   # no colon: 0 is a legal value
: "${SRC_SHA:?not set at run_metadata.json — the provenance block must precede this heredoc}"
: "${SRC_N:?not set at run_metadata.json}"
# GA-306: KB_SHA/KB_N are NOT asserted. e5294a2 removed the only code that set them, and these
# two assertions then aborted every launch at 5 s, before the container existed. The bundle keeps
# kb_src_sha256_16 / kb_files / kb_root as explicit nulls below; no variable is left to assert.
: "${CFG_SHA:?not set at run_metadata.json}"
: "${MERGED_SHA:?not set at run_metadata.json}"

# Keys are ADDED, never changed: three consumers read this file by key and some do arithmetic
# on the values. The five original keys keep their bytes. "config_name" keeps its meaning too,
# and that meaning is now stated: it is the name the LAUNCHER INTENDED. What the two processes
# actually loaded is recorded separately — the container's in preflight.json (a2 reads
# config.CFG_PATH, the file config.py actually read), the feed host's below.
# Every variable the extension declared must be SET before this heredoc, or the bundle records a
# value the process never received. This used to name one deployment's eight variables.
# GA-474 (owner 2026-09-10). ALL OF THEM, IN ONE MESSAGE. This exited on the FIRST missing name, so
# an extension declaring twelve unset variables cost twelve launches to discover -- measured today:
# the run died on the extension's first declared variable, then its second, then the next.
# Collect, then report. (The names are the extension's; this repository must not carry them.)
_missing_pass=""
for _v in ${EXT_ENV_PASS:-}; do
  eval "_isset=\${$_v+yes}"
  [ -n "${_isset:-}" ] || _missing_pass="$_missing_pass $_v"
done
if [ -n "$_missing_pass" ]; then
  echo "!! EXT_ENV_PASS names $(echo $_missing_pass | wc -w) variable(s) that are NOT SET:"
  for _v in $_missing_pass; do echo "     $_v"; done
  echo "   \`docker run -e VAR\` sends nothing when VAR is unset in the parent, so the container"
  echo "   would use its own default while run_metadata.json records this launcher's. Export them"
  echo "   in \$EXT_ENV_FILE ($EXT_ENV_FILE), or drop them from EXT_ENV_PASS."
  echo "   To run anyway with each of them empty:"
  echo "     $(for _v in $_missing_pass; do printf '%s= ' "$_v"; done)bash $0 $SCENE_ARG"
  exit 1
fi
# "machine" records WHICH MACHINE made this bundle. Absent until 2026-09-10, and its absence is why
# bundles from two machines cannot safely share one directory: nothing inside could tell them apart,
# so a reader comparing them would not know they were comparing different systems. The bundle now
# says so itself, which is stronger than keeping the directories apart and remembering why.
cat <<EOF > "$RUN_DIR/run_metadata.json"
{
  "run_id": "$RUN_ID",
  "scene": "$SCENE_ARG",
  "start_time": "$(date -Iseconds)",
  "machine": "$(hostname)",
  "config_name": "$CFG_NAME",
  "output_dir": "$RUN_DIR",
  "config_name_note": "the name the launcher intended; see config_resolved and preflight.json for what each process loaded",
  "config_resolved": {
    "feed_host_path": "$HERE/$CFG_NAME",
    "file_sha256_16": "$CFG_SHA",
    "merged_sha256_16": "$MERGED_SHA",
    "container_path": "recorded by the gate in preflight.json (a2.loaded_path)"
  },
  "seed": ${FEED_SEED:-7},
  "feed": {
    "motion_policy": "schedule",
    "schedule": "$FEED_SCHEDULE",
    "exploration_laps": $FEED_EXPLORATION_LAPS,
    "navigation_mode": "$FEED_MOVE_FN",
    "policy_note": "THE SAMPLING POLICY IS REMOVED (owner 2026-09-11). Every bundle from today on drives a precomputed Voronoi roadmap for exploration_laps identical laps. Bundles before this date carry walk_frames / dwell_frames / dwell_mode / mapping_seconds instead and measure a DIFFERENT experiment: the sampled agent moved on 1.0-1.8% of its frames with mapping_seconds 0. Never pool the two, and never difference them.",
    "fps": $FEED_FPS,
    "spawn_floor_requested": $([ -n "$FEED_SPAWN_FLOOR" ] && echo "$FEED_SPAWN_FLOOR" || echo null),
    "camera_pitch_deg": $FEED_CAMERA_PITCH_DEG,
    "camera_pitch_note": "negative looks DOWN, applied to rgb, depth and semantic together. 0 is the level camera every run before 2026-09-09 used.",
    "seed_source": "$SEED_SOURCE",
    "scene_source": "$SCENE_SOURCE",
    "draw_note": "drawn = this run chose it among the published per-floor maps (MAP_DRAW=1); pinned = the recipe named it. An A/B arm pins both.",
    "spawn_floor_note": "what was ASKED for. What the run actually mapped is measured from the map's own node poses into rtabmap.db.floor.json. If these two disagree the STAMP is right and this field records the intent that was not met.",
    "overlay": $FEED_OVERLAY,
    "show": $FEED_SHOW,
    "note": "how the agent moved. ABSENT from every bundle before 2026-08-31, so an older bundle's motion cannot be recovered from the bundle and must not be guessed from its date."
  },
  `# The extension's own policy keys are interpolated HERE, inside this object, so the bundle's
   # shape is unchanged and every reader of policy.<key> keeps working. This launcher does not
   # know what they are: EXT_POLICY_JSON is a comma-terminated JSON fragment the extension
   # supplies, or nothing at all when no extension is loaded. Owner ruling 2026-09-09.`
  "policy": {${EXT_POLICY_JSON:-}
    "merge_min_consecutive": ${MERGE_MIN_CONSECUTIVE:-2},
    "pose_source": "$FEED_POSE_SOURCE",
    "rtabmap_pub_loc_pose_only_when_localizing": true,
    "rtabmap_launch": "direct ros2 run rtabmap_slam rtabmap since 2026-09-07 21:40 (GA-359 C): rtabmap.launch.py cannot set pub_loc_pose_only_when_localizing and every earlier bundle's rtabmap.log echoes it false, so /rtabmap/localization_pose was published on every frame, localised or not.",
    "localization_pose_record": "logs/localization_pose.log",
    "localization_pose_record_note": "ros2 topic echo --csv --full-length of /rtabmap/localization_pose for the whole run, no header: stamp sec, nanosec, frame_id, position xyz, orientation xyzw, 36 covariance values. Gaps are the not-localised intervals; the covariance gate's threshold (perception, GA-359 C) is to be read from here.",
    "pose_source_note": "GA-359: simulator = boxes placed through the feed node's identity map->odom (Habitat's true pose as odometry AND localisation); rtabmap = rtabmap's map->odom correction applied. Bundles before 2026-09-07 19:45 ran with BOTH authorities publishing (publish_tf was an undeclared launch argument; publish_tf_map defaulted true): run 152446 had 54 of 747 detection rows 2-4 m off. A bundle with this key set is single-authority.",
    "room_frames": {"max": $ROOM_FRAME_MAX, "stride_m": $ROOM_FRAME_STRIDE_M,
                    "seam": "GA-350: object_manager_6 saves a room view on room entry and per stride_m of travel (at most max per room); the proposal carries room_frames/room_frame to the typer (vendor 4c0e0dc)."}},
  "provenance_intent": {
    "note": "host-side, taken BEFORE docker run. provenance_confirmed in preflight.json is taken after the container copies its sources, and is the authoritative record of what executed.",
    "graph_api_src_sha256_16": "$SRC_SHA", "graph_api_files": $SRC_N,
    "kb_src_sha256_16": null, "kb_files": null,
    "kb_root": null,
    "kb_note": "GA-306, 2026-09-06: the knowledge_bridge dependency is gone -- the e5 ConceptEmbedder it provided is vendored by the extension. Nothing is mounted at /kb and KB_SRC is read nowhere. Explicit nulls, not removed keys: bundles before this date carry real digests here, and a reader joining across them must be able to tell 'not applicable' from 'never stamped'.",
    "frozen_roots": ["graph_api"],
    "live_roots": ["found"],
    "live_root_note": "not copied into the container; on sys.path for the whole run, so sampled rather than asserted frozen"
  },
  "resolved_config": {
    "note": "OWNER RULING 16. The RESOLVED values that were in force, not hashes of the files they came from. A bundle must state its own configuration: config_name and the shas say WHICH files were read, and a reader still had to re-resolve them to learn what they said. GA-156 — per-detection archiving silently off — was found by hand for exactly this reason.",
    "sensor": {"width": $(python3 -c "import sys,yaml,os;c=yaml.safe_load(open(sys.argv[1])) or {};h=(c.get('habitat') or {});print(os.environ.get('FEED_WIDTH') or h.get('width',1280))" "$HERE/$CFG_NAME" 2>/dev/null || echo 1280),
               "height": $(python3 -c "import sys,yaml,os;c=yaml.safe_load(open(sys.argv[1])) or {};h=(c.get('habitat') or {});print(os.environ.get('FEED_HEIGHT') or h.get('height',960))" "$HERE/$CFG_NAME" 2>/dev/null || echo 960),
               "note": "raised from 640x480 by owner ruling 25. A gain measured here is a gain of the SYSTEM: resolution moves detector, segmentation, depth and describer together and cannot be attributed to one without a second arm."},
    "gt_semantic": ${FEED_GT_SEMANTIC:-0},
    "localize_db": $([ -n "${RTABMAP_LOCALIZE_DB:-}" ] && echo "\"$RTABMAP_LOCALIZE_DB\"" || echo null),
    "localization_regime": "fresh_map_per_launch",
    "tour_shape_note": "GA-434 / rule 73. WHICH SHAPE OF HOUSE RUN THIS BUNDLE BELONGS TO. Two exist and they are not comparable: relaunch_per_storey is one launch, one map and one bundle per storey, which is the owner's 2026-09-10 ruling; continuous_teleport is one launch touring every storey, whose map would straddle them and which owner ruling 25 refuses. A bundle set read as the wrong one would double-count objects across storeys or look like it lost them.",
    "tour_shape": "relaunch_per_storey",
    "tour_shape_note": "the only shape now. FEED_TOUR_ALL_FLOORS toured every storey in one continuous session; that machinery lived in the sampling tour and is removed (owner 2026-09-11). The feed host refuses the switch rather than ignoring it.",
    "house_id": $([ -n "${HOUSE_ID:-}" ] && echo "\"$HOUSE_ID\"" || echo null),
    "spawn_floor": $([ -n "${FEED_SPAWN_FLOOR:-}" ] && echo "$FEED_SPAWN_FLOOR" || echo null),
    "localize_db_note": "GA-336: localize_db points at a SCRATCH COPY deleted at exit, so the path alone identifies nothing. localize_db_source + localize_db_sha256_16 name the canonical file this run actually opened.",
    "localize_db_source": $([ -n "${LOCALIZE_DB_SOURCE:-}" ] && echo "\"$LOCALIZE_DB_SOURCE\"" || echo null),
    "localize_db_sha256_16": $([ -n "${LOCALIZE_DB_SHA:-}" ] && echo "\"$LOCALIZE_DB_SHA\"" || echo null),
    "bridge_port": ${BRIDGE_PORT:-null},
    "bridge_port_note": "the port the bridge bound (BRIDGE_PORT); null means BRIDGE_PORT was unset and the bridge used its own default. Asked for by agent2-dashboard 2026-09-06 (their 00015): the dashboard used to have to grep logs/bridge.log for it.",
    "effective_config": $(GRAPH_API_CONFIG="$HERE/$CFG_NAME" python3 -c "
import json, sys
sys.path.insert(0, sys.argv[1])
from config import CFG, CFG_PATH
def g(*path):
    cur = CFG
    for k in path:
        if not isinstance(cur, dict): return None
        cur = cur.get(k)
    return cur
print(json.dumps({
    '_source': 'the MERGED config the code reads (perception_module/config.py CFG), not the yaml '
               'file alone. Reading the yaml gave null for every key it does not set, and the '
               'module default then applied unseen: gvd_method read null while ridge was in '
               'force, and habitat.width read 1280 from my own code default while the merged '
               'config supplied 640 and the sensor stayed at 640x480.',
    '_config_file': CFG_PATH,
    'rooms.gvd_method': g('rooms', 'gvd_method'),
    # GA-204. The engine that decides a merge MUST be stamped. It was set in the config
    # rather than by environment precisely so the bundle would record it -- and this list is
    # hand-maintained, so a new key resolves at runtime and appears in no artefact. Run
    # 20260901_144539 ran with association.merge_engine absent from its own metadata.
    # NO BACKTICKS IN THIS HEREDOC: it is unquoted, so a backtick is command substitution.
    # This comment used them and bash reported 'association.merge_engine: command not found'
    # at the heredoc's opening line -- the trap the docker-block comment already documents.
    'association.merge_engine': g('association', 'merge_engine'),
    # The MERGE_ENGINE env override (passed into the container below) switches the engine
    # without touching the config, so the line above alone could name the other engine
    # (rules 2 and 5). This key records what will actually run. Added, never renamed.
    'association.merge_engine_effective': (__import__('os').environ.get('MERGE_ENGINE') or '').strip().lower() or g('association', 'merge_engine'),
    'association.merge_cost_ratio': g('association', 'merge_cost_ratio'),
    'association.merge_min_consecutive': g('association', 'merge_min_consecutive'),
    'association.merge_ontology_channel': g('association', 'merge_ontology_channel'),
    'association.merge_attribute_max_log_odds': g('association', 'merge_attribute_max_log_odds'),
    'association.association_margin_m': g('association', 'association_margin_m'),
    'similarity.label': g('similarity', 'label'),
    'similarity.color': g('similarity', 'color'),
    'similarity.material': g('similarity', 'material'),
    'similarity.description': g('similarity', 'description'),
    'archive.per_detection': g('archive', 'per_detection'),
    'crop.construction': g('crop', 'construction'),
    'perception.backend': g('perception', 'backend'),
    'perception.cloud_timeout_s': g('perception', 'cloud_timeout_s'),
    'habitat.width': g('habitat', 'width'),
    'habitat.height': g('habitat', 'height'),
    'habitat.min_floor_share': g('habitat', 'min_floor_share'),
    'habitat.floor_confinement': g('habitat', 'floor_confinement'),
    'habitat.floor_tolerance_m': g('habitat', 'floor_tolerance_m'),
    'habitat.single_floor': g('habitat', 'single_floor'),
}))" "$REPO/lost3dsg/src/perception_module" 2>/dev/null || echo '{"error": "config could not be resolved on the host; read config.yaml in the bundle"}')
  },
  "perception_service": {
    "endpoint": "$(python3 -c "import sys,yaml;c=yaml.safe_load(open(sys.argv[1]));print((c.get('perception') or {}).get('modal_endpoint',''))" "$HERE/$CFG_NAME" 2>/dev/null)",
    "identity": null,
    "identity_note": "MEASURED ABSENCE, 31 Aug: the endpoint is a URL, not an identity. A live call with a real 640x480 frame returned timing keys ['detector','sam2','total'] and NO model name, version or image digest. So no bundle can say which service answered, and a service that changed under a fixed URL would leave no trace. This field is where a real identity lands the day the server reports one; null means it reported none, not that nobody looked."
  },
  "environment": {
    "image_tag": "$IMAGE_TAG",
    "image_digest": "$IMAGE_DIGEST",
    "encoders": {"intfloat/e5-small-v2": "$ENC_E5",
                 "sentence-transformers/all-MiniLM-L6-v2": "$ENC_MINILM"},
    "encoder_pin_source": "refs/main (floating) — recorded, not asserted"
  },
  "ground_truth": {
    "path": "${GT_PATH:-unset}",
    "sha256_16": "$GT_SHA",
    "n_objects": "$GT_N",
    "note": "run-time INTENT. analyse_run.py opens this file offline and is the authoritative record of what was actually scored against."
  }
}
EOF
python3 -m json.tool "$RUN_DIR/run_metadata.json" > /dev/null   || { echo "!! run_metadata.json is not valid JSON — aborting rather than shipping an unreadable bundle"; exit 1; }

echo "    run bundle: $RUN_DIR"
echo "    (latest is repointed at the end, after a successful run)"

# The FEED_* names are exported above and stamped into run_metadata.json from those same
# values — one default per name, so the bundle cannot disagree with the process.
#
# NEVER put a comment between two continued lines here. `A=1 \` followed by `# ...` does not
# comment the line: the `#` swallows the continuation and A IS SILENTLY DROPPED. Measured, not
# reasoned about. `bash -n` passes it. I wrote exactly that bug into this spot on 31 Aug and it
# would have thrown away HABITAT_SCENE and HABITAT_DATASET, running the default scene under a

# bundle stamped with the requested one.
HABITAT_SCENE=${HABITAT_SCENE:-$DEF_SCENE} \
HABITAT_DATASET=${HABITAT_DATASET:-$DEF_DATASET} \
DISPLAY="${DISPLAY:-:1}" PYTHONUNBUFFERED=1 \
GRAPH_API_CONFIG="${GRAPH_API_CONFIG:-$HERE/$CFG_NAME}" \
  nohup "$HOME/miniconda3/envs/habitat_env/bin/python" "$HERE/habitat_feed_host.py" \
  > "$RUN_DIR/logs/feed_host.log" 2>&1 &
FEED_PID=$!
# habitat import + scene load can take >2 min on cold caches
for i in $(seq 1 90); do grep -q "listening" "$RUN_DIR/logs/feed_host.log" 2>/dev/null && break; sleep 2; done
grep -q "listening" "$RUN_DIR/logs/feed_host.log" || { echo "feed host failed:"; tail -20 "$RUN_DIR/logs/feed_host.log"; exit 1; }
echo "    feed host up"

# GA-464 (owner 2026-09-10). ONE RVIZ, AND IT IS THE LAUNCH FILE'S. Two were being started and
# neither knew about the other: habitat_launch.py declares use_rviz with default true (:70) and
# live_stack_container.sh passed only use_wall_detector and localization_mode, while this file also
# started the sibling container graphapi_rviz. MEASURED on the running system: rviz2 pid 415260 in
# graphapi_rviz and pid 416147 in graphapi_live, two containers, two windows. The sibling is gone.
#
# THE STACK CONTAINER HAD DISPLAY BUT NO WAY TO USE IT. It was given -e DISPLAY and neither the X
# socket nor a render device, so the surviving rviz was the one that could not draw. Both are added
# to the docker run below. GA-371's opt-out and its record are kept: RVIZ=0 for a headless host, no
# X socket means use_rviz:=false rather than a launch that dies, and run_metadata still carries
# rviz_started and rviz_reason so a bundle says whether anybody was watching.
RVIZ="${RVIZ:-1}"; RVIZ_STARTED=false; RVIZ_REASON=""; RVIZ_DISPLAY="${DISPLAY:-:1}"
if [ "$RVIZ" != "1" ]; then
  RVIZ_REASON="RVIZ=$RVIZ opt-out"
elif [ ! -S "/tmp/.X11-unix/X${RVIZ_DISPLAY#:}" ]; then
  RVIZ_REASON="no X socket for DISPLAY=$RVIZ_DISPLAY"
else
  RVIZ_STARTED=true
  RVIZ_REASON="habitat_launch.py use_rviz:=true (DISPLAY=$RVIZ_DISPLAY, dri=$([ -d /dev/dri ] && echo yes || echo no))"
fi
# Read by live_stack_container.sh and handed to the launch file, so the decision is made once here.
export USE_RVIZ=$([ "$RVIZ_STARTED" = "true" ] && echo true || echo false)
echo "    rviz: started=$RVIZ_STARTED ($RVIZ_REASON)"
python3 - "$RUN_DIR/run_metadata.json" "$RVIZ_STARTED" "$RVIZ_REASON" <<'PY'
import json, sys
p, started, reason = sys.argv[1], sys.argv[2] == "true", sys.argv[3]
d = json.load(open(p)); d["rviz_started"] = started; d["rviz_reason"] = reason   # GA-371, keys ADDED (rule 6)
json.dump(d, open(p, "w"), indent=2)
PY

# Asynchronous health & memory monitor
(
  while true; do
    echo "=== $(date) ===" >> "$RUN_DIR/logs/system_health.log"
    free -m >> "$RUN_DIR/logs/system_health.log"
    docker stats --no-stream graphapi_live >> "$RUN_DIR/logs/system_health.log" 2>/dev/null || true
    # per-process VRAM/CPU + per-model location inventory (JSON snapshot)
    python3 "$HERE/resource_monitor.py" --out "$OUT_DIR/model_resources.json" 2>/dev/null || true
    sleep 30
  done
) &
MON_PID=$!

echo ">>> ROS stack in container (web viewer -> http://localhost:${BRIDGE_PORT:-8081})"
# OWNER RULING 22, 2026-09-01: fifteen settings looked adjustable from the host and reached
# nothing. Every -e below is a variable some container-side module actually reads; docker passes
# an `-e NAME` only when NAME is set in the environment, so listing one costs nothing while it is
# unused and makes it work the first time somebody needs it.
#
# BRIDGE_PORT is why this matters: it was the only mitigation available for the 8081 port conflict
# and setting it host-side did nothing at all, silently.
#
# check_env_passthrough.py holds this list to the code: it ast-parses every container-side module
# for os.environ reads and fails if one is neither listed here, set inside the container, nor
# declared host-only. Do NOT put comments between the continued lines below — a comment after a
# `\` swallows the continuation, and a backtick-comment terminates an assignment prefix. Both were
# measured on 2026-08-31; both pass `bash -n`.
# FREEZE THE CONTAINER SCRIPT INTO THE RUN DIRECTORY, AND RUN THAT COPY.
#
# The container used to be started as `/graph_api/lost3dsg/test/live_stack_container.sh` -- the
# READ-ONLY MOUNT OF THE LIVE HOST TREE. Bash reads a script incrementally, by byte offset, so a
# host-side edit while the container is running moves the ground under the interpreter and it
# resumes at a shifted position.
#
# MEASURED 2026-09-11, and it cost a run: `live_stack_container.sh: line 691: syntax error near
# unexpected token 'then'` at 29 of 65 waypoints, while the host copy parsed cleanly and line 691
# held a `case`. Another lane wrote the file at 15:16 during a run that started at 15:04.
#
# a7 DID NOT AND COULD NOT CATCH IT. a7 freezes the tree copied into /ws/install; the one script
# that drives the whole container was executed from the mount, outside that freeze, and a7 passed
# at teardown while this happened.
#
# The copy lands in the bundle, so it is also provenance: the script that ran is the script the
# bundle holds, rather than whatever the host tree says afterwards.
cp "$HERE/live_stack_container.sh" "$RUN_DIR/live_stack_container.sh"
rm -f "$RUN_DIR/NOT_STARTED"   # GA-381: past every check; from here the directory is a real attempt
# FRAME_QUEUE_MAX, SCAN_COMPLETE_TOPIC, SCAN_MERGE_SETTLE_S and MOTION_POSITION_THRESHOLD
# are config-backed knobs whose
# ENVIRONMENT OVERRIDE needs this passthrough. Each has a home in the config, so the knob
# works without the -e line; without it the documented per-run override is a lever that looks
# connected and is not. check_env_passthrough.py found all three by reading them inside the
# container and not finding them on this list.
#
# RE-STAMP THE SOURCE IMMEDIATELY BEFORE THE CONTAINER STARTS.
#
# The provenance stamp above is taken ~490 lines earlier, and between the two sit the image build
# and the feed-host startup -- MINUTES. Any change under lost3dsg in that window made a7 compare
# the launcher's stamp against the container's hash of the same mount at a later moment, and fail.
#
# MEASURED 2026-09-11, three runs lost to it. On the third: launcher 8b5fc4925fef358c, container
# 0885d78908d97a83, and the host tree hashed 0885d78908d97a83 RIGHT THEN -- so the container and
# the live tree agreed and the LAUNCHER'S STAMP was the stale one. `find -newermt` showed nothing
# changed, which is consistent with a DELETION: a removed file leaves no mtime behind.
#
# THE HONEST STAMP IS THE ONE TAKEN WHEN THE CONTAINER COPIES THE TREE, not minutes before, so the
# stamp is refreshed here and the bundle is corrected to match. This is not a relaxation of a7:
# a7 still compares the launcher against the container, and the window it polices is now seconds
# instead of minutes. A change during THAT window still fails, and should.
read -r _SRC_SHA_NOW _SRC_N_NOW <<<"$(_tree_sha "$REPO/lost3dsg")"
if [ "$_SRC_SHA_NOW" != "$SRC_SHA" ]; then
  echo "    source changed during startup: $SRC_SHA ($SRC_N) -> $_SRC_SHA_NOW ($_SRC_N_NOW)"
  echo "    re-stamping: the bundle records the tree the container is about to copy."
  SRC_SHA="$_SRC_SHA_NOW"; SRC_N="$_SRC_N_NOW"
  export PREFLIGHT_EXPECT_SRC_SHA="graph_api=$SRC_SHA${EXT_SRC_SHAS:+,$EXT_SRC_SHAS}"
  python3 - "$RUN_DIR/run_metadata.json" "$SRC_SHA" "$SRC_N" <<'PYSTAMP' ||     { echo "!! could not correct the provenance in run_metadata.json — aborting rather than";       echo "   launching a run whose bundle names a tree it did not execute"; exit 1; }
import json, sys
path, sha, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = json.load(open(path))
d["graph_api_src_sha256_16"] = sha
d["graph_api_files"] = n
d.setdefault("provenance_notes", []).append(
    "graph_api_src re-stamped immediately before the container started; the earlier stamp was "
    "taken before the image build and the tree changed in between")
json.dump(d, open(path, "w"), indent=2)
PYSTAMP
fi

# NO COMMENT LINES INSIDE THIS COMMAND. Every line below is joined by a trailing backslash, so
# a "#" line here is not a comment -- docker receives "#" and each following word as ARGUMENTS.
# `bash -n` accepts it, because it is valid syntax; only the run fails. Done once, 2026-09-11.
docker run --name graphapi_live --rm --entrypoint bash --gpus all --network=host \
  -e OPENAI_API_KEY -e CFG_NAME -e MODAL_PERCEPTION_URL -e MERGE_ENGINE -e PERCEPTION_DEBUG \
  -e FRAME_QUEUE_MAX -e SCAN_COMPLETE_TOPIC -e SCAN_MERGE_SETTLE_S -e MOTION_POSITION_THRESHOLD \
  -e MERGE_MIN_CONSECUTIVE \
  -e RUN_START_EPOCH -e PREFLIGHT_EXPECT_POLICY -e PREFLIGHT_SKIP \
  -e RTABMAP_LOCALIZE_DB -e RTABMAP_CLOSE_TIMEOUT \
  -e FEED_HF_OFFLINE -e PREFLIGHT_HF_CACHE \
  -e FEED_SCHEDULE -e FEED_EXPLORATION_LAPS -e FEED_MOVE_FN -e FEED_POST_SCAN_HOOK \
  -e FEED_SPAWN_FLOOR -e WALL_DETECTOR -e BRIDGE_SERVICE_TIMEOUT \
  -e ROOM_FRAME_MAX -e ROOM_FRAME_STRIDE_M -e FEED_POSE_SOURCE \
 -e GRAPH_API_SRC -e GRAPH_API_TEST_SRC \
  -e KG_BRIDGE_SRC \
  -e BRIDGE_PORT -e BRIDGE_RAW_MAX_AGE -e BRIDGE_ANNOTATED_MAX_AGE -e BRIDGE_FEED_PROBE_BACKOFF \
  -e FEED_HOST -e FEED_PORT -e FEED_CTRL_HOST -e FEED_CTRL_PORT -e LOST3DSG_OUTPUT_DIR \
  -e GRAPH_API_AUTOSTART -e GRAPH_API_BASE_URL -e GRAPH_API_TIMEOUT \
  -e ROOM_VLM_MODEL -e OPENROUTER_API_KEY -e REGOLO_API_KEY \
  -e HABITAT_EXAMPLE_OBJECTS_DIR -e DISPLAY -e USE_RVIZ -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  $([ -d /dev/dri ] && echo "--device /dev/dri") \
  -e PREFLIGHT_EXPECT_CFG_SHA -e PREFLIGHT_EXPECT_MERGED_SHA -e PREFLIGHT_EXPECT_SRC_SHA \
  -e PREFLIGHT_EXPECT_CYCLE_S \
  -e ARCHIVE_DEPTH -e FEED_HFOV -e BRIDGE_OVERLAY -e BRIDGE_OVERLAY_CAM_FRAME -e BRIDGE_OVERLAY_MAP_FRAME \
  -e BRIDGE_OVERLAY_FAR -e BRIDGE_OVERLAY_MAX \
 -e OPENAI_BASE_URL \
  $EXT_E_ARGS \
  -v "$REPO":/graph_api:ro \
  -v "$WS_DIR":/ws \
  ${EXT_MOUNTS:-} \
  `# GA-295. THE MAP LIBRARY IS READ-ONLY, AND UNTIL NOW ONLY THE COMMENT SAID SO.
   # live_stack_container.sh has claimed since GA-158 that "the map is mounted read-only, not
   # copied", and printed a warning every run that it was writable. It was: the mount carried no :ro,
   # and in localization mode rtabmap is handed the canonical map AS ITS OWN database_path, so its
   # close path writes to it. The canonical hm3d_00861/rtabmap.db is 24 MB (5,870 pages) larger
   # than the 1,197,514,752 its provenance recorded on 31 Aug, and was last modified 2026-09-03
   # 22:51:58, during a run. Node, Data and integrity still match (1096/1096/ok), so this is not a
   # claim that the geometry changed -- it is a claim that a published artefact is not immutable.
   # The deeper mount wins, so runs still read the library and can no longer write it.` \
  -v "$WORKSPACE_ROOT/maps":"$EXT_MOUNT_POINT"/maps:ro \
  -v "$RUN_DIR":/ws/output \
  -v "$RUN_DIR/ros":/root/.ros \
  -v "${SAM_MODEL_DIR:-/DATA/models/efficientvit_sam}":/models/vitsam:ro \
  -v "${HF_SHARED_CACHE:-/DATA/huggingface_cache}":/models/hf \
  -v "$OUT_DIR":/out \
  "$IMAGE_TAG" /ws/output/live_stack_container.sh

# Post-run archive
# NOTHING TO COPY: $OUT_DIR IS $RUN_DIR since 2026-09-10 (one directory per run). The host-side
# logs are written into $RUN_DIR/logs/ directly, so there is no scratch directory to drain and no
# way for a run to archive its predecessor's files under its own source hashes.

# `latest` means THE LAST BUNDLE WORTH READING, so it moves here and only on a passing gate.
#
# It has claimed an aborted bundle three times. It was first moved at bundle-creation time; the
# fix conditioned it on run_metadata.json being valid JSON, and two aborted runs then claimed it
# anyway — because valid metadata says the launcher wrote a coherent header, and says NOTHING
# about whether a run happened. Both of those runs wrote a good header and then failed.
#
# The gate's verdict is the only thing that distinguishes them, and it is written by the process
# that actually looked. A bundle with no preflight.json never got that far; one with
# verdict "fail" or "skipped" was refused or was never checked. None of those is worth reading,
# and `latest` is what a tool follows when nobody told it which bundle to open.
_verdict=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('verdict','absent'))"              "$RUN_DIR/preflight.json" 2>/dev/null || echo absent)
  # THE CHILD MUST EXIT HERE. Without this line the engine falls off its own end, reaches the `fi`
  # and RUNS THE PARENT SECTION BELOW -- so every storey that finished started a WHOLE NEW HOUSE
  # RUN, each of which started a child that did it again. MEASURED 2026-09-11: one launch became
  # house_20260911_133641, _140421 and _143114, seven nested run_sim.sh shells deep, and it would not
  # have stopped on its own. The engine's last statement is an assignment, not an exit, so there
  # is nothing else to stop the fall-through.
  # The engine's own status, not a forced 0: the parent reads it to decide whether the storey
  # failed, and it stops the house on a failure rather than touring the next storey into the
  # same fault. Under `set -e` a non-zero would already have left the shell, so this is 0 in
  # practice -- but it stays faithful if a tolerated-failure command is ever added above.
  exit $?
fi   # end of the one-storey engine

# ======================================================================================
# THE PARENT. Arguments, the config, the storeys, then one child run per storey.
# ======================================================================================
set -euo pipefail
export LC_ALL=C     # the same reason as the engine: floor_%+.2f under it_IT.UTF-8 is a hard error
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ONE_STOREY=0
SCENE_ARG=""
SCHEDULE=""
CONFIG=""
_next_is_schedule=0
_next_is_config=0
for a in "$@"; do
  if [ "$_next_is_schedule" = "1" ]; then SCHEDULE="$a"; _next_is_schedule=0; continue; fi
  if [ "$_next_is_config" = "1" ]; then CONFIG="$a"; _next_is_config=0; continue; fi
  case "$a" in
    --schedule)   _next_is_schedule=1 ;;
    --config)     _next_is_config=1 ;;
    --one-storey) ONE_STOREY=1 ;;
    -h|--help)    sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)           echo "!! unknown option: $a. This script takes a scene, --one-storey," >&2
                  echo "   --config <file> and --schedule <file>." >&2
                  echo "   Every other setting belongs in the config file." >&2; exit 2 ;;
    *)            SCENE_ARG="$a" ;;
  esac
done

# --schedule WITHOUT A FILE MUST REFUSE, NOT FALL THROUGH. Measured 2026-09-10: `./run_sim.sh
# --schedule` with the filename forgotten left SCHEDULE empty and STARTED A FULL HOUSE RUN.
# An option that silently becomes a different command is worse than an unknown option.
if [ "$_next_is_schedule" = "1" ]; then
  echo "!! --schedule needs a file: ./run_sim.sh --schedule schedules/<name>.runs.yaml" >&2
  exit 2
fi
if [ "$_next_is_config" = "1" ]; then
  echo "!! --config needs a file: ./run_sim.sh --config schedules/configs/<name>.yaml" >&2
  exit 2
fi

# THE CONFIG, AND IT IS THE ONLY PLACE ONE IS CHOSEN (owner 2026-09-11). Both variables are set,
# because they are read in different places and disagreeing is how a bundle comes to name one file
# while loading another: GRAPH_API_CONFIG is what config.py opens, CFG_NAME is what the run echoes,
# stamps and hands to the container.
#
# THE DEFAULT IS THE TRACKED CONFIG. It used to be decided down in the engine -- regolo_config.yaml
# when an API key was present, smoke_config.yaml otherwise -- and regolo_config.yaml is UNTRACKED,
# so on a fresh clone that fallback named a file which does not exist.
CONFIG="${CONFIG:-${GRAPH_API_CONFIG:-$HERE/lost3dsg/src/perception_module/config.yaml}}"
[ -f "$CONFIG" ] || { echo "!! no such config file: $CONFIG" >&2; exit 2; }
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
export GRAPH_API_CONFIG="$CONFIG"
# CFG_NAME IS RESOLVED RELATIVE TO lost3dsg/test/, NOT A BARE FILENAME. The engine checks that
# "$HERE/$CFG_NAME" exists with HERE=lost3dsg/test, so a basename sent it looking in that directory
# and it aborted with "config.yaml does not exist" for a config sitting in src/perception_module.
# Measured on Gin 2026-09-10. A path relative to that directory resolves for the engine AND inside
# the container, which mounts the same layout at /graph_api.
export CFG_NAME="$(realpath --relative-to="$HERE/lost3dsg/test" "$CONFIG")"
echo "config: $CONFIG"

if [ -n "$SCHEDULE" ] && [ ! -f "$SCHEDULE" ]; then
  echo "!! no such schedule file: $SCHEDULE" >&2
  exit 2
fi

# A SCHEDULE OF RUNS, each with its own configuration. Owner instruction 2026-09-10. The driver
# writes one real config file per arm, passes it as GRAPH_API_CONFIG, runs this script once per
# arm, and records which arm produced which bundle. It is a separate file because it needs yaml
# and a manifest, and because "read config.yaml ALWAYS" means an arm must be a FILE rather than a
# pile of variables at launch time.
if [ -n "$SCHEDULE" ]; then
  exec python3 "$HERE/lost3dsg/test/schedule_runs.py" "$SCHEDULE" --runner "$0"
fi

# RULE 73 FORBIDS A CAP. A cap truncates a storey mid-tour and leaves a bundle that LOOKS
# finished, which is the one failure a reader cannot see. Refuse rather than unset it: the caller
# meant something by it.
if [ -n "${CAP_MIN:-}" ]; then
  echo "!! CAP_MIN=$CAP_MIN is set, and a base run has no cap (rule 73)." >&2
  echo "   Each storey ends when its tour completes. Clear CAP_MIN." >&2
  exit 2
fi
# FEED_TOUR_ALL_FLOORS asked for the continuous teleporting tour, which is removed with the
# sampling policy (owner 2026-09-11). The feed host refuses it too; this catches it before N
# children start.
if [ "${FEED_TOUR_ALL_FLOORS:-0}" != "0" ]; then
  echo "!! FEED_TOUR_ALL_FLOORS=$FEED_TOUR_ALL_FLOORS: the continuous teleporting tour is removed" >&2
  echo "   with the sampling policy (owner 2026-09-11). This script IS how a house is toured now:" >&2
  echo "   one run per storey, one map per storey (ruling 25). Clear the variable." >&2
  exit 2
fi

# THE SETUP IS CHECKED HERE, WHERE THE PERSON IS, rather than failing three layers down with a
# message about a mount. install.sh writes config.local.yaml; without it a modal-backend run has
# no endpoint, and the failure would otherwise arrive as a container error.
if [ ! -f "$HERE/lost3dsg/test/env.local.sh" ] && [ ! -f "$HERE/config.local.yaml" ]; then
  echo "!! Not installed yet: no config.local.yaml (or lost3dsg/test/env.local.sh)." >&2
  echo "   Run ./install.sh once, then fill in the values it names." >&2
  exit 2
fi

# ---- THE STOREYS, RESOLVED BEFORE ANY RUN STARTS -------------------------------------
# Owner 2026-09-11: "The storey schedule should come before as a multi-storey run is actually
# multiple runs."
#
# THE PUBLISHED MAPS NAME THE STOREYS, so there is no discovery launch. There used to be one: the
# first launch ran unpinned and the storeys were read afterwards from its own bev_data.json. That
# contradicted the engine, which REFUSES an unpinned spawn whenever the scene has per-floor maps
# and no whole-scene map -- it will not map from scratch by accident. MEASURED 2026-09-11:
# house_20260911_131347 died on exactly that, exit 1, before a single frame.
SCENE_NAME="${SCENE_ARG:-${SCENE:-hm3d_00861}}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$HERE}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE_ROOT/results}"
FLOORS="${HOUSE_FLOORS:-}"
_floor_source="HOUSE_FLOORS"
if [ -z "$FLOORS" ]; then
  _mapdir="$WORKSPACE_ROOT/maps/$SCENE_NAME"
  FLOORS="$(ls -d "$_mapdir"/floor_* 2>/dev/null | sed -E 's#.*/floor_##' | sort -g | tr '\n' ' ')"
  _floor_source="the published maps in $_mapdir"
fi
FLOORS="$(echo $FLOORS)"      # collapse the trailing space so ${FLOORS%% *} is exact

# --one-storey: ONE run, and it still has to be pinned when the maps are per-floor.
if [ "$ONE_STOREY" = "1" ]; then
  FLOORS="${FEED_SPAWN_FLOOR:-${FLOORS%% *}}"
  _floor_source="--one-storey (${_floor_source})"
fi

HOUSE_ID="house_$(date +%Y%m%d_%H%M%S)"
HOUSE_DIR="$RESULTS_DIR/$HOUSE_ID"
mkdir -p "$HOUSE_DIR"
echo ">>> RUN $HOUSE_ID — scene $SCENE_NAME"
if [ -n "$FLOORS" ]; then
  echo "    storeys: $FLOORS   (from $_floor_source)"
else
  # A scene with no published maps has nothing to pin, and the engine only refuses an unpinned
  # spawn when per-floor maps EXIST. One unpinned run is the right answer here, not a refusal.
  echo "    storeys: <none published> — one unpinned run, which will map from scratch"
fi
echo "    manifest: $HOUSE_DIR/manifest.json"

storeys_done=()
bundles=()
statuses=()

run_storey() {   # $1 = floor, or "" for an unpinned spawn
  local floor="$1" stamp bundle rc=0
  stamp="$(date +%Y%m%d_%H%M%S)"
  # The stamp is the bundle's name and the engine REFUSES a name already taken, so two storeys
  # starting inside the same second cannot land in one bundle.
  while [ -e "$RESULTS_DIR/${stamp}_${SCENE_NAME}" ]; do sleep 1; stamp="$(date +%Y%m%d_%H%M%S)"; done
  echo ""
  echo ">>> STOREY ${floor:-<unpinned>} — running (bundle stamp $stamp)"
  if [ -n "$floor" ]; then
    GRAPH_API_STOREY_CHILD=1 RUN_TIMESTAMP="$stamp" HOUSE_ID="$HOUSE_ID" FEED_SPAWN_FLOOR="$floor" \
      bash "$0" ${SCENE_ARG:+"$SCENE_ARG"} || rc=$?
  else
    GRAPH_API_STOREY_CHILD=1 RUN_TIMESTAMP="$stamp" HOUSE_ID="$HOUSE_ID" \
      bash "$0" ${SCENE_ARG:+"$SCENE_ARG"} || rc=$?
  fi
  bundle="$(ls -d "$RESULTS_DIR/${stamp}_"* 2>/dev/null | head -1 || true)"
  storeys_done+=("${floor:-unpinned}")
  bundles+=("${bundle:-none}")
  statuses+=("$rc")
  if [ "$rc" -ne 0 ]; then
    echo "!! STOREY ${floor:-<unpinned>} FAILED (exit $rc). Stopping here."
    echo "   A storey usually fails for a reason the next storey would hit too, and four identical"
    echo "   failures cost four runs to learn once. The manifest records what did run."
  fi
  return "$rc"
}

house_rc=0
if [ -z "$FLOORS" ]; then
  run_storey "" || house_rc=$?
else
  for z in $FLOORS; do
    run_storey "$z" || { house_rc=$?; break; }
  done
fi

python3 - "$HOUSE_DIR/manifest.json" "$HOUSE_ID" "$house_rc" "$SCENE_NAME" "$CONFIG" \
         "${storeys_done[@]}" -- "${bundles[@]}" -- "${statuses[@]}" <<'PY'
import json, sys
path, house_id, rc, scene, config = sys.argv[1:6]
rest = sys.argv[6:]
a = rest.index("--"); b = rest.index("--", a + 1)
storeys, bundles, statuses = rest[:a], rest[a+1:b], rest[b+1:]
json.dump({
    "house_id": house_id,
    "scene": scene,
    "config": config,
    "policy": "rule 73: one run, one map, one bundle, per storey",
    "exit_status": int(rc),
    "complete": int(rc) == 0,
    "storeys": [{"floor": s, "bundle": bu, "exit_status": int(st)}
                for s, bu, st in zip(storeys, bundles, statuses)],
    "note": "N bundles, not one. Anything that spans the house -- total coverage, an object seen "
            "on two storeys, the duplicate rate -- is a post-hoc join across these bundles. Each "
            "bundle's map has its own SLAM origin and they are NOT in a common frame.",
}, open(path, "w"), indent=2)
print(f"wrote {path}")
PY

echo ""
echo ">>> RUN $HOUSE_ID ${storeys_done[*]} — $( [ "$house_rc" -eq 0 ] && echo COMPLETE || echo "INCOMPLETE (exit $house_rc)")"
echo "    manifest: $HOUSE_DIR/manifest.json"
exit "$house_rc"
