#!/usr/bin/env bash
# Live demo on this machine: habitat renders on the host (conda habitat_env),
# the ROS 2 stack runs in the graphapi-run:humble-ga290 container (patched rtabmap, GA-290)
# over a TCP feed.
# Watch: web viewer at http://localhost:${BRIDGE_PORT:-8081} and snapshots in $OUT_DIR.
#   ./live_run.sh [scene]  # foreground; ctrl-C stops everything
# scene: hm3d_00861 (default) | hm3d_00337 | hm3d_00770 | mp3d_17DRP
# HABITAT_SCENE/HABITAT_DATASET env vars still override everything.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

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
WORKSPACE_ROOT=${WORKSPACE_ROOT:-$(cd "$REPO/../.." && pwd)}
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
  docker rm -f graphapi_rviz >/dev/null 2>&1 || true   # GA-371: the RViz sibling started below
  [ -n "$FEED_PID" ] && kill -9 "$FEED_PID" 2>/dev/null || true
  # Archive the HOST-written artefacts here rather than only after the container exits. The
  # per-frame viewpoint series and the feed host's own log are written on this side, and the
  # post-run copy below never runs when the script is stopped with Ctrl-C — which is the
  # documented way to stop it. feed_host.log reached 1 of 21 shipped bundles for this reason.
  if [ -n "${RUN_DIR:-}" ] && [ -d "$RUN_DIR" ]; then
    cp "$OUT_DIR"/*.json "$OUT_DIR"/*.jsonl "$RUN_DIR/" 2>/dev/null || true
    cp "$OUT_DIR"/*.log "$RUN_DIR/logs/" 2>/dev/null || true
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
    # $OUT_DIR is a DIFFERENT mount (/out). Copying from there would always find nothing and
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
#   tour_complete  the feed wrote feed_ended.json and the container closed on it (rule 73's normal end)
#   mapping_time   a mapping run reached its own deadline, which is also a normal end
#   node_death     a watched node exited, whatever its status -- 0 included, which is why the NODE
#                  and not the status is what discriminates
#   unrecorded     no terminating_node.json: killed from outside, or dead before the watch loop.
#                  NOT "unknown": it says the container never reached its own end.
_tn = d["terminating_node"].get("node")
d["terminating_node"]["ended"] = ({"FEED_ENDED": "tour_complete",
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
if [ -n "${OPENAI_API_KEY:-}" ]; then
  CFG_NAME=${CFG_NAME:-regolo_config.yaml}
else
  CFG_NAME=${CFG_NAME:-smoke_config.yaml}
fi
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
export OUT_DIR=${OUT_DIR:-$WORKSPACE_ROOT/results/${RUN_TIMESTAMP}_${SCENE_ARG}}
mkdir -p "$OUT_DIR"
echo "    live output: $OUT_DIR"
# Stamped BEFORE the bundle directory is created, so every artefact the run legitimately
# writes is newer than it. The gate's a5 probe fails on anything older — a scratch directory
# left dirty by the previous run, or a file copied in by hand.
RUN_START_EPOCH=$(date +%s)
export RUN_START_EPOCH

# The scratch directory is never cleared between runs, and its *.json and *.jsonl are copied
# into the bundle below. A run that dies early therefore archives its PREDECESSOR's files
# under its own source hashes. Rename rather than delete: if the previous run's archiving
# failed, this is the only copy of its artefacts, and an inode costs nothing against that.
if [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
  mv "$OUT_DIR" "${OUT_DIR}.prev-${RUN_START_EPOCH}"
  mkdir -p "$OUT_DIR"
  echo "    scratch: previous contents moved to ${OUT_DIR}.prev-${RUN_START_EPOCH}"
fi

RUN_ID="${RUN_TIMESTAMP}_${SCENE_ARG}"
RUNS_DIR=${RUNS_DIR:-$WORKSPACE_ROOT/runs}
RUN_DIR="$RUNS_DIR/$RUN_ID"
# A bundle directory that already holds a run is never reused: two runs in one directory produce a
# bundle whose files come from both and whose metadata describes one.
if [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
  echo "!! $RUN_DIR already exists and is not empty. RUN_TIMESTAMP=$RUN_TIMESTAMP is already taken."
  exit 1
fi
# GA-258b. EXPORTED, because the FEED HOST needs it. The host process reads
# merge_pending.json to decide how long to dwell, and that file is written by the container
# into /ws/output -- which is bind-mounted to $RUN_DIR, not to $OUT_DIR (/out). The feed host
# was building the path from GRAPH_API_OUTPUT_DIR, which only exists INSIDE the container, so
# on the host it resolved to a bare relative filename and never opened. Measured on run
# 20260902_125130: every dwell line read "? merges pending (sweep None)" and every waypoint
# ran to the 90-frame cap.
export RUN_DIR
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/crops" "$RUN_DIR/snapshots"
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
if [ -z "${EXT_ENV_FILE:-}" ] && grep -qE '^\s*filter:\s*"[^"]+"' "$HERE/$CFG_NAME" 2>/dev/null; then
  echo "!! $CFG_NAME wires an extension filter but EXT_ENV_FILE is unset."
  echo "   The extension's variables would never reach the container and the run would die"
  echo "   at the first proposal that needs one. Set EXT_ENV_FILE, or clear hooks.filter."
  exit 1
fi

# CAMERA PITCH, in degrees, negative looks DOWN. It reaches the feed host and the bundle: a
# setting that changes what the camera SEES and is not recorded is the shape that made six days
# of tour runs unreadable.
export FEED_CAMERA_PITCH_DEG="${FEED_CAMERA_PITCH_DEG:-0}"
export ROOM_FRAME_MAX="${ROOM_FRAME_MAX:-5}"
export ROOM_FRAME_STRIDE_M="${ROOM_FRAME_STRIDE_M:-1.5}"
# GA-359 (owner 2026-09-07 ~18:20 "switch to rtabmap localised poses"; design plan/14). The pose
# source the boxes are placed in. simulator = the feed node's static identity map->odom is the only
# authority and rtabmap does not publish TF (today's behaviour, made explicit); rtabmap = rtabmap
# publishes map->odom and the feed node must not (perception's half, not landed yet: do NOT set
# rtabmap before it lands, or two authorities publish again). Read by live_stack_container.sh and
# by habitat_feed_node.py (once perception lands its half); stamped as pose_source.
export FEED_POSE_SOURCE="${FEED_POSE_SOURCE:-simulator}"
case "$FEED_POSE_SOURCE" in simulator|rtabmap) ;; *) echo "!! FEED_POSE_SOURCE=$FEED_POSE_SOURCE is neither simulator nor rtabmap"; exit 1 ;; esac

# Feed geometry. These were interpolated ONLY into the launch line 160 lines below and appeared
# NOWHERE in the bundle — a run recorded its seed and nothing else about how the agent moved.
#
# FEED_DWELL DEFAULT IS 0 AS OF 2026-08-31, by the owner, relayed by the orchestrator in the
# owner's words: "I think we should try removing dwell completely and try making it work from
# there." The ruling stands on its own; the two numbers that were quoted alongside it here do
# not, and both were struck the same afternoon by the lanes that checked them.
#
# STRUCK, and NOT to be restored to this comment:
#
#   "dwell held the agent still 20 s of every 22 s in run 19" — NOT MEASURED FOR RUN 19. That
#   bundle has no feed block, no feed_stats.json, and a feed_host.log that prints phase NAMES
#   and never the numbers. It is an estimate from the then-default of 60, and its own config.yaml
#   cannot confirm it (see below). Experiment lane, from the artefacts.
#
#   "0/5 merges under 0.925 versus 87% under the old gate" — the COUNT is right and the
#   INFERENCE is not. Run 19 compared 20 pairs, refused 19 on similarity, and the highest
#   similarity observed across all of them was 0.797. The 0.85-0.925 band is EMPTY, so the old
#   threshold would have refused every one of these pairs too. That run measures the scene and
#   the detections, not the knob. Testing lane, recovered from logs/om6.log — object_manager_6's
#   stdout, symlinked into the bundle at live_stack_container.sh:57. NOT perception.log, which I
#   cited first and which holds zero of those lines: om6.log 20, perception.log 0.
#
#   CONFIRMED AGAIN on run A (20260831_174209) with a second, larger sample: 137 pairs reaching a
#   decision, similarity-refused ceiling 0.827. The 0.85-0.925 gap is empty there too, so the
#   0.85 -> 0.925 change would have refused ZERO additional pairs on either run. Two independent
#   samples now say the knob is inert on this scene.
#
# A BUNDLE'S config.yaml IS NOT EVIDENCE OF ITS FEED SETTINGS. Every bundle carries a copy, so
# it looks like dwell was always recorded. The environment wins over it at habitat_feed_host.py
# :505, and the copy is of the file, not of what took effect. Measured: bundle 20260831_033330
# has config.yaml mapping_seconds 0.0 while its feed_stats.json says 150.0 and its log announces
# a 150 s mapping phase. Where a pre-stamp bundle has feed_stats.json, phase_dwell_frames is
# written AFTER the override and is the recoverable value (20260826_112549 and run 13: both 60).
# Where it does not, the setting is simply absent and must not be inferred from the run date.
#
# dwell=0 STARTS A NEW BUNDLE FAMILY: runs before this line are not comparable to runs after it.
# That is why these values are now IN run_metadata.json — the family boundary belongs in the
# artefact, not in the message that announced it, and not in a file that records the intent
# rather than the effect.
#
# EXPECTED COST, to be measured and not assumed: the agent never stops, so every frame carries
# motion blur that a dwell frame did not. If association degrades, that is a finding to record,
# not a reason to quietly restore 60.
export FEED_SEED="${FEED_SEED:-7}"
export FEED_FPS="${FEED_FPS:-3}"
export FEED_WALK="${FEED_WALK:-6}"
export FEED_DWELL="${FEED_DWELL:-0}"
# GA-339 (owner ruling 2026-09-07 ~13:50). ADAPTIVE dwell by default: after each walk burst the
# feed HOLDS a still camera until the object manager's merge_pending.json says nothing is pending,
# bounded by FEED_DWELL_MAX. FEED_DWELL (fixed frames) is IGNORED in adaptive mode and only read
# under FEED_DWELL_MODE=fixed. 18 = gate 0.5 s + one ~5 s cycle at 3 f/s; 90 = 30 s, the owner's cap
# (raised from 45 on 2026-09-07 ~16:55: run 152446 capped 27 of 39 holds with pending work still owed).
# Bundles at 90 are a new family against 152446 (45).
# Adaptive bundles are a NEW FAMILY, stamped below as dwell_family.
# MAPPING_ONLY runs no object manager, so merge_pending.json never exists and every adaptive hold
# caps at FEED_DWELL_MAX with the signal absent by construction: 360/450/360 stationary frames on the
# three 8 Sep mapping runs, ~14 % of the tour (PLAN_1.3 §56.2, orchestrator follow-up 3). Fixed dwell
# with FEED_DWELL 0 is the mapping default; an explicit FEED_DWELL_MODE still wins.
export FEED_DWELL_MODE="${FEED_DWELL_MODE:-$([ "${MAPPING_ONLY:-0}" = "1" ] && echo fixed || echo adaptive)}"
export FEED_DWELL_MIN="${FEED_DWELL_MIN:-18}"
export FEED_DWELL_MAX="${FEED_DWELL_MAX:-90}"
export FEED_DWELL_SIGNAL_MAX_AGE_S="${FEED_DWELL_SIGNAL_MAX_AGE_S:-10}"
# GA-330. Ground truth ON by default. The scene ships its semantic mesh, the feed host renders
# it, the feed node publishes /gt/semantic_instance and the archive joins it per detection --
# and the switch below was 0 in every one of the first 12 bundles, so not one row was ever
# labelled ("no semantic frame" on 100% of rows). The cost is a third render per frame on the
# host; the archive refuses on any shape mismatch rather than guessing. Set 0 to opt out.
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
    export FEED_MAPPING_SECONDS="${FEED_MAPPING_SECONDS:-0}"
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
if [ "$MAPPING_ONLY" = "1" ]; then
  export FEED_MAPPING_SECONDS="${FEED_MAPPING_SECONDS:-900}"
  # THE CAP MUST COVER THE GATE AS WELL AS THE TOUR (rule 55: two settings that must agree get a
  # probe). run_capped.sh counts CAP_MIN from the container appearing; the container's mapping
  # deadline counts FEED_MAPPING_SECONDS + 60 from after build + gate, which took 1:21 / 2:52 / 4:32
  # on the three 8 Sep launches; the close needs up to 150 s. Run 150019 was capped 1 s before its
  # own timer and labelled capped=true with a finished tour. CAP_MIN is run_capped.sh's; it reaches
  # here through the environment when the recipe sets it, and an unset CAP_MIN means no cap.
  if [ -n "${CAP_MIN:-}" ]; then
    case "$CAP_MIN" in (*[!0-9]*|"") echo "!! CAP_MIN='$CAP_MIN' is not a whole number of minutes. Refusing."; exit 1;; esac
    _cap_need=$(( (${FEED_MAPPING_SECONDS%.*} + 60 + 300 + 150 + 59) / 60 ))
    if [ "$CAP_MIN" -lt "$_cap_need" ]; then
      echo "!! MAPPING_ONLY with CAP_MIN=$CAP_MIN: the cap must cover build+gate (<=5 min) + ${FEED_MAPPING_SECONDS%.*}+60 s tour deadline + 150 s close = CAP_MIN >= $_cap_need. Refusing to start a run whose finished tour would be labelled capped."
      exit 1
    fi
  fi
else
  export FEED_MAPPING_SECONDS="${FEED_MAPPING_SECONDS:-150}"
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
IMAGE_TAG=${IMAGE_TAG:-graphapi-run:humble-ga290}
IMAGE_DIGEST=$(docker image inspect -f '{{.Id}}' "$IMAGE_TAG" 2>/dev/null || echo "unknown")
# GA-437 (2026-09-10). THE STAMP MUST READ THE CACHE THE RUN USES. This was
# $WORKSPACE_ROOT/.hf_cache, which made WORKSPACE_ROOT do double duty as the data root AND the model
# cache; after the workspace moved to a neutral directory it names nothing, and _enc_rev below would
# have stamped every encoder revision as unknown. The container loads its weights from the mount at
# /models/hf, whose host side is HF_SHARED_CACHE, so that is the directory whose refs describe the
# run. Same value the container now sets HF_HOME to (live_stack_container.sh).
HF_CACHE=${HF_CACHE:-${HF_SHARED_CACHE:-/DATA/huggingface_cache}}
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
# Six NUMERIC positions in the feed block below. An empty expansion yields `"dwell_frames": ,`
# and the validator kills the run — which is the correct outcome, but these refuse first and say
# which name is missing.
: "${FEED_SEED:?not set at run_metadata.json — the feed exports must precede this heredoc}"
: "${FEED_FPS:?not set at run_metadata.json}"
: "${FEED_WALK:?not set at run_metadata.json}"
: "${FEED_DWELL?not set at run_metadata.json}"   # no colon: 0 is the point of this variable
: "${FEED_DWELL_MODE:?not set at run_metadata.json}"   # GA-339
: "${FEED_DWELL_MIN:?not set at run_metadata.json}"
: "${FEED_DWELL_MAX:?not set at run_metadata.json}"
: "${FEED_DWELL_SIGNAL_MAX_AGE_S:?not set at run_metadata.json}"
: "${ROOM_FRAME_MAX:?not set at run_metadata.json}"   # GA-350
: "${ROOM_FRAME_STRIDE_M:?not set at run_metadata.json}"
: "${FEED_POSE_SOURCE:?not set at run_metadata.json}"   # GA-359
: "${FEED_MAPPING_SECONDS:?not set at run_metadata.json}"
: "${MAPPING_ONLY?not set at run_metadata.json}"
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
for _v in ${EXT_ENV_PASS:-}; do
  eval "_isset=\${$_v+yes}"
  [ -n "${_isset:-}" ] || { echo "!! $_v is declared in EXT_ENV_PASS but not set at run_metadata.json"; exit 1; }
done
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
    "walk_frames": $FEED_WALK,
    "tour_waypoints": ${FEED_TEST_TOUR:-0},
    "tour_scan_frames": ${FEED_TEST_TOUR_SCAN:-12},
    "tour_note": "GA-256. 0 means NO TOUR: the agent turns in place (walk radius 0) or wanders a disc around its spawn, and never leaves the room it started in. Run 20260901_174810 recorded total_distance_m 0.0 over 1,566 steps for exactly that reason, which is why coverage, room segmentation and the held-pool resolution rate could not be measured from it. A positive value is the number of farthest-point-sampled waypoints toured on the traversed storey.",
    "dwell_frames": $FEED_DWELL,
    "dwell_mode": "$FEED_DWELL_MODE",
    "dwell_min_frames": $FEED_DWELL_MIN,
    "dwell_max_frames": $FEED_DWELL_MAX,
    "dwell_signal_path": "$RUN_DIR/merge_pending.json",
    "dwell_signal_max_age_s": $FEED_DWELL_SIGNAL_MAX_AGE_S,
    "dwell_family": "GA-339, 2026-09-07: dwell_mode adaptive holds a STILL camera after each walk burst until merge_pending.json reads pending 0 (fresh), bounded by dwell_max_frames (45 in 20260907_152446, 90 from 2026-09-07 ~17:00). dwell_frames is IGNORED when dwell_mode is adaptive. Adaptive bundles are a NEW family: not comparable with dwell_frames 0 (2026-08-31 to 2026-09-07) or 60 (before). Per-run counters are in feed_stats.json (dwell_episodes, dwell_capped, dwell_released_on_zero, dwell_unknown_frames).",
    "fps": $FEED_FPS,
    "mapping_seconds": $FEED_MAPPING_SECONDS,
    "mapping_only": $([ "$MAPPING_ONLY" = "1" ] && echo true || echo false),
    "spawn_floor_requested": $([ -n "$FEED_SPAWN_FLOOR" ] && echo "$FEED_SPAWN_FLOOR" || echo null),
    "camera_pitch_deg": $FEED_CAMERA_PITCH_DEG,
    "camera_pitch_note": "negative looks DOWN, applied to rgb, depth and semantic together. 0 is the level camera every run before 2026-09-09 used.",
    "seed_source": "$SEED_SOURCE",
    "scene_source": "$SCENE_SOURCE",
    "draw_note": "drawn = this run chose it among the published per-floor maps (MAP_DRAW=1); pinned = the recipe named it. An A/B arm pins both.",
    "spawn_floor_note": "what was ASKED for. What the run actually mapped is measured from the map's own node poses into rtabmap.db.floor.json. If these two disagree the STAMP is right and this field records the intent that was not met.",
    "mapping_only_note": "true means NO DETECTOR RAN. A mapping bundle with zero detections is a mapping run, not a detection run that found nothing -- the two are otherwise indistinguishable from the artefacts, which is the failure that cost run 19 its merge question.",
    "overlay": $FEED_OVERLAY,
    "show": $FEED_SHOW,
    "note": "how the agent moved. ABSENT from every bundle before 2026-08-31, so a run's dwell setting cannot be recovered from an older bundle and must not be guessed from its date.",
    "dwell_note": "dwell_frames 0 means the agent never stops. Bundles with dwell_frames 0 are a DIFFERENT FAMILY from bundles with 60 and must not be pooled with them or differenced against them."
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
    "tour_shape": "$([ "${FEED_TOUR_ALL_FLOORS:-0}" != "0" ] && echo continuous_teleport || echo relaunch_per_storey)",
    "house_id": $([ -n "${HOUSE_ID:-}" ] && echo "\"$HOUSE_ID\"" || echo null),
    "spawn_floor": $([ -n "${FEED_SPAWN_FLOOR:-}" ] && echo "$FEED_SPAWN_FLOOR" || echo null),
    "localize_db_note": "GA-336: localize_db points at a SCRATCH COPY deleted at exit, so the path alone identifies nothing. localize_db_source + localize_db_sha256_16 name the canonical file this run actually opened.",
    "localize_db_source": $([ -n "${LOCALIZE_DB_SOURCE:-}" ] && echo "\"$LOCALIZE_DB_SOURCE\"" || echo null),
    "localize_db_sha256_16": $([ -n "${LOCALIZE_DB_SHA:-}" ] && echo "\"$LOCALIZE_DB_SHA\"" || echo null),
    "bridge_port": ${BRIDGE_PORT:-null},
    "bridge_port_note": "the port the bridge bound (BRIDGE_PORT); null means BRIDGE_PORT was unset and the bridge used its own default. Asked for by agent2-dashboard 2026-09-06 (their 00015): the dashboard used to have to grep logs/bridge.log for it.",
    "mapping_seconds_effective": $FEED_MAPPING_SECONDS,
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
    'association.merge_cost_ratio': g('association', 'merge_cost_ratio'),
    'association.merge_min_consecutive': g('association', 'merge_min_consecutive'),
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
echo "    (latest is repointed at the end, and only if the gate passes)"

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
  > "$OUT_DIR/feed_host.log" 2>&1 &
FEED_PID=$!
# habitat import + scene load can take >2 min on cold caches
for i in $(seq 1 90); do grep -q "listening" "$OUT_DIR/feed_host.log" 2>/dev/null && break; sleep 2; done
grep -q "listening" "$OUT_DIR/feed_host.log" || { echo "feed host failed:"; tail -20 "$OUT_DIR/feed_host.log"; exit 1; }
echo "    feed host up"

# GA-371 (owner 2026-09-08 13:05): every launch also opens RViz, beside Habitat and the dashboard.
# Sibling container graphapi_rviz on the host network (view_rviz.sh), started here so DDS discovery
# sees the stack's topics as they appear. RVIZ=0 opts out (headless hosts). Absent display or
# opt-out is RECORDED in run_metadata (rviz_started false + rviz_reason), never skipped silently.
# The log rides the existing $OUT_DIR/*.log archive into logs/rviz.log. Stopped in cleanup().
RVIZ="${RVIZ:-1}"; RVIZ_STARTED=false; RVIZ_REASON=""; RVIZ_DISPLAY="${DISPLAY:-:1}"   # same default the feed host uses
if [ "$RVIZ" != "1" ]; then
  RVIZ_REASON="RVIZ=$RVIZ opt-out"
elif [ ! -S "/tmp/.X11-unix/X${RVIZ_DISPLAY#:}" ]; then
  RVIZ_REASON="no X socket for DISPLAY=$RVIZ_DISPLAY"
else
  LOG="$OUT_DIR/rviz.log" DISPLAY="$RVIZ_DISPLAY" IMAGE_TAG="$IMAGE_TAG" \
    setsid nohup bash "$HERE/view_rviz.sh" >/dev/null 2>&1 < /dev/null &
  for _i in $(seq 1 15); do docker ps --format '{{.Names}}' | grep -qx graphapi_rviz && break; sleep 2; done
  if docker ps --format '{{.Names}}' | grep -qx graphapi_rviz; then
    RVIZ_STARTED=true; RVIZ_REASON="graphapi_rviz up (DISPLAY=$RVIZ_DISPLAY, dri=$([ -d /dev/dri ] && echo yes || echo no))"
  else
    RVIZ_REASON="graphapi_rviz not up 30 s after start; see logs/rviz.log"
  fi
fi
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
    echo "=== $(date) ===" >> "$OUT_DIR/system_health.log"
    free -m >> "$OUT_DIR/system_health.log"
    docker stats --no-stream graphapi_live >> "$OUT_DIR/system_health.log" 2>/dev/null || true
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
rm -f "$RUN_DIR/NOT_STARTED"   # GA-381: past every check; from here the directory is a real attempt
docker run --name graphapi_live --rm --entrypoint bash --gpus all --network=host \
  -e OPENAI_API_KEY -e CFG_NAME -e MODAL_PERCEPTION_URL -e MERGE_ENGINE -e PERCEPTION_DEBUG \
  -e MERGE_MIN_CONSECUTIVE \
  -e RUN_START_EPOCH -e PREFLIGHT_EXPECT_POLICY -e PREFLIGHT_SKIP \
  -e MAPPING_ONLY -e FEED_MAPPING_SECONDS -e RTABMAP_LOCALIZE_DB -e RTABMAP_CLOSE_TIMEOUT \
  -e FEED_HF_OFFLINE -e PREFLIGHT_HF_CACHE \
  -e FEED_SPAWN_FLOOR -e WALL_DETECTOR -e BRIDGE_SERVICE_TIMEOUT \
  -e ROOM_FRAME_MAX -e ROOM_FRAME_STRIDE_M -e FEED_POSE_SOURCE \
 -e GRAPH_API_SRC -e GRAPH_API_TEST_SRC \
  -e KG_BRIDGE_SRC \
  -e BRIDGE_PORT -e BRIDGE_RAW_MAX_AGE -e BRIDGE_ANNOTATED_MAX_AGE -e BRIDGE_FEED_PROBE_BACKOFF \
  -e FEED_HOST -e FEED_PORT -e FEED_CTRL_HOST -e FEED_CTRL_PORT -e LOST3DSG_OUTPUT_DIR \
  -e GRAPH_API_AUTOSTART -e GRAPH_API_BASE_URL -e GRAPH_API_TIMEOUT \
  -e ROOM_VLM_MODEL -e OPENROUTER_API_KEY -e REGOLO_API_KEY \
  -e HABITAT_EXAMPLE_OBJECTS_DIR -e DISPLAY \
  -e PREFLIGHT_EXPECT_CFG_SHA -e PREFLIGHT_EXPECT_MERGED_SHA -e PREFLIGHT_EXPECT_SRC_SHA \
  -e PREFLIGHT_EXPECT_CYCLE_S \
  -e ARCHIVE_DEPTH -e FEED_HFOV -e BRIDGE_OVERLAY -e BRIDGE_OVERLAY_CAM_FRAME -e BRIDGE_OVERLAY_MAP_FRAME \
  -e BRIDGE_OVERLAY_FAR -e BRIDGE_OVERLAY_MAX \
 -e OPENAI_BASE_URL \
  $EXT_E_ARGS \
  -v "$REPO":/graph_api:ro \
  -v graphapi_ws:/ws \
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
  -v "${SAM_MODEL_DIR:-/DATA/models/efficientvit_sam}":/models/vitsam:ro \
  -v "${HF_SHARED_CACHE:-/DATA/huggingface_cache}":/models/hf \
  -v "$OUT_DIR":/out \
  "$IMAGE_TAG" /graph_api/lost3dsg/test/live_stack_container.sh

# Post-run archive
cp "$OUT_DIR"/*.log "$RUN_DIR/logs/" 2>/dev/null || true
cp "$OUT_DIR"/*.json "$RUN_DIR/" 2>/dev/null || true
cp "$OUT_DIR"/*.jsonl "$RUN_DIR/" 2>/dev/null || true

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
