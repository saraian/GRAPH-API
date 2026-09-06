#!/usr/bin/env bash
# Live demo on this machine: habitat renders on the host (conda habitat_env),
# the ROS 2 stack runs in the graphapi-run:humble-ga290 container (patched rtabmap, GA-290)
# over a TCP feed.
# Watch: web viewer at http://localhost:8081 and snapshots in $OUT_DIR.
#   ./live_run.sh [scene]  # foreground; ctrl-C stops everything
# scene: hm3d_00861 (default) | hm3d_00337 | hm3d_00770 | mp3d_17DRP
# HABITAT_SCENE/HABITAT_DATASET env vars still override everything.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
# WHERE FOUND IS. Derived from this script's own location, not hardcoded: the submodule sits at
# <FOUND>/vendor/graph-api, so two levels above $REPO is the FOUND checkout whatever it is called
# and wherever it lives. $FOUND_ROOT was written into ten places and a clone anywhere else could
# not run at all — the launcher would look for maps, results and the found/ package on a machine
# that has none of them.
#
# Override only to run against a FOUND checkout other than the one this submodule is inside.
FOUND_ROOT=${FOUND_ROOT:-$(cd "$REPO/../.." && pwd)}
[ -d "$FOUND_ROOT/found" ] || { echo "!! FOUND_ROOT=$FOUND_ROOT has no found/ package."; \
  echo "   This script expects to live at <FOUND>/vendor/graph-api/lost3dsg/test, or FOUND_ROOT set."; exit 1; }

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
  # The map library lives on the HOST at $FOUND_ROOT/maps/<scene>/, not inside a bundle — a
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
print(f'floor_{z:+.2f}')" "${db}.floor.json")
  local dest="$FOUND_ROOT/maps/${SCENE_ARG}/${fl}"
  mkdir -p "$dest"
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
  if [ "$verdict" = "pass" ] && [ -n "$measured" ]; then
    ln -sfn "$RUN_DIR" "$FOUND_RUNS_DIR/latest"
    echo ">>> gate passed, measured output present ($measured) -- $RUN_DIR is now latest"
  else
    echo ">>> latest NOT moved: preflight '$verdict', measured artefact '${measured:-none}'."
    echo "    it still points at $(readlink "$FOUND_RUNS_DIR/latest" 2>/dev/null || echo '<unset>')"
  fi
}

cleanup() {
  # PID-scoped: only kill what THIS instance started. A pattern-based
  # pkill here would let two concurrent live_run.sh instances destroy
  # each other's feed host (seen live 2026-08-25).
  [ -n "$MON_PID" ] && kill -9 "$MON_PID" 2>/dev/null || true
  [ -n "$FEED_PID" ] && kill -9 "$FEED_PID" 2>/dev/null || true
  # Archive the HOST-written artefacts here rather than only after the container exits. The
  # per-frame viewpoint series and the feed host's own log are written on this side, and the
  # post-run copy below never runs when the script is stopped with Ctrl-C — which is the
  # documented way to stop it. feed_host.log reached 1 of 21 shipped bundles for this reason.
  if [ -n "${RUN_DIR:-}" ] && [ -d "$RUN_DIR" ]; then
    cp "$OUT_DIR"/*.json "$OUT_DIR"/*.jsonl "$RUN_DIR/" 2>/dev/null || true
    cp "$OUT_DIR"/*.log "$RUN_DIR/logs/" 2>/dev/null || true
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
    ( cd "$FOUND_ROOT" && python3 -m tools.class_counts "$RUN_DIR" ) 2>&1 \
      | sed 's/^/    /' || echo "    class_counts failed (non-fatal)"

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

# Setup persistent FOUND run bundle (never overwritten across runs)
RUN_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCENE_ARG=${1:-hm3d_00861}

# Live run output. A TIMESTAMPED DIRECTORY under $FOUND_ROOT/results/, never /tmp.
# Owner ruling, relayed to this lane rather than given to it directly (rule 8's second half):
#   "no output should go to the temp directory, always in a timestamped experiment results dir
#    inside the results/ dir inside the $FOUND_ROOT directory."
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
export OUT_DIR=${OUT_DIR:-$FOUND_ROOT/results/${RUN_TIMESTAMP}_${SCENE_ARG}}
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
FOUND_RUNS_DIR=${FOUND_RUNS_DIR:-$FOUND_ROOT/runs}
RUN_DIR="$FOUND_RUNS_DIR/$RUN_ID"
# GA-258b. EXPORTED, because the FEED HOST needs it. The host process reads
# merge_pending.json to decide how long to dwell, and that file is written by the container
# into /ws/output -- which is bind-mounted to $RUN_DIR, not to $OUT_DIR (/out). The feed host
# was building the path from GRAPH_API_OUTPUT_DIR, which only exists INSIDE the container, so
# on the host it resolved to a bare relative filename and never opened. Measured on run
# 20260902_125130: every dwell line read "? merges pending (sweep None)" and every waypoint
# ran to the 90-frame cap.
export RUN_DIR
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/crops" "$RUN_DIR/snapshots"
echo "    run bundle: $RUN_DIR (symlinked as $FOUND_RUNS_DIR/latest)"

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

# Every knob the admission policy reads, EXPORTED with its default rather than forwarded bare.
# `docker run -e VAR` sends nothing when VAR is unset in the parent environment, so a bare
# forward silently ships the container's own idea of the default while the bundle records the
# launcher's. Exporting here makes the two the same value, and it is what lets the pre-flight
# gate compare an intention against what actually arrived instead of echoing what it finds.
export FOUND_ENFORCE="${FOUND_ENFORCE:-0}"
export FOUND_HOLD_BAND="${FOUND_HOLD_BAND:-0.05}"
# GA-263, owner ruling: 30 -> 4. See found/admission.py for why the floor stopped
# doing anything once _envelope_key was made class-first.
export FOUND_MIN_SUPPORT="${FOUND_MIN_SUPPORT:-4}"
export FOUND_ROOM_ENFORCE="${FOUND_ROOM_ENFORCE:-0}"
export FOUND_ALIGNER="${FOUND_ALIGNER:-kg}"
# EMPTY means "use the built-in extension". found/kg_align.py:88-91 reads this as a PATH when
# it is non-empty and raises FileNotFoundError if that path is absent — so the literal string
# "default" was passed as a filename and every gated run failed a1 with
# `FOUND_ONTOLOGY_EXT set to default, which does not exist`.
#
# The word came from run_metadata.json's policy block, where "default" is a human-readable
# LABEL. I exported the label as the value. The bundle still records "default"; the process
# receives the empty string that actually means it.
# Owner rulings 13 and 15. EXPORTED AND PASSED, not merely stamped.
#
# These arrived recorded in run_metadata.json but neither exported nor on the docker run -e list.
# Today that is harmless because both sides default to the same thing — found/dims.py's
# _DEFAULT_CORPUS_ORDER is ("abo","metrictree") and kg_align defaults aliases to 1, matching the
# ${VAR:-default} the bundle stamps. THE AGREEMENT IS COINCIDENTAL, which is the GA-99 shape
# exactly: two paths that agree by accident until one of them moves.
#
# The live consequence is sharper than a future one. dims.py:60 says the point of making the
# corpus order configurable is that an ablation arm can be attributed — "otherwise the arm would
# be labelled and not applied". Without the passthrough that is the CURRENT state: setting
# FOUND_CORPUS_ORDER host-side would be written into the bundle and never reach the code.
export FOUND_CORPUS_ORDER="${FOUND_CORPUS_ORDER:-}"
export FOUND_KG_ALIASES="${FOUND_KG_ALIASES:-}"
export FOUND_ONTOLOGY_EXT="${FOUND_ONTOLOGY_EXT:-}"
export FOUND_STORE_PATH="${FOUND_STORE_PATH:-/ws/output/knowledge_graph.ttl}"
export FOUND_SCENE="${FOUND_SCENE:-$SCENE_ARG}"

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
    _mapdir=$(printf "$FOUND_ROOT/maps/%s/floor_%+.2f" "$SCENE_ARG" "$FEED_SPAWN_FLOOR")
  else
    _mapdir="$FOUND_ROOT/maps/$SCENE_ARG"
  fi
  # A CHECKED FALLBACK, not a guess. The canonical hm3d map lives at the SCENE level rather than
  # under floor_<z>, because it was published before per-floor publication existed. Falling back to
  # it blindly would localize a floor-1.35 run against whatever that file happens to be. So the
  # fallback is allowed only when the map's OWN STAMP says it covers the requested floor — the
  # nearest_scene_floor that stamp_floor.py measured from its node poses.
  if [ ! -f "$_mapdir/rtabmap.db" ] && [ -n "$FEED_SPAWN_FLOOR" ] \
     && [ -f "$FOUND_ROOT/maps/$SCENE_ARG/rtabmap.db.floor.json" ]; then
    if python3 -c "import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if abs(float(d.get('nearest_scene_floor') or 1e9) - float(sys.argv[2])) < 1e-6 else 1)" \
        "$FOUND_ROOT/maps/$SCENE_ARG/rtabmap.db.floor.json" "$FEED_SPAWN_FLOOR" 2>/dev/null; then
      _mapdir="$FOUND_ROOT/maps/$SCENE_ARG"
      echo "    scene-level map stamped for floor $FEED_SPAWN_FLOOR — using it"
    fi
  fi
  if [ -f "$_mapdir/rtabmap.db" ]; then
    export RTABMAP_LOCALIZE_DB="${_mapdir#$FOUND_ROOT}"
    export RTABMAP_LOCALIZE_DB="/found${RTABMAP_LOCALIZE_DB}/rtabmap.db"
    export FEED_MAPPING_SECONDS="${FEED_MAPPING_SECONDS:-0}"
    echo "    localizing against $_mapdir/rtabmap.db (mapping phase 0s)"
  else
    echo "    NO published map at $_mapdir — this run will MAP from scratch."
    echo "    That is the fallback, not the intent: publish a map for this scene and floor and"
    echo "    subsequent runs will localize instead of re-mapping."
  fi
fi
if [ "$MAPPING_ONLY" = "1" ]; then
  export FEED_MAPPING_SECONDS="${FEED_MAPPING_SECONDS:-900}"
else
  export FEED_MAPPING_SECONDS="${FEED_MAPPING_SECONDS:-150}"
fi
export FEED_OVERLAY="${FEED_OVERLAY:-1}"
export FEED_SHOW="${FEED_SHOW:-1}"

# What the launcher INTENDS the policy to be. The gate compares this against the environment
# actually present inside the container, rather than echoing whatever it finds there — an echo
# is what let an "enforcing" run be a pass-through for weeks.
PREFLIGHT_EXPECT_POLICY="FOUND_ENFORCE=$FOUND_ENFORCE,FOUND_HOLD_BAND=$FOUND_HOLD_BAND"
PREFLIGHT_EXPECT_POLICY="$PREFLIGHT_EXPECT_POLICY,FOUND_MIN_SUPPORT=$FOUND_MIN_SUPPORT"
PREFLIGHT_EXPECT_POLICY="$PREFLIGHT_EXPECT_POLICY,FOUND_ROOM_ENFORCE=$FOUND_ROOM_ENFORCE"
PREFLIGHT_EXPECT_POLICY="$PREFLIGHT_EXPECT_POLICY,FOUND_ALIGNER=$FOUND_ALIGNER"
export PREFLIGHT_EXPECT_POLICY
echo "    policy: enforce=$FOUND_ENFORCE hold_band=$FOUND_HOLD_BAND \
min_support=$FOUND_MIN_SUPPORT rooms_enforced=$FOUND_ROOM_ENFORCE \
aligner=$FOUND_ALIGNER ontology_ext=${FOUND_ONTOLOGY_EXT:-default} \
corpus_order=${FOUND_CORPUS_ORDER:-<code default>} kg_aliases=${FOUND_KG_ALIASES:-1}"

# ---- provenance ---------------------------------------------------------------------------
# WHICH CODE produced this bundle. The digests come from preflight_gate.py rather than from a
# `find | xargs cat` here, so the launcher and the gate cannot drift apart — and so a root
# matching no files ABORTS instead of yielding e3b0c442..., the sha256 of nothing, which is a
# plausible sixteen-hex provenance stamp for a hash that covered zero files.
#
# The roots are typed. Only $REPO/lost3dsg is copied into the container at startup, so only it
# has a freeze point; $FOUND_ROOT/found is live on the path for the whole run and is SAMPLED,
# never asserted frozen. knowledge_bridge was a third root until GA-306 vendored the one class
# FOUND used into found/concept_embedder.py; it is no longer read, mounted or sampled.
_tree_sha() {
  local out
  out=$(python3 "$HERE/preflight_gate.py" --print-tree-sha "$1")     || { echo "!! cannot hash $1 — aborting rather than stamping an unrecorded run"; exit 1; }
  echo "$out"
}
read -r SRC_SHA SRC_N   <<<"$(_tree_sha "$REPO/lost3dsg")"
read -r FOUND_SHA FOUND_N <<<"$(_tree_sha $FOUND_ROOT/found)"
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
PREFLIGHT_EXPECT_CYCLE_S=$(python3 "$HERE/last_frame_age_rejected.py" "$FOUND_RUNS_DIR" 2>/dev/null || echo "")
export PREFLIGHT_EXPECT_CYCLE_S
[ -n "$PREFLIGHT_EXPECT_CYCLE_S" ] && \
  echo "    last run REJECTED a frame at ${PREFLIGHT_EXPECT_CYCLE_S}s (a10 checks max_frame_age_s against it)"

MERGED_SHA=$(GRAPH_API_CONFIG="$HERE/$CFG_NAME" python3 "$HERE/preflight_gate.py" --print-merged-sha)   || { echo "!! cannot compute the merged-config sha — aborting rather than passing an empty expectation"; exit 1; }
echo "    sources: graph-api $SRC_SHA ($SRC_N)  found $FOUND_SHA ($FOUND_N)  kb $KB_SHA ($KB_N)"
echo "    config:  file $CFG_SHA  merged $MERGED_SHA"

# Handed to the gate, which recomputes them INSIDE the container after the source copy. A
# difference means an edit landed in the window and the run is not the code stamped here.
export PREFLIGHT_EXPECT_SRC_SHA="graph_api=$SRC_SHA"
export PREFLIGHT_EXPECT_CFG_SHA="$CFG_SHA"
export PREFLIGHT_EXPECT_MERGED_SHA="$MERGED_SHA"

# WHICH ENVIRONMENT answered. The encoders are NOT in the image: live_stack_container.sh
# exports HF_HOME=/found/.hf_cache, so weights come from a host cache at runtime and two runs
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
HF_CACHE=${HF_CACHE:-$FOUND_ROOT/.hf_cache}
# >>> TEST-EXTRACT _enc_rev  (test_env_stamp.sh sources the block between these markers.
# It guessed the boundary with a sed pattern twice and was wrong twice: /^$/ swallowed the
# call sites below, and /; }$/ ran to end-of-file because this definition is a single line,
# so the range's START line also matched its terminator. The boundary is stated now, not
# inferred. Do not remove the markers.)
_enc_rev() { cat "$HF_CACHE/hub/models--$1/refs/main" 2>/dev/null              || cat "$HF_CACHE/models--$1/refs/main" 2>/dev/null || echo "unknown"; }
# <<< TEST-EXTRACT _enc_rev
ENC_E5=$(_enc_rev intfloat--e5-small-v2)
ENC_MINILM=$(_enc_rev sentence-transformers--all-MiniLM-L6-v2)

# WHICH GROUND TRUTH the numbers will be scored against. found/gt.py reads FOUND_SCENE_INSTANCE
# and has no default, so an unset or stale value silently rescopes every recall figure while the
# bundle still looks complete.
GT_PATH=${FOUND_SCENE_INSTANCE:-}
GT_SHA="unset"; GT_N="unset"
if [ -n "$GT_PATH" ] && [ -f "$GT_PATH" ]; then
  GT_SHA=$(sha256sum "$GT_PATH" | cut -c1-16)
  GT_N=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1])).get('objects') or []))" "$GT_PATH" 2>/dev/null || echo unknown)
  [ "$GT_N" = "0" ] && echo "!! WARNING: ground truth $GT_PATH contains 0 objects — every recall figure will be vacuous"
elif [ -n "$GT_PATH" ]; then
  echo "!! FOUND_SCENE_INSTANCE=$GT_PATH does not exist — aborting"; exit 1
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
: "${FOUND_ENFORCE:?not set at run_metadata.json — the policy exports must precede this heredoc}"
: "${FOUND_HOLD_BAND:?not set at run_metadata.json}"
: "${FOUND_MIN_SUPPORT:?not set at run_metadata.json}"
: "${FOUND_ROOM_ENFORCE:?not set at run_metadata.json}"
: "${FOUND_ALIGNER:?not set at run_metadata.json}"
: "${FOUND_ONTOLOGY_EXT?not set at run_metadata.json}"   # no colon: empty is a legal value here
# Six NUMERIC positions in the feed block below. An empty expansion yields `"dwell_frames": ,`
# and the validator kills the run — which is the correct outcome, but these refuse first and say
# which name is missing.
: "${FEED_SEED:?not set at run_metadata.json — the feed exports must precede this heredoc}"
: "${FEED_FPS:?not set at run_metadata.json}"
: "${FEED_WALK:?not set at run_metadata.json}"
: "${FEED_DWELL?not set at run_metadata.json}"   # no colon: 0 is the point of this variable
: "${FEED_MAPPING_SECONDS:?not set at run_metadata.json}"
: "${MAPPING_ONLY?not set at run_metadata.json}"
: "${FEED_SPAWN_FLOOR?not set at run_metadata.json}"   # no colon: empty means "no floor requested"   # no colon: 0 is a legal value
: "${SRC_SHA:?not set at run_metadata.json — the provenance block must precede this heredoc}"
: "${SRC_N:?not set at run_metadata.json}"
: "${FOUND_SHA:?not set at run_metadata.json}"
: "${FOUND_N:?not set at run_metadata.json}"
: "${KB_SHA:?not set at run_metadata.json}"
: "${KB_N:?not set at run_metadata.json}"
: "${CFG_SHA:?not set at run_metadata.json}"
: "${MERGED_SHA:?not set at run_metadata.json}"

# Keys are ADDED, never changed: three consumers read this file by key and some do arithmetic
# on the values. The five original keys keep their bytes. "config_name" keeps its meaning too,
# and that meaning is now stated: it is the name the LAUNCHER INTENDED. What the two processes
# actually loaded is recorded separately — the container's in preflight.json (a2 reads
# config.CFG_PATH, the file config.py actually read), the feed host's below.
cat <<EOF > "$RUN_DIR/run_metadata.json"
{
  "run_id": "$RUN_ID",
  "scene": "$SCENE_ARG",
  "start_time": "$(date -Iseconds)",
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
    "fps": $FEED_FPS,
    "mapping_seconds": $FEED_MAPPING_SECONDS,
    "mapping_only": $([ "$MAPPING_ONLY" = "1" ] && echo true || echo false),
    "spawn_floor_requested": $([ -n "$FEED_SPAWN_FLOOR" ] && echo "$FEED_SPAWN_FLOOR" || echo null),
    "spawn_floor_note": "what was ASKED for. What the run actually mapped is measured from the map's own node poses into rtabmap.db.floor.json. If these two disagree the STAMP is right and this field records the intent that was not met.",
    "mapping_only_note": "true means NO DETECTOR RAN. A mapping bundle with zero detections is a mapping run, not a detection run that found nothing -- the two are otherwise indistinguishable from the artefacts, which is the failure that cost run 19 its merge question.",
    "overlay": $FEED_OVERLAY,
    "show": $FEED_SHOW,
    "note": "how the agent moved. ABSENT from every bundle before 2026-08-31, so a run's dwell setting cannot be recovered from an older bundle and must not be guessed from its date.",
    "dwell_note": "dwell_frames 0 means the agent never stops. Bundles with dwell_frames 0 are a DIFFERENT FAMILY from bundles with 60 and must not be pooled with them or differenced against them."
  },
  "policy": {"enforce": $FOUND_ENFORCE, "hold_band": $FOUND_HOLD_BAND,
             "min_support": $FOUND_MIN_SUPPORT, "rooms_enforced": $FOUND_ROOM_ENFORCE,
    "corpus_order_note": "empty FOUND_CORPUS_ORDER means the code default in found/dims.py, standard,hssd,metrictree,abo,procthor as of GA-266, and the field then says so rather than naming an order. GA-282: this note claimed the hardcoded abo,metrictree fallback was PAST while line 638 still carried it, so every bundle up to and including 20260903_110622 records corpus_order abo,metrictree for a run that used standard(125) hssd(63) metrictree(56) abo(56) by its own decision records. The note outlived the fix it described. Read the corpus cited in each decision's margins, never this field, for any bundle stamped before 2026-09-03.",
    "merge_min_consecutive": ${MERGE_MIN_CONSECUTIVE:-2},
             "aligner": "$FOUND_ALIGNER", "ontology_ext": "${FOUND_ONTOLOGY_EXT:-default}",
             "corpus_order": "${FOUND_CORPUS_ORDER:-<code default: standard,hssd,metrictree,abo,procthor>}",
             "kg_aliases": ${FOUND_KG_ALIASES:-1},
             "policy_note": "corpus_order and kg_aliases were added 2026-09-01 (owner rulings 13, 15). ABSENT from every earlier bundle, so an older run's corpus order is metrictree,abo and its alias count is 0 -- read, never guessed from the date."},
  "provenance_intent": {
    "note": "host-side, taken BEFORE docker run. provenance_confirmed in preflight.json is taken after the container copies its sources, and is the authoritative record of what executed.",
    "graph_api_src_sha256_16": "$SRC_SHA", "graph_api_files": $SRC_N,
    "found_src_sha256_16": "$FOUND_SHA", "found_files": $FOUND_N,
    "kb_src_sha256_16": null, "kb_files": null,
    "kb_root": null,
    "kb_note": "GA-306, 2026-09-06: FOUND no longer imports knowledge_bridge -- the e5 ConceptEmbedder it used is vendored at found/concept_embedder.py. Nothing is mounted at /kb and KB_SRC is read nowhere. Explicit nulls, not removed keys: bundles before this date carry real digests here, and a reader joining across them must be able to tell 'not applicable' from 'never stamped'.",
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

echo ">>> ROS stack in container (web viewer -> http://localhost:8081)"
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
docker run --name graphapi_live --rm --entrypoint bash --gpus all --network=host \
  -e OPENAI_API_KEY -e CFG_NAME -e MODAL_PERCEPTION_URL -e MERGE_ENGINE -e PERCEPTION_DEBUG \
  -e MERGE_MIN_CONSECUTIVE \
  -e FOUND_ENFORCE -e FOUND_HOLD_BAND -e FOUND_MIN_SUPPORT -e FOUND_ROOM_ENFORCE \
  -e FOUND_ALIGNER -e FOUND_ONTOLOGY_EXT -e FOUND_STORE_PATH -e FOUND_SCENE \
  -e RUN_START_EPOCH -e PREFLIGHT_EXPECT_POLICY -e PREFLIGHT_SKIP \
  -e MAPPING_ONLY -e FEED_MAPPING_SECONDS -e RTABMAP_LOCALIZE_DB -e RTABMAP_CLOSE_TIMEOUT \
  -e FEED_SPAWN_FLOOR -e FOUND_CORPUS_ORDER -e FOUND_KG_ALIASES -e WALL_DETECTOR -e BRIDGE_SERVICE_TIMEOUT \
  -e FOUND_EMBED_MODEL -e FOUND_KG_TOP -e FOUND_KG_Z -e FOUND_LEXICAL -e FOUND_ONTOLOGY \
  -e FOUND_ROOM_TYPES_PATH -e FOUND_ROOM_VLM_API_KEY -e FOUND_ROOM_VLM_BASE_URL \
  -e FOUND_ROOM_VLM_MODEL -e FOUND_SCENE_INSTANCE -e GRAPH_API_SRC -e GRAPH_API_TEST_SRC \
  -e KG_BRIDGE_SRC \
  -e BRIDGE_PORT -e BRIDGE_RAW_MAX_AGE -e BRIDGE_ANNOTATED_MAX_AGE -e BRIDGE_FEED_PROBE_BACKOFF \
  -e FEED_HOST -e FEED_PORT -e LOST3DSG_OUTPUT_DIR \
  -e GRAPH_API_AUTOSTART -e GRAPH_API_BASE_URL -e GRAPH_API_TIMEOUT \
  -e ROOM_VLM_MODEL -e OPENROUTER_API_KEY -e REGOLO_API_KEY \
  -e FOUND_ADJUDICATE -e FOUND_ADJUDICATE_BASE_URL -e FOUND_ADJUDICATE_MODEL \
  -e HABITAT_EXAMPLE_OBJECTS_DIR -e DISPLAY \
  -e PREFLIGHT_EXPECT_CFG_SHA -e PREFLIGHT_EXPECT_MERGED_SHA -e PREFLIGHT_EXPECT_SRC_SHA \
  -e PREFLIGHT_EXPECT_CYCLE_S \
  -v "$REPO":/graph_api:ro \
  -v graphapi_ws:/ws \
  -v $FOUND_ROOT:/found \
  `# GA-295. THE MAP LIBRARY IS READ-ONLY, AND UNTIL NOW ONLY THE COMMENT SAID SO.
   # live_stack_container.sh has claimed since GA-158 that "the map is mounted read-only, not
   # copied", and printed a warning every run that it was writable. It was: /found carried no :ro,
   # and in localization mode rtabmap is handed the canonical map AS ITS OWN database_path, so its
   # close path writes to it. /DATA/FOUND/maps/hm3d_00861/rtabmap.db is 24 MB (5,870 pages) larger
   # than the 1,197,514,752 its provenance recorded on 31 Aug, and was last modified 2026-09-03
   # 22:51:58, during a run. Node, Data and integrity still match (1096/1096/ok), so this is not a
   # claim that the geometry changed -- it is a claim that a published artefact is not immutable.
   # The deeper mount wins, so runs still read the library and can no longer write it.` \
  -v "$FOUND_ROOT/maps":/found/maps:ro \
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
