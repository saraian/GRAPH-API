#!/usr/bin/env bash
# GA-434 / RULE 73. A BASE RUN IS A HOUSE: one launch, one map, one bundle, PER STOREY.
#
# Owner, 2026-09-10: "no caps this time, we need to perform a full house tour, all storeys (if we
# finish a storey, just teleport to the next storey). This is our base run policy from now on."
# Owner, same day, on how the mapping is arranged: ONE MAPPING SESSION PER STOREY. Ruling 25 stands,
# so no map may straddle storeys -- a 2D occupancy grid cannot represent two, and the walls of the
# upper storey cut through the rooms of the lower one on a merged grid.
#
# SO "TELEPORT TO THE NEXT STOREY" IS BUILT AS "RELAUNCH ON THE NEXT STOREY", and almost nothing
# new was needed for it:
#   - the container's output directory IS the bundle and it is fresh per launch, so the world
#     model, the decision log and the store start EMPTY per storey by construction. No reset code,
#     no handshake, and no transform between two SLAM origins.
#   - habitat_launch.py hardcodes database_path with --delete_db_on_start, so each launch's
#     rtabmap.db IS that storey's map, and live_run.sh already archives it with its integrity mark,
#     its floor stamp and its params-sha.
#   - FEED_SPAWN_FLOOR already spawns on a named storey and REFUSES rather than landing on another.
# The alternative -- one continuous feed process teleporting between storeys -- is REMOVED (owner
# 2026-09-11). It lived in the sampling tour, which chose the next storey and walked to it; a
# schedule is one storey by construction. FEED_TOUR_ALL_FLOORS now refuses rather than doing
# nothing, so this script is the only way to tour a house.
#
# WHAT ENDS EACH LAUNCH, now that rule 73 forbids a cap: the feed host writes feed_ended.json when
# the storey's waypoints are exhausted and its settle period has passed, and live_stack_container.sh
# watches for that file and closes the stack through the normal archive path. Without it a no-cap
# run tours the storey and then turns in place forever.
#
# THE COST THE OWNER ACCEPTED: a house run produces N BUNDLES, not one. Anything that spans the
# house -- total coverage, an object seen on two storeys, the duplicate rate -- is a post-hoc join
# across them. manifest.json below is what that join reads.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$HERE/../.." && pwd)}"
# ONE DIRECTORY PER RUN, IN THE REPOSITORY (owner instruction 2026-09-10). Was
# $WORKSPACE_ROOT/runs; with the workspace now defaulting to the checkout that path holds nothing.
RUNS_DIR="${RESULTS_DIR:-$WORKSPACE_ROOT/results}"

# RULE 73 FORBIDS A CAP, and a cap set here would silently truncate a storey mid-tour and leave a
# bundle that looks finished. Refuse rather than unset it: the caller meant something by it.
if [ -n "${CAP_MIN:-}" ]; then
  echo "!! CAP_MIN=$CAP_MIN is set. Rule 73: a base run has no cap — each storey ends when its"
  echo "   tour is complete (feed_ended.json). Clear CAP_MIN, or run one storey with live_run.sh."
  exit 1
fi
# FEED_TOUR_ALL_FLOORS asked for the teleporting tour, which is removed with the sampling policy
# (owner 2026-09-11). The feed host refuses it too; this one catches it before N launches start.
if [ "${FEED_TOUR_ALL_FLOORS:-0}" != "0" ]; then
  echo "!! FEED_TOUR_ALL_FLOORS=$FEED_TOUR_ALL_FLOORS: the continuous teleporting tour is removed"
  echo "   with the sampling policy (owner 2026-09-11). This script IS how a house is toured now:"
  echo "   one launch per storey, one map per storey (ruling 25). Clear the variable."
  exit 1
fi

HOUSE_ID="house_$(date +%Y%m%d_%H%M%S)"
HOUSE_DIR="$RUNS_DIR/$HOUSE_ID"
mkdir -p "$HOUSE_DIR"
echo ">>> HOUSE RUN $HOUSE_ID"
echo "    manifest: $HOUSE_DIR/manifest.json"

# THE STOREYS. Given ones are used as given; otherwise the FIRST launch discovers them, because the
# clustering that names a storey needs a loaded navmesh and that is what a launch already has. The
# first launch is a real storey bundle, not a probe: it spawns where the seed sends it and the other
# storeys are toured after, so nothing is wasted to find them.
FLOORS="${HOUSE_FLOORS:-}"
storeys_done=()
bundles=()
statuses=()

run_storey() {   # $1 = floor or "" for the seeded spawn; the rest are live_run.sh's own arguments
  local floor="$1" stamp bundle rc=0
  shift   # WITHOUT THIS the floor is passed to live_run.sh as its first argument, which is the scene.
  stamp="$(date +%Y%m%d_%H%M%S)"
  # The stamp is the bundle's name and live_run.sh REFUSES a name already taken, so two storeys
  # starting inside the same second cannot land in one bundle.
  while [ -e "$RUNS_DIR/${stamp}_${SCENE:-hm3d_00861}" ]; do sleep 1; stamp="$(date +%Y%m%d_%H%M%S)"; done
  echo ""
  echo ">>> STOREY ${floor:-<seeded spawn>} — launching (bundle stamp $stamp)"
  if [ -n "$floor" ]; then
    RUN_TIMESTAMP="$stamp" HOUSE_ID="$HOUSE_ID" FEED_SPAWN_FLOOR="$floor" \
      bash "$HERE/live_run.sh" "$@" || rc=$?
  else
    RUN_TIMESTAMP="$stamp" HOUSE_ID="$HOUSE_ID" bash "$HERE/live_run.sh" "$@" || rc=$?
  fi
  bundle="$(ls -d "$RUNS_DIR/${stamp}_"* 2>/dev/null | head -1 || true)"
  storeys_done+=("${floor:-seeded}")
  bundles+=("${bundle:-none}")
  statuses+=("$rc")
  if [ "$rc" -ne 0 ]; then
    echo "!! STOREY ${floor:-<seeded spawn>} FAILED (exit $rc). Stopping the house here."
    echo "   A storey usually fails for a reason the next storey would hit too, and four identical"
    echo "   failures cost four launches to learn once. The manifest records what did run."
  fi
  return "$rc"
}

first_rc=0
run_storey "" "$@" || first_rc=$?

# The floors the run itself measured, from its own bundle. Not a second clustering: bev_data.json is
# written by the same code that chose the spawn, so the driver and the run agree by construction.
first_bundle="${bundles[0]}"
if [ -z "$FLOORS" ] && [ -f "$first_bundle/bev_data.json" ]; then
  # `floors` holds the accepted STOREYS; `floor_detail.levels` holds the clusters that were
  # rejected as stair landings and galleries. Touring a level would map a 1 x 0.4 m landing and
  # report it as a storey, which is the fault GA-93 was corrected for. Verified against
  # 20260909_004443_hm3d_00861: floors [-1.59, 1.21], levels [1.81, 2.21].
  FLOORS="$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
fl = d.get("floors") or [f["z"] for f in (d.get("floor_detail") or {}).get("floors") or []]
print(" ".join(f"{float(z):+.2f}" for z in sorted(float(z) for z in fl)))' \
    "$first_bundle/bev_data.json" 2>/dev/null || true)"
  echo ""
  echo ">>> storeys measured by the first launch: ${FLOORS:-<none found in bev_data.json>}"
fi
if [ -z "$FLOORS" ]; then
  echo "!! No storeys to tour: neither HOUSE_FLOORS nor $first_bundle/bev_data.json named any."
  echo "   The first storey's bundle stands on its own; the house is one storey by default."
fi

# The storey the first launch actually toured. From feed_stats.json's floor_guard, which records
# the ANCHOR the guard held the tour to -- the nearest derived storey, not wherever the spawn
# happened to land, so it names a storey in the same terms as the list above. run_metadata.json
# carries no spawn floor when none was requested, which is exactly this case.
first_floor=""
if [ -f "$first_bundle/feed_stats.json" ]; then
  first_floor="$(python3 -c '
import json, sys
g = (json.load(open(sys.argv[1])).get("floor_guard") or {})
z = g.get("floor_y")
print("" if z is None else f"{float(z):+.2f}")' "$first_bundle/feed_stats.json" 2>/dev/null || true)"
fi
[ -n "$first_floor" ] && echo "    first launch toured storey $first_floor"

house_rc="$first_rc"
if [ "$first_rc" -eq 0 ]; then
  for z in $FLOORS; do
    [ -n "$first_floor" ] && [ "$z" = "$first_floor" ] && { echo ">>> storey $z already toured by the first launch"; continue; }
    run_storey "$z" "$@" || { house_rc=$?; break; }
  done
fi

python3 - "$HOUSE_DIR/manifest.json" "$HOUSE_ID" "$house_rc" "${storeys_done[@]}" -- "${bundles[@]}" -- "${statuses[@]}" <<'PY'
import json, sys
path, house_id, rc = sys.argv[1], sys.argv[2], sys.argv[3]
rest = sys.argv[4:]
a = rest.index("--"); b = rest.index("--", a + 1)
storeys, bundles, statuses = rest[:a], rest[a+1:b], rest[b+1:]
json.dump({
    "house_id": house_id,
    "policy": "rule 73: one launch, one map, one bundle, per storey",
    "exit_status": int(rc),
    "complete": int(rc) == 0,
    "storeys": [{"floor": s, "bundle": bu, "exit_status": int(st)}
                for s, bu, st in zip(storeys, bundles, statuses)],
    "note": "N bundles, not one. Anything that spans the house -- coverage, an object seen on two "
            "storeys, the duplicate rate -- is a post-hoc join across these bundles. Each bundle's "
            "map has its own SLAM origin and they are NOT in a common frame.",
}, open(path, "w"), indent=2)
print(f"wrote {path}")
PY

echo ""
echo ">>> HOUSE RUN $HOUSE_ID ${storeys_done[*]} — $( [ "$house_rc" -eq 0 ] && echo COMPLETE || echo "INCOMPLETE (exit $house_rc)")"
echo "    manifest: $HOUSE_DIR/manifest.json"
exit "$house_rc"
