# Multi-floor tour proposal and GRAPH-API certification gate

Status: DESIGN, not implemented or certified. Baseline rollout is on hold.
Owner sequencing on 2026-09-13: first demonstrate this approach in an actual
multi-floor GRAPH-API run; only after certification notify the orchestrator,
then assess whether the same approach is suitable for Clio/HOV-SG. Notification
is explicitly conditional on successful certification. No notification sent.

## Architecture

Keep a persistent Habitat scene, dynamic-object executor, building coordinate
frame and acquisition clock. A small external floor-session coordinator imports
the existing ScheduledTour and selects a complete schedule per floor. It owns
floor transitions; neither baseline algorithms nor the scheduled-tour source
should be copied or modified to emulate this orchestration.

Keep one 2D occupancy map and localization state per floor. Initially use a clean
RTAB-Map process/session switch, with an explicit database path and mapping versus
localization mode, instead of relying on an unverified live database-switch API.
Do not restart the simulator at the floor boundary: that would reset dynamic
objects and change the experimental world.

A building-wide 3D scene graph is a separate consumer. Convert observations through
recorded transforms before using a common building frame; never concatenate
independent local map coordinates. Floor height alone is not a map transform:
independent sessions can differ in horizontal translation and yaw too.

Within one continuous physical session, preserve odometry across normal stair/elevator
travel. Example active chain: building -> map_floor_1 -> odom -> base_link -> camera.
Only one localization authority publishes the active map-to-odom transform. A simulator
teleport or odometry restart is an explicit new odometry epoch, with recorded alignment;
never present a teleport as ordinary odometry. Keep old TF samples/messages out of the
new epoch. ROS REP105 describes per-floor map frames and preserving odometry through
map transitions: https://github.com/ros-infrastructure/rep/blob/master/rep-0105.rst

For a simulator-pose experiment, the Habitat-to-building transform is known and
recorded. For a localization experiment, that transform/initialization must come
from the declared prior map or localization method; simulator truth remains an
evaluation channel and must not silently rescue failed localization.

## Transition sequence

1. Finish the current floor's tour segment and stop accepting its new map/graph
   updates. Drain pending work or explicitly reject late work using floor_id,
   session_id, input timestamp and transform epoch. Save the floor state.
2. Mark TRANSITION. Execute a recorded teleport first for transport-isolation tests,
   or traverse a validated stair path for a continuous-walking experiment. A 2D
   floor planner does not plan the stair leg. Transition RGB-D may be archived but
   must not enter either floor's 2D occupancy grid.
3. Activate the destination floor's database and localizer. Start a fresh map for
   first-visit mapping, or load its preserved map for revisits/localization. Clear
   map-dependent caches, scan matching history and navigation goals. Do not clear
   a valid continuous odometry stream just because the map changes.
4. Establish the destination pose and its building alignment. Require fresh
   timestamped pose data, the intended floor/map identity, unique TF authority and
   a declared localization-quality criterion. A failure leaves acquisition for
   map/graph updates paused; it must not fall back to a different floor or pose source.
5. Commit ACTIVE_FLOOR and resume the imported per-floor tour. The dashboard changes
   the active 2D map while its 3D building view and global replay clock remain stable.

Dynamic script waypoint triggers need floor_id as well as stop/lap because stop0
exists on every floor. Timed script actions retain their declared clock semantics;
do not quietly restart/pause a wall-clock script during localizer startup. Record
all actions during transitions and the observation gap. Choose and record the same
transition policy for every later baseline comparison.

## Existing implementation gaps found directly in this checkout

- tools/baselines/runtime.py: acquire() explicitly selects --floor, imports
  load_schedule(), and sets FEED_TOUR_ALL_FLOORS=0. This adapter is per-floor today.
- Imported Gin habitat_feed_host.py refuses FEED_TOUR_ALL_FLOORS=1. Executed on Gin:
  exit1 with its explicit continuous-session refusal. No baseline or GPU run started.
- ScheduledTour and load_schedule() operate on one storey. Existing schedule files
  do not provide stair-crossing routes.
- run_sim.sh already orchestrates separate whole-stack runs per storey, with separate
  bundles. This isolates maps but is not a persistent-world continuous multi-floor run.
- lost3dsg/launch/habitat_launch.py currently passes a fixed database path and
  --delete_db_on_start. Its localization_mode controls TF selection; that alone does
  not prove database reuse/localization mode. Floor switching must expose and verify
  actual database/mode configuration, especially on a return to an earlier floor.
- Current replay normalization selects one trajectory level from the first scan.
  Before multi-floor replay, extend the common schema with per-floor trajectories,
  floor-qualified events, transitions and coordinate transforms; do not silently
  display only the first floor while replaying a full-building acquisition.

## Certification experiment

Use a different scene from824 with at least two substantial traversable storeys
and multiple rooms. Choose it from actual navmesh/semantic evidence. Run full
per-floor schedules, floor0 -> floor1 -> floor0, without restarting Habitat.
Save a certification bundle containing:

- Exact source/config/container identities, schedules, map/database hashes and floor IDs.
- One monotonic RGB-D/pose recording, per-frame floor/session/transform epoch, plus
  transition and localization-readiness events.
- Dynamic spawn/move/remove events spanning the boundary, demonstrating persistent
  world state and unambiguous floor-qualified waypoint triggers.
- Actual completed scans and skipped/unreachable targets, per-room coverage evidence.
- TF authority/pose-age/localization-quality traces around both transitions.
- Per-floor map contents, graph coordinates and map hashes before/after revisits.

Pass requires completed schedules with zero skipped points; no floor mixing in 2D
maps; successful destination localization with the declared pose source; no late
old-floor measurements accepted; correct full rigid transforms; and a return to
floor0 that reuses the intended floor0 state. Reproduce a stale prior-floor message
and a wrong-map or failed-localization case in an isolated check and verify refusal.

A successful teleport experiment certifies floor/session switching only. It does
not certify autonomous stair navigation; certify that separately with a real
connected stair route and continuous odometry. Report exactly which was tested.

Only after these checks pass, send the orchestrator the bundle path, source hashes,
scene/floor sequence, measured pass/fail results and limitations. Then assess native
Clio/HOV continuous 3D processing separately. Do not inject GT room/floor assignments
into native baseline algorithms or call per-floor output concatenation a native
whole-building result.

## Fallback audit retained

artifacts/baselines/single-floor-scene-audit.json records a read-only scan of ten
annotated candidate scenes, using imported schedule_batch.scene_storeys plus native
Habitat PathFinder on the actual navmeshes. Six pass the deliberately conservative
initial screen; four have flags (minor levels, off-floor sampled area, or goal snap
beyond0.25m). This is not a certified ten-scene cohort or completed baseline runs.
Scene00813 is a different clean candidate:10 annotated regions with floor objects,
42 scan stops,131 trajectory points, all consecutive navmesh routes connected.
Room annotation counts are not a claim that every room has an observed scan.

Gin had only2.5GB free at the initial check. Plan recording/database capacity before
a full run; do not delete other lanes' files. Baseline adaptation remains gated on
GRAPH-API certification regardless of available space.
