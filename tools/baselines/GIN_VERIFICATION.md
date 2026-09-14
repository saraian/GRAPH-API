# Gin integration verification — 2026-09-13

Integration root: `/home/phd_student/Musumeci/baseline-integration` on SSH host `Gin`.
All paths below are relative to that root. These are integration checks, not
accuracy, dynamic tracking, timing, full-house coverage or baseline-eligibility results.

The shared source was imported from
`/home/phd_student/Musumeci/perception_parallel_realrun_20260913T0845Z/source`.
Acquisition manifests record the selected module paths and SHA-256 hashes. That
checkout was read-only during this work. Thin `run_scheduled.py` imports were added
to `Musumeci/Clio-Baseline` and `Musumeci/HOV-Baseline`; their algorithms were unchanged.
The selected source can be changed with `--graph-api-root` without editing either
baseline wrapper. The compiler dependency root can be selected separately.

## Current shared acquisition

`verification-824b`: 667 RGB-D/pose frames, 15 trajectory points, 7 completed scans,
one lap, no skipped points. The actual shared controller completed banana spawn,
move at stop 2/lap 0, and removal at stop 4/lap 0. The executor reported no active
objects at completion. Positions and support metadata came from the colleague's
compiled `household_experiments_scene_824.json`; the fixture selects one object's
three actions and adds explicit waypoint triggers. Placement was not recompiled.

The frame manifest and all four file directories have matching counts. Elapsed
frame timestamps strictly increase. `script_result.json` and `object_actions.jsonl`
agree on the three successful actions and their waypoint/lap triggers. This proves
execution; it does not establish whether either baseline recognizes the moved banana.
Diagnostic captures use the robot camera, not the ROS2 wrapper's object-centric views.

`verification-004` additionally exercises two laps: 361 frames, four scans,
move at stop 0/lap 0, remove at stop 0/lap 1, zero skipped points.
`verification-003` exercised all 70 trajectory points/33 scans on the selected
00861 upper floor and a timed wait. Its prototype nominal timestamps are not timing evidence.

## Native HOV-SG

`hov-verification-824/baseline_result.json`: complete, hierarchy requested.
Native `Graph` processed every 25th frame (27 of 667), producing 381418 points
and 184 masks. Saved JSON artifacts contain 1 floor, 2 rooms, 184 objects;
floor-to-room and room-to-object references were independently checked against
all saved IDs. Counts agree with `hov-824.log`.

Image: `sha256:48a13f2774f82874bd642fa34f0b68fd3c8ff7612295046117cdb73bb5c5eaed`.
Models were loaded from existing checkpoint files in a network-disabled container.

## Native Clio

`clio-verification-824/baseline_result.json`: complete on the SAME
`verification-824b` recording used by HOV. The ROS1 bag contains 667 messages on
each of RGB, depth, CameraInfo and TF. Native task messages contain the requested
three names and three feature vectors. The saved backend graph has 1330 nodes,
including 106 uppercase `O` task-clustered objects. Playback rate was 0.5.
The run completed with the current adapters; no post-hoc validator repair was needed.
Image: `sha256:8adb0781cf61385baff635e80ab84a05ba951ea7345ef43b3200eaa0e49c7aa9`.

Earlier `clio-verification-005` completed on prototype `verification-002`:
361 messages on each RGB/depth/CameraInfo/TF topic, three verified task embeddings,
880 backend nodes including 73 native uppercase `O` task-clustered objects.
The saved graph was revalidated after correcting the adapter's initial lowercase
object-prefix expectation. Its revalidation report does not retain an image ID.
This older recording has nominal timestamps and does not verify current timed waits.

The adapter fixes native launch argument forwarding and localhost ROS addressing,
provides camera calibration, and uses Clio's documented task-file interface.
The native list parser reads literal `~{prefix}` and otherwise returns empty tasks.
`clio-verification-003` was corrected to `complete: false` because it had no task
embeddings or task-clustered objects. Segment/pose nodes are insufficient evidence.

## Known limitation and checks

HOV runs `hov-verification-001`, `002`, and `003` on scene 00861 failed in native
`Graph.segment_floors`: only one height peak survived, so indexing the floor pair
raised `IndexError`. Run 003 used the full selected upper-floor tour and produced
399 masks before that failure. No synthetic floor, modified floor heuristic or
replacement hierarchy was introduced. Scene 00824 proves the input adapter can
feed successful native hierarchy construction; it does not resolve the 00861 case.

All nine Python/shell adapter files matched Gin byte-for-byte.
Six executor/transport tests passed, including unreachable/backwards triggers,
lap matching and concurrent timed waits. Ruff and shell syntax checks passed.
Real compiler, batch-generator and schedule-generator `--help` imports succeeded
with the correct dependency root. The ROS2 `run_habitat_script` command is exposed
but was not executed: acquisition calls its shared executor directly.

Only regenerable files from this task's failed attempts were removed for disk space:
Clio 001/002 input bags and HOV 001/002/003 full feature tensors. Logs, configurations,
recordings, point clouds and masks were retained. No commits or pushes were made.
