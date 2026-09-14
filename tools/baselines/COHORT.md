# Ten-scene baseline input preparation

The owner requested five single-floor and five multi-floor FOUND scenes, complete
tours, expected object-change connectors and coverage of physical free floor space.
The common entry point is:

```bash
MPLCONFIGDIR=/tmp/baseline-mpl python -m tools.baselines.prepare_cohort \
  --output artifacts/baselines/cohort-new
```

Use the Habitat Python environment with local GPU access for native GT and rendered
placement validation. The command imports the canonical FOUND placement generator
and compiler, the canonical floor detector and the shared waypoint-trigger parser.
It performs no LLM request, native baseline inference or scheduled acquisition.

The inputs retain every stop and its order from the canonical per-scene schedules.
Goals are projected with native `PathFinder.snap_point`; the source hash and maximum
displacement are recorded, and a displacement above 0.5 m is refused. Every per-floor
leg and lap closure is checked. Multi-floor candidates additionally need connected
cross-floor navmesh paths and agreement between schedule and detected floor counts.
Candidates with unscheduled minor levels are rejected for this cohort.

Object changes use a banana on compiler-validated furniture supports. Each eligible
floor gets an ordered spawn, relocation and removal with explicit `at_waypoint`
barriers. The plot's robot position is the corresponding scan position; its object
position is read directly from the compiled step. Zero-duration wait steps prevent
fixed settling delays from replacing the waypoint timing. Physics/support/clear-view
placement checks come from the canonical compiler. A floor lacking a valid support
pair is explicitly marked as having no planned changes. Every selected scene must
have at least one complete object-change sequence. These are expected changes, not
execution outcomes; the recording must later capture actual acknowledgement times
and robot/object poses.

`tour_coverage.py` adds a `coverage` element to each floor's schedule and an
area-weighted `coverage` summary to the scene schedule. It uses a 0.10 m grid over
the full static scene bounds. Native semantic floor heights identify floor supports;
collision rays determine supported floor and clear 1.5 m vertical body columns.
There is no navmesh mask or robot-radius erosion in the free-area denominator.
The record also states how much free and covered area lies outside the navmesh.

A cell is covered when a scheduled 360-degree scan can see its floor surface through
a collision-tested line of sight within the schedule's coverage radius (3 m for this
cohort), with a 1.5 m camera, 90-degree horizontal FOV and 640×480 aspect ratio.
Vertical FOV and scheduled tilts are applied. Motion frames and stair transitions
are not credited. Grid resolution, range, camera assumptions, cell counts, area,
percentage, cumulative per-stop area and hashed grid/source references are stored.
This is planned static geometric visibility, not measured camera coverage, semantic
accuracy or a guarantee of observing every object. The original `coverage_share`
field is the older navmesh-radius estimate; use the new `coverage` element for the
physical free-floor measure. Stairs outside a scheduled floor-height band and
missing mesh surfaces are not floor cells in that band's denominator.

PNG colors: green covered free floor, amber uncovered free floor, gray floor
obstacles. Thin triangles show the native navmesh. Cyan lines are complete tours;
purple dashed lines show available cross-floor navmesh paths. Numbered squares
mark robot trigger positions; colored dotted arrows connect them to object changes.
Orange dashed arrows show object relocation. `index.html` links the full PNGs and
`overview.png` provides a contact sheet. `validation.json` records input-plan checks.

Multi-floor baseline execution remains gated on the separate lane's actual GRAPH-API
certification. Geodesic connectivity, compiler success and these PNGs do not certify
stair traversal, localization, dynamic world persistence or native baseline success.
No inner Clio/HOV algorithm is changed by any of these tools.
