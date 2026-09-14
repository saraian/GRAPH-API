# Coverage optimization and nominal duration

`python -m tools.baselines.optimize_tours --output artifacts/baselines/cohort-10-optimized`
uses the prepared cohort's physical floor masks, original trajectories and dynamic
scripts. Run with Habitat Python. It writes a new cohort and leaves source artifacts
intact. `--scenes` selects a diagnostic subset; `--fps` defaults to the acquisition
adapter's 3 fps. The output directory must be new.

The optimizer keeps the original 10 cm floor domain, 3 m horizontal range, camera
height and collision visibility rules. It first adds a horizontal and a -45 degree
camera rotation at each original stop through the canonical executor's `scan_plan`.
Then it samples a deterministic 0.5 m lattice, projects candidates to the native
navmesh, and rejects disconnected or cross-storey paths. Greedy set cover chooses
extra viewpoints until the tour sees 99.5% of the candidate set's visible area.
Cheapest geodesic insertion retains original trajectory order and floor transfer
endpoints. Added stop IDs are new; dynamic scripts and existing stop IDs are intact.
The PNG event visit numbers are recalculated after insertion.

The total map coverage denominator never excludes an unseen cell merely because
no candidate can see it. The candidate visibility ceiling is a separate diagnostic,
not a global optimum proof. Planned target poses may differ from the native
follower's actual stopping poses within its arrival tolerance. Static geometry
coverage is not measured acquisition coverage or semantic object recall.

Duration comes from the imported, unchanged `ScheduledTour` and native Habitat
navigation, counting every tick through three laps without rendering or wall-clock
sleep. It includes travel, in-place turns, arrival ticks and both scan rotations.
The first lap excludes a final return to the start, as the native executor does;
subsequent laps include the path back to the first point. `frames / fps` gives a
nominal action-clock estimate. Actual acquisition adds rendering, recording, IPC,
startup and object-script overhead. Native baseline processing is separate. For
multi-floor scenes the reported time sums floor sessions and excludes transfers
and localization/map initialization. A skipped waypoint makes a full-tour duration
invalid. This diagnostic does not certify multi-floor localization or dynamic runs.

Original schedule `budget`, `waypoints` and similar inherited values describe the
source planner. Use the new `duration`, `coverage` and `coverage_optimization`
records and manifest scan counts for optimized plans. The original source is
hash-referenced. No inner Clio or HOV algorithm is modified.

For a separate source checkout (including Gin), pass `--graph-api-root` to select
the canonical executor explicitly. Prepared asset paths must also exist on that
machine. `optimize_cohort_batch` runs isolated CPU processes with `--workers 3` and
can reuse validated completed subsets with `--reuse <cohort.json>`.

Completed ten-scene result: `artifacts/baselines/cohort-10-optimized/`.
The middle and upper floors of 00820 additionally received a 0.25 m refinement;
its driver and original result are retained. Final coverage is 91.599% there and
95.064–99.707% in the other nine scenes. The total nominal action time at 3 fps is
315.95 minutes for one pass per floor, or 948.33 minutes for three laps per floor.
The PNG gallery and `summary.csv` show the per-scene durations. All native dry
navigation checks completed three laps without skipped waypoints. This remains
separate from multi-floor runtime/localization certification.

`verify_optimized_tours` checks the saved grids, native frame counts, floor/scene
sums, stable triggers and source hashes without changing the plans. Negative
controls reject a wrong percentage, wrong duration, altered original point and
duplicate stop ID. Actual sensor tilt was checked with rendered RGB at one824
viewpoint (`artifacts/baselines/coverage-tilt-check/`). The source snapshot records
the planning code before the final CLI portability option and spacing-caption fix.
