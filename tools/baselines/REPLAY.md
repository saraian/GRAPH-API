# Reduced baseline replay dashboard

The same external readers and renderer power the four-panel browser dashboard and
MP4 exports. No baseline inference, clustering, graph-building, segmentation, or
object-update algorithm is modified or invoked by replay.

The dashboard has baseline selection, play/pause, a time slider, playback speed,
video download, and inspectable state/provenance. It has no simulator controls.

## Open the prepared Clio and HOV-SG replays

In the official dashboard, open **Tools → Baseline Replay**.
The active local service is `http://127.0.0.1:8082/baseline-replay/`.
The mount is available in both live and archived dashboard modes. All replay API and
vendor requests stay under that same URL prefix. Override the default two example
models with `BASELINE_REPLAY_MODELS` (a JSON array of paths); optionally set
`BASELINE_REPLAY_VIDEO_ROOT`. Models load lazily when the tool is first opened.
Restart the dashboard service after changing those settings or replacing models.

For a standalone server, from `/DATA/GRAPH-API`:

```bash
python3 -m tools.baselines.replay_server \
  artifacts/baselines/clio.replay.json artifacts/baselines/hovsg.replay.json \
  --video-root artifacts/baselines --port 8097
```

Open `http://127.0.0.1:8097/`. The server binds localhost by default.
The normalized models reference the copied recording images by absolute path.
Keep `artifacts/baselines/source/` with them, or normalize again on the destination
machine. The task's recordings and generated media are gitignored.

## Normalize another run

```bash
python3 -m tools.baselines.replay_model --baseline clio \
  --recording "$RECORDING" --result "$NATIVE_CLIO_RESULT" \
  --schedule "$EXACT_ACQUISITION_SCHEDULE" --output "$NEW_CLIO_REPLAY_JSON"
python3 -m tools.baselines.replay_model --baseline hovsg \
  --recording "$RECORDING" --result "$NATIVE_HOV_RESULT" \
  --schedule "$EXACT_ACQUISITION_SCHEDULE" --output "$NEW_HOV_REPLAY_JSON"
python3 -m tools.baselines.replay_render "$REPLAY_JSON" \
  --video "$NEW_VIDEO.mp4" --fps 10 --speed 1
```

Use `--map-ply` for a Habitat-world point cloud used only as minimap context.
The prepared examples use HOV-SG's saved cloud for both maps. It is not ground
truth and does not enter either baseline or an evaluator.

Dependencies: Python, NumPy, Pillow, FastAPI/Uvicorn, FFmpeg with libx264, and
DejaVu Sans fonts. Baseline model environments and GPUs are unnecessary for replay.
MP4 completion is recorded in its adjacent JSON; the dashboard offers only
completed videos for download.

## Shared format and extension boundary

`replay_model.READERS` contains small external readers. Clio reads native DSG JSON;
HOV reads native floor/room/object JSON and PLY files. The `dashboard` reader accepts
`graph_data.json`, a saved response from the existing dashboard's `/graph_data`
endpoint. It preserves its nodes and edges and converts the map frame to the common
Habitat world frame. That response and the shared acquisition must describe the
same scene. A future baseline needs one reader into this format, not another UI.

`graphapi.baseline_replay.v1` carries ordered frames/poses, acknowledged scheduled
actions, the selected tour, graph nodes/edges, optional recorded graph snapshots,
optional measured metric rows, and source paths/hashes. Visualization uses Habitat
Y-up internally. Clio/dashboard Z-up positions are converted explicitly. The
server's `/api/graph/{id}` converts back to the existing dashboard's Cytoscape
`elements` format, providing a small integration boundary for the full dashboard.
The official dashboard mounts this app at `/baseline-replay/` and adds a Tools link.
Its perception viewer and native baseline algorithms are unchanged.

Optional `--snapshots` is JSONL of actual recorded snapshots with `time_s`,
`scope: "recorded_snapshot"`, `source`, `nodes`, and `edges` in the normalized
coordinate convention. Replay uses the last recorded snapshot at or before the
cursor. Before the first snapshot it shows no graph. It never interpolates a graph.

Optional `--metrics` is JSONL using acquisition-clock `time_s`:

```json
{"time_s": 10.5, "metrics": {"avg_latency_ms": {"value": 83.2, "unit": "ms", "source": "observer/latency.jsonl", "scope": "Measured source-frame receipt to native result publication; 12 matched frames"}}}
```

That is a schema example, not a result from these runs. Supported displayed keys:
`stage_latency_ms`, `objects_added`, `avg_latency_ms`, `accuracy`, `precision`,
`recall`. Each value must identify its unit, source and scope. For quality scores,
scope must identify the task, ground-truth population, matching rule and denominator.
Ground-truth annotations belong in external evaluation; they must not be supplied
to the baseline inference algorithm. A metric row may include a descriptive `label`.

For future live recording, external observers can record existing published
outputs or completed exported files into these same snapshot/metric streams.
These observers and a live mode are not implemented by this replay server.
It serves completed normalized models; it does not claim to follow a running baseline.

## What the prepared videos measure

Both replay the same 667-frame, approximately 63.8-second scene-00824 acquisition.
Object spawn, move at stop 2, and remove at stop 4 come from acknowledged native
controller responses. The first following image is used as the action's display
boundary; the action cannot appear on the preceding frame. Moving backwards on the
slider reconstructs schedule state from the beginning.

Clio's native graph contains 106 object nodes, 100 place nodes and one room node
in the displayed layers, with 544 native edges. HOV's hierarchy contains 184 objects,
two rooms and one floor, with 186 containment edges. Graph labels and links come
from the native outputs. Robot-pose/segment layers in Clio and HOV's navigation
graph are outside this graph panel. The map shows the recorded robot trajectory.

Both graphs are **final snapshots**. They are explicitly labeled and remain fixed;
creation times, deleted nodes, merges and intermediate graph states cannot be
recovered from these exports. Native first-observed timestamps do not establish
when the final clustered graph nodes were created. The camera overlay reprojects
final object bounds, capped at the 12 nearest visible candidates for readability.
It is not per-frame detector output and does not perform depth-occlusion testing.

Clio's `graph/active_window/all_timing_raw.csv` provides 663 positive native stage
timings. Their cumulative mean is shown by source-frame timestamp, ending at
14.168 ms. This excludes neural inference and end-to-end latency. Neither bundle
records the matching/evaluation data needed for accuracy, precision or recall.
Those values and model-object addition counts remain N/A. Schedule counters are
labeled separately from model counts.

## Verification

```bash
python3 -m unittest tools.baselines.test_runtime tools.baselines.test_replay -v
python3 -m ruff check tools/baselines
```

Tests include asymmetric camera-axis/behind-camera projection, exact action
boundaries, backward scrubbing, no future graph snapshots, native timer unit
conversion and rejection of negative durations, actual HTTP graph counts, replay
isolation, missing metrics and refusal of write requests. Artifact-dependent tests
explicitly skip when the real local Gin copies are absent.

Prepared-output verification (2026-09-13): 13 tests passed with no skips; Ruff
passed. Chromium exercised both baselines, play/pause, action-aligned scrubbing,
state reset on switching and mobile layout with zero page errors. Both 639-frame
H.264 exports decode fully at 1600×1000, 10 fps, duration 63.9 seconds. Evidence is
`artifacts/baselines/verification.json` and `browser-check.json`.

The browser server was verified locally. Code is mirrored to Gin; its Habitat
Python environment has NumPy/Pillow but lacks FastAPI/Uvicorn, so running the HTTP
server there requires a separate environment with those dependencies. Local replay
uses read-only copies of Gin's artifacts and does not depend on that environment.

## Native 3D display

The original four-panel replay now embeds native 3D in its graph quadrant, alongside
the recorded camera, dynamic schedule and metrics. There is one playback timeline and
no separate 3D mode link. The 3D camera frustum and trail use recorded poses, including
backward seek resets; final native geometry is not animated into a fabricated history.

The graph quadrant uses different native representations:

- HOV-SG: native object PLY clouds, distinct colors for object IDs, real room/floor
  hierarchy. These colors are display choices, not semantic predictions.
- Clio: native object meshes and RGB colors, with separate object/place/room graph
  layers. Unnamed objects retain their native IDs; no task labels are inferred.
- Both: orbit/pan/zoom, fit, geometry/boxes/graph/layer-label controls, layer separation,
  object selection linked to a single projected camera box, timeline and PNG export.

This is a custom view of native saved results inspired by the supplied references,
not an embedded PyVista/RViz process or a claim to reproduce the paper's exact UI.
The geometry remains explicitly final while the camera timeline advances. Graph edges
are native; vertical graph-layer offsets are presentation only. First appearance is
unrecorded. Selection does not establish visibility or occlusion.

Export native geometry next to the normalized model:

```bash
python3 -m tools.baselines.replay_scene artifacts/baselines/clio.replay.json \
  artifacts/baselines/source/clio-verification-824 artifacts/baselines/clio.replay.scene.json
python3 -m tools.baselines.replay_scene artifacts/baselines/hovsg.replay.json \
  artifacts/baselines/source/hov-verification-824 artifacts/baselines/hovsg.replay.scene.json
```

Exports refuse to overwrite a file. `/api/scene/{id}` verifies the export's model hash
and returns HTTP 409 for a mismatched model. A missing export returns 404; an absent
export does not masquerade as a scene with zero objects. The shared browser uses local
`lost3dsg/dashboard/vendor_three/{three.module.js,OrbitControls.js}`, with no CDN.
New baseline geometry adapters can target `graphapi.baseline_scene.v1` without changing
this viewer or running inference.

Native artifacts inspected on 2026-09-13:

| Baseline | Native objects | Native vertices | Displayed vertices |
|---|---:|---:|---:|
| HOV-SG | 184 | 1,230,066 | 172,810 |
| Clio | 106 | 896,826 | 248,107 |

Counts sum object geometry, including overlaps; they are not unique scene points.
Sampling is deterministic for display only, with original/displayed counts and input
hashes in the sidecar. Clio's native version 1.0.6 uses object-center offsets; the
adapter also honors the native 1.0.0 lower-corner convention.

**Clio artifact limitation:** 35 of these 106 native object meshes have face indices
outside their saved vertex array. The viewer explicitly shows their saved vertices
instead of fabricating topology. The other 71 use native triangles. Source meshes,
face indices, baseline code and algorithm outputs remain untouched. The viewer's
caption, object inspector and provenance disclose this limitation.

## Coverage and recording gaps in the verification fixture

The prepared scene824 input uses 15 trajectory points and 7 scan stops, one lap.
The full source schedule has 137 points and 48 scan stops. The fixture retained stale
full-schedule summary values (`waypoints: 48`, `coverage_share: 1.0`, `navigable_m2: 81.4`);
those values do not measure coverage of the shortened recording. The replay uses the
actual trajectory and recorded scans. Do not change the archived schedule to correct
those summaries: its hash identifies what ran.

The minimap background is a partial HOV reconstruction, not a navmesh. No full-scene
navmesh audit has been completed, and this image cannot establish missing navigation
coverage. Scheduled banana positions are not guaranteed to be visible from the short
tour. Full-experiment validation needs the complete selected tour, a navmesh/topology
export, per-room visit counts and visibility checks at each object-change event.

Current boxes are actual Clio final native AABBs, or bounds computed externally from
HOV's final native object clouds. They are not saved per-frame detections. To track
boxes in appearance order, add external recording of native outputs: input frame ID
and acquisition timestamp, result completion time, native detection/object ID,
2D bbox or mask, 3D geometry, native label/score when present, camera pose/calibration,
and native merge/removal events when exposed. Preserve the distinction between first
detection, first graph insertion and scheduled ground-truth spawn. Existing final-only
bundles cannot reconstruct that history. Such observers must use native output hooks
or topics without altering baseline detection, association or graph algorithms.

Accuracy/precision/recall also require explicit matched evaluation records. Model
additions and inference/end-to-end latency remain N/A until their native outputs are
recorded. The existing Clio stage timer does not substitute for inference latency.

The standalone `/scene` URL remains for compatibility; the dashboard opens the four
quadrants directly. Existing MP4 exports predate the embedded WebGL panel and are
not offered as videos of the updated page. The updated layout was verified in Chromium
on the official8082 service, including synchronized pose/trail motion and mobile stacking.

## Metrics correction (2026-09-13)

`replay_metrics.py` evaluates final native object AABBs against native Habitat GT
in the same Y-up coordinate frame. At 3D IoU >= 0.25 it finds a maximum-cardinality
one-to-one association, then maximizes IoU among those associations. Every prediction
counts in the precision denominator (including missing boxes), and every native GT
instance counts in recall. Duplicate predictions cannot claim the same GT object.
The report includes TP/FP/FN, denominators, pairs, threshold and input/source hashes.

These are **whole-scene class-agnostic final-box diagnostics**, not paper benchmark
scores, semantic accuracy or live detection metrics. Structural instances and unseen
rooms remain in the GT denominator. Clio task clusters and HOV object instances also
have different granularity. Short-tour recall is therefore strongly coverage-sensitive.
Static GT evaluation refuses a final scene with active scheduled dynamic objects.

```bash
python -m tools.baselines.replay_metrics \
  --model artifacts/baselines/clio.replay.json \
  --ground-truth artifacts/baselines/selected-scenes-20260913-v2/00824-Dd4bFSTQ8gi.gt.json \
  --output artifacts/baselines/clio.replay.metrics.json
```

Reports are separate `*.replay.metrics.json` sidecars. Their model SHA must match;
the server refuses a mismatched report. `/api/state` keeps cursor `metrics` separate
from `final_metrics` and exposes `final_evaluation` with matching evidence. Final
scores are labelled as final at every cursor position. Replay models/native geometry
are not rewritten by evaluation. Missing values show specific reasons. Nonfinite,
negative or out-of-range scores and invalid metric clocks are rejected.

For the existing scene824 short tour, both results were evaluated against 392 native
GT objects. Clio: TP20/FP86/FN372, precision18.868%, recall5.102%, F1 8.032%.
HOV: TP39/FP145/FN353, precision21.196%, recall9.949%, F1 13.542%.
These numbers have the restricted diagnostic meaning above.

Native wrappers now write `execution_timing.json` with measured `perf_counter`
wall durations for **future runs**. HOV records model initialization, feature-map
construction, hierarchy and total wall time. Clio total includes startup, slowed
bag playback, configured drain and shutdown/save; it excludes bag conversion.
These durations are not per-frame latency or a fair throughput comparison. Completed
old runs have no new timing retroactively assigned. Normalization imports this file
as a separately labelled final run duration. Per-frame input-to-result latency still
needs paired external observation timestamps; model additions still need native
identity history. No inner baseline algorithm is instrumented or changed.

The selected two-scene cohort and reproduction command are in `SELECTED_SCENES.md`.
