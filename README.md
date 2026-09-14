# GRAPH-API — `main`

The LOST-3DSG perception and world-model stack: a containerised ROS 2 stack, a Habitat simulator
feed, a run harness with a pre-flight gate, cloud perception, and a generic extension seam that an
external package plugs into through configuration only.

Upstream LOST-3DSG — the paper, the authors, and the ROS 2 install on a real robot — is documented
in [`lost3dsg/README.md`](lost3dsg/README.md).

**Start at [Quick start](#quick-start): `./install.sh`, `./run_sim.sh` for Habitat, or
`./run_tiago.sh` for a physical TIAGo / TIAGo RGB-D bag.**

## What runs where

```
habitat_feed_host.py   HOST      renders the scene (habitat-sim, GPU + display), walks a coverage
                                 tour, serves RGB-D + pose over TCP :7799, control surface :7790
habitat_feed_node.py   container relays the feed to ROS 2 topics /camera/* and TF
rtabmap                container SLAM; localization mode against a prebuilt map; map->odom TF
perception_2.py        container detection cycle (fires when the robot is stopped): VLM label
                                 call -> detector/segmenter backend -> 3D boxes from depth ->
                                 async VLM crop describer -> /bbox_3d, /object_descriptions
object_manager_6.py    container association (locality, then similarity), merge sweep, the
                                 hooks seam (admission filter), POST to the bridge
graph_api_bridge.py    container FastAPI world model + viewer on :8081
detector backend       Modal     OWLv2 + SAM + CLIP on a cloud GPU (perception.backend: modal),
                                 or the local EfficientViT-SAM path (backend: local)
VLM                    regolo    any OpenAI-compatible endpoint; vlm.base_url / vlm.model
```

## Quick start

Five scripts at the top of the repository. Nothing else is needed for Habitat;
TIAGo additionally needs the private bundle described below.

```bash
./install.sh          # once. Finds what this machine has and writes your settings file.
./run_sim.sh              # a base run: the whole house, every storey, no time limit.
./run_sim_headless.sh     # the same run on a machine with no screen.
./run_tiago.sh physical      # fresh physical TIAGo run.
./run_tiago.sh bag BAG        # fresh offline TIAGo RGB-D bag run.
./eval.sh             # score the newest run against the scene's ground truth.
```

For a physical TIAGo or an offline RGB-D recording, use the concise TIAGo
launcher instead of the Habitat runner:

```bash
./run_tiago.sh physical
./run_tiago.sh bag BAG_NAME
./run_tiago.sh bag BAG_NAME --rtabmap
./run_tiago.sh physical --resume
./run_tiago.sh bag BAG_NAME --rtabmap --resume
```

The concise TIAGo modes are fresh by default: an existing tmux stack is
stopped, `/ws/output` is archived under `/ws/runs/tiago_<timestamp>`, and a new
run is started. Use `--resume` to reuse the current session/output instead.
The compatibility action `start` means resume; `new` explicitly means fresh:

```bash
./run_tiago.sh start  # compatibility resume form
./run_tiago.sh new    # explicit fresh form
```

The public TIAGo runtime files are under [`tiago/`](tiago/). The PAL image,
ISO and keys are private; place them under [`TIAGO_ISO/`](TIAGO_ISO/) as
described in its setup note. Relative bag names use the repository-local
`bags/` directory by default; `TIAGO_BAG_DIR` can override it. The compatibility
launcher `./run_tiago.sh --help` documents physical DDS, bag replay, fresh
RTAB-Map filtering, RViz, VLM authentication and automatic container creation.

`install.sh` **discovers** the container image, the renderer, the scene library, the model cache and
the results directory, then writes them into your settings file and names the one value no search can
find —
the labelling endpoint, which is a credential. Without it, it configures the local detector so a run
works anyway. It ends by telling you which run script this machine needs. It is safe to run again.

`./run_sim.sh hm3d_00861` picks a scene for one run. `./run_sim.sh --one-storey` does a single storey.
**Everything else is a setting, not a flag.**

### One run with a config of your own

```bash
./run_sim.sh --config schedules/configs/01_reference.yaml
./run_sim_headless.sh --config schedules/configs/03_size_gate.yaml
```

`--config` takes any config file. It sets both variables the launcher reads, so the bundle can never
name one file while loading another.

### Several runs, each with its own configuration

```bash
./run_sim_headless.sh --schedule schedules/full.runs.yaml
```

`schedules/full.runs.yaml` is the set of runs we have to perform, and each arm **names its own
complete config file** in `schedules/configs/` — so what runs is the file you reviewed, not a copy
generated from a list of overrides:

| config | what differs |
|---|---|
| `01_reference.yaml` | the reference run: local detector, rtabmap pose, a mapping phase, no size gate |
| `02_no_mapping_phase.yaml` | the control: `mapping_seconds: 0`, which is what recent runs used |
| `03_size_gate.yaml` | the reference plus the size-envelope filter, annotating only |
| `04_ground_truth_pose.yaml` | ground-truth pose, to separate localisation error from perception error |
| `05_noise_floor.yaml` | identical to the reference, run three times |

**`mapping_seconds` is not a duration knob, it is a motion policy.** At 0 the whole run uses the
detection cycle, which gives the agent six frames in ninety-six to move and spends most of those
turning. The archive splits on that one setting: runs with a mapping phase move in 20–26% of frames,
runs without it in 1.0–1.8%. That is why `01` and `02` differ only there.

**`repeat: 3` on the noise-floor arm is a prerequisite, not an extra.** No two archived runs are
comparable *and* normalisable, so the spread between runs of one configuration cannot be recovered
by analysis. Until it is measured, no figure from any arm supports a regression claim.

An arm may instead give `config:` with dotted keys (`perception.backend: local`) to override the
base, and `env:` for the few settings that have no config key yet. **An arm may not give both a
config file and overrides** — with both, a reader of the bundle cannot tell which won.

One bundle per run, and a `manifest.json` naming which arm produced which bundle and how it ended.
A failed arm does not stop the schedule, and the manifest is rewritten after every arm, so a sweep
stopped halfway still says what it did. Arms run in the order written: one GPU, one container.

### What you must fill in

`install.sh` creates the settings file and names these. They are per-machine, so no committed value
can be right for yours.

| value | what it is |
|---|---|
| `WORKSPACE_ROOT` | the directory holding `maps/`, `runs/` and `results/` |
| `HF_SHARED_CACHE` | where the models are cached |
| `HM3D_ROOT` | the scene library |
| `SAM_MODEL_DIR` | the two EfficientViT-SAM `.onnx` files |
| `IMAGE_TAG` | only if your container image is not tagged `graphapi-run:humble` |
| `MODAL_PERCEPTION_URL` | the labelling endpoint. **Required:** both shipped configs set `perception.backend` to `modal`, so without it the run fails its gate at probe a4. It is a credential — the URL alone spends the account's GPU budget — so it is never committed. |

**Never let `WORKSPACE_ROOT` derive itself.** The launcher computes it as two levels above the
checkout, which from this repository is `/` — a run would write its bundle to `/runs` and publish
maps to `/maps`. The launcher refuses that, which is why the value must be set.

### If the machine has no display

**A run will pass its gate and then die.** `rviz`, the camera window and the box overlay all default
to on, and each one aborts without an X server. Measured on a headless lab machine on 2026-09-10:
six attempts, all ended with `rviz2 exited with status -6` **after** preflight had passed, because
the launcher treats a missing node as fatal.

**Use `./run_sim_headless.sh` instead of `./run_sim.sh`.** It takes the same arguments and turns off the two
things that need a window:

```bash
./run_sim_headless.sh                 # the whole house
./run_sim_headless.sh --one-storey    # a single storey
```

It leaves the box overlay on, because the overlay draws into the frame the dashboard serves over
HTTP and opens no window — so you still see the boxes in the browser.

`install.sh` detects whether a display exists and tells you which of the two scripts to use.

## Where settings live

| file | what belongs there | committed? |
|---|---|---|
| `config.yaml` | every choice about a run: the tour, the camera, mapping, perception | yes |
| `config.local.yaml` | your machine's paths and your credentials | **no, and never** |

`config.yaml` is the source of truth. A missing key falls back to the default in `config.py`.
`config.local.yaml` wins key by key and announces which keys it changed on stderr, so a run is never
silently different from the file you read.

**`config.local.yaml` must sit beside the config that is actually LOADED**, which is the one
`GRAPH_API_CONFIG` names — not beside the tracked `config.yaml`. An override next to the wrong file
is ignored without a word.

**Known defect, and it fails the gate: with a local override in force, probe a2 cannot find the
config.** The config module reports the path as `"<config> + <local>"`, and a2 tries to open that
string as a file, so the probe skips and a skipped probe fails the gate. Until that is fixed, a run
that must pass the gate cannot use a local override.

**A credential is not a setting.** The labelling endpoint URL alone spends the account's GPU budget,
so it lives in the local file. It sat in the tracked config once and was therefore committed; that
must not happen again.

*The tracked config is at `lost3dsg/src/perception_module/config.yaml` while it is being moved to
the root, and the local values are still in `lost3dsg/test/env.local.sh`. `install.sh` creates
whichever is current. The plan is in `.handoff/RUN_SURFACE_SPEC_2026-09-10.md`.*

| section | what it decides |
|---|---|
| `habitat` | the scene, how the agent moves, how many storeys it tours, the camera |
| `perception` | which detector backend runs, and what it is allowed to label |
| `association` | when two sightings are the same object |
| `similarity` | the weights of that decision; they must sum to 1.0 |
| `frames` | which TF frame is which — `frames.camera` must be an OPTICAL frame |
| `vlm` | the labelling service, its timeout and its retries |
| `rooms`, `walls` | room splitting and wall detection |
| `hooks` | the extension seam, below |

---

# Reference

Nothing below is a step. It is here to be looked up.

## What a base run does

It tours a storey until its waypoints are exhausted, closes that storey's map, moves to the next
storey and starts a fresh map. **You get one bundle per storey, not one per run**, plus a
`manifest.json` naming every storey with its bundle and how that launch ended.

**Every launch builds its own map.** Nothing localises against a previously published map, so two
storeys are never compared against the same map. That is deliberate: one map spanning two storeys
puts the upper walls on top of the lower rooms. **The map library is therefore unused under this
configuration** — publishing a per-floor map, the canonical read-only mount and asking to localise
against one are all unreachable, and the variable that used to ask for it now refuses rather than
being quietly ignored.

**Before any node starts,** `preflight_gate.py` (probes a1 to a13) asserts that what is about to run
is what was asked for: the config the container loaded is the one the launcher intended, the executed
tree is the mounted one, every model the run loads is already cached, a detection round-trip
completes. A skipped probe fails the gate. Then the nodes start: feed, rtabmap, perception, object
manager, bridge.

## Exploration schedules

A schedule is one scene's roadmap and the order to walk it. **It is the only motion policy** — the
sampling tour that used to stand behind it was removed on 2026-09-11, so a run without a schedule
would publish frames from a robot that never moves. `run_sim.sh` refuses to start one.

**You do not normally generate a schedule by hand.** `run_sim.sh` builds or reuses the scene's
schedule before it starts anything, and caches it in `$WORKSPACE_ROOT/schedules`. Generate one
yourself when you want a variant, a scene the launcher does not know, or the whole dataset at once.

### How it is built

1. The navmesh is rendered as a top-down grid of the storey, one cell per `--mpp`, and eroded by
   `--robot-radius` so a waypoint is somewhere the robot fits.
2. The **generalized Voronoi diagram** of that free space is drawn — the set of points equidistant
   from two or more obstacles, which runs down the middle of every corridor and doorway — and
   thinned to one pixel wide.
3. Waypoints are placed along it every `--spacing`, merged when closer than `--merge-radius`, and
   every junction gets one.
4. The visiting order is a tour over roadmap distances (`--route-order`), starting from the
   busiest junction. The path between two stops is simplified and straightened.
5. Each stop turns a full circle. One lap is the file; the run repeats it `--laps` times.

### The procedure

```bash
PY=$HOME/miniconda3/envs/habitat_env/bin/python      # the environment with habitat-sim

# One scene. --ensure reuses a cached schedule whose settings match, and prints SCHEDULE_FILE=.
$PY lost3dsg/test/schedule_batch.py \
    --navmesh /path/to/scene/NAME.basis.navmesh \
    --scene-id hm3d_00861 --ensure --out-dir "$WORKSPACE_ROOT/schedules"

# Every scene under a root, plus an index.json summarising them.
$PY lost3dsg/test/schedule_batch.py \
    --scene-root /path/to/hm3d-val-habitat-v0.2 --out-dir "$WORKSPACE_ROOT/schedules"

# The covering variant: adds stops until every navigable cell is seen. Separate file.
$PY lost3dsg/test/schedule_batch.py --navmesh ... --covering --ensure --out-dir ...
```

**Measured: 1.3 to 2.0 s a scene, 5 to 15 s with `--covering`** — so all 100 val scenes take about
four minutes plain, and half an hour covering. The output is
`<scene-id>.schedule.json`, or `<scene-id>_covering.schedule.json` for the variant, holding **every
storey of the scene** — a storey is found from a height histogram of navigable samples, and stair
landings and galleries are skipped rather than toured.

**The generator is deterministic.** The same navmesh and the same settings give byte-identical
trajectories; only `scene_id` and `navmesh` change with the path you pass.

**`--ensure` decides on the settings digest, not the file name.** A schedule built at a different
`--merge-radius` is rebuilt rather than reused. **The digest does NOT cover the generator's own
source**, so after changing `voronoi_roadmap.py` or `schedule_batch.py` you must pass
`--regenerate`. `habitat.regenerate_schedule: true` in the config makes `run_sim.sh` do it.

### Parameters

**Geometry — what the roadmap looks like.**

| | default | what it decides |
|---|---|---|
| `--mpp` | 0.05 m | grid cell size. Smaller sees narrower gaps and costs time as its square |
| `--robot-radius` | 0.25 m | free space is eroded by this, so waypoints fit a robot and not a point |
| `--spacing` | 2.0 m | distance between waypoints. Larger means fewer stops and less coverage |
| `--merge-radius` | 0.75 m | two waypoints closer than this become one |
| `--simplify` | 0.20 m | path simplification tolerance. A shortcut leaving free space is rejected |
| `--step` | 0.15 m | the agent's `move_forward`. Turns metres into frames, so it must match the config |
| `--min-area` | 5.0 m² | a storey smaller than this is not toured |
| `--max-bridge` | 2.0 m | the longest gap between two ridge pieces that may be joined |
| `--max-room-path` | 12.0 m | the longest path used to join a seeded room back to the roadmap |

**The route — how long a lap takes.**

| | default | what it decides |
|---|---|---|
| `--laps` | 3 | complete passes. The laps are IDENTICAL: a difference between two is a difference in the world, not the route |
| `--route-order` | `2opt` | nearest neighbour then 2-opt over roadmap distances. `dfs` is the pre-2026-09-11 order, which drives every backtrack |
| `--smooth-path` / `--no-smooth-path` | on | drop any path point its neighbours can see past. Corner turning cost as much as driving before this |
| `--seed` | 7 | chooses the root among equally-connected candidates. Same seed, same schedule |

**Coverage — what "100%" means.** Three models, and a number is meaningless without the model
beside it. Never compare across them.

| `--coverage-model` | what counts as covered |
|---|---|
| `los` *(default)* | what a stop can SEE: the ray to the point is unobstructed, no distance limit. The simulator's depth sensor has none |
| `los_range` | the same ray test, stopped at `--coverage-range` (8.0 m, CHOSEN, not measured) |
| `radius` | free space within `--coverage-radius` (3.0 m), straight-line **through walls**. What every schedule before 2026-09-11 used. Kept so old numbers reproduce, not because it is right |

| | default | what it decides |
|---|---|---|
| `--covering` | off | keep adding stops until `--coverage-target` is met, by greedy set cover. Writes the `_covering` variant: a promise rather than a measurement |
| `--coverage-target` | 1.0 | the share `--covering` tops up to |
| `--max-extra-stops` | 60 | a ceiling on what `--covering` may add, so one bad storey cannot make a schedule nobody can run |

**The scan — and it decides whether a merge can commit at all.** A merge needs
`merge_min_consecutive` (2) consecutive detection cycles on the same pair, and a cycle takes about
4.3 s. A full 360° scan costs `360 / --turn-step-deg` frames at 3 f/s:

| turn step | frames | seconds | cycles |
|---|---|---|---|
| 10° *(default)* | 36 | 12.0 | **2.79** |
| 20° | 18 | 6.0 | **1.40 — no stop can ever merge** |

20° was tried on 2026-09-11 for the 36% of run time scans cost, and reverted the same day. A run
confirmed it: **43 merges before the end of lap zero at 10°, against 5 in a whole tour at 20°.**

| | default | what it decides |
|---|---|---|
| `--turn-step-deg` | 10.0 | degrees per turn action. `habitat.turn_step_deg` must match it, or the budget describes a run that did not happen |
| `--cycle-seconds` | 4.3 | a detection cycle. MEASURED on bundles `20260911_133641` and `_140421`. The scan floor derives from it, so re-measure and pass the new number rather than editing anything else |
| `--min-scan-cycles` | 2.0 | no stop turns through fewer frames than this many cycles. Matches `merge_min_consecutive` |
| `--adaptive-scan` / `--full-scan` | full | adaptive turns only through the arc holding unseen ground. Faster, and it left 649 of 1900 stops below the merge threshold — the MEAN was a comfortable 3.5 cycles and the tail was not |
| `--stepped-scan` | off | hold each heading still for a whole cycle instead of turning every frame. A continuous turn holds a heading for ONE frame, so a scan is a drive-through; stepped makes it a scan |
| `--scan-hold-frames` | 0 | frames per heading under `--stepped-scan`. 0 derives one whole cycle |
| `--scan-tilts` | `0` | one full rotation per tilt. `30,0` is the two-rotation ask and doubles the bill |

**Two frame-rate knobs, and neither changes what the agent does.** `--fps` (3.0) turns the frame
budget into the minutes printed per storey. `--fps-for-budget` (3.0) turns the scan floor into
frames. Both should equal `habitat.fps`, or the schedule's arithmetic describes a different run.

**`--stepped-scan` costs 13x and is off for that reason**: over 71 storeys, three laps, 22.0 h as
built against 150.0 h stepped and 287.5 h with two tilts. The tilt rotates the sensor node and **no
run has confirmed it** — check the published frames actually tilt before quoting a two-rotation run.

**Ground-truth rooms.** `--gt-manifest` takes an HM3D room manifest and adds a stop in any room that
has none. Only **36 of the 100** HM3D val scenes are annotated (`hm3d_annotated_val_basis.
scene_dataset_config.json` lists them); 221 scenes are annotated across all splits. Without the
manifest, `room_seeds` is 0 and only geometric coverage is checked.

## What you get

**One directory per run, in the repository:** `results/<stamp>_<scene>/`. The live output and the
bundle are the same directory — there is no second copy and nothing is moved at the end.

It holds `run_metadata.json` (the resolved settings), `preflight.json`, `detections.jsonl`,
`hook_decisions.jsonl`, `actual_perceptions.json`, `frames/`, `depth/`, `cropped_images/`,
`knowledge_graph.ttl`, `rtabmap.db`, `ros/` (the mapper's own directory) and one log per node under
`logs/`. `results/latest` points at the newest run that passed its gate.

Everything a run reads or writes is a **host directory mounted into the container**, so it is all
reachable from outside: the bundle at `/ws/output`, the mapper's directory at `/root/.ros`, the map
library at `maps/`, the model cache and the build tree. No run data lives in docker-managed storage.

**Watch a run:** the viewer at `http://localhost:8081`.

## Scoring a run

```bash
./eval.sh                    # the newest run
./eval.sh results/<stamp>_<scene>     # a particular one
```

Four steps, into `<bundle>/eval/`: the scene's ground truth, the join to what the run recorded, the
metrics, and an HTML view of the boxes. **The scene comes from the bundle, not from this file** — a
hardcoded scene path is how an evaluation comes to describe a different house.

## Stopping a run

A completed tour is what ends a launch normally. There is no time limit, so nothing else stops it.

**To end one early, create `feed_ended.json` in that storey's bundle directory.** The container
watches for it and closes through the normal archive path, so rtabmap closes its database and the
bundle is complete. `run_metadata.json` then records `terminating_node.ended: operator_abort`, so a
run ended by hand can never be read as a finished house tour.

`docker stop -t 150 graphapi_live` is the last resort, for a stack that has stopped responding. It
kills the nodes where they stand; the grace period lets rtabmap try to close its database, and a
shorter one tears the write.

**How a launch ended is a field, not a log line.** `terminating_node.ended` is one of
`tour_complete`, `operator_abort`, `mapping_time`, `node_death`, `unrecorded`. `unrecorded` does not
mean unknown — it means the container never reached its own end. A watched node exiting with status
0 is `node_death`, not a clean finish: which node stopped decides, not its exit status.

## The layers under `run_sim.sh`

There is one layer fewer than there used to be. `run_sim.sh` now holds the whole launch: it works out
which storeys the house has, then performs one run per storey. It does that by calling itself once
per storey, so each storey still gets a clean process of its own.

| | |
|---|---|
| `install.sh`, `run_sim.sh`, `run_sim_headless.sh`, `run_tiago.sh`, `eval.sh` | what a person runs |
| `lost3dsg/test/live_stack_container.sh` | inside the container. Nobody calls it by hand |

`lost3dsg/test/live_run.sh` and `lost3dsg/test/run_house.sh` were deleted on 2026-09-11. If a step
anywhere names either of them, that step is out of date.

## Other things you can run

```bash
./lost3dsg/test/smoke_test.sh      # build, import and startup only; no cloud calls
cd lost3dsg/test && UPDATE_BASELINE=1 ./nonregression.sh && ./nonregression.sh
```

## What install.sh does, if you would rather do it by hand

1. `docker build -t graphapi-run:humble .` — about 16 GB: ROS 2 Humble, rtabmap, navigation2, torch.
2. A conda environment named `habitat_env` with habitat-sim in it. The launcher calls
   `$HOME/miniconda3/envs/habitat_env/bin/python` by absolute path, so another prefix is not a
   substitute. The renderer needs a GPU and an X display.
3. Cache all five models, then **load each one with the hub switched off** to prove none will be
   fetched during a run. Warming alone is not proof: a cache in the old flat layout is found by a
   file search and not by the loader, which reads `$HF_HOME/hub` and nothing else.
4. Create the local settings file and name every value to fill in.
5. Verify each of the above — including that an X display exists — and refuse with a cause.

**One thing to know about a fresh clone:** the historical simulation entry points may still need
`chmod +x` when checked out from older revisions, or they can be called with `bash`. The new
`run_tiago.sh` is stored executable and can be invoked directly.

## Extension seam (`hooks.py`)

`lost3dsg/src/perception_module/hooks.py` ships pass-through blueprints (`Filter`,
`Refiner`, `Reevaluation`, `Store`, `DecisionLog`) that an external package subclasses. They
are loaded by dotted path from config; this tree never imports the package.

```yaml
hooks:
  search_paths: ["/path/to/extension"]
  filter:  "<pkg>.filter:<Class>"            # per-proposal admission: ADMIT / REJECT / ABSTAIN
  refiner: "pkg.module:ClassName"            # second look at a node + neighbours (default: none)
  store:   "pkg.module:ClassName"            # persistence adapter (default: SQLite temporal map)
```

A proposal carries labels, the 3D box (AABB and PCA-oriented), `room_id`, a tagged room
frame and `crop_path`. Unconfigured, behaviour is unchanged. Every decision is logged to
`output/hook_decisions.jsonl` with a `decision_id` that the object manager writes back on
the object, so a verdict can be joined to the object it judged. Self-checks: `python3
hooks.py`, `python3 box_view.py`.

## Layout

| path | holds |
|---|---|
| `lost3dsg/src/perception_module/` | the nodes (`perception_2.py`, `object_manager_6.py`, `object_services.py`, `room_manager.py`, `graph_api_bridge.py`), `config.py` / `config.yaml`, `hooks.py`, `cloud/` (Modal client + service), `viewer/` |
| `run_sim.sh` | the whole launch: the storeys, the gate, the host feed, the container, the archive |
| `run_tiago.sh` | physical TIAGo or RGB-D bag launch, DDS setup, container bootstrap, RViz and SLAM |
| `lost3dsg/test/` | `live_stack_container.sh`, `habitat_feed_host.py`, `preflight_gate.py`, `smoke_test.sh`, `nonregression.sh`, the configs, `resource_monitor.py` |
| `lost3dsg/msg`, `lost3dsg/srv` | the ROS 2 interfaces; `ObjectDescription.msg` carries `crop_path` |
| `Dockerfile` | the `graphapi-run:humble` image |

## Known ways a run ends early

| log line | cause | action |
|---|---|---|
| `rtabmap.log`: `[FATAL] Rtabmap.cpp:4090::process() Condition (_optimizedPoses.find(...)) not met`, exit -6 | an rtabmap assertion in localization mode; recurred twice; unreported upstream, no parameter workaround found | the bundle is intact up to that point; re-run, or rebuild the map |
| `perception.log`: `VLM label call failed ... strike N/3` | the VLM endpoint was unreachable; the cycle is skipped and counted | transient: next cycle retries; `vlm_strikes_max` in a row end the run on purpose |
| `om6.log`: `INPUT SILENCE` | no proposals reached the object manager for the configured window | read `feed_host.log` for a disconnected client and `perception.log` for the first frame |
| the pre-flight gate refuses | a probe found the running identity is not the intended one | `preflight.json` names the probe and the numbers |
