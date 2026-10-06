# COGS-3DSG

COGS-3DSG is a ROS 2 system for building and maintaining open-vocabulary 3D scene graphs from RGB-D observations. Its perception pipeline detects and segments objects, projects the resulting observations into 3D, and associates them across frames to maintain a persistent world model. It also models the spatial structure of the environment, considering rooms, walls, doors, and windows.

The repository integrates the pipeline with Habitat simulation and with a physical TIAGo.

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
VLM                    regolo    OpenAI-compatible or Gemini endpoint; vlm.base_url / vlm.model
```

The VLM seam also supports Gemini's Vertex AI `generateContent` API. Use the Vertex base URL and
model shown below; the adapter builds the `/publishers/google/models/<model>:generateContent`
path, sends the image as Gemini `inlineData`, authenticates with
`x-goog-api-key`, and maps the existing structured-scene schema to Gemini's JSON response schema.
The provider is detected automatically from `aiplatform.googleapis.com`, but setting it explicitly
makes the deployment intent clear:

```yaml
vlm:
  provider: gemini
  base_url: "https://aiplatform.googleapis.com/v1"
  model: "gemini-3.8-flash"
  api_key: ""                 # keep credentials in config.local.yaml, or use GEMINI_API_KEY
  thinking_level: low          # sent as generationConfig.thinkingConfig.thinkingLevel
```

`GEMINI_API_KEY` and `GOOGLE_API_KEY` are accepted environment variables. Existing
OpenAI-compatible configurations continue to use the Chat Completions transport unchanged.

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

Among the complete configuration files currently stored in `schedules/configs/`, only the
following are effectively distinct:

| configuration | difference from `01_reference.yaml` |
|---|---|
| `01_reference.yaml` | reference configuration: local detector, rtabmap pose, scheduled exploration and `envelope_size:SizeFilter` |
| `04_ground_truth_pose.yaml` | uses ground-truth pose and disables the admission filter |

A run schedule executes a sequence of configurations, called arms. Each arm either references a
complete YAML file with `config_file`, or modifies the base configuration through `config`; the two
forms cannot be combined. Environment variables may be supplied through `env` when no corresponding
configuration key exists.

Each arm produces its own result bundle. The sweep manifest records the configuration, output
directory and outcome of every run. Arms execute sequentially, and a failed arm does not prevent
the remaining runs from starting.

### Machine-specific settings

`install.sh` discovers the paths and resources available on the current machine and writes them to
`lost3dsg/test/env.local.sh`. Review this file after installation and edit only the values that
could not be detected correctly. The file is not committed because these paths vary between
machines.

If `MODAL_PERCEPTION_URL` is not provided, the installer configures the local perception backend
instead.

| value | what it is |
|---|---|
| `WORKSPACE_ROOT` | the directory holding `maps/`, `runs/` and `results/` |
| `HF_SHARED_CACHE` | where the models are cached |
| `HM3D_ROOT` | the scene library |
| `SAM_MODEL_DIR` | the two EfficientViT-SAM `.onnx` files |
| `IMAGE_TAG` | only if your container image is not tagged `graphapi-run:humble` |
| `MODAL_PERCEPTION_URL` | optional cloud labelling endpoint. It is treated as a credential and is never committed; when it is absent, the installer configures the local perception backend. |

`WORKSPACE_ROOT` defaults to the repository directory. Set it explicitly in
`lost3dsg/test/env.local.sh` only when maps, schedules and results should be stored elsewhere.

### Running without a display

On a machine without an X display, use the headless launcher:

```bash
./run_sim_headless.sh                 # run the whole house
./run_sim_headless.sh --one-storey    # run a single storey
```

The headless launcher accepts the same arguments as `run_sim.sh` and disables RViz and the local
camera preview. The browser overlay remains available because it is rendered into the streamed
image and does not require a local display.

`install.sh` checks whether an X display is available and recommends the appropriate launcher. The
standard launcher also skips RViz automatically when no X socket is detected.

## Configuration

The default shared configuration is `lost3dsg/src/perception_module/config.yaml`. A different
complete YAML file can be selected through `GRAPH_API_CONFIG`. Keys omitted from the selected file
use the defaults defined in `config.py`.

A `config.local.yaml` file placed beside the selected configuration is merged on top of it. The
loader reports the local override and the keys it changes.

Machine-specific paths and credentials are stored separately in `lost3dsg/test/env.local.sh`, which
is generated by `install.sh`. Both local files are excluded from version control.

| file | purpose | version controlled |
|---|---|---|
| `config.yaml` | shared run configuration | yes |
| `config.local.yaml` | local YAML overrides | no |
| `env.local.sh` | machine paths, environment variables and credentials | no |

The main configuration sections are:

| section | purpose |
|---|---|
| `habitat` | scene, exploration, storeys and camera |
| `perception` | detector backend and detection policy |
| `association` | matching observations across frames |
| `similarity` | weights used during object matching |
| `frames` | ROS and camera reference frames |
| `vlm` | vision-language model connection and retries |
| `rooms`, `walls` | room segmentation and wall detection |
| `hooks` | optional extension components |

---

## What a base run does

A base simulation processes each discovered storey as an independent launch. The robot follows
that storey's exploration schedule, and the launch writes its own result bundle and RTAB-Map
database. A top-level manifest records the bundle and outcome associated with each storey.

By default, every storey starts a new mapping session. An existing map is used for localisation only
when `RTABMAP_LOCALIZE_DB` is provided explicitly, in which case the launcher works on a writable
copy of that map. Published per-storey maps may also be used to identify the storeys available in a
scene. Keeping the mapping sessions separate prevents geometry from different floors from being
projected into the same map.

Before the stack starts, the preflight gate validates the selected configuration, mounted source
tree, required models and perception backend. Checks that require live ROS data, such as the camera
transform, feed progress and pose authority, run after startup. A failed or skipped applicable
check prevents the execution from being accepted as a measured run.

The runtime stack consists of the Habitat feed, RTAB-Map, the perception node, the object manager
and the HTTP bridge.

## Exploration schedules

A schedule is one scene's roadmap and the order to walk it. **It is the only motion policy**

**You do not normally generate a schedule by hand.** `run_sim.sh` builds or reuses the scene's
schedule before it starts anything, and caches it in `$WORKSPACE_ROOT/schedules`. Generate one
yourself when you want a variant, a scene the launcher does not know, or the whole dataset at once.

### How a schedule is generated

1. The navmesh is sampled as a top-down grid with resolution `--mpp`. Only the largest connected
   navigable region is retained, with `--robot-radius` used to keep the route clear of obstacles.
2. A generalized Voronoi ridge is extracted from the free space, reduced to a one-pixel-wide
   skeleton and converted into a connected roadmap.
3. Waypoints are placed along the roadmap at `--spacing` intervals. Nearby points are merged using
   `--merge-radius`, and additional stops are added where needed to cover junctions and rooms.
4. The route starts from one of the most connected waypoints. `--route-order` selects either the
   depth-first route or its 2-opt refinement, both measured along the roadmap. Path simplification
   and smoothing preserve navigability.
5. The generated trajectory describes one lap. At each stop the robot normally performs a full
   360-degree scan, and the runtime repeats the trajectory according to `--laps`.

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

The following options control the geometry of the roadmap:

| option | default | purpose |
|---|---:|---|
| `--mpp` | 0.05 m | Resolution of the top-down navigation grid. Smaller cells preserve narrower passages but require more computation and memory. |
| `--robot-radius` | 0.25 m | Minimum obstacle clearance used when extracting the Voronoi ridge. |
| `--spacing` | 2.0 m | Approximate distance between waypoints along the ridge. |
| `--merge-radius` | 0.75 m | Distance below which nearby waypoints are merged; it is also used when deciding whether a junction already has a stop. |
| `--simplify` | 0.20 m | Tolerance used to simplify each route segment. A shortcut is rejected if it leaves navigable space. |
| `--step` | 0.15 m | Forward-motion step used to convert route length into a frame estimate. It should match the simulator's movement step. |
| `--min-area` | 5.0 m² | Minimum navigable area required to generate a schedule for a storey. |
| `--max-bridge` | 2.0 m | Maximum straight-line gap that may be bridged between disconnected ridge components. |
| `--max-room-path` | 12.0 m | Maximum free-space path used to connect a room seed to the roadmap. |

Route construction and repetition are controlled by:

| option | default | purpose |
|---|---:|---|
| `--laps` | 3 | Number of laps used when calculating the schedule budget. It should match `habitat.exploration_laps`, which controls repetition at runtime. |
| `--route-order` | `2opt` | Uses either the depth-first order (`dfs`) or a 2-opt refinement of that order, with distances measured along the roadmap. |
| `--smooth-path` / `--no-smooth-path` | on | Enables or disables removal of intermediate path points when the resulting shortcut remains navigable. |
| `--seed` | 7 | Selects the starting waypoint when several candidates have the same maximum degree. |
| `--revisit-scan-deg` | 0.0° | Rotation performed when the route returns to a waypoint that has already been scanned. |
| `--revisit-offset-m` | 0.0 m | Offset applied to the observation position on later visits to the same waypoint. |

Coverage is a geometric estimate computed from the generated stops. The selected model must be
reported with the resulting coverage value:

| model | definition |
|---|---|
| `los` *(default)* | Counts navigable cells connected to a stop by an unobstructed ray, without a range limit. |
| `los_range` | Applies the same line-of-sight test, limited by `--coverage-range`. |
| `radius` | Counts cells within `--coverage-radius` without checking whether a wall lies between the stop and the cell. |

| option | default | purpose |
|---|---:|---|
| `--coverage-range` | 8.0 m | Range limit used by the `los_range` model. |
| `--coverage-radius` | 3.0 m | Radius used by the `radius` model. |
| `--covering` | off | Creates the `_covering` variant and greedily adds stops in an attempt to reach the coverage target. |
| `--coverage-target` | 1.0 | Target fraction of navigable cells for the covering variant. |
| `--max-extra-stops` | 60 | Maximum number of stops that the covering procedure may add. The target may remain unmet when this limit is reached or no candidate adds coverage. |

The scan options determine how long the robot observes each stop:

| option | default | purpose |
|---|---:|---|
| `--turn-step-deg` | 10.0° | Rotation applied by each turn action. It must match `habitat.turn_step_deg`. |
| `--cycle-seconds` | 4.3 s | Estimated duration of one detection cycle, used to calculate the minimum scan length. |
| `--min-scan-cycles` | 2.0 | Minimum number of detection-cycle durations allocated to a scan. |
| `--fps-for-budget` | 3.0 | Converts the minimum scan duration into frames. Because the resulting floor is executed at runtime, it should match `habitat.fps`. |
| `--adaptive-scan` / `--full-scan` | full scan | Selects an estimated useful viewing arc or a full 360-degree scan. The minimum scan length still applies in both modes. |
| `--stepped-scan` | off | Holds each heading for several frames instead of turning on every frame. |
| `--scan-hold-frames` | 0 | Frames held at each heading in stepped mode. Zero derives the value from `--cycle-seconds` and `--fps-for-budget`. |
| `--scan-tilts` | `0` | Comma-separated camera tilts; the scan is repeated once for each value. |
| `--fps` | 3.0 | Converts the estimated frame count into the duration printed for each storey; it does not change the trajectory. |

`--gt-manifest` supplies HM3D room annotations. When possible, the generator connects a room that
has no roadmap stop to the existing graph and records rooms that could not be reached. Without a
manifest, room-specific seeding is disabled and coverage is evaluated only from the navigation
grid.

## What you get

Each storey launch writes directly to `results/<stamp>_<scene>/`. This directory is both the live
output location and the retained result bundle; no post-run copy is required. A full-house run also
records a manifest linking each storey to its bundle and outcome.

Every bundle contains `run_metadata.json`, logs and the available preflight and termination
records. Depending on the enabled components and how far the launch progressed, it may also contain
detections, perception decisions, captured frames and depth images, cropped images, the knowledge
graph and the RTAB-Map database.

`results/latest` is updated only when the preflight gate passed, the bundle contains measured
output and the teardown source check did not fail.

Persistent run data is stored in host directories mounted into the container. The bundle is mounted
at `/ws/output`, its ROS directory at `/root/.ros`, and the published map library is mounted
read-only from `$WORKSPACE_ROOT/maps`.

The bridge control page is available at `http://localhost:8081` by default. To open the full live
dashboard, start it separately:

```bash
python3 lost3dsg/dashboard/replay_server.py --mode live --port 8086
```

Then open `http://localhost:8086`.

## Scoring a run

Pass the bundle explicitly because the current default in `eval.sh` still points to `runs/latest`:

```bash
./eval.sh results/latest
./eval.sh results/<stamp>_<scene>
./eval.sh --force results/<stamp>_<scene>  # rebuild the ground-truth manifest
```

The evaluation writes its outputs to `<bundle>/eval/` and runs six stages:

1. generate or reuse the scene's ground-truth manifest;
2. join the ground truth with the observations recorded in the bundle;
3. compute the evaluation metrics;
4. generate an HTML visualization of the matched boxes;
5. compute timing metrics, when the required timing data is available;
6. generate a PDF report, when a Python environment with `reportlab` is available.

The scene identity is read from the bundle. Its files are taken from the resolved paths in the
bundle's `config.yaml` when available; otherwise they are located through the local dataset root,
such as `HM3D_ROOT`. The evaluation stops if the resolved scene does not match the scene recorded
in `run_metadata.json`.

## Stopping a run

A base launch has no time limit: `run_sim.sh` rejects `CAP_MIN` outside the dedicated debug mode.
Normally, the launch ends when the storey's scheduled tour and final settling period are complete.
It may also end earlier if a monitored process exits.

To request an early but orderly shutdown, create `feed_ended.json` in the active bundle:

```bash
touch results/<stamp>_<scene>/feed_ended.json
```

A marker without `reason: house_tour_complete` is recorded as an operator abort. The container then
runs its normal shutdown procedure, giving RTAB-Map an opportunity to close its database and run the
post-run checks.

If the stack no longer responds, stop the container directly:

```bash
docker stop -t 150 graphapi_live
```

The default 150-second grace period allows up to 120 seconds for RTAB-Map to close, followed by a
30-second margin. If `RTABMAP_CLOSE_TIMEOUT` is changed, use a Docker timeout of at least that value
plus 30 seconds.

At the end of the launch, `run_sim.sh` copies the termination record into
`run_metadata.json` as `terminating_node.ended`. Its possible values are `tour_complete`,
`operator_abort`, `mapping_time`, `node_death` and `unrecorded`. A monitored process that exits is
classified as `node_death` even when its exit status is zero. `unrecorded` means that the container
did not write a termination record before it stopped.

## Launch structure

`run_sim.sh` contains both the parent launcher and the single-storey execution engine. The parent
selects the requested storeys from `HOUSE_FLOORS`, the published per-storey maps or, in
`--multi-floor` mode, `MULTI_FLOOR_SEQUENCE`. If no storey information is available, it starts one
unconstrained mapping run.

For each storey or ordered floor visit, the parent invokes `run_sim.sh` again with
`GRAPH_API_STOREY_CHILD=1`. The child creates the bundle, runs the preflight checks, starts the
Habitat feed and launches the ROS stack in the container. In the default mode, each storey gets an
independent feed, container and RTAB-Map session. With `--multi-floor`, the Habitat world and
acquisition clock persist across visits, while each visit still receives its own RTAB-Map session
and bundle.

| script | purpose |
|---|---|
| `install.sh` | prepares the local configuration and required resources |
| `run_sim.sh` | runs the simulation, one bundle per storey or ordered floor visit |
| `run_sim_headless.sh` | invokes `run_sim.sh` with the local graphical interfaces disabled |
| `run_tiago.sh` | runs the stack with a physical TIAGo or a TIAGo RGB-D bag |
| `eval.sh` | evaluates an existing result bundle |
| `lost3dsg/test/live_stack_container.sh` | starts and monitors the ROS processes inside the container |

`lost3dsg/test/live_run.sh` and `lost3dsg/test/run_house.sh` no longer exist. Instructions that
refer to either script are obsolete.

## Other things you can run

```bash
./lost3dsg/test/smoke_test.sh      # build, import and startup only; no cloud calls
cd lost3dsg/test && UPDATE_BASELINE=1 ./nonregression.sh && ./nonregression.sh
```

## What `install.sh` checks and configures

Run the installer with:

```bash
./install.sh
```

The script performs the following steps:

1. Reuses a compatible container image when one is already available, or builds the configured
   image from the repository.
2. Locates the Python interpreter for the Habitat renderer and verifies that it can import
   `habitat_sim`. Set `CONDA_PY` to use an interpreter outside the searched Conda installations.
3. Verifies that the MiniLM and CLIP snapshots can be resolved from the Hugging Face cache with
   network access disabled. It also checks that `l2_encoder.onnx` and `l2_decoder.onnx` exist in
   `SAM_MODEL_DIR`. The installer validates these files but does not download them.
4. Discovers the workspace, dataset, model and cache paths and writes them to
   `lost3dsg/test/env.local.sh` when that file does not already exist.
5. Creates `config.local.yaml` with the local perception backend when no cloud endpoint or existing
   local override is available, then validates the resulting setup.

A missing X display does not fail the installation. In that case, the script recommends
`run_sim_headless.sh`; otherwise it recommends the standard simulation launcher.

The top-level launch and utility scripts are executable and can be invoked directly.

## Extension hooks

`lost3dsg/src/perception_module/hooks.py` defines three configurable extension interfaces:

- `Filter`, which accepts, rejects or provisionally admits a new proposal;
- `Refiner`, which may revise an existing object using its neighbouring objects;
- `Store`, which persists object events and exposes object and history queries.

Implementations are selected with `module:ClassName` references and imported dynamically.
`search_paths` may add directories to Python's module search path.

```yaml
hooks:
  search_paths: ["/path/to/extension"]
  filter: "pkg.filter:CustomFilter"
  refiner: "pkg.refiner:CustomRefiner"
  store: "pkg.store:CustomStore"
  decisions_log: ""
```

An empty filter selects the pass-through implementation, an empty refiner disables refinement, and
an empty store uses the built-in SQLite temporal map. The tracked configuration currently selects
`envelope_size:SizeFilter`, whose behaviour is controlled by `size_check`.

A filter receives the label, bounding-box data, appearance attributes, description, room identifier,
available room frames, crop path and observation metadata. Admission and revision records are
written to `<bundle>/hook_decisions.jsonl` unless `decisions_log` specifies another path. Each
admission receives a `decision_id`; when it creates an object, a later `link` record associates that
decision with the resulting object identifier.

The built-in self-checks can be run with:

```bash
python3 lost3dsg/src/perception_module/hooks.py
python3 lost3dsg/src/perception_module/box_view.py
```

## Layout

| path | purpose |
|---|---|
| `lost3dsg/src/perception_module/` | perception, association, room modelling, persistence, configuration, extension hooks and visualization code |
| `lost3dsg/msg/`, `lost3dsg/srv/` | ROS 2 messages and services; `ObjectDescription.msg` includes `crop_path` |
| `lost3dsg/test/` | simulation feed, container launcher, preflight checks, monitoring tools and test scripts |
| `run_sim.sh` | simulation parent launcher and single-storey execution engine |
| `run_sim_headless.sh` | headless wrapper for `run_sim.sh` |
| `run_tiago.sh` | physical TIAGo and RGB-D bag launcher |
| `eval.sh` | offline bundle evaluation |
| `schedules/` | run schedules, complete run configurations and generated exploration schedules |
| `Dockerfile` | container image definition |

## Common early termination signals

| signal | meaning | what to check |
|---|---|---|
| `preflight.json` reports `verdict: fail` | an applicable preflight or post-start check failed | inspect the failed probe and its recorded evidence |
| `perception.log` reports `[VLM] ... strike N/M` | the scene-labelling call failed on consecutive cycles | check the configured VLM service; `perception.vlm_strikes_max` controls termination |
| `perception.log` reports `[SEGMENTATION] ... strike N/M` | the remote segmentation backend failed on consecutive cycles | check the selected backend; `perception.detector_strikes_max` controls termination |
| `om6.log` reports `[INPUT] ENDING THE RUN` | no `/bbox_3d` message arrived after the configured timeout and required robot stops | inspect `perception.log` and the feed logs to find where production stopped |
| `rtabmap.log` contains the `_optimizedPoses` assertion | the active RTAB-Map build did not avoid the known localization failure | verify `IMAGE_TAG` and the RTAB-Map build; treat the resulting bundle as incomplete |

The final cause is recorded in `run_metadata.json` under `terminating_node`.
