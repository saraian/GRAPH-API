# Scheduled dynamic inputs for Clio and HOV-SG

The baselines import one shared adapter. The adapter imports the selected GRAPH-API
checkout's `ScheduledTour`, `load_schedule`, `DynamicObjectController`, and
`HabitatScriptRunner`. It contains no copied tour, object-action interpreter, or
placement compiler. `scene_script`, `generate_scene_batch`, `schedule_batch`, and
the ROS2 `run_habitat_script` CLI are available through `source-tool`.

Updating the selected checkout changes the next acquisition. A running acquisition
records source hashes and refuses completion if those files change during the run.
Keep the checkout stable while collecting an input. Existing recordings stay fixed:
updating code does not regenerate old RGB-D data.

## Install the thin baseline entry points

Run on Gin from the integration root (currently
`/home/phd_student/Musumeci/baseline-integration`):

```bash
python3 -m tools.baselines.install_entrypoints \
  --integration-root "$PWD" \
  --clio-root /home/phd_student/Musumeci/Clio-Baseline \
  --hovsg-root /home/phd_student/Musumeci/HOV-Baseline
```

Only `run_scheduled.py` is added to each baseline. It imports
`tools.baselines.entrypoint`. Set `BASELINE_INTEGRATION_ROOT` if the shared adapter
moves. `--graph-api-root` separately selects the evolving tour/dataset checkout.
The old Gin `Musumeci/baseline/GRAPH-API` checkout lacks `wait_waypoint`; it must be
updated before it can serve as that source. Validation initially uses the newer
`Musumeci/perception_parallel_realrun_20260913T0845Z/source` tree read-only.

## Generate inputs with the shared tools

Use the Habitat environment for the compiler and schedule generator. The compiler
can have a separate canonical root: Gin's GRAPH-API copy of `scene_script.py` imports
`habitat_value`, which its sibling GRAPH-API `config.py` does not export. Its dataset
submodule has the matching configuration module. Use `--dataset-tools-root` to
import the complete dataset tool dependency set instead of mixing those versions. Their own
`--help` describes current arguments, so this adapter does not maintain a second
copy of their options. The ROS2 runner needs a ROS2 environment.

```bash
python -m tools.baselines.runtime --graph-api-root "$GRAPH_API_ROOT" \
  source-tool schedule_batch --help
python -m tools.baselines.runtime --graph-api-root "$GRAPH_API_ROOT" \
  --dataset-tools-root "$GRAPH_API_ROOT/lost3dsg/FOUND-Dataset" \
  source-tool scene_script --help
python -m tools.baselines.runtime --graph-api-root "$GRAPH_API_ROOT" \
  --dataset-tools-root "$GRAPH_API_ROOT/lost3dsg/FOUND-Dataset" \
  source-tool generate_scene_batch --help
```

`scene_script.py` compiles placements; `generate_scene_batch.py` coordinates their
batch generation; `run_habitat_script.py` wraps the shared `script_runner.py` in
ROS2. Acquisition uses that same executor's callbacks directly, so Clio's ROS1
runtime and HOV-SG's non-ROS runtime do not need a ROS1/ROS2 bridge.

## Acquire once, use the same observations in both baselines

Run with Habitat-Sim installed. Supply matching scene, dataset configuration,
object templates, compiled action script and scene schedule. Select one storey
explicitly; repeat per storey for a full house experiment.

```bash
python -m tools.baselines.runtime --graph-api-root "$GRAPH_API_ROOT" acquire \
  --scene "$SCENE" --dataset "$DATASET_CONFIG" --objects "$OBJECT_CONFIG_DIR" \
  --schedule "$SCHEDULE" --script "$COMPILED_SCRIPT" --floor 1.21 --laps 2 \
  --config "$GRAPH_API_CONFIG" --output "$NEW_RECORDING"
```

The shared script runs concurrently with the tour. Ordinary `wait` actions use wall
time. `at_waypoint: {stop: 3, lap: 1}` waits for the real scan event (zero-based lap).
One thread owns Habitat; worker callbacks submit object commands to it. Failed
actions, impossible triggers, skipped tour points, missing APIs and source changes
prevent `complete: true`. Preflight rejects a compiled script naming another scene.

Outputs include native HOV `rgb/`, millimetre `depth/`, single-line Habitat/OpenGL
camera-to-world `pose/`, float-metre `depth_m/`, ordered `frames.jsonl`, scan events,
object command/results, script result and `acquisition.json`. Extra post-action
captures use the robot camera and are recorded explicitly. They are not the ROS
runner's object-centric diagnostic photographs. Camera HFOV is 90 degrees to match
HOV's native loader. Frame timestamps are acquisition elapsed times.

## Run the native baselines

```bash
bash tools/baselines/gin.sh hovsg "$RECORDING" "$NEW_HOV_RESULT" \
  --skip-frames 10 --record-native-observations
bash tools/baselines/gin.sh clio "$RECORDING" "$NEW_CLIO_RESULT" \
  --rate 1 --segmentation-confidence 0.25 --semantic-mapping-only
```

These launch isolated containers with networking disabled, GPU 0 by default,
read-only source/model mounts and a fresh output directory. Override `BASELINE_GPU`,
`BASELINE_REPOS_ROOT`, `CLIO_IMAGE` or `HOV_IMAGE` when needed. Clio uses localhost ROS
addresses inside its container. Cache FastSAM-x and ViT-B/32 first under the shared
`models/` and `clip/` directories. HOV uses its existing repository checkpoints.

Clio's adapter writes ROS1 RGB, float-metre depth, calibrated CameraInfo and TF. It
rotates the Habitat world to Z-up and converts the camera to ROS optical axes. It
loads native launch/config files, overriding camera calibration and model paths,
then plays the bag and asks native Clio to save its graph on graceful shutdown.
Tasks are passed as YAML files: this native task server's list parser reads the
literal `~{prefix}` parameter and otherwise publishes empty embeddings. Verification
checks the empty task feature inputs and the native lower-case `s` semantic
primitives, including their features, 3-D boxes, meshes, activity state and graph
history. Task-free Clio intentionally emits no task-clustered uppercase `O` nodes.
The launcher gives ROS ten minutes for a large native graph to serialize during
graceful shutdown and rejects runs with less than 95% semantic-image output coverage.

HOV's adapter imports its native `Graph`, loads its native configuration and runs
feature mapping plus hierarchy construction. Its read-only observer saves aligned
SAM masks, mask CLIP features, projected 3-D boxes and per-call timing for every
sampled frame. After native construction it saves the native CLIP category space
and native room-view vote needed by evaluation. It does not replace a failed
hierarchy with a synthetic graph. `--feature-map-only` is an explicit narrower
experiment.

`run_pair.py` can run Clio and HOV-SG sequentially or concurrently on explicitly
distinct physical GPUs. Pack A assigns Clio to GPU 1 and HOV-SG to GPU 0; sharing one
GPU is refused in parallel mode. Both must declare their evaluation artifacts ready, both native repositories must remain clean, and both
end-to-end reports must finish before the pair reaches `evaluation_complete`.
Large reproducible intermediates (`input.bag` and `full_feats.pt`) are checksummed
and removed only after the corresponding native output passes. After both evaluations
and the pair audit pass, the Pack A driver also hashes and reclaims consumed depth and
semantic frame arrays plus the raw Clio output bag. RGB, poses, frame/action/static GT,
graphs, compressed graph history, HOV observations, metrics, configuration and all
cleanup receipts remain, so the reduced dashboard replay and pack aggregation remain
fully reproducible without filling Gin's system disk.

The external evaluator uses the official GRAPH-API table-II/III/IV/VI/VII code and
full HM3D GT. Table V retrieval/navigation is excluded from this experiment. HOV-SG
supports the hierarchy/object tables; task-free Clio reports its unsupported native
floor/room/object-instance tables as unsupported and keeps its semantic-primitive
geometry as a descriptive proxy. Scheduled appearance, movement and removal are
reconstructed from dynamic semantic pixels plus native temporal outputs. No missing
metric is converted to zero.

`pack_a_driver.py` is resumable at scene boundaries and gives each failed native
pair a separate attempt directory while reusing its verified recording. It never
rewrites a failed attempt as success. Pack A records at 3 FPS and explicitly samples
HOV-SG every 50 frames (about every 16.7 seconds); Clio receives every frame. This
adapter sampling policy and both GPU assignments are persisted in the run receipts.
`pack_a_monitor.py` is intended for a systemd
timer with `OnUnitActiveSec=20min`; it verifies or recreates the historical
`clio-gpu` development shell, records GPU/disk/progress state, and restarts an
inactive Pack A driver. It removes only containers named in the current Pack A pair
receipt before a retry. If Pack A's resource journal and native logs all stop changing
for four hours, it restarts that exact driver and its receipted containers. The driver
service also uses a 20-minute failure restart. `freeze_pack.py` hashes every staged
input, adapter/model file and imported simulator source before launch; the driver
rechecks those hashes before every scene. Unknown or frozen-input failures remain
visible in `pack-status.json` and the append-only `monitor.jsonl`.

Gin currently has user linger disabled. For an overnight run, invoke the monitor
with `--direct` from the supplied 20-minute user crontab entry. Direct mode validates
the exact Pack A driver PID and command line, detaches a replacement driver into its
own process group, and applies the same receipted-container cleanup and stale-progress
policy without depending on a login-scoped user service manager.

A complete acquisition or a valid bag is not evidence that a native baseline ran.
Read each result's `baseline_result.json` and logs. HOV's native floor segmenter failed on the tested scene-00861 recordings,
including a complete upper-floor tour, because it identified only one height peak.
Scene 00824 completed hierarchy construction. Failed hierarchies remain failures
even when their object feature maps were saved.

## Checks

```bash
python3 -m unittest tools.baselines.test_runtime -v
python3 -m ruff check tools/baselines
bash -n tools/baselines/gin.sh
```

These check the import/executor seam and failure controls. Real Gin outcomes and
remaining limitations are recorded in the baselines lane handoff, not inferred
from unit-test success.

## Verified Gin fixture

See [GIN_VERIFICATION.md](GIN_VERIFICATION.md) for results and their scope.
To repeat the scene-00824 acquisition with the existing fixture files on Gin:

```bash
cd /home/phd_student/Musumeci/baseline-integration
export GRAPH_API_ROOT=/home/phd_student/Musumeci/perception_parallel_realrun_20260913T0845Z/source
/home/phd_student/miniconda3/envs/habitat_env/bin/python \
  /home/phd_student/Musumeci/HOV-Baseline/run_scheduled.py \
  --graph-api-root "$GRAPH_API_ROOT" acquire \
  --scene /home/phd_student/Musumeci/HOV-Baseline/data/hm3d/val/00824-Dd4bFSTQ8gi/Dd4bFSTQ8gi.basis.glb \
  --dataset /home/phd_student/Musumeci/HOV-Baseline/data/hm3d/hm3d_annotated_basis.scene_dataset_config.json \
  --objects "$PWD/objects/configs" --schedule "$PWD/baseline-824.schedule.json" \
  --script "$PWD/baseline-824.script.json" --config "$PWD/verification.config.yaml" \
  --floor 0.07 --laps 1 --fps 30 --width 640 --height 480 \
  --output "$PWD/verification-824-next"
bash tools/baselines/gin.sh hovsg "$PWD/verification-824-next" "$PWD/hov-824-next" --skip-frames 25
bash tools/baselines/gin.sh clio "$PWD/verification-824-next" "$PWD/clio-824-next" --rate 0.5
```

The fixture selects 15 trajectory points (7 scan stops) and one object's compiled
spawn/move/remove sequence. It verifies integration; it is not a full-house study.
For an experiment, supply the complete schedule and compiled script, and run each
storey separately. The output directories must be new. Gin's disk is nearly full;
account for RGB-D recordings, ROS bags and HOV per-point feature tensors before a
larger run.

## Baseline replay dashboard and videos

The reduced dashboard and four-panel MP4 renderer share the same external bundle
readers. See [REPLAY.md](REPLAY.md) for the prepared Clio/HOV replays, commands,
shared schema, measured metrics and limitations. Replay never modifies native
baseline algorithms.
