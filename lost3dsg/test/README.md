# Running lost3dsg — test harness

Everything runs in the ROS 2 Humble container (`graphapi-run:humble`); the Habitat
simulator renders on the host. All commands below from `lost3dsg/test/`.

## Prerequisites

- Docker image `graphapi-run:humble` (ROS 2 Humble + rtabmap + the Python deps).
- Host conda env with `habitat-sim` (the feed renders on the host GPU).
- HM3D example scenes (public, no token) — e.g. under `/DATA/habitat_matterport/hm3d_example`
  with `hm3d_annotated_basis.scene_dataset_config.json`. `habitat_sim` needs the FULL
  `.basis.glb` path as the scene id.
- EfficientViT-SAM ONNX models and a HuggingFace cache directory (mounted by the
  scripts; adjust the paths at the top of `live_run.sh` for your machine).

## Configuration

All runtime values live in `src/perception_module/config.yaml`; missing keys fall back
to defaults that reproduce the historical behaviour, so a checkout without the yaml
runs as before. Point the env var `GRAPH_API_CONFIG` at an alternate yaml to override
anything without touching the tree (the merge is recursive — a 5-line override file is
enough). `test/smoke_config.yaml` is the config the harness uses.

Key sections: `vlm` (OpenAI-compatible endpoint, model, timeout, retries, fallback
labels used when the endpoint is unreachable), `embedding` (word2vec path), `similarity`
(association weights), `association` (thresholds; `partial_view_fusion` keeps a clipped
re-observation from shrinking an established box), `frames.camera` (MUST be an optical
frame), `paths`, `tf`, `rooms.default_room_id`, `habitat` (scenes; `single_floor` keeps
the tour on the start storey — HM3D navmeshes join storeys through the stairs),
`hooks` (see below).

## Smoke test (build + import + startup)

```bash
docker run --rm --entrypoint bash --gpus all \
  -v /path/to/GRAPH-API:/graph_api:ro -v graphapi_ws:/ws \
  -v /path/to/vitsam:/models/vitsam:ro -v /path/to/hf_cache:/models/hf \
  graphapi-run:humble /graph_api/lost3dsg/test/smoke_test.sh
```

Builds with colcon, checks the message interfaces, imports every node module from the
installed tree, then briefly starts `object_manager_6`. Non-zero exit on first failure.

## Live run on Matterport (full stack)

```bash
./live_run.sh
```

Starts the host-side Habitat feed (`habitat_feed_host.py` — coverage tour with a
mapping phase, then walk/dwell detection) and the container stack: feed node → rtabmap
(RELIABLE QoS on rgb/depth/camera_info/odom and an `odom->base_link` TF are required,
or rtabmap silently receives nothing) → perception → object manager → web viewer.

Watch: **http://localhost:8081** (viewer), `./view_rviz.sh` (rviz in a sibling
container), snapshots and logs in `$OUT_DIR` (default `/tmp/graphapi_live`).

Useful env overrides (defaults in the script): `HM3D_ROOT`, `HABITAT_SCENE`,
`HABITAT_DATASET`, `FEED_SEED`, `FEED_FPS`, `FEED_WALK`/`FEED_DWELL` (frames moving /
stationary per cycle — perception only fires while stationary),
`FEED_MAPPING_SECONDS` (initial pure-mapping tour), `FEED_SHOW`/`FEED_OVERLAY`
(camera window with the belief's 3D boxes projected in, visibility-tested),
`GRAPH_API_CONFIG` (host-side config override), `RTABMAP_GRID_ARGS` (grid hygiene).

## Non-regression

```bash
UPDATE_BASELINE=1 ./nonregression.sh   # once, on a trusted run -> commit the baseline
./nonregression.sh                     # compares object count, distinct labels,
                                       # per-axis belief span (15% tolerance, TOL=)
```

The baseline file is per scene+duration (`baseline_<scene>_<secs>.json`).

## Extension seam (`hooks.py`)

`src/perception_module/hooks.py` ships pass-through blueprints an external package can
subclass, loaded by dotted path from config:

```yaml
hooks:
  search_paths: ["/path/to/your/package"]
  filter:  "pkg.module:ClassName"   # per-proposal admission (default: admit all)
  refiner: "pkg.module:ClassName"   # second look at a node + neighbours (default: none)
  store:   "pkg.module:ClassName"   # persistence adapter (default: SQLite temporal map)
```

Unconfigured, behaviour is unchanged. Decisions and proposed revisions are logged to
`output/hook_decisions.jsonl`. Self-checks: `python3 hooks.py`, `python3 box_view.py`.
