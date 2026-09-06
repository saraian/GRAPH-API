# GRAPH-API — `dev/lost3dsg-cleanup`

The LOST-3DSG perception and world-model stack, on the branch that
[FOUND](https://github.com/EmanueleMusumeci/FOUND) vendors as `vendor/graph-api`. Upstream
LOST-3DSG (paper, authors, ROS 2 install on a real robot) is documented in
[`lost3dsg/README.md`](lost3dsg/README.md). This file documents what this branch adds: a
containerised stack, a Habitat simulator feed, a run harness with a pre-flight gate, cloud
perception, and a generic extension seam that an external package (FOUND) plugs into
through configuration only.

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

## Install

1. **The container image** (~16 GB: ROS 2 Humble, rtabmap, navigation2, torch):

   ```bash
   docker build -t graphapi-run:humble .        # IMAGE_TAG overrides the name
   ```

2. **habitat-sim on the host**, in a conda env named `habitat_env`. The launcher calls
   `$HOME/miniconda3/envs/habitat_env/bin/python` directly. Install per
   https://github.com/facebookresearch/habitat-sim; the renderer needs the GPU and an X
   display (`DISPLAY`, default `:1`).

   The installed package is all `habitat_feed_host.py` needs. The two older host nodes,
   `habitat_camera_node.py` and `habitat_camera_objects_node.py`, also need a habitat-sim
   **source checkout**: each inserts the hardcoded `/root/exchange/habitat-sim/examples`
   into `sys.path` and imports `HabitatSimInteractiveViewer` from `viewer` there. Mount or
   symlink a checkout at that path, at the commit the installed package was built from —
   `examples/viewer.py` calls the Magnum bindings, whose event and renderer classes are
   renamed between versions, so a mismatched tree fails at import. Read the pair with
   `python -c "import habitat_sim; print(habitat_sim.__version__)"`,
   `conda list -n habitat_env habitat-sim` and `git -C <checkout> log -1 --format=%H`.
   The pair our runs use (read on the runner host 2026-09-06): `habitat_sim` **0.3.2**, PyPI wheel,
   conda env `habitat_env` (py3.9, `habitat-sim-mutex 1.0 headless_bullet`, channel `aihabitat`).
   No source checkout exists on the runner and the live path never needs one; for the two older
   host nodes, check out tag **`v0.3.2`**, not a nightly.

3. **Scene data.** Public example scenes, no token:

   | var | default | holds |
   |---|---|---|
   | `HM3D_ROOT` | `/DATA/habitat_matterport/hm3d_example` | `00861-GLAQ4DNUx5U/…basis.glb` + `hm3d_annotated_basis.scene_dataset_config.json` |
   | `MP3D_ROOT` | `/DATA/habitat_matterport/versioned_data/mp3d_example_scene_1.1` | the MP3D example scene |

   Optional: `SAM_MODEL_DIR` (EfficientViT-SAM ONNX weights, local backend only),
   `HF_SHARED_CACHE` (a HuggingFace cache mounted into the container).

4. **Keys.** Export them in the shell that starts the launcher; never write one into a
   tracked file. The launcher passes each into the container by name.

   | var | used for |
   |---|---|
   | `REGOLO_API_KEY` | the VLM. Copied into `OPENAI_API_KEY`, which the OpenAI-compatible client reads. |
   | `MODAL_PERCEPTION_URL` | the detector service when `perception.backend: "modal"`. |
   | `OPENROUTER_API_KEY` | only if `vlm.base_url` points at OpenRouter. |

5. **The Modal detector service** (once per account):

   ```bash
   pip install modal && modal token new
   modal deploy lost3dsg/src/perception_module/cloud/modal_perception.py
   ```

   The deploy prints the endpoint URL; export it as `MODAL_PERCEPTION_URL`. A redeploy
   replaces the running detector for every run that points at that URL, so deploy from a
   committed tree and record the commit. The service runs OWLv2-L + SAM 2.1 + CLIP on an
   L4, per-second billing, scale-to-zero; a warm request is ~100 ms of GPU inside a
   ~1.3-5.5 s HTTP round trip (transport-bound, not compute-bound).

6. **FOUND** (optional, the admission layer). Clone it next to this checkout, or use this
   tree as its submodule; the seam is configured in step "Extension seam" below.

## Configuration

All runtime values live in `lost3dsg/src/perception_module/config.yaml`. Missing keys
fall back to the defaults in `config.py`. Point `GRAPH_API_CONFIG` at another yaml to
override anything; the merge is recursive, so a five-line file is enough. The harness ships
two: `lost3dsg/test/regolo_config.yaml` (the measured configuration: Modal backend, regolo
VLM) and `lost3dsg/test/smoke_config.yaml` (no cloud calls). `CFG_NAME` selects one of them
for `live_run.sh`; the bundle records the resolved values.

| section | keys that matter |
|---|---|
| `vlm` | `base_url`, `model`, `timeout`, `retries` (backoff 1 s / 2 s / 4 s between attempts), `crop_timeout` |
| `perception` | `backend` (`modal` / `managed` / `local`), `modal_endpoint`, `vlm_strikes_max` (default 3: consecutive cycles whose label call failed before the run ends; 0 disables) |
| `association` | `search_radius`, `merge_min_evidence`, `tracking_fallback_radius_m` (locality gate on the tracking path when an object has no covariance yet), `partial_view_fusion` |
| `similarity` | the `lost_similarity` term weights; must sum to 1.0 |
| `frames` | `frames.camera` must be an OPTICAL frame |
| `habitat` | scenes, `single_floor` |
| `hooks` | the extension seam, below |

## Run

All commands from `lost3dsg/test/`. Details in [`lost3dsg/test/README.md`](lost3dsg/test/README.md).

```bash
./live_run.sh                    # hm3d_00861, the default scene; foreground; Ctrl-C archives the bundle
./live_run.sh mp3d_17DRP
MAPPING_ONLY=1 FEED_SPAWN_FLOOR=1.35 ./live_run.sh hm3d_00861   # build the localization map first
./smoke_test.sh                  # build + import + startup, no cloud calls
UPDATE_BASELINE=1 ./nonregression.sh ; ./nonregression.sh      # non-regression against a trusted run
```

**Order of events.** The launcher stamps the source tree and the resolved config, starts the
host feed, then the container: feed node, rtabmap, perception, object manager, bridge. Before
any node starts, `preflight_gate.py` (probes a1-a8) asserts the identity of what is about to
run: the real aligner answered, the config the container loaded is the one the launcher
intended, the executed tree is the mounted one, a detection round-trip completes. A skipped
probe fails the gate. A run localizes against the published map for its scene and floor and
maps only when none exists.

**Cap a run** with `docker stop -t 150 graphapi_live`. The 150 s grace lets rtabmap close its
database (`RTABMAP_CLOSE_TIMEOUT`, 120 s); a shorter grace tears the write.

**Watch:** viewer at http://localhost:8081 (`BRIDGE_PORT`), `./view_rviz.sh` for rviz in a
sibling container. Frequently used feed knobs: `FEED_SEED`, `FEED_FPS`, `FEED_WALK` /
`FEED_DWELL` (frames moving / stationary; perception fires only while stationary),
`FEED_MAPPING_SECONDS`, `FEED_SHOW` / `FEED_OVERLAY` (camera window with the belief's
boxes projected in), `FEED_SPAWN_FLOOR`, `FEED_GT_SEMANTIC`.

**What a run leaves behind:** the bundle under `runs/<stamp>_<scene>/` (when launched from
FOUND; `OUT_DIR` otherwise): `run_metadata.json` (resolved config), `preflight.json`,
`detections.jsonl`, `hook_decisions.jsonl`, `frames/`, `depth/`, `cropped_images/`,
`knowledge_graph.ttl`, `rtabmap.db`, and one log per node under `logs/`.

## Extension seam (`hooks.py`)

`lost3dsg/src/perception_module/hooks.py` ships pass-through blueprints (`Filter`,
`Refiner`, `Reevaluation`, `Store`, `DecisionLog`) that an external package subclasses. They
are loaded by dotted path from config; this tree never imports the package.

```yaml
hooks:
  search_paths: ["/path/to/FOUND"]
  filter:  "found.filter:OntologicalFilter"   # per-proposal admission: ADMIT / REJECT / ABSTAIN
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
| `lost3dsg/test/` | `live_run.sh`, `live_stack_container.sh`, `habitat_feed_host.py`, `preflight_gate.py`, `smoke_test.sh`, `nonregression.sh`, the two configs, `resource_monitor.py` |
| `lost3dsg/msg`, `lost3dsg/srv` | the ROS 2 interfaces; `ObjectDescription.msg` carries `crop_path` |
| `Dockerfile` | the `graphapi-run:humble` image |
| `lost3dsg/src/perception_module/old/` | superseded nodes, not built |

## Known ways a run ends early

| log line | cause | action |
|---|---|---|
| `rtabmap.log`: `[FATAL] Rtabmap.cpp:4090::process() Condition (_optimizedPoses.find(...)) not met`, exit -6 | an rtabmap assertion in localization mode; recurred twice; unreported upstream, no parameter workaround found | the bundle is intact up to that point; re-run, or rebuild the map |
| `perception.log`: `VLM label call failed ... strike N/3` | the VLM endpoint was unreachable; the cycle is skipped and counted | transient: next cycle retries; `vlm_strikes_max` in a row end the run on purpose |
| `om6.log`: `INPUT SILENCE` | no proposals reached the object manager for the configured window | read `feed_host.log` for a disconnected client and `perception.log` for the first frame |
| the pre-flight gate refuses | a probe found the running identity is not the intended one | `preflight.json` names the probe and the numbers |
