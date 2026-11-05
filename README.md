# GRAPH-API

A ROS 2 scene-graph pipeline for Habitat and TIAGO. Requires Bash, Docker and
NVIDIA container support. No host Python or virtual environment is needed.

Commands that use a remote VLM check for an API key before starting the pipeline.
Export `REGOLO_API_KEY` in the terminal that launches the run. Local VLM servers,
TIAGO `--no-perception`, help, dashboards and replay do not need a VLM key.

## CLI help

Every command includes option descriptions and usage examples. The
[complete help reference](#complete-cli-help) is included below.

```bash
./graphapi --help
./graphapi run sim --help
./graphapi run tiago bag --help
./graphapi dashboard --help
./graphapi stop --help
```

## Habitat

Download the [public HM3D example](https://github.com/matterport/habitat-matterport-3dresearch#-downloading-hm3d-v02),
which needs no account. Run these commands from the
repository; change `HM3D_DIR` to choose where the files are stored:

```bash
HM3D_DIR="$HOME/datasets/hm3d_example"
mkdir -p "$HM3D_DIR"
./graphapi --help >/dev/null  # prepare the small CLI Docker image
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$HM3D_DIR:/data" --entrypoint bash graphapi-cli:latest -c '
    set -euo pipefail
    base=https://github.com/matterport/habitat-matterport-3dresearch/raw/main/example
    for archive in hm3d-example-habitat-v0.2.tar hm3d-example-configs.tar \
      hm3d-example-semantic-annots-v0.2.tar hm3d-example-semantic-configs-v0.2.tar; do
      curl -fsSL "$base/$archive" | tar -xf - -C /data
    done
    cp /data/hm3d_annotated_example_basis.scene_dataset_config.json \
       /data/hm3d_annotated_basis.scene_dataset_config.json
  '
./graphapi setup sim --dataset "$HM3D_DIR"
export REGOLO_API_KEY='your-key'
./graphapi run sim --scene hm3d_00861 --detach
```

Already downloaded HM3D? Use `./graphapi setup sim --dataset /path/to/hm3d`.
That folder must contain the scene folders and
`hm3d_annotated_basis.scene_dataset_config.json`. Setup prepares Docker and models;
the download commands above also run in Docker.

<details>
<summary>Download more HM3D scenes</summary>

Request [Matterport dataset access](https://matterport.com/habitat-matterport-3d-research-dataset),
then create a [Matterport API token](https://my.matterport.com/settings/account/devtools).
After the setup above, download the validation split using Habitat's
[official downloader](https://github.com/facebookresearch/habitat-sim/blob/v0.3.3/DATASETS.md#downloading-hm3d-with-the-download-utility).
Replace `TOKEN_ID` and `TOKEN_SECRET` with the token's ID and secret; these are
separate from your VLM API key:

```bash
HM3D_DATA="$HOME/datasets/hm3d"
mkdir -p "$HM3D_DATA"
docker run --rm -it --user "$(id -u):$(id -g)" \
  -v "$HM3D_DATA:/data" --entrypoint python3 graphapi-sim:latest \
  -m habitat_sim.utils.datasets_download --uids hm3d_val_v0.2 --data-path /data \
  --username TOKEN_ID --password TOKEN_SECRET
cp "$HM3D_DATA/scene_datasets/hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json" \
   "$HM3D_DATA/scene_datasets/hm3d/val/hm3d_annotated_basis.scene_dataset_config.json"
./graphapi setup sim --dataset "$HM3D_DATA/scene_datasets/hm3d/val"
```

Use `--uids hm3d` to download all available splits. Register the split containing
your scene and copy its `hm3d_annotated_SPLIT_basis.scene_dataset_config.json` to
`hm3d_annotated_basis.scene_dataset_config.json` inside that split, as above.

</details>

## TIAGO

Supply **your private PAL Docker**. Reuse a prepared container:

```bash
./graphapi setup tiago --container tiago-127-dev
```

To create one, supply these files (not included in this repository) and run
`./graphapi setup tiago --pal-bundle /private/TIAGO_ISO`.

Keep any other files required by your private builder in the same bundle.
Once setup is complete:

```bash
export REGOLO_API_KEY='your-key'
./graphapi run tiago physical --detach
./graphapi run tiago bag /data/bags/example --no-record --detach
```

`physical` connects to a robot and starts RTAB-Map. `bag` uses its recorded map;
add `--map-source slam` for a new map or `--loop` to repeat playback.
The bag folder must contain `metadata.yaml`.

Perception and recording are on by default. `--no-perception` skips detection;
`--no-record` skips recording. TIAGO needs recording space in `/ws/output`.
Movement during detection can discard that cycle.

## View results and stop

```bash
./graphapi dashboard --mode live          # http://127.0.0.1:8082
./graphapi status                         # find RUN_ID
./graphapi logs RUN_ID --follow
./graphapi stop RUN_ID                    # leave the dashboard open
./graphapi status RUN_ID                  # check that shutdown finished
./graphapi dashboard latest --mode replay
./graphapi eval latest
```

Click **ANNOTATIONS** for the latest annotated image. Replace `RUN_ID` with an
ID from `status`. `latest` selects the last completed run.
**Open RViz** uses the active run's camera, map and frames. Habitat shows the
agent position; TIAGO shows the robot model using your private PAL image.

Settings live in `config/`; local paths go in ignored `config/local.yaml`.
Use `./graphapi tools list` for other tools and `./graphapi COMMAND --help`
for more options and examples.

## Complete CLI help

The output below includes every public command and subcommand. Expand a
command to see all its options and examples. Run the same `--help` command
in your terminal for the current help.

<!-- Generated from graphapi_cli.cli; keep in sync with CLI --help. -->

<details>
<summary><code>./graphapi --help</code></summary>

```text
usage: graphapi [-h] [--project DIRECTORY] [--local FILE] [--config FILE]
                [--json]
                {init,setup,run,doctor,batch,status,logs,stop,attach,dashboard,view,eval,tools,launch,baseline,cloud,tiago,maps}
                ...

Run GRAPH-API in Docker: Habitat scenes, TIAGO robots and recorded TIAGO bags.
Use COMMAND --help to see its options and examples.
Replace paths with your own paths and RUN_ID with an ID from ./graphapi status.

positional arguments:
  {init,setup,run,doctor,batch,status,logs,stop,attach,dashboard,view,eval,tools,launch,baseline,cloud,tiago,maps}
    init                Create local settings. Keep settings that already
                        exist.
    setup               Prepare Docker, dependencies and data paths for a run.
    run                 Start a Habitat scene, a physical TIAGO robot or a
                        recorded TIAGO bag.
    doctor              Check Docker, settings and the files needed for your
                        chosen run.
    batch               Run the experiments listed in a YAML schedule, one
                        after another.
    status              Show running or starting runs and dashboards. Use
                        --all to include stopped runs.
    logs                Read the startup log for a run. Use --follow to see
                        new lines as they arrive.
    stop                Stop a run or dashboard and let it finish saving its
                        files.
    attach              Watch a run log until the run stops. Press Ctrl+C to
                        stop watching.
    dashboard           Open the web dashboard. The default address is
                        http://127.0.0.1:8082.
    view                Open RViz on this computer for a running pipeline.
    eval                Evaluate saved run results inside Docker. Write
                        reports into the run folder.
    tools               Find and run the other tools included in this
                        repository.
    launch              Start a TIAGO ROS launch file. Gazebo uses the public
                        image; bag launches need private PAL Docker.
    baseline            Run or prepare comparison methods using their existing
                        tools.
    cloud               Deploy or run the existing perception service on
                        Modal.
    tiago               Manage private PAL Docker and the physical robot
                        network.
    maps                List the saved RTAB-Map databases in the workspace.

options:
  -h, --help            show this help message and exit
  --project DIRECTORY   repository folder; normally found automatically
  --local FILE          local paths and Docker settings (default:
                        config/local.yaml)
  --config FILE         pipeline settings; otherwise use the default for the
                        chosen run
  --json                print results as JSON; send progress messages to
                        stderr

Examples:
  Prepare Habitat:
    ./graphapi setup sim --dataset /path/to/hm3d

  Run a Habitat scene in the background:
    ./graphapi run sim --scene hm3d_00861 --detach

  Run a recorded TIAGO bag (requires private PAL Docker):
    ./graphapi run tiago bag /data/bags/example --no-record --detach

  Open the live dashboard:
    ./graphapi dashboard --mode live

  Find a run and stop it:
    ./graphapi status
    ./graphapi stop RUN_ID
```

</details>

<details>
<summary><code>./graphapi init --help</code></summary>

```text
usage: graphapi init [-h] [--workspace DIRECTORY]

Create local settings. Keep settings that already exist.

options:
  -h, --help            show this help message and exit
  --workspace DIRECTORY
                        folder for results, maps, schedules and build files

Examples:
  Create config/local.yaml:
    ./graphapi init

  Choose where to save run files:
    ./graphapi init --workspace /data/graphapi
```

</details>

<details>
<summary><code>./graphapi setup --help</code></summary>

```text
usage: graphapi setup [-h] [--dataset DATASET] [--models MODELS]
                      [--cache CACHE] [--workspace WORKSPACE]
                      [--pal-bundle PAL_BUNDLE] [--container CONTAINER]
                      [--baseline-repos BASELINE_REPOS] [--rebuild]
                      [{sim,gazebo,tiago,baselines,cloud,all}]

Prepare Docker, dependencies and data paths for a run.
Missing public dependencies and model files may be installed or downloaded.
TIAGO needs your private PAL container or private build files.

positional arguments:
  {sim,gazebo,tiago,baselines,cloud,all}
                        what to prepare (default: sim); all prepares every
                        listed mode

options:
  -h, --help            show this help message and exit
  --dataset DATASET     HM3D folder containing scene folders and the scene
                        dataset config JSON
  --models MODELS       folder for the VitSAM encoder and decoder model files
  --cache CACHE         folder for downloaded Hugging Face models
  --workspace WORKSPACE
                        folder for results, maps, schedules and build files
  --pal-bundle PAL_BUNDLE
                        private PAL build folder; see the TIAGO section in
                        README.md
  --container CONTAINER
                        name of your private PAL container (for example:
                        tiago-127-dev)
  --baseline-repos BASELINE_REPOS
                        folder containing the comparison-method repositories
  --rebuild             rebuild public Docker images even if they already
                        exist

Examples:
  Prepare Habitat:
    ./graphapi setup sim --dataset /path/to/hm3d

  Choose model storage and the workspace:
    ./graphapi setup sim --dataset /path/to/hm3d --models /data/models --workspace /data/graphapi

  Register private PAL build files:
    ./graphapi setup tiago --pal-bundle /private/TIAGO_ISO

  Use a prepared private PAL container:
    ./graphapi setup tiago --container tiago-127-dev
```

</details>

<details>
<summary><code>./graphapi run --help</code></summary>

```text
usage: graphapi run [-h] {sim,tiago,bag} ...

Start a Habitat scene, a physical TIAGO robot or a recorded TIAGO bag.
Perception and recording are on by default. Use --detach to run in the background.
Export REGOLO_API_KEY='your-key' before starting. A missing VLM key stops startup.

positional arguments:
  {sim,tiago,bag}
    sim            Run a Habitat scene. By default, use the simulator position
                   and run without windows.
    tiago          Choose how to run TIAGO. Both choices require your private
                   PAL Docker.
    bag            Older spelling of run tiago bag. Prefer run tiago bag.

options:
  -h, --help       show this help message and exit

Examples:
  Habitat:
    ./graphapi run sim --scene hm3d_00861 --detach

  Physical TIAGO:
    ./graphapi run tiago physical

  Recorded TIAGO bag:
    ./graphapi run tiago bag /data/bags/example --no-record --detach
```

</details>

<details>
<summary><code>./graphapi run sim --help</code></summary>

```text
usage: graphapi run sim [-h] [--profile NAME] [--config FILE] [--gpu INDEX]
                        [--gui] [--no-record] [--no-perception] [--detach]
                        [--scene NAME] [--one-storey]
                        [--floor HEIGHT | --floors HEIGHTS | --visits HEIGHTS]
                        [--transforms FILE] [--tour-schedule FILE]
                        [--pose {simulator,rtabmap}] [--map FILE]
                        [--mapping-only]

Run a Habitat scene. By default, use the simulator position and run without windows.
You do not need --profile sim-gt: that is the default.
Run setup sim first. Full perception needs REGOLO_API_KEY in your terminal.

options:
  -h, --help            show this help message and exit
  --profile NAME        use a named settings file from config/; sim-gt is
                        already the default
  --config FILE         use this YAML settings file for the run
  --gpu INDEX           GPU number to use, for example 0
  --gui                 open visualization windows on this computer
  --no-record           do not save another ROS topic recording; keep normal
                        results and logs
  --no-perception       run mapping without object detection; no annotations
                        will be produced
  --detach              run in the background and print an ID for status, logs
                        and stop
  --scene NAME          scene to load, for example hm3d_00861
  --one-storey          run one floor instead of separate runs for each saved
                        floor
  --floor HEIGHT        run one floor at this height in metres; for negative
                        values use --floor=-1.59
  --floors HEIGHTS      run these floor heights, for example --floors="-1.59
                        1.21"
  --visits HEIGHTS      visit floors in this order, including repeats; for
                        example --visits="0 2.8 0"; needs --transforms
  --transforms FILE     JSON file describing how each floor map fits into the
                        building
  --tour-schedule FILE  JSON movement schedule for the scene or floor visits
  --pose {simulator,rtabmap}
                        where the camera position comes from (default:
                        simulator)
  --map FILE            use an existing RTAB-Map database; work on a copy,
                        keeping the original
  --mapping-only        build a map without object detection

Examples:
  Run one scene in the background:
    ./graphapi run sim --scene hm3d_00861 --one-storey --detach

  Open visualization windows without saving another recording:
    ./graphapi run sim --scene hm3d_00861 --gui --no-record

  Choose one floor by height in metres:
    ./graphapi run sim --scene hm3d_00861 --floor=-1.59

  Build a map without object detection:
    ./graphapi run sim --scene hm3d_00861 --mapping-only
```

</details>

<details>
<summary><code>./graphapi run tiago --help</code></summary>

```text
usage: graphapi run tiago [-h] {physical,bag} ...

Choose how to run TIAGO. Both choices require your private PAL Docker.
physical connects to a robot. bag replays a recording without a robot.

positional arguments:
  {physical,bag}
    physical      Connect to a physical TIAGO robot using private PAL Docker.
    bag           Replay a recorded TIAGO bag using private PAL Docker. No
                  robot connection is needed.

options:
  -h, --help      show this help message and exit

Examples:
  Physical robot:
    ./graphapi run tiago physical --detach

  Recorded bag:
    ./graphapi run tiago bag /data/bags/example --no-record --detach
```

</details>

<details>
<summary><code>./graphapi run tiago physical --help</code></summary>

```text
usage: graphapi run tiago physical [-h] [--profile NAME] [--config FILE]
                                   [--gpu INDEX] [--gui] [--no-record]
                                   [--no-perception] [--detach]
                                   [--container NAME] [--resume]
                                   [--map-source {slam,robot}]

Connect to a physical TIAGO robot using private PAL Docker.
RTAB-Map builds the application map by default.
Perception and recording are on. Full perception needs REGOLO_API_KEY.

options:
  -h, --help            show this help message and exit
  --profile NAME        use a named settings file from config/
  --config FILE         use this YAML settings file for the run
  --gpu INDEX           GPU number to use, for example 0
  --gui                 open visualization windows on this computer
  --no-record           do not save another ROS topic recording; keep normal
                        results and logs
  --no-perception       disable object detection and the object manager; no
                        annotations will be produced
  --detach              run in the background and print an ID for status, logs
                        and stop
  --container NAME      private PAL container name; otherwise use local
                        settings
  --resume              reuse the current TIAGO session and output; does not
                        restart an old run ID
  --map-source {slam,robot}
                        slam: build a map with RTAB-Map (default); robot: use
                        the robot's map and position

Examples:
  Start the robot pipeline in the background:
    ./graphapi run tiago physical --detach

  Show RViz:
    ./graphapi run tiago physical --gui

  Check cameras and mapping without object detection:
    ./graphapi run tiago physical --no-perception --no-record

  Reuse the current TIAGO session and output:
    ./graphapi run tiago physical --container tiago-127-dev --resume
```

</details>

<details>
<summary><code>./graphapi run tiago bag --help</code></summary>

```text
usage: graphapi run tiago bag [-h] [--profile NAME] [--config FILE]
                              [--gpu INDEX] [--gui] [--no-record]
                              [--no-perception] [--detach] [--container NAME]
                              [--resume] [--map-source {slam,recorded}]
                              [--rate RATE] [--loop]
                              BAG_DIRECTORY

Replay a recorded TIAGO bag using private PAL Docker. No robot connection is needed.
Use the map and transforms saved in the bag by default.
Perception and recording are on. Full perception needs REGOLO_API_KEY.
In the dashboard, click ANNOTATIONS to keep the last completed detection visible.

positional arguments:
  BAG_DIRECTORY         TIAGO bag folder containing metadata.yaml

options:
  -h, --help            show this help message and exit
  --profile NAME        use a named settings file from config/
  --config FILE         use this YAML settings file for the run
  --gpu INDEX           GPU number to use, for example 0
  --gui                 open visualization windows on this computer
  --no-record           do not save another ROS topic recording; keep normal
                        results and logs
  --no-perception       disable object detection and the object manager; no
                        annotations will be produced
  --detach              run in the background and print an ID for status, logs
                        and stop
  --container NAME      private PAL container name; otherwise use local
                        settings
  --resume              reuse the current TIAGO session and output; does not
                        restart an old run ID
  --map-source {slam,recorded}
                        recorded: use the bag's map and transforms (default);
                        slam: build a new map with RTAB-Map
  --rate RATE           playback speed: 1 is normal, 0.5 is half speed
                        (default: 1)
  --loop                restart the bag when it ends; keep playing until you
                        stop the run

Examples:
  Test perception without saving another recording:
    ./graphapi run tiago bag /data/bags/example --no-record --detach

  Replay repeatedly until you stop the run:
    ./graphapi run tiago bag /data/bags/example --loop --no-record --detach

  Check cameras and the map without object detection:
    ./graphapi run tiago bag /data/bags/example --no-perception --no-record

  Build a new map with RTAB-Map:
    ./graphapi run tiago bag /data/bags/example --map-source slam

  Play at half speed:
    ./graphapi run tiago bag /data/bags/example --rate 0.5
```

</details>

<details>
<summary><code>./graphapi run bag --help</code></summary>

```text
usage: graphapi run bag [-h] [--profile NAME] [--config FILE] [--gpu INDEX]
                        [--gui] [--no-record] [--no-perception] [--detach]
                        [--container NAME] [--resume]
                        [--map-source {slam,recorded}] [--rate RATE] [--loop]
                        BAG_DIRECTORY

Older spelling of run tiago bag. Prefer run tiago bag.
Replay a recorded TIAGO bag using private PAL Docker. No robot connection is needed.
Use the map and transforms saved in the bag by default.
Perception and recording are on. Full perception needs REGOLO_API_KEY.
In the dashboard, click ANNOTATIONS to keep the last completed detection visible.

positional arguments:
  BAG_DIRECTORY         TIAGO bag folder containing metadata.yaml

options:
  -h, --help            show this help message and exit
  --profile NAME        use a named settings file from config/
  --config FILE         use this YAML settings file for the run
  --gpu INDEX           GPU number to use, for example 0
  --gui                 open visualization windows on this computer
  --no-record           do not save another ROS topic recording; keep normal
                        results and logs
  --no-perception       disable object detection and the object manager; no
                        annotations will be produced
  --detach              run in the background and print an ID for status, logs
                        and stop
  --container NAME      private PAL container name; otherwise use local
                        settings
  --resume              reuse the current TIAGO session and output; does not
                        restart an old run ID
  --map-source {slam,recorded}
                        recorded: use the bag's map and transforms (default);
                        slam: build a new map with RTAB-Map
  --rate RATE           playback speed: 1 is normal, 0.5 is half speed
                        (default: 1)
  --loop                restart the bag when it ends; keep playing until you
                        stop the run

Examples:
  Test perception without saving another recording:
    ./graphapi run tiago bag /data/bags/example --no-record --detach

  Replay repeatedly until you stop the run:
    ./graphapi run tiago bag /data/bags/example --loop --no-record --detach

  Check cameras and the map without object detection:
    ./graphapi run tiago bag /data/bags/example --no-perception --no-record

  Build a new map with RTAB-Map:
    ./graphapi run tiago bag /data/bags/example --map-source slam

  Play at half speed:
    ./graphapi run tiago bag /data/bags/example --rate 0.5
```

</details>

<details>
<summary><code>./graphapi doctor --help</code></summary>

```text
usage: graphapi doctor [-h] [--mode {sim,tiago-physical,tiago-bag,tiago,bag}]
                       [--live]

Check Docker, settings and the files needed for your chosen run.
Use --live for extra checks inside the private TIAGO container.

options:
  -h, --help            show this help message and exit
  --mode {sim,tiago-physical,tiago-bag,tiago,bag}
                        what to check (default: sim); use tiago-physical for a
                        robot or tiago-bag for a recording
  --live                also check Python and model loading in the private
                        TIAGO container

Examples:
  Check Habitat:
    ./graphapi doctor --mode sim

  Check TIAGO bag replay:
    ./graphapi doctor --mode tiago-bag --live

  Check physical TIAGO:
    ./graphapi doctor --mode tiago-physical --live
```

</details>

<details>
<summary><code>./graphapi batch --help</code></summary>

```text
usage: graphapi batch [-h] [--continue-on-failure] [--force] [--dry-run]
                      [--gpus GPUS]
                      FILE

Run the experiments listed in a YAML schedule, one after another.
Skip experiments that already completed unless you use --force.

positional arguments:
  FILE                  YAML file listing the experiments to run

options:
  -h, --help            show this help message and exit
  --continue-on-failure
                        run the next experiment even if one fails
  --force               run completed experiments again instead of skipping
                        them
  --dry-run             show the planned experiments without starting them
  --gpus GPUS           GPU numbers to assign in turn, for example 0,1
                        (default: 0)

Examples:
  See the planned experiments without running them:
    ./graphapi batch config/example.runs.yaml --dry-run

  Run the schedule:
    ./graphapi batch config/full.runs.yaml

  Continue after a failed experiment:
    ./graphapi batch config/full.runs.yaml --continue-on-failure
```

</details>

<details>
<summary><code>./graphapi status --help</code></summary>

```text
usage: graphapi status [-h] [--all] [--json] [RUN_ID]

Show running or starting runs and dashboards. Use --all to include stopped runs.
Copy an ID from this list to use with logs, stop or view.

positional arguments:
  RUN_ID      ID from status; omit to list active runs

options:
  -h, --help  show this help message and exit
  --all       include completed, failed and interrupted operations
  --json      print results as JSON

Examples:
  Show active runs and dashboards:
    ./graphapi status

  Show one run:
    ./graphapi status RUN_ID

  Show past runs too:
    ./graphapi status --all

  Get JSON output:
    ./graphapi status --json
```

</details>

<details>
<summary><code>./graphapi logs --help</code></summary>

```text
usage: graphapi logs [-h] [--follow] [RUN_ID]

Read the startup log for a run. Use --follow to see new lines as they arrive.
If more than one run or dashboard is active, give its ID.

positional arguments:
  RUN_ID      ID from status, or active if exactly one run or dashboard is
              active (default: active)

options:
  -h, --help  show this help message and exit
  --follow    keep watching new log lines; attach already does this

Examples:
  Read a run log:
    ./graphapi logs RUN_ID

  Watch new log lines:
    ./graphapi logs RUN_ID --follow
```

</details>

<details>
<summary><code>./graphapi stop --help</code></summary>

```text
usage: graphapi stop [-h] [--json] [RUN_ID]

Stop a run or dashboard and let it finish saving its files.
Use its ID when several are active. Stopping a run leaves its dashboard open.
DRAINING means shutdown is in progress; check status until it finishes.

positional arguments:
  RUN_ID      ID from status, or active if exactly one run or dashboard is
              active (default: active)

options:
  -h, --help  show this help message and exit
  --json      print results as JSON

Examples:
  Find the ID, then stop that run:
    ./graphapi status
    ./graphapi stop RUN_ID

  Check that shutdown finished:
    ./graphapi status RUN_ID

  Stop the only active run or dashboard:
    ./graphapi stop active
```

</details>

<details>
<summary><code>./graphapi attach --help</code></summary>

```text
usage: graphapi attach [-h] [--follow] [RUN_ID]

Watch a run log until the run stops. Press Ctrl+C to stop watching.
This does not open a shell or stop the run.

positional arguments:
  RUN_ID      ID from status, or active if exactly one run or dashboard is
              active (default: active)

options:
  -h, --help  show this help message and exit
  --follow    keep watching new log lines; attach already does this

Examples:
  Watch a run:
    ./graphapi attach RUN_ID
```

</details>

<details>
<summary><code>./graphapi dashboard --help</code></summary>

```text
usage: graphapi dashboard [-h] [--port PORT] [--host HOST]
                          [--mode {auto,live,replay}]
                          [RUN_ID_OR_FOLDER]

Open the web dashboard. The default address is http://127.0.0.1:8082.
live shows the running pipeline; replay shows saved results.
Click ANNOTATIONS to see the last completed annotated image in live mode.

positional arguments:
  RUN_ID_OR_FOLDER      saved run ID or results folder (default: latest
                        completed run)

options:
  -h, --help            show this help message and exit
  --port PORT           web address port (default: 8082)
  --host HOST           address to listen on (default: 127.0.0.1, this
                        computer only)
  --mode {auto,live,replay}
                        live: running pipeline; replay: saved results; auto:
                        choose at startup (default: auto)

Examples:
  Show a running pipeline:
    ./graphapi dashboard --mode live

  Show the last completed run:
    ./graphapi dashboard latest --mode replay

  Show a specific saved run:
    ./graphapi dashboard RUN_ID --mode replay

  Use another port:
    ./graphapi dashboard --mode live --port 8083
```

</details>

<details>
<summary><code>./graphapi view --help</code></summary>

```text
usage: graphapi view [-h] [RUN_ID]

Open RViz on this computer for a running pipeline.
If several pipelines are running, give the ID from status.

positional arguments:
  RUN_ID      run ID; omit when only one pipeline is running

options:
  -h, --help  show this help message and exit

Examples:
  Open RViz for the only running pipeline:
    ./graphapi view

  Open RViz for a specific run:
    ./graphapi view RUN_ID
```

</details>

<details>
<summary><code>./graphapi eval --help</code></summary>

```text
usage: graphapi eval [-h] [--force] [RUN_ID_OR_FOLDER]

Evaluate saved run results inside Docker. Write reports into the run folder.
latest means the last completed run, not a run still in progress.

positional arguments:
  RUN_ID_OR_FOLDER  run ID or results folder (default: latest completed run)

options:
  -h, --help        show this help message and exit
  --force           rebuild ground-truth data even if it already exists

Examples:
  Evaluate the last completed run:
    ./graphapi eval latest

  Evaluate a specific run:
    ./graphapi eval RUN_ID

  Rebuild the ground-truth data used for evaluation:
    ./graphapi eval RUN_ID --force
```

</details>

<details>
<summary><code>./graphapi tools --help</code></summary>

```text
usage: graphapi tools [-h] {list,run} ...

Find and run the other tools included in this repository.

positional arguments:
  {list,run}

options:
  -h, --help  show this help message and exit

Examples:
  List available tools:
    ./graphapi tools list

  Show tests and internal tools too:
    ./graphapi tools list --all

  Preview a tool command:
    ./graphapi tools run --dry-run launch-simulation
```

</details>

<details>
<summary><code>./graphapi tools list --help</code></summary>

```text
usage: graphapi tools list [-h] [--all]

List tool names and where they run.

options:
  -h, --help  show this help message and exit
  --all       include tests and tools used internally by the pipeline

Examples:
  List user-facing tools:
    ./graphapi tools list

  Include tests and internal tools:
    ./graphapi tools list --all
```

</details>

<details>
<summary><code>./graphapi tools run --help</code></summary>

```text
usage: graphapi tools run [-h] [--dry-run] NAME ...

Run a tool by its name from tools list.
Put tool options after -- to pass them to the tool.

positional arguments:
  NAME          tool name from ./graphapi tools list
  TOOL_OPTIONS  options for the tool; put -- before them

options:
  -h, --help    show this help message and exit
  --dry-run     print the command without starting the tool

Examples:
  Preview a launch command:
    ./graphapi tools run --dry-run launch-simulation -- map_yaml:=/data/maps/office.yaml

  Ask a tool for its own help:
    ./graphapi tools run schedule-runs -- --help
```

</details>

<details>
<summary><code>./graphapi launch --help</code></summary>

```text
usage: graphapi launch [-h] name ...

Start a TIAGO ROS launch file. Gazebo uses the public image; bag launches need private PAL Docker.
Use run tiago bag for the main recorded-bag pipeline.

positional arguments:
  name            tiago-gazebo, tiago-navigation or tiago-bag-slam; older
                  launch names also work
  LAUNCH_OPTIONS  ROS arguments after --, written as NAME:=VALUE

options:
  -h, --help      show this help message and exit

Examples:
  Start TIAGO in Gazebo:
    ./graphapi launch tiago-gazebo -- map_yaml:=/data/maps/office.yaml

  Show arguments for a bag launch:
    ./graphapi launch tiago-bag-slam --help

  Replay a bag with the existing SLAM launch:
    ./graphapi launch tiago-bag-slam -- bag_path:=/bags/example
```

</details>

<details>
<summary><code>./graphapi baseline --help</code></summary>

```text
usage: graphapi baseline [-h] {run,acquire,pair,remote} ...

Run or prepare comparison methods using their existing tools.

positional arguments:
  {run,acquire,pair,remote}

options:
  -h, --help            show this help message and exit

Examples:
  Run a comparison method on a recording:
    ./graphapi baseline run clio /data/recording /data/clio-results

  See baseline run options:
    ./graphapi baseline run --help
```

</details>

<details>
<summary><code>./graphapi baseline run --help</code></summary>

```text
usage: graphapi baseline run [-h] {clio,hovsg,dynamicgsg} RECORDING OUTPUT ...

Run a comparison method on a recording. Save its results in OUTPUT.

positional arguments:
  {clio,hovsg,dynamicgsg}
                        comparison method to run
  RECORDING             input recording folder
  OUTPUT                folder for the comparison results
  METHOD_OPTIONS        extra options after --

options:
  -h, --help            show this help message and exit

Examples:
  Run CLIO:
    ./graphapi baseline run clio /data/recording /data/clio-results

  Run HOV-SG:
    ./graphapi baseline run hovsg /data/recording /data/hovsg-results
```

</details>

<details>
<summary><code>./graphapi baseline acquire --help</code></summary>

```text
usage: graphapi baseline acquire [-h] ...

Collect input data using the existing baseline acquisition tool.

positional arguments:
  TOOL_OPTIONS  options for the existing tool; put -- before them

options:
  -h, --help    show this help message and exit

Examples:
  Show the acquisition tool options:
    ./graphapi baseline acquire -- --help
```

</details>

<details>
<summary><code>./graphapi baseline pair --help</code></summary>

```text
usage: graphapi baseline pair [-h] ...

Compare two runs using the existing baseline pairing tool.

positional arguments:
  TOOL_OPTIONS  options for the existing tool; put -- before them

options:
  -h, --help    show this help message and exit

Examples:
  Show the pairing tool options:
    ./graphapi baseline pair -- --help
```

</details>

<details>
<summary><code>./graphapi baseline remote --help</code></summary>

```text
usage: graphapi baseline remote [-h] ...

Run the existing baseline tool on a remote machine.

positional arguments:
  TOOL_OPTIONS  options for the existing tool; put -- before them

options:
  -h, --help    show this help message and exit

Examples:
  Show the remote tool options:
    ./graphapi baseline remote -- --help
```

</details>

<details>
<summary><code>./graphapi cloud --help</code></summary>

```text
usage: graphapi cloud [-h] {deploy,run,serve} ...

Deploy or run the existing perception service on Modal.

positional arguments:
  {deploy,run,serve}  Modal action to perform
  OPTIONS             extra Modal options after --

options:
  -h, --help          show this help message and exit

Examples:
  Deploy the service:
    ./graphapi cloud deploy

  Run the service:
    ./graphapi cloud run
```

</details>

<details>
<summary><code>./graphapi tiago --help</code></summary>

```text
usage: graphapi tiago [-h]
                      {network,check,build,shell,start,new,stop,attach} ...

Manage private PAL Docker and the physical robot network.
For new pipeline runs, use run tiago physical or run tiago bag.

positional arguments:
  {network,check,build,shell,start,new,stop,attach}
                        container action, or network to check/apply robot
                        network settings
  OPTIONS               container name or action options; use network status
                        to inspect the network

options:
  -h, --help            show this help message and exit

Examples:
  Check the private container:
    ./graphapi tiago check tiago-127-dev

  Open a container shell:
    ./graphapi tiago shell tiago-127-dev

  Check the physical robot network settings:
    ./graphapi tiago network status
```

</details>

<details>
<summary><code>./graphapi maps --help</code></summary>

```text
usage: graphapi maps [-h] {list}

List the saved RTAB-Map databases in the workspace.

positional arguments:
  {list}      list saved map databases

options:
  -h, --help  show this help message and exit

Examples:
  List saved maps:
    ./graphapi maps list
```

</details>
