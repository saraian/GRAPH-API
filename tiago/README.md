# TIAGo launcher

[`../run_tiago.sh`](../run_tiago.sh) is the public entry point for the physical
TIAGo stack and offline TIAGo RGB-D recordings. It supports a concise
`physical`/`bag` interface while retaining the compatibility actions. The files
in this directory are the non-secret runtime layer: ROS helpers, configuration
overlays, the TF conflict relay and the shared RViz layout.

The concise `physical` and `bag` commands start a fresh run by default. If a
previous tmux session or `/ws/output` exists, the launcher stops the session,
archives the output under `/ws/runs/tiago_<timestamp>`, and starts clean. Add
`--resume` when the intention is to reuse the current session and output. The
low-level `start` action remains a compatibility spelling for resume, while
`new` remains the explicit low-level spelling for a fresh run.

The PAL development image is private. Put the private bundle under
`../TIAGO_ISO/`, or set `TIAGO_ISO_DIR`, before the first run. The launcher
expects the private `uni-sap-rome-build-docker.sh`; the ISO and PAL keys remain
private and are never copied into Git. A PAL APT key package is needed when the
private builder has to build a new image.

## Physical TIAGo

```bash
./run_tiago.sh physical
./run_tiago.sh physical --resume
```

The default is `tiago-127-dev`, ROS domain `1`, robot address `10.68.0.1`, and
automatic host DDS preparation. Use `SKIP_TIAGO_HOST_DDS=1` after the host
firewall and route have already been configured. `FOUND_START_RVIZ=1` opens the
shared Habitat-style RViz layout in a separate tmux window. Set
`FOUND_START_PERCEPTION=0` for a dashboard/RTAB-Map-only run.

The private builder is called automatically when the default container does not
exist. Set `TIAGO_AUTO_CREATE=0` to require an already-created container. A
checkout nested at `<FOUND>/vendor/graph-api` uses that outer FOUND root; a
standalone clone gets an ignored mount shim so the private builder still mounts
the current checkout at `/graph_api`.

## Offline RGB-D bag

The default bag root is `<repo>/bags`. A relative bag argument is resolved
there, and the host directory is mounted by the container builder at `/bags`.
The `bags` directory may be a symlink to another storage volume; set
`TIAGO_BAG_DIR` to override it on a different machine. An absolute bag path is
also accepted; its parent directory is used for the mount unless
`TIAGO_BAG_DIR` is explicitly set.

```bash
./run_tiago.sh bag BAG_NAME
./run_tiago.sh bag BAG_NAME --resume
```

The normal bag profile replays the recorded `/map`. To run fresh RTAB-Map SLAM,
use the recording without the recorded map as a competing global transform:

```bash
./run_tiago.sh bag BAG_NAME --rtabmap --no-perception --rviz
./run_tiago.sh bag BAG_NAME --rtabmap --resume
```

All mode options can be combined with either lifecycle choice. For example,
`--rtabmap`, `--rviz`, `--no-perception`, `--rate` and `--loop` configure the
run; `--resume` only controls whether the existing run is reused. A resumed bag
must match the bag and profile already running in the container; use the default
fresh form when switching recordings or changing the run's map ownership.

The equivalent explicit path is `./run_tiago.sh bag bags/BAG_NAME`.

In fresh-SLAM mode the launcher automatically omits `/map` from playback and
relays `/tf` and `/tf_static` after dropping transforms touching `map`. This
prevents the recorded `map -> odom` transform from competing with RTAB-Map.
The launcher selects `TIAGO_BAG_FILTER_CONFLICTING=1` for `--rtabmap` and `0`
for recorded-map mode; the variable can be set explicitly when using the
environment form, but it must agree with the selected map profile. Customize
the dropped frame names with `TIAGO_BAG_TF_DROP_FRAMES=map,...`.

## Actions and useful overrides

```text
./run_tiago.sh physical
./run_tiago.sh physical --resume [--rviz]
./run_tiago.sh bag BAG [--resume] [--rtabmap] [--rviz]
./run_tiago.sh check                 # runtime/VitSAM check
./run_tiago.sh build                 # build lost3dsg into /ws
./run_tiago.sh firewall              # physical host DDS setup
./run_tiago.sh start                 # compatibility: resume current run
./run_tiago.sh new                   # explicit: archive /ws/output and restart
./run_tiago.sh stop
./run_tiago.sh attach
./run_tiago.sh shell
```

The environment-driven compatibility forms are also lifecycle-explicit:

```bash
TIAGO_BAG_PATH=/path/to/bag ./run_tiago.sh start  # resume
TIAGO_BAG_PATH=/path/to/bag ./run_tiago.sh new    # fresh
```

`TIAGO_RVIZ_CONFIG` selects a custom RViz file inside the container. The
default layout shows RTAB-Map output, camera/depth data, current and persistent
object bounding boxes, object text labels, room polygons, wall boundaries and
door/window markers. `REGOLO_API_KEY` can be exported ahead of a perception run;
otherwise the launcher prompts for it without writing it to a YAML file.
