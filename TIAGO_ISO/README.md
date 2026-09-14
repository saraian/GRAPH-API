# Private TIAGo setup

`run_tiago.sh` is the public launcher. The PAL development image and its
credentials are intentionally not part of this repository.

Put the private TIAGo bundle in this directory (or set `TIAGO_ISO_DIR` to
another directory), including at least:

```text
TIAGO_ISO/
├── ROS2_Alum.iso
├── keys/                         # PAL/ROS credentials as supplied privately
├── uni-sap-rome-build-docker.sh  # private image/container builder
└── ...                           # any private files required by that builder
```

The builder needs a PAL APT key package when it has to build the image. The
launcher accepts it in any of these locations:

```text
TIAGO_ISO/found-docker/pal-apt-keys.deb
TIAGO_ISO/pal-apt-keys.deb
TIAGO_ISO/keys/pal-apt-keys.deb
```

The public runtime helpers, configuration overlays, TF relay and RViz layout
are under `tiago/`; they are staged automatically for the private builder and
copied into the container on every `check`, `build`, `start` and `new`. The
concise `physical` and `bag` forms select `new` by default; `--resume` selects
the existing-run `start` behavior.

Typical first run:

```bash
./run_tiago.sh physical
```

To reuse the current physical tmux session and `/ws/output` instead of starting
fresh:

```bash
./run_tiago.sh physical --resume
```

The default container is `tiago-127-dev`. If it does not exist, the launcher
calls the private builder with `--create --robot` and mounts this checkout at
`/graph_api`. Set `TIAGO_AUTO_CREATE=0` if the container must be created by
hand. A clone that is not nested at `<FOUND>/vendor/graph-api` is supported;
the launcher creates an ignored mount shim automatically.

For a bag, put it below the repository-local `bags/` directory (or make that
directory a symlink to local storage) and use its name:

```bash
./run_tiago.sh bag BAG_NAME
```

The bag form is fresh by default as well. Add `--resume` only when continuing
the same bag/profile:

```bash
./run_tiago.sh bag BAG_NAME --resume
```

Use `FOUND_START_RTABMAP=1` for fresh RGB-D SLAM. That mode removes the
recorded global-map topic and filters conflicting map transforms before
RTAB-Map receives the robot/camera TF. The concise equivalent is:

```bash
./run_tiago.sh bag BAG_NAME --rtabmap
```

The default bag root is `<repo>/bags`; `TIAGO_BAG_DIR` can override it for a
different machine. The low-level environment form remains supported, for
example:

```bash
TIAGO_BAG_PATH=/absolute/path/to/bag ./run_tiago.sh new    # fresh
TIAGO_BAG_PATH=/absolute/path/to/bag ./run_tiago.sh start  # resume
```

See `./run_tiago.sh --help` for the complete physical, bag, RViz, VLM and
container environment variables.
