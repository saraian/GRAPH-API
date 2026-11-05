# Private TIAGo setup

`graphapi run tiago physical` and `graphapi run tiago bag BAG_DIRECTORY` are the
public TIAGO workflows. The PAL development image and its
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

See [the TIAGO setup instructions](../README.md#tiago).

Typical first run:

```bash
./graphapi run tiago physical
```

To reuse the current physical tmux session and `/ws/output` instead of starting
fresh:

```bash
./graphapi run tiago physical --resume
```

The default container is `tiago-127-dev`. If it does not exist, the launcher
calls the private builder with `--create --robot` and mounts this checkout at
`/graph_api`. Set `TIAGO_AUTO_CREATE=0` if the container must be created by
hand. A clone that is not nested at `<FOUND>/vendor/graph-api` is supported;
the launcher creates an ignored mount shim automatically.

For offline TIAGO bag replay, pass the directory containing `metadata.yaml`.
Relative CLI paths resolve from the checkout root; an absolute path makes the
input explicit. The parent directory is used as the bag mount root when creating
a container. An existing container must have a mount covering the selected bag.

```bash
./graphapi run tiago bag /data/bags/BAG_NAME
./graphapi run tiago bag /data/bags/BAG_NAME --resume
./graphapi run tiago bag /data/bags/BAG_NAME --map-source slam
```

The default is recorded map/TF. Fresh SLAM excludes the recorded map topic and
filters conflicting global-map transforms. Neither bag mode connects to the
physical robot. Perception and recording default on in both physical and bag
workflows; add `--no-perception` or `--no-record` for an explicit opt-out.

An already prepared private PAL container can be reused without this ISO/builder
bundle. The bundle is needed when creating a missing container or image; this
repository does not supply it. See `./graphapi run tiago physical --help`,
`./graphapi run tiago bag --help` and the
[TIAGO workflow guide](../tiago/README.md).
