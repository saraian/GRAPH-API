# TIAGO workflows

Use `./graphapi run tiago physical` for the physical robot and
`./graphapi run tiago bag BAG_DIRECTORY` for offline TIAGO RGB-D replay.
Both run in the runner’s private PAL Docker container. Bag replay is an offline
input to the pipeline; it does not simulate a robot in Gazebo or connect to one.
`run tiago` requires a workflow. The historical `run bag` spelling is an alias
for `run tiago bag`; `run_tiago.sh` is a compatibility adapter.

| Workflow | Default map source | Default node configuration |
| --- | --- | --- |
| Physical robot | Application RTAB-Map (`found_map`, `/rtabmap/map`) | `config/tiago_robot.yaml` |
| Offline bag | Recorded map and TF (`map`, `/map`) | `config/tiago_bag.yaml` |
| Offline bag with `--map-source slam` | Fresh RTAB-Map (`map`, `/rtabmap/map`) | `config/tiago_bag_rtabmap.yaml` |

Perception and topic recording default on; GUI defaults off. Use
`--no-perception`, `--no-record` and `--gui` to select these explicitly.
The physical target defaults to `tiago-127-dev`, ROS domain `1`, robot address
`10.68.0.1`, and automatic host DDS preparation. Set the existing robot/topic
variables in `config/local.yaml` when the target differs. `--map-source robot`
disables application RTAB-Map and requires a node configuration matching the
robot’s actual map topic and frame.

```bash
./graphapi doctor --mode tiago-physical
./graphapi run tiago physical --detach
./graphapi doctor --mode tiago-bag
./graphapi run tiago bag /data/bags/example --map-source recorded --detach
./graphapi run tiago bag /data/bags/example --map-source slam --no-record --detach
./graphapi status OPERATION_ID
./graphapi logs OPERATION_ID --follow
./graphapi stop OPERATION_ID
```

The bag argument is a directory containing `metadata.yaml`; relative CLI paths
resolve from the checkout root. Its parent is mounted as the bag root. An
existing private container must already have a mount covering that recording.
Bag replay skips physical DDS/firewall setup and uses an allocated replay ROS
domain. Fresh SLAM omits recorded `/map` and filters transforms touching `map`
from `/tf` and `/tf_static` before RTAB-Map receives them. Existing TF relay and
mapping parameters are preserved.

Fresh runs are default: the launcher closes the existing private tmux session
and archives `/ws/output` before starting clean. `--resume` explicitly reuses
that session/output; it must match the bag and profile already running. It does
not resume a stopped historical operation. The CLI reserves the private
container for one managed acquisition at a time.

The runner must provide the private PAL container or private build inputs.
See the [TIAGO setup instructions](../README.md#tiago).
An already prepared container can be reused without supplying the ISO, builder
or keys again. Source resolution selects this checkout through the existing
container mounts, including when `/graph_api` points to an older checkout.
The non-secret helpers under `tiago/found-docker/` are copied on each preparation;
canonical node configuration and RViz files are in `config/`.

```bash
./graphapi setup tiago --pal-bundle /private/TIAGO_ISO --container tiago-127-dev
./graphapi tiago check
./graphapi tiago build
./graphapi tiago shell
./graphapi tiago network status
```

`REGOLO_API_KEY` supplies the current VLM credential; otherwise the private
launcher prompts without writing it into YAML. Recordings are written to the
private container’s `/ws/output/recording` before final output is copied to the
configured results directory. Put the private `/ws` mount on a drive with enough
space; moving `results` alone does not move that live recording.

The separate existing ROS launch variants have explicit names:
`launch tiago-gazebo` starts the Gazebo/AMCL/navigation stack;
`launch tiago-navigation` starts map/AMCL/navigation without spawning Gazebo.
They use the public Gazebo image from `setup gazebo`.
`launch tiago-bag-slam`, `tiago-bag-slam-1` and `tiago-bag-slam-2` expose
historical bag ROS launch files in private PAL Docker. Use `run tiago bag` for
the managed pipeline above. Historical launch names remain compatibility aliases.

See the [CLI help examples](../README.md#cli-help) for all commands and the existing
compatibility environment variables in `./run_tiago.sh --help`.
