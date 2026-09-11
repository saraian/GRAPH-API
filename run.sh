#!/usr/bin/env bash
# THE ONLY SCRIPT A PERSON RUNS.
#
# Owner, 2026-09-10: "Run guide is so complex. I'm pretty sure we can simplify things to just ONE
# config file (which anyway should be in the project root), ONE install script and ONE launch
# script." This is that launch script. Everything under lost3dsg/test/ is an internal layer:
#   run.sh -> lost3dsg/test/run_house.sh   one launch per storey
#          -> lost3dsg/test/live_run.sh    one launch
#          -> live_stack_container.sh      inside the container
# Nobody is asked to call those. If a step in the README names one of them, the step is wrong.
#
# WHAT A BASE RUN IS (rule 73, owner 2026-09-10): the whole house, every storey, NO CAP, one
# mapping session per storey. It produces ONE BUNDLE PER STOREY, not one per run.
#
#   ./run.sh                 the scene from the config file
#   ./run.sh hm3d_00861      that scene, this run only
#   ./run.sh --one-storey    a single storey, the only variant worth a flag
#
# DELIBERATELY THIN. Every knob lives in config.yaml, so this file must not grow a flag per
# setting -- that is the complexity the owner asked us to remove. A setting you cannot find in
# config.yaml is a bug in config.yaml, not a missing flag here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ONE_STOREY=0
SCENE_ARG=""
SCHEDULE=""
CONFIG=""
_next_is_schedule=0
_next_is_config=0
for a in "$@"; do
  if [ "$_next_is_schedule" = "1" ]; then SCHEDULE="$a"; _next_is_schedule=0; continue; fi
  if [ "$_next_is_config" = "1" ]; then CONFIG="$a"; _next_is_config=0; continue; fi
  case "$a" in
    --schedule)   _next_is_schedule=1 ;;
    --config)     _next_is_config=1 ;;
    --one-storey) ONE_STOREY=1 ;;
    -h|--help)    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*)           echo "!! unknown option: $a. This script takes a scene and --one-storey." >&2
                  echo "   Every other setting belongs in config.yaml." >&2; exit 2 ;;
    *)            SCENE_ARG="$a" ;;
  esac
done

# --schedule WITHOUT A FILE MUST REFUSE, NOT FALL THROUGH. Measured 2026-09-10: `./run.sh
# --schedule` with the filename forgotten left SCHEDULE empty and STARTED A FULL HOUSE RUN.
# An option that silently becomes a different command is worse than an unknown option.
if [ "$_next_is_schedule" = "1" ]; then
  echo "!! --schedule needs a file: ./run.sh --schedule schedules/<name>.runs.yaml" >&2
  exit 2
fi
if [ "$_next_is_config" = "1" ]; then
  echo "!! --config needs a file: ./run.sh --config schedules/configs/<name>.yaml" >&2
  exit 2
fi
# A CUSTOM CONFIG FILE FOR ONE RUN. Owner instruction 2026-09-10. It sets BOTH variables the
# launcher uses, because they are read in different places and disagreeing is how a bundle comes to
# name one file while loading another: GRAPH_API_CONFIG is what config.py opens, CFG_NAME is what
# the launcher echoes, stamps and hands to the container.
if [ -n "$CONFIG" ]; then
  [ -f "$CONFIG" ] || { echo "!! no such config file: $CONFIG" >&2; exit 2; }
  CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
  export GRAPH_API_CONFIG="$CONFIG"
  export CFG_NAME="$(basename "$CONFIG")"
  echo "config: $CONFIG"
fi
if [ -n "$SCHEDULE" ] && [ ! -f "$SCHEDULE" ]; then
  echo "!! no such schedule file: $SCHEDULE" >&2
  exit 2
fi

# A SCHEDULE OF RUNS, each with its own configuration. Owner instruction 2026-09-10. The driver
# writes one real config file per arm, passes it as GRAPH_API_CONFIG, runs this script once per
# arm, and records which arm produced which bundle. It is a separate file because it needs yaml
# and a manifest, and because "read config.yaml ALWAYS" means an arm must be a FILE rather than a
# pile of variables at launch time.
if [ -n "$SCHEDULE" ]; then
  exec python3 "$HERE/lost3dsg/test/schedule_runs.py" "$SCHEDULE" --runner "$0"
fi

# REFUSE A CAP HERE TOO, not only in run_house.sh. A person who sets CAP_MIN in front of this
# command means something by it, and rule 73 says a base run has none: a cap truncates a storey
# mid-tour and leaves a bundle that LOOKS finished, which is the one failure a reader cannot see.
if [ -n "${CAP_MIN:-}" ]; then
  echo "!! CAP_MIN=$CAP_MIN is set, and a base run has no cap (rule 73)." >&2
  echo "   Each storey ends when its tour completes. Clear CAP_MIN, or use --one-storey." >&2
  exit 2
fi

# THE SETUP IS CHECKED HERE, WHERE THE PERSON IS, rather than failing three layers down with a
# message about a mount. install.sh writes config.local.yaml; without it the run has no endpoint
# and no workspace, and the failure would otherwise arrive as a container error.
if [ ! -f "$HERE/lost3dsg/test/env.local.sh" ] && [ ! -f "$HERE/config.local.yaml" ]; then
  echo "!! Not installed yet: no config.local.yaml (or lost3dsg/test/env.local.sh)." >&2
  echo "   Run ./install.sh once, then fill in the values it names." >&2
  exit 2
fi

# CALLED THROUGH bash ON PURPOSE, not as ./script. MEASURED ON GIN 2026-09-10: git records
# live_run.sh, run_house.sh and live_stack_container.sh as mode 100644, so on ANY fresh clone
# `./live_run.sh` fails with "Permission denied". They only look executable in a working tree whose
# filesystem is permissive, which is why the README's documented command has never been run from a
# clean clone. Invoking the interpreter removes the dependency on the mode entirely.
if [ "$ONE_STOREY" = "1" ]; then
  exec bash "$HERE/lost3dsg/test/live_run.sh" ${SCENE_ARG:+"$SCENE_ARG"}
fi
# run_house.sh reads the scene from SCENE; a positional argument here overrides the config for one
# run. When neither is given the config file decides, which is the point of the config file.
exec env ${SCENE_ARG:+SCENE="$SCENE_ARG"} bash "$HERE/lost3dsg/test/run_house.sh"
