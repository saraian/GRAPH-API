#!/usr/bin/env bash
# WATCH THE RUN THAT IS HAPPENING ON THE LAB MACHINE, from here.
#
#   ./connect_gin.sh              open the viewer on http://localhost:8081
#   ./connect_gin.sh --status     say what is running over there, then exit
#   ./connect_gin.sh --port 9000  use a different local port
#   ./connect_gin.sh --host Gin   a different ssh host (default: $GIN_HOST, else Gin)
#
# WHY A TUNNEL AND NOT A URL. The run's viewer binds to 127.0.0.1 on the lab machine, on purpose:
# the machine is shared with eight people's work and the bridge has no authentication. So it is
# reachable only by forwarding the port over ssh, and no firewall change would do instead.
#
# WHAT THIS IS NOT. It is not the replay dashboard. This is the RUN's own viewer -- live camera,
# the belief's boxes, the graph as it is built. The replay dashboard reads bundle DIRECTORIES
# through GRAPH_API_RUNS_DIR, and those live on the lab machine; pointing it at an ongoing run
# means syncing the bundle down, which is a different job.
set -uo pipefail
HOST="${GIN_HOST:-Gin}"
PORT=8081
REMOTE_PORT=8081
REMOTE_EXPLICIT=0
STATUS_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --status)      STATUS_ONLY=1 ;;
    --port)        shift; PORT="${1:?--port needs a number}" ;;
    --remote-port) shift; REMOTE_PORT="${1:?--remote-port needs a number}"; REMOTE_EXPLICIT=1 ;;
    --host)        shift; HOST="${1:?--host needs a name}" ;;
    -h|--help)     sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)             echo "!! unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

# 1. CAN WE REACH IT AT ALL? A tunnel that cannot connect fails with a bare "closed by remote
#    host" ten seconds later; saying so here costs one round trip and names the cause.
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" true 2>/dev/null; then
  echo "!! cannot ssh to '$HOST'." >&2
  echo "   Check the host name (set GIN_HOST, or pass --host) and that your key is loaded." >&2
  exit 1
fi

# Read the same operation record as the remote CLI; never guess its newest bundle.
if [ -n "${GIN_PROJECT_ROOT:-}" ]; then
  printf -v remote_command '%q --json status active' "${GIN_PROJECT_ROOT%/}/graphapi"
else
  remote_command='graphapi --json status active'
fi
remote_status=$(ssh -o BatchMode=yes "$HOST" "$remote_command" 2>/dev/null || true)
remote_bridge=$(printf '%s' "$remote_status" | python3 -c 'import json,sys,urllib.parse
try:
 data=json.load(sys.stdin); print(urllib.parse.urlparse(data.get("bridge_url", "")).port or "")
except (ValueError,TypeError): pass')
echo "== on $HOST"
if [ -n "$remote_bridge" ]; then
  printf '%s\n' "$remote_status"
  [ "$REMOTE_EXPLICIT" = 1 ] || REMOTE_PORT="$remote_bridge"
elif [ "$REMOTE_EXPLICIT" != 1 ]; then
  echo "Cannot select an active remote operation. Set GIN_PROJECT_ROOT to its checkout, install graphapi on its PATH, or provide --remote-port for a legacy run." >&2
  exit 1
else
  echo "legacy explicit bridge port: $REMOTE_PORT"
fi
[ "$STATUS_ONLY" = 0 ] || exit 0

echo "== forwarding $HOST:$REMOTE_PORT -> http://localhost:$PORT"
echo "   Ctrl-C closes the tunnel. The run is unaffected either way."
echo
# -N: no remote command, this is a tunnel. ServerAlive*: a dropped link is reported rather than
# hanging silently, which matters over the jump host.
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
     -L "${PORT}:localhost:${REMOTE_PORT}" "$HOST"
