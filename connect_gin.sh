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
STATUS_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --status)      STATUS_ONLY=1 ;;
    --port)        shift; PORT="${1:?--port needs a number}" ;;
    --remote-port) shift; REMOTE_PORT="${1:?--remote-port needs a number}" ;;
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

# 2. WHAT IS ACTUALLY RUNNING THERE. Reported before the tunnel opens, because a viewer showing
#    nothing and a viewer that is not connected look identical in a browser.
echo "== on $HOST"
ssh -o BatchMode=yes "$HOST" '
  c=$(docker ps --format "{{.Names}} {{.Status}}" | grep -i graphapi || true)
  if [ -n "$c" ]; then echo "   container: $c"; else echo "   container: NONE — no run is in progress"; fi
  b=$(ls -dt ~/Musumeci/*/GRAPH-API/results/2026* 2>/dev/null | head -1)
  if [ -n "$b" ]; then
    echo "   newest bundle: $(basename "$b")"
    echo "   size: $(du -sh "$b" 2>/dev/null | cut -f1)   last written: $(date -r "$b" +%H:%M:%S)"
    m=$(ls "$b"/../../monitor*.log 2>/dev/null | head -1)
    python3 - "$b" <<PY
import json, os, sys
b = sys.argv[1]
def j(n, d=None):
    try: return json.load(open(os.path.join(b, n)))
    except Exception: return d
f = j("feed_stats.json", {}) or {}
print("   moved %s m, waypoints %s" % (f.get("total_distance_m"), f.get("tour_waypoints_reached")))
p = os.path.join(b, "hook_decisions.jsonl")
print("   decisions:", sum(1 for _ in open(p)) if os.path.exists(p) else 0)
c = os.path.join(b, "perception_latencies.jsonl")
print("   perception cycles timed:", sum(1 for _ in open(c)) if os.path.exists(c) else 0)
PY
  else
    echo "   no bundle found under ~/Musumeci/*/GRAPH-API/results/"
  fi
  if curl -s -m 4 -o /dev/null -w "   viewer on :'"$REMOTE_PORT"' -> HTTP %{http_code}\n" "http://localhost:'"$REMOTE_PORT"'/"; then :; else
    echo "   viewer on :'"$REMOTE_PORT"' -> no answer"
  fi
' 2>/dev/null

[ "$STATUS_ONLY" = "1" ] && exit 0

# 3. IS THE LOCAL PORT FREE? Forwarding onto a busy port silently gives you the OTHER service,
#    which is how somebody ends up reading yesterday's dashboard and calling it the live run.
if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -qE "[:.]$PORT\b"; then
  echo "!! local port $PORT is already in use — something else would answer, not the run." >&2
  echo "   Pass --port <free port>, or stop what is listening." >&2
  exit 1
fi

echo
echo "== forwarding $HOST:$REMOTE_PORT -> http://localhost:$PORT"
echo "   Ctrl-C closes the tunnel. The run is unaffected either way."
echo
# -N: no remote command, this is a tunnel. ServerAlive*: a dropped link is reported rather than
# hanging silently, which matters over the jump host.
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
     -L "${PORT}:localhost:${REMOTE_PORT}" "$HOST"
