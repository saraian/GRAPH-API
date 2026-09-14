#!/usr/bin/env bash
# A bounded, GT-isolated capture run for GA-493. This is not a base-run entry point.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${GRAPH_API_CONFIG:?set GRAPH_API_CONFIG to the GA-493 debug config}"
[ -f "$CONFIG" ] || { echo "!! no debug config: $CONFIG" >&2; exit 2; }

CAP_MIN="$(python3 -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])) or {}; print((c.get("run") or {}).get("cap_min", ""))' "$CONFIG")"
case "$CAP_MIN" in
  ''|*[!0-9]*) echo "!! run.cap_min must be a positive integer" >&2; exit 2 ;;
esac
[ "$CAP_MIN" -gt 0 ] && [ "$CAP_MIN" -le 15 ] || {
  echo "!! GA-493 debug cap must be between 1 and 15 minutes" >&2; exit 2;
}

python3 - "$CONFIG" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
checks = {
    "run.gt_semantic=false": (c.get("run") or {}).get("gt_semantic") is False,
    "archive.per_detection=false": (c.get("archive") or {}).get("per_detection") is False,
    "perception_parallel.bbox_backend=cuda":
        (c.get("perception_parallel") or {}).get("bbox_backend") == "cuda",
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("GA-493 debug config refusal: " + ", ".join(failed))
PY

if docker ps --format '{{.Names}}' | grep -qx graphapi_live; then
  echo "!! graphapi_live is already running" >&2
  exit 1
fi

export CAP_MIN GA493_DEBUG_MODE=1 FEED_GT_SEMANTIC=0
export PERCEPTION_EXECUTABLE=perception_parallel.py
export GA493_REPLAY_CAPTURE_DIR="${GA493_REPLAY_CAPTURE_DIR:-/ws/output/ga493_replay}"
export GA493_REPLAY_CAPTURE_MAX_CYCLES="${GA493_REPLAY_CAPTURE_MAX_CYCLES:-12}"
export GA493_REPLAY_CAPTURE_MAX_BYTES="${GA493_REPLAY_CAPTURE_MAX_BYTES:-5368709120}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
export RESULTS_DIR
CAP_FIRED_MARKER="$RESULTS_DIR/.ga493_cap_fired_$$"
export CAP_FIRED_MARKER

bash "$ROOT/run_sim.sh" "$@" &
RUN_PID=$!
(
  waited=0
  until docker ps --format '{{.Names}}' | grep -qx graphapi_live; do
    kill -0 "$RUN_PID" 2>/dev/null || exit 0
    [ "$waited" -lt 900 ] || { echo "!! graphapi_live did not start within 900 seconds" >&2; exit 1; }
    sleep 5
    waited=$((waited + 5))
  done
  deadline=$(( $(date +%s) + CAP_MIN * 60 ))
  while kill -0 "$RUN_PID" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      date +%s > "$CAP_FIRED_MARKER"
      # The container closes both replay owners, then RTAB-Map. Give both bounded stages their
      # declared budgets plus 30 seconds for the launch tree and log copy.
      stop_timeout="${STOP_TIMEOUT:-$(( ${GA493_CAPTURE_CLOSE_TIMEOUT:-30} + ${RTABMAP_CLOSE_TIMEOUT:-120} + 30 ))}"
      docker stop -t "$stop_timeout" graphapi_live >/dev/null
      exit 0
    fi
    sleep 5
  done
) &
WATCH_PID=$!

wait "$RUN_PID"
RC=$?
kill "$WATCH_PID" 2>/dev/null || true
wait "$WATCH_PID" 2>/dev/null || true
exit "$RC"
