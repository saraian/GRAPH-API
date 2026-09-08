#!/usr/bin/env bash
# Sample the feed node (or any node) INSIDE the running container with py-spy, so a CPU-pegged
# process is explained by its hot frames instead of argued from diffs. Written 2026-09-07 for the
# feed regression measured across runs C/E/F/G (95 -> 31/39/34 frames at T+150 s; the consumer
# habitat_feed_node.py at 83% CPU, the renderer at 15%). ptrace from the host is denied, so this
# runs py-spy from a --privileged exec. Output: <bundle>/logs/pyspy_<pattern>.txt (top 20 s + dump).
#   usage: profile_feed_node.sh <container> <bundle_dir> [process pattern] [seconds]
set -euo pipefail
C="$1"; OUT="$2"; PAT="${3:-habitat_feed_node}"; SECS="${4:-20}"
# MEASURED 2026-09-07 on run 20260907_004128: `pgrep -f habitat_feed_node` returns 294 296 — 294
# is the `ros2 run` WRAPPER, idle in subprocess.wait(), and `head -1` picked it. The profile was
# 17 lines of a sleeping parent while the real node sat at 94% CPU. Match the INSTALLED path so
# the wrapper cannot win; fall back to the bare pattern for a node that is not run from /ws.
PID="$(docker exec "$C" pgrep -f "lib/lost3dsg/$PAT" | head -1)"
[ -n "$PID" ] || PID="$(docker exec "$C" pgrep -f "$PAT" | head -1)"
[ -n "$PID" ] || { echo "no process matching '$PAT' in $C"; exit 1; }
mkdir -p "$OUT/logs"
docker exec --privileged "$C" sh -c "python3 -m pip install -q py-spy >/dev/null 2>&1 || true; \
  py-spy dump --pid $PID; echo; echo '--- top, ${SECS}s ---'; \
  timeout ${SECS} py-spy top --pid $PID --nonblocking 2>/dev/null | tail -40 || true; \
  echo '--- ps ---'; ps -o pid,pcpu,rss,etime,args -p $PID" > "$OUT/logs/pyspy_${PAT}.txt" 2>&1 || true
echo "wrote $OUT/logs/pyspy_${PAT}.txt ($(wc -l < "$OUT/logs/pyspy_${PAT}.txt") lines)"
