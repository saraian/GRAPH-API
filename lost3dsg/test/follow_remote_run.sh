#!/usr/bin/env bash
# Follow a run on another machine from a local dashboard.
#
# WHY THIS EXISTS. The dashboard's replay mode already follows a bundle that is still being
# written -- it re-reads the run directory on a timer -- so "streaming" a remote run needs no
# new protocol, only the run directory kept in step. That is an rsync in a loop.
#
# WHAT IT COSTS, measured against Gin on 2026-09-10: a whole bundle is 1.9 MB, of which the
# JSON and JSONL the dashboard reads is 504 KB, and the first pull moved 67 KB compressed in
# about a second over the ProxyJump. Later passes send only the deltas. frames/ and depth/ are
# EXCLUDED by default because they are the only heavy parts (a local run with archiving on is
# 1.3 GB); pass --with-frames if you want the replay timeline as well as the graph and map.
#
# THE LIVE CAMERA IS A DIFFERENT THING and this script does not do it. /feed and the D-pad are
# served by the bridge inside the container, so they need a port forward to a RUNNING run:
#     ssh -N -L 8081:127.0.0.1:8081 Gin
#     BRIDGE_PORT=8081 python3 lost3dsg/dashboard/replay_server.py --mode live --port 8086
# That works only while the run is up; this script works during and after it.
set -euo pipefail

HOST=${1:-Gin}
REMOTE_RUNS=${REMOTE_RUNS:-/home/phd_student/Musumeci/gin_data/runs}
LOCAL_RUNS=${LOCAL_RUNS:-/DATA/GRAPH-API/results}
INTERVAL=${INTERVAL:-10}
EXCLUDES=(--exclude 'frames/' --exclude 'depth/' --exclude 'logs/' --exclude '*.db')
[ "${2:-}" = "--with-frames" ] && EXCLUDES=(--exclude 'logs/' --exclude '*.db')

# THE NEWEST REMOTE BUNDLE, asked for by name rather than guessed: a run started while this
# script is sleeping must be picked up, and the newest directory is the run in progress.
# THE TRAILING SLASH IS STRIPPED, and that is not a detail. `ls -dt` prints directories with
# one, and to rsync `src/` means "the CONTENTS of src" while `src` means "the directory src".
# With the slash left on, the first run of this script emptied a bundle's 21 files loose into
# the runs directory instead of creating the bundle -- next to 134 real bundles, where they
# looked like debris nobody could attribute.
newest() { ssh -o BatchMode=yes "$HOST" "ls -dt $REMOTE_RUNS/2026*/ 2>/dev/null | head -1" | sed 's:/*$::'; }

B=$(newest)
[ -n "$B" ] || { echo "!! no run directories under $HOST:$REMOTE_RUNS"; exit 1; }
NAME=$(basename "$B")
echo ">>> following $HOST:$B"
echo "    into $LOCAL_RUNS/$NAME   (every ${INTERVAL}s; Ctrl-C to stop)"
# The machine stamp is read from the bundle itself, so a followed run can never be
# mistaken for a local one. Printed once, at the start.
STAMP=$(ssh -o BatchMode=yes "$HOST" "grep -o '\"machine\": *\"[^\"]*\"' $B/run_metadata.json 2>/dev/null | head -1")
echo "    ${STAMP:-(no machine stamp in run_metadata.json)}"

mkdir -p "$LOCAL_RUNS"
while true; do
  rsync -az "${EXCLUDES[@]}" -e 'ssh -o BatchMode=yes' "$HOST:$B" "$LOCAL_RUNS/" || \
    echo "!! rsync failed; retrying in ${INTERVAL}s"   # a dropped link must not end the follow
  # Re-check which run is newest: the remote may have started the next one.
  NB=$(newest || true)
  if [ -n "$NB" ] && [ "$NB" != "$B" ]; then
    B=$NB; NAME=$(basename "$B"); echo ">>> remote moved to a new run: $NAME"
  fi
  sleep "$INTERVAL"
done
