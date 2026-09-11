#!/usr/bin/env bash
# WATCH A RUN WHILE IT HAPPENS. Same arguments as lost3dsg/test/run_monitor.py:
#   ./monitor.sh                the newest run in results/
#   ./monitor.sh <bundle>       that one
#   ./monitor.sh --once         one snapshot
#   ./monitor.sh --interval 30  seconds between lines
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/lost3dsg/test/run_monitor.py" "$@"
