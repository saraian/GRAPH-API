#!/bin/bash
# FEED_REMOTE_CMD for run_sim.sh on a stack host with no GPU: runs habitat_feed_host.py on Modal
# (remote_feed_modal.py) with this process's feed environment. exec, so run_sim.sh's kill of
# FEED_PID stops `modal run`, and with it the remote feed.
HERE=$(cd "$(dirname "$0")" && pwd)
# EU, near a European stack host: measured 31.7 MB/s through the tunnel against 1.4 MB/s from the default
# (US) region, and the frame stream needs about 8 MB/s.
export REMOTE_FEED_REGION=${REMOTE_FEED_REGION:-eu}
# Python 3.9.13 (the laptop habitat_env) needs an image builder that still supports 3.9; 2025.06 refuses it.
export MODAL_IMAGE_BUILDER_VERSION=${MODAL_IMAGE_BUILDER_VERSION:-2024.10}
mkdir -p "$RUN_DIR/logs"
spec="$RUN_DIR/logs/remote_feed_spec.json"
python3 - "$spec" <<'PY'
import json, os, re, sys
env = {k: v for k, v in os.environ.items()
       if re.match(r"(FEED_|HABITAT_)|(GRAPH_API_CONFIG|RUN_DIR|OUT_DIR|PYTHONUNBUFFERED)$", k)}
files = [p for p in (env.get("GRAPH_API_CONFIG"), env.get("FEED_SCHEDULE")) if p and os.path.isfile(p)]
json.dump({"env": env, "files": files}, open(sys.argv[1], "w"), indent=1)
PY
exec modal run "$HERE/remote_feed_modal.py::main" --spec "$spec"
