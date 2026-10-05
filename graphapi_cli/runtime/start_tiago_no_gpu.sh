#!/usr/bin/env bash
# Compatibility for the public TIAGO Gazebo launcher; GUI/display setup stays explicit.
set -euo pipefail
exec "${GRAPHAPI_ROOT:?run through graphapi}/graphapi" launch simulation "$@"
