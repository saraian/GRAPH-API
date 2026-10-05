#!/usr/bin/env bash
# Compatibility entrypoint for the graphapi CLI.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/graphapi" setup sim "$@"
