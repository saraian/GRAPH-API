#!/usr/bin/env bash
# Compatibility entrypoint; launch ownership and configuration live in graphapi.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/graphapi" legacy eval "$@"
