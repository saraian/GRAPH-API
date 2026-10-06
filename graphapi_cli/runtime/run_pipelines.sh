#!/usr/bin/env bash
# Run one managed Habitat pipeline per GPU. Each run has its own ID and resources.
# Usage: ./graphapi tools run run-pipelines -- 0,1 hm3d_00337 hm3d_00861
set -uo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo "Run one Habitat scene per GPU, using the normal graphapi run sim workflow."
    echo "Usage: ./graphapi tools run run-pipelines -- GPU_LIST SCENE [SCENE ...]"
    echo "Example: ./graphapi tools run run-pipelines -- 0,1 hm3d_00337 hm3d_00861"
    exit 0
fi
GPUS="${GPUS:-}"
if [[ ${1:-} =~ ^[0-9]+(,[0-9]+)*$ ]]; then GPUS=$1; shift; fi
[[ ${1:-} == -- ]] && shift
[[ -n $GPUS && $# -gt 0 ]] || { echo "Give GPU numbers and at least one scene; use --help for an example." >&2; exit 2; }
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if (( $# > ${#GPU_ARR[@]} )); then
    echo "Give at least one GPU per scene." >&2
    exit 2
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
declare -a PIDS=()
i=0
for scene in "$@"; do
    gpu=${GPU_ARR[$i]}
    log="${LOG_DIR:-/tmp}/pipeline_gpu${gpu}_${STAMP}.log"
    echo "Starting scene=$scene gpu=$gpu; log=$log"
    args=(run sim --scene "$scene" --gpu "$gpu")
    [[ -z ${GRAPH_API_CONFIG:-} ]] || args+=(--config "$GRAPH_API_CONFIG")
    RVIZ=0 FEED_SHOW=0 PYTHONPATH="${GRAPHAPI_ROOT:?run through graphapi}${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m graphapi_cli.cli "${args[@]}" > "$log" 2>&1 &
    PIDS+=("$!")
    i=$((i + 1))
    # Keep build starts staggered, as in the existing parallel launcher.
    (( i == $# )) || sleep 20
done
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=$?; done
exit "$rc"
