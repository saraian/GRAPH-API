#!/usr/bin/env bash
# Run SEVERAL isolated GRAPH-API pipelines at once, one container and one GPU each.
#
#   ./run_pipelines.sh 0,1,2,3 <scene> [<scene> ...]      one pipeline per GPU
#   GPUS=0,1 ./run_pipelines.sh -- hm3d_00337 hm3d_00824
#
# WHY THIS EXISTS. run_sim.sh launches ONE stack with a fixed container name, `--gpus all`, and
# `--network=host`. Three things therefore have to differ per pipeline, and the third is the one
# that bites silently:
#
#   1. container name   -- a duplicate fails loudly on "name already in use". Harmless.
#   2. GPU              -- `--gpus all` makes every run fight for every card.
#   3. ROS_DOMAIN_ID    -- with --network=host and no domain, two stacks share DDS domain 0.
#                          They discover each other's nodes and one run's perception can consume
#                          the other run's frames. NOTHING CRASHES. The bundle looks complete and
#                          the numbers are wrong. This is why the script REFUSES to launch two
#                          pipelines on the same domain rather than trusting the caller.
#
# The ports are already environment-driven and are offset per pipeline:
# FEED_PORT 7799+i, FEED_CTRL_PORT 7790+i, BRIDGE_PORT 8081+i.
#
# THROUGHPUT, MEASURED NOT ASSUMED. The GA-493 replay measured a 7.1955 s median cycle of which
# the VLM was 5.882 s -- 82%. config.yaml points the VLM at https://api.regolo.ai/v1, a REMOTE
# service. Local GPU work is SAM 280.7 ms, geometry 43.35 ms and the CUDA encoder 3.032 ms. So the
# GPU is NOT the bottleneck and N pipelines do NOT give N times the throughput unless that remote
# endpoint serves N concurrent streams. Watch its latency and error rate before adding pipelines.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPUS="${GPUS:-}"
if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then GPUS="$1"; shift; fi
[ "${1:-}" = "--" ] && shift
[ -n "$GPUS" ] || { echo "usage: $0 <gpu-list> <scene> [<scene> ...]   e.g. $0 0,1,2,3 hm3d_00337" >&2; exit 2; }
[ $# -gt 0 ] || { echo "give at least one scene" >&2; exit 2; }

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [ $# -gt ${#GPU_ARR[@]} ]; then
    echo "!! $# scenes but only ${#GPU_ARR[@]} GPUs. Two pipelines on one card will fight for VRAM." >&2
    echo "   Give more GPUs, or fewer scenes." >&2
    exit 2
fi

# REFUSE A CARD SOMEONE ELSE IS USING. This is a shared machine. The check asserts on
# `nvidia-smi -q -d PIDS`, which lists EVERY process type: --query-compute-apps returns empty for a
# graphics context, so it reports a busy card as free (measured on Gin, 2026-09-14, GA-509).
for g in "${GPU_ARR[@]}"; do
    busy=$(nvidia-smi -i "$g" -q -d PIDS 2>/dev/null | awk '/^ *Process ID *:/ {n++} END {print n+0}')
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ "${busy:-1}" != "0" ]; then
        echo "!! GPU $g is BUSY: $busy process(es), ${used:-?} MiB used. Refusing to launch on it." >&2
        nvidia-smi -i "$g" -q -d PIDS 2>/dev/null | sed -n '/Processes/,$p' >&2
        exit 2
    fi
    echo "   GPU $g free (${used:-?} MiB)"
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
declare -a PIDS=() NAMES=()
i=0
for scene in "$@"; do
    gpu="${GPU_ARR[$i]}"
    # The domain is derived from the GPU index, so two pipelines cannot collide by construction
    # and the mapping is reproducible across restarts.
    domain=$(( 20 + gpu ))
    name="graphapi_p${gpu}_${STAMP}"
    log="${LOG_DIR:-/tmp}/pipeline_gpu${gpu}_${STAMP}.log"
    echo ">> pipeline $i: scene=$scene gpu=$gpu domain=$domain container=$name"
    echo "   log: $log"
    GRAPH_API_CONTAINER_NAME="$name" \
    GRAPH_API_GPUS="device=$gpu" \
    ROS_DOMAIN_ID="$domain" \
    FEED_PORT=$(( 7799 + i )) \
    FEED_CTRL_PORT=$(( 7790 + i )) \
    BRIDGE_PORT=$(( 8081 + i )) \
    HABITAT_SCENE="$scene" \
        nohup "$HERE/run_sim_headless.sh" > "$log" 2>&1 &
    PIDS+=("$!"); NAMES+=("$name")
    i=$(( i + 1 ))
    # Stagger the starts. Both stacks otherwise build the colcon workspace from the same mount at
    # once, and the preflight gate reads a tree another container is still writing.
    sleep 20
done

echo
echo "launched ${#PIDS[@]} pipeline(s): ${NAMES[*]}"
echo "watch:  docker ps --filter name=graphapi_p"
echo "stop:   docker stop ${NAMES[*]}"
rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=$?; done
echo "all pipelines finished, worst rc=$rc"
exit "$rc"
