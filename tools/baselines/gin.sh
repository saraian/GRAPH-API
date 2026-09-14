#!/usr/bin/env bash
# Run on Gin. Sources and models are read-only mounts; only a fresh result dir is writable.
set -euo pipefail
if (($# < 3)); then
    echo "Usage: $0 clio|hovsg|dynamicgsg RECORDING_OR_EXPORT OUTPUT [native adapter options...]" >&2
    exit 2
fi
baseline=$1
recording=$(realpath "$2")
output=$(realpath -m "$3")
shift 3
integration=${BASELINE_INTEGRATION_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
root_base=${BASELINE_REPOS_ROOT:-/home/phd_student/Musumeci}
if [[ -e $output ]]; then
    echo "Output already exists: $output" >&2
    exit 2
fi
mkdir -p "$(dirname "$output")"
gpu=${BASELINE_GPU:-0}
# Refuse to launch onto a card another lane is already using.
#
# HOST-SIDE on purpose. Inside the container nvidia-smi cannot resolve host process identity,
# so a container-side guard reports the card free whatever is running on it.
#
# `--query-compute-apps` is NOT sufficient and is the trap this replaces. A Habitat acquisition
# holds a GRAPHICS context, not a compute one, so that query returns EMPTY on a busy card.
# Measured 2026-09-14 on Gin GPU 1 against a live baseline acquisition: --query-compute-apps
# printed nothing, while `-q -d PIDS` printed "Process ID: 2177348 / Type: G / 627 MiB".
# `-q -d PIDS` lists every process type and is the one to assert on.
#
# memory.used is reported for context, not asserted: an idle card here reads 34 MiB, so a
# threshold on it would need a magic number. The process count is the assertion.
#
# Set BASELINE_GPU_GUARD=0 to launch onto a busy card deliberately.
guard_gpu() {
    local gpu=$1 busy used
    if [[ ${BASELINE_GPU_GUARD:-1} != 1 ]]; then
        echo "GPU guard DISABLED by BASELINE_GPU_GUARD for device $gpu" >&2
        return 0
    fi
    busy=$(nvidia-smi -i "$gpu" -q -d PIDS | awk '/^ *Process ID *:/ {n++} END {print n+0}')
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    if ((busy > 0)); then
        echo "GPU $gpu is BUSY: $busy process(es), $used MiB used. Refusing to launch." >&2
        nvidia-smi -i "$gpu" -q -d PIDS | sed -n '/Processes/,$p' >&2
        exit 2
    fi
    echo "GPU guard: device $gpu is free ($busy processes, $used MiB used)" >&2
}
guard_gpu "$gpu"
common=(--rm --network none --gpus "device=$gpu"
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/integration:/baseline
    -e BASELINE_INTEGRATION_ROOT=/integration
    -v "$integration:/integration:ro" -v "$recording:/recording:ro"
    -v "$(dirname "$output"):/results" -w /baseline)
if [[ -n ${BASELINE_CONTAINER_NAME:-} ]]; then
    common+=(--name "$BASELINE_CONTAINER_NAME")
fi
# Pin the container's CPU set on many-core hosts. Measured on the 256-core DGX A100 (2026-09-14):
# rtabmap exits 1 silently at ~117-141 iterations when it sees all cores (thread-pool allocation
# fails), and runs past 520 when pinned to 32. Give concurrent containers DISJOINT ranges.
if [[ -n ${BASELINE_CPUSET:-} ]]; then
    common+=(--cpuset-cpus "$BASELINE_CPUSET")
fi
container_rc=0
release_output_owner() {
    local image_id=$1
    local output_parent output_name
    [[ -d $output ]] || return 0
    output_parent=$(dirname "$output")
    output_name=$(basename "$output")
    docker run --rm --network none -v "$output_parent:/results" \
        --entrypoint chown "$image_id" "$(id -u):$(id -g)" "/results/$output_name"
}
case "$baseline" in
    hovsg)
        image_id=$(docker image inspect --format '{{.Id}}' "${HOV_IMAGE:-hov-baseline:clean}")
        docker run "${common[@]}" -e BASELINE_IMAGE_ID="$image_id" -v "$root_base/HOV-Baseline:/baseline:ro" \
            "$image_id" python /baseline/run_scheduled.py run \
            --baseline-root /baseline --recording /recording --output "/results/$(basename "$output")" "$@" \
            || container_rc=$?
        ;;
    clio)
        image_id=$(docker image inspect --format '{{.Id}}' "${CLIO_IMAGE:-clio-baseline:noetic}")
        docker run "${common[@]}" -e BASELINE_IMAGE_ID="$image_id" -v "$root_base/Clio-Baseline:/baseline:ro" \
            -e ROS_HOSTNAME=127.0.0.1 -e ROS_MASTER_URI=http://127.0.0.1:11311 \
            --entrypoint bash "$image_id" -lc '
                source /opt/ros/noetic/setup.bash
                source /root/catkin_ws/devel/setup.bash
                export PATH=/root/environments/clio_ros/bin:$PATH
                exec /root/environments/clio_ros/bin/python /baseline/run_scheduled.py run "$@"
            ' baseline --baseline-root /baseline --recording /recording \
            --output "/results/$(basename "$output")" --models /integration/models \
            --clip-cache /integration/clip "$@" || container_rc=$?
        ;;
    dynamicgsg)
        image_id=$(docker image inspect --format '{{.Id}}' "${DYNAMICGSG_IMAGE:-dynamicgsg-baseline:cu121}")
        model_root=${DYNAMICGSG_MODEL_ROOT:-$root_base/Dynamic-GSG-models}
        baseline_root=${DYNAMICGSG_BASELINE_ROOT:-$root_base/Dynamic-GSG-Upstream-Execfix-ID32-73c2dba}
        variant=${DYNAMICGSG_VARIANT:-upstream-execfix}
        profile=${DYNAMICGSG_PROFILE:-configs/realsense/dgsg.py}
        [[ -d $model_root/models ]] || { echo "Missing DynamicGSG model cache: $model_root/models" >&2; exit 2; }
        [[ -d $model_root/huggingface ]] || { echo "Missing DynamicGSG Hugging Face cache: $model_root/huggingface" >&2; exit 2; }
        [[ -d $model_root/torch ]] || { echo "Missing DynamicGSG Torch cache: $model_root/torch" >&2; exit 2; }
        [[ -d $baseline_root/.git ]] || { echo "Missing DynamicGSG checkout: $baseline_root" >&2; exit 2; }
        dynamic_args=(--variant "$variant" --profile "$profile")
        if [[ -n ${DYNAMICGSG_DYNAMIC_START_FRAME:-} ]]; then
            dynamic_args+=(--dynamic-start-frame "$DYNAMICGSG_DYNAMIC_START_FRAME")
        fi
        if [[ -n ${DYNAMICGSG_SOURCE_PATCH_SHA256+x} ]]; then
            source_patch=$DYNAMICGSG_SOURCE_PATCH_SHA256
        elif [[ $variant == upstream-execfix ]]; then
            source_patch=4766e876fbba97c50d57b3c5c4e2c10f48cbda4e038fadf5c4b4eceec96d5deb
        else
            source_patch=
        fi
        if [[ -n $source_patch ]]; then
            dynamic_args+=(--source-patch-sha256 "$source_patch")
        fi
        docker run "${common[@]}" -e BASELINE_IMAGE_ID="$image_id" \
            -e MPLBACKEND=Agg \
            -v "$baseline_root:/baseline:ro" \
            -v "$model_root/models:/models:ro" \
            -v "$model_root/huggingface:/root/.cache/huggingface:ro" \
            -v "$model_root/torch:/root/.cache/torch:ro" \
            "$image_id" python3 -m tools.baselines.dynamicgsg_run \
            --baseline-root /baseline --dataset-export /recording \
            --output "/results/$(basename "$output")" --models /models \
            "${dynamic_args[@]}" "$@" || container_rc=$?
        ;;
    *) echo "Unknown baseline: $baseline" >&2; exit 2 ;;
esac
release_output_owner "$image_id"
exit "$container_rc"
