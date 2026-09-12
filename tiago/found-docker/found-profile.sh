# FOUND / LOST-3DSG environment for the PAL TIAGo development image.
# The FOUND checkout is bind-mounted at runtime (/found); this file only sets paths.

if [ -n "${FOUND_PROFILE_LOADED:-}" ]; then
  return 0 2>/dev/null || exit 0
fi
export FOUND_PROFILE_LOADED=1

# PAL/ROS setup files assume nounset is off.
_found_had_nounset=0
case $- in *u*) _found_had_nounset=1; set +u ;; esac
if [ -f /opt/pal/alum/setup.bash ]; then
  # shellcheck disable=SC1091
  . /opt/pal/alum/setup.bash --
elif [ -f /opt/ros/humble/setup.bash ]; then
  # shellcheck disable=SC1091
  . /opt/ros/humble/setup.bash --
fi
if [ -f /ws/install/setup.bash ]; then
  # shellcheck disable=SC1091
  . /ws/install/setup.bash --
fi
[ "$_found_had_nounset" = 1 ] && set -u
unset _found_had_nounset

export FOUND_ROOT="${FOUND_ROOT:-/found}"
export GRAPH_API_SRC="${GRAPH_API_SRC:-/graph_api/lost3dsg}"
export GRAPH_API_WS="${GRAPH_API_WS:-/ws}"
export GRAPH_API_OUTPUT_DIR="${GRAPH_API_OUTPUT_DIR:-/ws/output}"
export HF_HOME="${HF_HOME:-/models/hf}"
export SAM_MODEL_DIR="${SAM_MODEL_DIR:-/models/vitsam}"
export PYTHONUNBUFFERED=1
# The host's Xauthority file is inside the runtime-directory bind mount in the
# Docker container. Resolve it when the caller did not explicitly provide an
# XAUTHORITY path, so GUI tools such as rviz2 can authenticate to DISPLAY=:0.
if [ -z "${XAUTHORITY:-}" ] && [ -n "${LOCAL_XDG_RUNTIME_DIR:-}" ]; then
  for _found_xauthority in "$LOCAL_XDG_RUNTIME_DIR"/xauth_*; do
    if [ -f "$_found_xauthority" ]; then
      export XAUTHORITY="$_found_xauthority"
      break
    fi
  done
  unset _found_xauthority
fi
# The PAL image ships PyTorch with CUDA 12 libraries under its Python
# environment.  ONNX Runtime GPU also needs to find those libraries, plus the
# isolated cuDNN 9 tree used by the CUDA 12 provider.  Keep this separate from
# PyTorch's cuDNN 8 installation.
_found_onnxruntime_cuda_libs="/opt/onnxruntime-cuda12-libs/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib"
export LD_LIBRARY_PATH="${_found_onnxruntime_cuda_libs}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset _found_onnxruntime_cuda_libs
# PAL robots use CycloneDDS. Do not override RMW_IMPLEMENTATION here.
export PYTHONPATH="${FOUND_ROOT}:${GRAPH_API_SRC}/src/perception_module${PYTHONPATH:+:$PYTHONPATH}"
