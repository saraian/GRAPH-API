FROM ros:humble
ENV DEBIAN_FRONTEND=noninteractive

# Base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates python3-pip \
    python3-vcstool python3-rosdep python3-colcon-common-extensions \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
    ros-humble-rtabmap-ros \
    ros-humble-xacro ros-humble-robot-state-publisher ros-humble-joint-state-publisher-gui \
    ros-humble-rmw-cyclonedds-cpp ros-humble-rmw-fastrtps-cpp \
    ros-humble-tf-transformations \
    ros-humble-image-view ros-humble-teleop-twist-keyboard \
    ros-humble-moveit \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-gripper-controllers \
    ros-humble-joint-trajectory-controller \
    ros-humble-joint-state-broadcaster \
    ros-humble-gazebo-ros2-control \
    ros-humble-ros-gz \
    ros-humble-sdformat-urdf \
    ros-humble-ros2controlcli \
    ros-humble-controller-interface \
    ros-humble-hardware-interface-testing \
    ros-humble-ament-cmake-clang-format \
    ros-humble-ament-cmake-clang-tidy \
    ros-humble-controller-manager \
    ros-humble-ros2-control-test-assets \
    ros-humble-hardware-interface \
    ros-humble-control-msgs \
    ros-humble-backward-ros \
    ros-humble-generate-parameter-library \
    ros-humble-realtime-tools \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-kinematics \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-pinocchio \
    libignition-gazebo6-dev \
    libignition-plugin-dev \
    libpoco-dev \
    libgtest-dev libgmock-dev \
    && rm -rf /var/lib/apt/lists/*

# Runtime inference dependencies.  The unified Regolo VLM supplies the 2D boxes;
# sentence-transformers is used for semantic matching and EfficientViT-SAM uses the
# CUDA ONNX Runtime provider for the 3D lift.  Keep these in the image so a run never
# falls back because an optional Python package is missing.
# ONNX Runtime 1.18.1 is built against the CUDA 11 ABI.  PyTorch in this image
# remains on CUDA 12.1, so keep the CUDA 11 math/runtime libraries side by side;
# the image's existing CUDA 12 cuDNN 8 library satisfies both consumers.
RUN apt-get remove -y python3-sympy

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN python3 -m pip install --no-cache-dir \
    "setuptools<80" \
    "packaging>=22" \
    "numpy>=1.24.0,<2.0" \
    "scipy>=1.15.0,<1.16" \
    "Pillow>=10.0.0" \
    "PyYAML>=6.0" \
    "sentence-transformers==3.0.1" \
    "gensim>=4.3.0" \
    "webcolors" \
    "torchvision==0.29.1" \
    "timm>=0.9.0" \
    "onnx>=1.14.0" \
    "onnxsim>=0.4.0" \
    "segment-anything>=1.0" \
    "matplotlib" \
    "uvicorn" \
    "fastapi" \
    "openai>=1.40.0" \
    "onnxruntime-gpu==1.18.1" \
    "nvidia-cublas-cu11" \
    "nvidia-cudnn-cu11==8.9.6.50" \
    "nvidia-cuda-runtime-cu11" \
    "nvidia-curand-cu11" \
    "nvidia-cufft-cu11"

# Make the CUDA libraries bundled in the image discoverable by both PyTorch and
# ONNX Runtime.  The cu11/cu12 wheels use the same package directories but keep
# different SONAMEs (for example libcublas.so.11 and libcublas.so.12), so this
# is safe for the mixed PyTorch/cu12 + ONNX Runtime/cu11 stack above.
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib"

RUN rosdep update

# ── TIAGo workspace ────────────────────────────────────────────────────────────
ENV WS=/root/tiago_public_ws
RUN mkdir -p ${WS}/src
WORKDIR ${WS}

RUN vcs import --input https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/humble-devel/tiago_public.repos src

RUN rosdep install --from-paths src -y --ignore-src --skip-keys="moveit" || true
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

WORKDIR /root/exchange
CMD ["/bin/bash"]
