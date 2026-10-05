# Complete public simulation runtime. Both Habitat and ROS execute inside Docker.
# Gazebo/Franka retain their existing separate public image (root Dockerfile).
FROM ros:humble
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-rtabmap-ros ros-humble-cv-bridge ros-humble-tf2-ros \
    ros-humble-rosbag2 ros-humble-rosbag2-storage-default-plugins ros-humble-rmw-cyclonedds-cpp \
    python3-pip python3-colcon-common-extensions git curl ca-certificates \
    libopencv-dev libpcl-dev libeigen3-dev ros-humble-libg2o ros-humble-gtsam \
    build-essential cmake ros-humble-rviz2 \
    libegl1 libgl1 libglib2.0-0 wget bzip2 patch docker.io openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*
COPY lost3dsg/test/patches/rtabmap-0.23.7-ga290-guard.patch /tmp/rtabmap-guard.patch
RUN source /opt/ros/humble/setup.bash && \
    version=$(dpkg-query -W -f='${Version}' ros-humble-rtabmap | cut -d- -f1) && \
    curl -fsSL "https://codeload.github.com/introlab/rtabmap/tar.gz/refs/tags/$version" -o /tmp/rtabmap.tar.gz && \
    mkdir /tmp/rtabmap && tar -xf /tmp/rtabmap.tar.gz -C /tmp/rtabmap --strip-components=1 && \
    cd /tmp/rtabmap && patch -p1 < /tmp/rtabmap-guard.patch && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_APP=OFF -DBUILD_EXAMPLES=OFF && \
    grep -Eq 'g2o_DIR:PATH=.*' build/CMakeCache.txt && \
    ! grep -q 'g2o_DIR:PATH=.*NOTFOUND' build/CMakeCache.txt && \
    ! grep -q 'GTSAM_DIR:PATH=.*NOTFOUND' build/CMakeCache.txt && \
    cmake --build build --target rtabmap_core -j 4 && \
    library=$(find build -name "librtabmap_core.so.$version" -print -quit) && test -n "$library" && \
    target=$(find /opt/ros/humble/lib -name "librtabmap_core.so.$version" -print -quit) && test -n "$target" && \
    cp "$library" "$target" && rm -rf /tmp/rtabmap /tmp/rtabmap.tar.gz
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib"
COPY lost3dsg/requirements.txt /tmp/requirements.txt
# Use ROS's cv_bridge-compatible OpenCV instead of replacing it with a pip wheel.
RUN python3 -m pip install --upgrade "pip>=24,<26" && sed '/^opencv-python/d' /tmp/requirements.txt > /tmp/ros-requirements.txt && \
    python3 -m pip install --no-cache-dir -r /tmp/ros-requirements.txt \
    'numpy==1.26.4' 'scipy==1.11.4' 'torch==2.2.2' 'torchvision==0.17.2' \
    'transformers==4.44.2' 'httpx==0.27.2' 'sentence-transformers==3.0.1' \
    nvidia-cublas-cu11 nvidia-cuda-runtime-cu11 nvidia-curand-cu11 nvidia-cufft-cu11 \
    rdflib pyoxigraph webcolors fastapi uvicorn requests httpx reportlab \
    matplotlib networkx scikit-image scikit-learn flask pandas modal psutil
# Build the existing Habitat release for ROS's Python ABI. Legacy ROS+Habitat nodes
# and the normal feed/evaluation now share one fully containerized environment.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev ninja-build libjpeg-dev libglm-dev libgl1-mesa-dev libegl1-mesa-dev \
    xorg-dev freeglut3-dev && rm -rf /var/lib/apt/lists/* && \
    git clone --depth 1 --branch v0.3.3 https://github.com/facebookresearch/habitat-sim.git /tmp/habitat-sim && \
    cd /tmp/habitat-sim && \
    python3 -m pip install --no-cache-dir -r requirements.txt 'numpy==1.26.4' && \
    python3 setup.py build_ext --parallel 4 install --headless --bullet && \
    mkdir -p /opt/habitat/bin && ln -s /usr/bin/python3 /opt/habitat/bin/python && \
    rm -rf /tmp/habitat-sim
COPY lost3dsg /opt/graphapi_ws/src/lost3dsg
COPY config /opt/graphapi_ws/config
# Compatibility resource symlinks resolve relative to the copied package.
RUN cp -r /opt/graphapi_ws/config /opt/graphapi_ws/src/config && \
    find /opt/graphapi_ws/src/lost3dsg -type l -exec sh -c 'for link; do cp -L "$link" "$link.copy" && mv "$link.copy" "$link"; done' sh {} + && \
    source /opt/ros/humble/setup.bash && cd /opt/graphapi_ws && \
    colcon build --packages-select lost3dsg --cmake-args -DCMAKE_BUILD_TYPE=Release
# Runtime dependency used by perception_2. Keep this final layer independent of
# the expensive Habitat/RTAB-Map source builds when updating ROS dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-tf-transformations && rm -rf /var/lib/apt/lists/*
ARG RUNTIME_BUILD_REVISION=manual
LABEL graphapi.runtime="2" graphapi.runtime.build="${RUNTIME_BUILD_REVISION}"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
WORKDIR /graph_api
CMD ["bash"]
