FROM ros:humble
ENV DEBIAN_FRONTEND=noninteractive

# Base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    python3-vcstool python3-rosdep python3-colcon-common-extensions \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox \
    ros-humble-xacro ros-humble-robot-state-publisher ros-humble-joint-state-publisher-gui \
    ros-humble-rmw-cyclonedds-cpp ros-humble-rmw-fastrtps-cpp \
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

RUN rosdep update

# ── TIAGo workspace ────────────────────────────────────────────────────────────
ENV WS=/root/tiago_public_ws
RUN mkdir -p ${WS}/src
WORKDIR ${WS}

RUN vcs import --input https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/humble-devel/tiago_public.repos src

RUN rosdep install --from-paths src -y --ignore-src --skip-keys="moveit" || true
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install

# ── Franka Emika Panda workspace ───────────────────────────────────────────────
ENV FRANKA_WS=/root/franka_ws
RUN mkdir -p ${FRANKA_WS}/src
WORKDIR ${FRANKA_WS}/src

RUN git clone -b humble https://github.com/frankarobotics/franka_ros2.git

WORKDIR ${FRANKA_WS}
RUN vcs import src < src/franka_ros2/dependency.repos --recursive --skip-existing
RUN rosdep install --from-paths src --ignore-src --rosdistro humble -y \
    --skip-keys="gz_ros2_control realsense2_camera realsense2_description joy" || true
RUN . /opt/ros/humble/setup.sh && \
    . ${WS}/install/setup.sh && \
    colcon build --symlink-install \
        --packages-select libfranka \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF && \
    colcon build --symlink-install \
        --packages-skip libfranka franka_gazebo_hardware franka_gazebo_bringup franka_bringup franka_fr3_moveit_config franka_ros2 \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF

# Source everything in bashrc
RUN echo ". /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo ". ${WS}/install/setup.bash" >> ~/.bashrc && \
    echo ". ${FRANKA_WS}/install/setup.bash" >> ~/.bashrc

WORKDIR /root/exchange
CMD ["/bin/bash"]
