#!/usr/bin/env python3
"""
Launch file per il sistema lost3dsg completo:

  1) rtabmap (SLAM, senza odometria visiva, TF non pubblicato)
  2) perception_2.py         (ros2 run lost3dsg perception_2.py)
  3) object_manager_6.py     (ros2 run lost3dsg object_manager_6.py)
  4) graph_api_bridge.py     (script Python standalone, non un nodo ROS2)
  5) rviz2

USO
---
    ros2 launch lost3dsg full_system.launch.py

    # opzioni:
    ros2 launch lost3dsg full_system.launch.py use_rviz:=false
    ros2 launch lost3dsg full_system.launch.py graph_api_bridge_dir:=/altro/path

INSTALLAZIONE
-------------
Copia questo file in:
    ~/exchange/lost3dsg/src/lost3dsg/launch/full_system.launch.py

Nel setup.py del pacchetto lost3dsg assicurati che la cartella launch/
venga installata, ad esempio:

    import os
    from glob import glob
    ...
    data_files=[
        ...
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],

Poi ricompila:
    cd ~/exchange/lost3dsg
    colcon build --packages-select lost3dsg
    source install/setup.bash
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
    DeclareLaunchArgument,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ------------------------------------------------------------
    # Argomenti configurabili da CLI
    # ------------------------------------------------------------
    graph_api_bridge_dir_arg = DeclareLaunchArgument(
        'graph_api_bridge_dir',
        default_value=os.path.expanduser(
            '~/exchange/lost3dsg/src/perception_module'
        ),
        description="Cartella contenente graph_api_bridge.py",
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description="Avvia rviz2 se true",
    )

    perception_delay_arg = DeclareLaunchArgument(
        'perception_delay',
        default_value='3.0',
        description="Secondi di attesa prima di avviare i nodi di percezione",
    )

    bridge_delay_arg = DeclareLaunchArgument(
        'bridge_delay',
        default_value='5.0',
        description="Secondi di attesa prima di avviare graph_api_bridge.py",
    )

    rtabmap_output_arg = DeclareLaunchArgument(
        'rtabmap_output',
        default_value='log',
        description="'screen' per vedere i log di rtabmap in terminale, "
                    "'log' per mandarli solo su file (~/.ros/log/...)",
    )

    graph_api_bridge_dir = LaunchConfiguration('graph_api_bridge_dir')
    use_rviz = LaunchConfiguration('use_rviz')
    perception_delay = LaunchConfiguration('perception_delay')
    bridge_delay = LaunchConfiguration('bridge_delay')
    rtabmap_output = LaunchConfiguration('rtabmap_output')

    # ------------------------------------------------------------
    # 1) rtabmap
    # ------------------------------------------------------------
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rtabmap_launch'),
                'launch',
                'rtabmap.launch.py',
            )
        ),
        launch_arguments={
            'visual_odometry': 'false',
            'odom_topic': '/odom',
            'rgb_topic': '/camera/rgb',
            'depth_topic': '/camera/depth',
            'camera_info_topic': '/camera/camera_info',
            'approx_sync': 'true',
            'rtabmap_viz': 'false',
            'publish_tf': 'false',
            'database_path': '/root/.ros/rtabmap.db',
            'rtabmap_args': '--delete_db_on_start --RGBD/NeighborLinkRefining false',
            'output': rtabmap_output,
        }.items(),
    )

    # ------------------------------------------------------------
    # 2) Nodi ROS2 del pacchetto lost3dsg
    # ------------------------------------------------------------
    perception_node = Node(
        package='lost3dsg',
        executable='perception_2.py',
        name='perception_2',
        output='screen',
    )

    object_manager_node = Node(
        package='lost3dsg',
        executable='object_manager_6.py',
        name='object_manager_6',
        output='screen',
    )

    # ------------------------------------------------------------
    # 3) graph_api_bridge.py — script standalone (non un nodo ROS2
    #    installato), lanciato con ExecuteProcess impostando la cwd.
    # ------------------------------------------------------------
    graph_api_bridge = ExecuteProcess(
        cmd=['python3', 'graph_api_bridge.py'],
        cwd=graph_api_bridge_dir,
        output='screen',
        name='graph_api_bridge',
    )

    # ------------------------------------------------------------
    # 4) rviz2
    # ------------------------------------------------------------
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        condition=IfCondition(use_rviz),
    )

    # ------------------------------------------------------------
    # Ritardi: diamo tempo a rtabmap di avviarsi prima di far
    # partire percezione e bridge, per evitare race condition sui
    # topic. Regola i valori con perception_delay / bridge_delay.
    # ------------------------------------------------------------
    delayed_perception = TimerAction(
        period=perception_delay,
        actions=[perception_node, object_manager_node],
    )

    delayed_bridge = TimerAction(
        period=bridge_delay,
        actions=[graph_api_bridge],
    )

    return LaunchDescription([
        graph_api_bridge_dir_arg,
        use_rviz_arg,
        perception_delay_arg,
        bridge_delay_arg,
        rtabmap_output_arg,
        rtabmap_launch,
        delayed_perception,
        delayed_bridge,
        rviz_node,
    ])
