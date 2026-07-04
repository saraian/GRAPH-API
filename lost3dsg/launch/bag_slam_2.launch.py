import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():

    # ── 1. Argomenti ──────────────────────────────────────────────────────────
    bag_path    = LaunchConfiguration('bag_path')
    rviz_config = LaunchConfiguration('rviz_config')
    slam_params = LaunchConfiguration('slam_params')

    # ── 2. Descrizione robot (solo per robot_description, NON per TF) ─────────
    xacro_file = '/root/tiago_public_ws/src/tiago_robot/tiago_description/robots/tiago.urdf.xacro'
    robot_description_content = Command(['xacro ', xacro_file])

    # ── 3. Riproduzione BAG ───────────────────────────────────────────────────
    # La bag contiene /tf e /tf_static → li lasciamo passare dalla bag.
    # --clock pubblica il sim-time, --rate 1.0 = velocità reale.
    play_bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', bag_path,
            '--clock',
            '--rate', '1.0',
        ],
        output='screen'
    )

    # ── 4. Robot State Publisher ──────────────────────────────────────────────
    # IMPORTANTE: reindirizzamo /tf e /tf_static su topic ignorati perché
    # la bag li pubblica già con i timestamp corretti.
    # RSP serve solo per rendere disponibile /robot_description ai nodi che lo
    # leggono (es. RViz per visualizzare il modello 3D).
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        remappings=[
            ('/tf',        '/tf_rsp_ignored'),
            ('/tf_static', '/tf_static_rsp_ignored'),
        ],
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_description_content,
        }],
        output='screen'
    )

    # ── 5. SLAM Toolbox ───────────────────────────────────────────────────────
    # Ritardo di 10 s: dà tempo alla bag di iniziare a pubblicare /clock e /tf
    # prima che SLAM tenti di fare lookup delle trasformazioni.
    slam_node = TimerAction(period=10.0, actions=[
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            parameters=[
                slam_params,
                {
                    'use_sim_time': True,
                    'scan_topic':   '/scan',        # presente nella bag (8358 msg)
                    'odom_frame':   'odom',
                    'base_frame':   'base_footprint',
                    'map_frame':    'map',
                    'mode':         'mapping',
                }
            ],
            output='screen'
        )
    ])

    # ── 6. RViz2 ──────────────────────────────────────────────────────────────
    # Ritardo di 15 s: aspetta che SLAM abbia iniziato a costruire la mappa.
    rviz_node = TimerAction(period=15.0, actions=[
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': True}],
            output='screen'
        )
    ])

    # ── 7. Launch Description ─────────────────────────────────────────────────
    return LaunchDescription([
        DeclareLaunchArgument(
            'bag_path',
            default_value='/root/exchange/portone'
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value='/root/exchange/lost3dsg/rviz/default.rviz'
        ),
        DeclareLaunchArgument(
            'slam_params',
            default_value='/root/tiago_public_ws/src/pmb2_navigation/pmb2_2dnav/config/nav_public_sim.yaml'
        ),

        play_bag,
        rsp_node,
        slam_node,
        rviz_node,
    ])
