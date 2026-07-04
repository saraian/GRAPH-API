import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node

def generate_launch_description():

    # 1. Configurazione Percorsi e Argomenti
    bag_path = LaunchConfiguration('bag_path')
    rviz_config = LaunchConfiguration('rviz_config')
    slam_params = LaunchConfiguration('slam_params')

    # Caricamento del modello Robot (Xacro) - Assicurati che il percorso sia corretto
    xacro_file = '/root/tiago_public_ws/src/tiago_robot/tiago_description/robots/tiago.urdf.xacro'
    robot_description_content = Command(['xacro ', xacro_file])

    # 2. Riproduzione BAG (Velocità 1.0 per stabilità dello SLAM)
    play_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--clock', '--rate', '1'],
        output='screen'
    )

    # 3. Robot State Publisher (Pubblica i giunti del robot dalla bag)
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': True, 
            'robot_description': robot_description_content
        }]
    )

    # 4. SLAM Toolbox (Avvio ritardato di 5 secondi per far stabilizzare il clock della bag)
    slam_node = TimerAction(period=5.0, actions=[
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            parameters=[
                slam_params,
                {'use_sim_time': True},
                {'scan_topic': '/scan'},
                {'odom_frame': 'odom'},
                {'base_frame': 'base_footprint'},
                {'map_frame': 'map'}
            ],
            output='screen'
        )
    ])

    # 5. RViz2 (Avvio ritardato di 8 secondi)
    rviz_node = TimerAction(period=8.0, actions=[
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': True}],
            output='screen'
        )
    ])

    return LaunchDescription([
        # Argomenti modificabili da riga di comando
        DeclareLaunchArgument('bag_path', default_value='/root/exchange/portone'),
        DeclareLaunchArgument('rviz_config', default_value='/root/exchange/lost3dsg/rviz/default.rviz'),
        DeclareLaunchArgument('slam_params', default_value='/root/tiago_public_ws/src/pmb2_navigation/pmb2_2dnav/config/nav_public_sim.yaml'),
        
        # Esecuzione dei nodi
        play_bag,
        rsp_node,
        slam_node,
        rviz_node
    ])
