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

    # Xacro: Caricamento pulito per evitare crash di macro
    xacro_file = '/root/tiago_public_ws/src/tiago_robot/tiago_description/robots/tiago.urdf.xacro'
    robot_description_content = Command(['xacro ', xacro_file])

    # 2. Riproduzione BAG (Clock attivo e rate controllato per stabilità TF)
    play_bag = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag_path, '--clock', '--rate', '0.2'],
        output='screen'
    )

    # 3. Robot State Publisher
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'use_sim_time': True, 
            'robot_description': robot_description_content
        }]
    )

    # 4. CATENA DI TRASFORMAZIONI STATICHE (I "PONTI")
    # Questa sezione incolla i pezzi che RViz vede "scollegati" (rossi)
    
    # Ponte Base: collega il movimento alla struttura del robot
    tf_base = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='fix_base',
        arguments=['0', '0', '0', '0', '0', '0', 'base_footprint', 'torso_fixed_link'],
        parameters=[{'use_sim_time': True}]
    )

    # --- RAMO MANO HEY5 ---
    # Collega il braccio (7) alla radice della mano (hand_link)
    tf_wrist_to_hand = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='fix_hand_root',
        arguments=['0', '0', '0', '0', '0', '0', 'arm_7_link', 'hand_link'],
        parameters=[{'use_sim_time': True}]
    )

    # Alias per i link della mano (molti link della bag cercano 'hand' come parent)
    tf_hand_alias = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='fix_hand_alias',
        arguments=['0', '0', '0', '0', '0', '0', 'hand_link', 'hand'],
        parameters=[{'use_sim_time': True}]
    )

    # --- RAMO GRIPPER ---
    # Collega il braccio (7) al gripper_link (quello che dava errore "to map")
    tf_wrist_to_gripper = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='fix_gripper_link',
        arguments=['0', '0', '0', '0', '0', '0', 'arm_7_link', 'gripper_link'],
        parameters=[{'use_sim_time': True}]
    )

    # Ponte per il frame di grasping (fondamentale per lo SLAM e la navigazione)
    tf_grasping_fix = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='fix_grasping',
        arguments=['0.1', '0', '0', '0', '0', '0', 'gripper_link', 'hand_grasping_frame'],
        parameters=[{'use_sim_time': True}]
    )

    # 5. SLAM Toolbox (Avvio ritardato per attendere la stabilità del clock)
    slam_node = TimerAction(period=5.0, actions=[
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            parameters=[
                slam_params,
                {'use_sim_time': True},
                {'scan_topic': '/scan'}
            ],
            output='screen'
        )
    ])

    # 6. RViz2 (L'ultimo ad avviarsi per mostrare tutto già pronto)
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
        DeclareLaunchArgument('bag_path', default_value='/root/exchange/portone'),
        DeclareLaunchArgument('rviz_config', default_value='/root/exchange/lost3dsg/rviz/default.rviz'),
        DeclareLaunchArgument('slam_params', default_value='/root/tiago_public_ws/src/pmb2_navigation/pmb2_2dnav/config/nav_public_sim.yaml'),
        play_bag,
        rsp_node,
        tf_base,
        tf_wrist_to_hand,
        tf_hand_alias,
        tf_wrist_to_gripper,
        tf_grasping_fix,
        slam_node,
        rviz_node
    ])
