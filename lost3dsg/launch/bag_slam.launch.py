import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    bag_path = LaunchConfiguration('bag_path')
    rviz_config = LaunchConfiguration('rviz_config')
    slam_params = LaunchConfiguration('slam_params')

    declare_bag_path = DeclareLaunchArgument(
        'bag_path',
        default_value='/root/exchange/lost3dsg/porte',
        description='Path to the ROS2 bag'
    )

    declare_rviz_config = DeclareLaunchArgument(
        'rviz_config',
        default_value='/root/exchange/lost3dsg/rviz/default.rviz',
        description='Path to RViz config'
    )

    declare_slam_params = DeclareLaunchArgument(
        'slam_params',
        default_value='/root/tiago_public_ws/src/pmb2_navigation/pmb2_2dnav/config/nav_public_sim.yaml',
        description='Path to SLAM params'
    )

    play_bag = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', bag_path,
            '--clock',
            '--rate', '0.5',
            '--topics',
            '/scan', '/tf', '/tf_static',
            '/head_front_camera/rgb/image_raw',
            '/head_front_camera/depth/image_raw',
            '/head_front_camera/depth/camera_info',
            '/head_front_camera/rgb/camera_info',
            '/joint_states',
            '/robot_description',
            '/mobile_base_controller/odom',
        ],
        output='screen'
    )

# 1. SLAM Toolbox: Togli il TimerAction, fallo partire SUBITO
    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            slam_params,
            {'use_sim_time': True},
            {'scan_topic': '/scan'},
        ],
    )

    # 2. RViz: Lascialo pure con un piccolo delay o subito
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}],
    )

    # 3. La BAG: Questa deve partire per ULTIMA con un delay
    # Rimuovi '--delay', '5' dagli argomenti di play_bag se lo avevi messo lì
    delayed_bag = TimerAction(
        period=5.0,
        actions=[play_bag]
    )

    return LaunchDescription([
        declare_bag_path,
        declare_rviz_config,
        declare_slam_params,
        slam_node,    # Parte al secondo 0
        rviz_node,    # Parte al secondo 0
        delayed_bag,  # Parte al secondo 5
    ])
