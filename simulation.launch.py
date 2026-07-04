from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_pal.include_utils import include_scoped_launch_py_description

def generate_launch_description():
    # 1. Path dei tuoi file
    map_yaml = LaunchConfiguration('map_yaml')

    declare_map_yaml = DeclareLaunchArgument(
        'map_yaml',
        default_value='/root/tiago_public_ws/src/pal_maps/maps/tiago_world/my_map_1.yaml',
        description='La tua mappa salvata'
    )

    # 2. Tiago Gazebo - MODIFICATO (Spawn nell'origine)
    tiago_gazebo = include_scoped_launch_py_description(
        pkg_name='tiago_gazebo',
        paths=['launch', 'tiago_gazebo.launch.py'],
        launch_arguments={
            'is_public_sim': 'True',
            'world_name': 'tiago_world',  
            'navigation': 'False',
            'slam': 'False',
            'moveit': 'True',
            'rviz': 'False',
            'tuck_arm': 'True',
            'x': '0.5',
            'y': '-0.1',
            'z': '0.0',
            'yaw': '0.0',
        }
    )

    # 3. Map Server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml}, {'use_sim_time': True}]
    )

    # 4. AMCL - MODIFICATO (0.5 metri a sinistra)
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'base_frame_id': 'base_footprint'},
            {'odom_frame_id': 'odom'},
            {'global_frame_id': 'map'},
            {'scan_topic': '/scan_raw'},
            # ---- SETTAGGIO POSIZIONE INIZIALE ----
            {'set_initial_pose': True},
            {'initial_pose.x': 0.5},
            {'initial_pose.y': -0.1},
            {'initial_pose.yaw': 0.0},
            {'initial_pose.yaw': 0.0},
        ]
    )

    # 5. Lifecycle Manager
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': ['map_server', 'amcl']},
        ]
    )

    # 6. RViz
    rviz = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', '/root/exchange/lost3dsg/rviz/default.rviz'],
                parameters=[{'use_sim_time': True}],
                output='screen',
            )
        ]
    )

    return LaunchDescription([
        declare_map_yaml,
        tiago_gazebo,
        map_server,
        amcl,
        lifecycle_manager,
        rviz,
    ])
