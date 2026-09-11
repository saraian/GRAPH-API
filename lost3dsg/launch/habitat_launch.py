#!/usr/bin/env python3
"""
Launch file per il sistema lost3dsg completo:

  1) rtabmap (SLAM RGB-D di default; ground-truth selezionabile)
  2) perception_2.py         (ros2 run lost3dsg perception_2.py)
  3) object_manager_6.py     (ros2 run lost3dsg object_manager_6.py)
  4) graph_api_bridge.py     (script Python standalone, non un nodo ROS2)
  5) rviz2

USO
---
    ros2 launch lost3dsg habitat_launch.py

    # opzioni:
    ros2 launch lost3dsg habitat_launch.py use_rviz:=false
    ros2 launch lost3dsg habitat_launch.py localization_mode:=ground_truth
    ros2 launch lost3dsg habitat_launch.py use_wall_detector:=true

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
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (EnvironmentVariable, LaunchConfiguration,
                                  PathJoinSubstitution, PythonExpression)
from launch_ros.actions import Node


def generate_launch_description():

    # ------------------------------------------------------------
    # Argomenti configurabili da CLI
    # ------------------------------------------------------------
    graph_api_bridge_dir_arg = DeclareLaunchArgument(
        'graph_api_bridge_dir',
        # DEFERS TO GRAPH_API_SRC_DIR, for the same reason metrics_output_dir defers to
        # GRAPH_API_OUTPUT_DIR below: a literal here is a path from one container layout, and this
        # one does not use it. /root/exchange/lost3dsg/src/perception_module does not exist in the
        # image; the sources are at /graph_api/lost3dsg/src/perception_module.
        #
        # MEASURED: this is the `cwd` of habitat_metrics_collector and of this file's copy of
        # graph_api_bridge, so both die at startup with
        #   FileNotFoundError: [Errno 2] No such file or directory:
        #   '/root/exchange/lost3dsg/src/perception_module'
        # The bridge survives because live_stack_container.sh starts its own. The collector does
        # not, and NO BUNDLE HAS EVER CARRIED ITS REPORT: 0 of 113 bundles hold
        # risultati_operativi.json. It failed loudly into the launch log every run and nobody read
        # that far, which is why a missing artefact looked like an artefact nobody wanted.
        #
        # The literal stays as the last resort, so a launch with nothing exported behaves as before.
        default_value=EnvironmentVariable('GRAPH_API_SRC_DIR',
                                          default_value='/root/exchange/lost3dsg/src/perception_module'),
        description="Cartella contenente graph_api_bridge.py",
    )

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description="Avvia rviz2 se true",
    )

    use_wall_detector_arg = DeclareLaunchArgument(
        'use_wall_detector',
        default_value='true',
        description="Avvia wall_detector.py se true",
    )

    collect_metrics_arg = DeclareLaunchArgument(
        'collect_metrics',
        default_value='true',
        description='Raccoglie e salva automaticamente le metriche operative alla chiusura',
    )

    metrics_output_dir_arg = DeclareLaunchArgument(
        'metrics_output_dir',
        # DEFERS TO GRAPH_API_OUTPUT_DIR, and that is the whole point. Line 264 below does
        # SetEnvironmentVariable('GRAPH_API_OUTPUT_DIR', metrics_output_dir), so a plain constant
        # here does not fall back to the exported value -- it OVERWRITES it for every node this
        # file starts. live_stack_container.sh:165 exports /ws/output, which is the bind mount to
        # the run bundle; the launch file replaced it with /root/exchange/output, which is the
        # container's own writable layer and is discarded with the container.
        #
        # MEASURED on run 20260910_164624: perception logged "[ARCHIVE] per-detection archiving ON
        # -> /root/exchange/output" and wrote frames/, depth/ and detections.jsonl there, so the
        # bundle held none and the dashboard's replay timeline was empty. Same shape as GA-463
        # (rtabmap's database written to /root/.ros, outside every mount).
        #
        # The literal stays as the last resort, so a launch with nothing exported behaves as before.
        default_value=EnvironmentVariable('GRAPH_API_OUTPUT_DIR',
                                          default_value='/root/exchange/output'),
        description='Directory condivisa degli artefatti e del report operativo',
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

    localization_mode_arg = DeclareLaunchArgument(
        'localization_mode',
        default_value='rtabmap',
        choices=['rtabmap', 'ground_truth'],
        description="Sorgente della posa: SLAM RGB-D RTAB-Map oppure ground truth Habitat",
    )

    odom_args_arg = DeclareLaunchArgument(
        'odom_args',
        default_value='--Odom/Strategy 1 --Odom/GuessMotion false --Odom/ResetCountdown 0',
        description="Odometry frame-to-frame senza reset automatici che spezzano la mappa",
    )

    graph_api_bridge_dir = LaunchConfiguration('graph_api_bridge_dir')
    use_rviz = LaunchConfiguration('use_rviz')
    use_wall_detector = LaunchConfiguration('use_wall_detector')
    collect_metrics = LaunchConfiguration('collect_metrics')
    metrics_output_dir = LaunchConfiguration('metrics_output_dir')
    perception_delay = LaunchConfiguration('perception_delay')
    bridge_delay = LaunchConfiguration('bridge_delay')
    rtabmap_output = LaunchConfiguration('rtabmap_output')
    localization_mode = LaunchConfiguration('localization_mode')
    use_rtabmap_tf = PythonExpression([
        "'true' if '", localization_mode, "' == 'rtabmap' else 'false'"
    ])

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
            # Habitat publishes encoder-like dead reckoning on /odom.
            'visual_odometry': 'false',
            'odom_topic': '/odom',
            'frame_id': 'base_link',
            'vo_frame_id': 'odom',
            # Vuoto: il nodo SLAM sincronizza il messaggio /odom con RGB-D.
            # Usare "odom" qui forza invece un lookup TF al timestamp dell'immagine:
            # quando rgbd_odometry e' in ritardo, quel transform non esiste ancora.
            'odom_frame_id': '',
            'odom_args': LaunchConfiguration('odom_args'),
            'rgb_topic': '/camera/rgb',
            'depth_topic': '/camera/depth',
            'camera_info_topic': '/camera/camera_info',
            # Habitat pubblica RGB, depth, camera_info e odometria con lo stesso stamp.
            'approx_sync': 'false',
            'rtabmap_viz': 'false',
            'publish_tf_odom': 'false',
            'publish_tf_map': use_rtabmap_tf,
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

    wall_detector_node = Node(
        package='lost3dsg',
        executable='wall_detector.py',
        name='wall_detector',
        output='screen',
        condition=IfCondition(use_wall_detector),
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

    metrics_collector = ExecuteProcess(
        cmd=[
            'python3', 'habitat_metrics_collector.py',
            '--run-dir', metrics_output_dir,
            '--output', PathJoinSubstitution([
                metrics_output_dir, 'risultati_operativi.json'
            ]),
        ],
        cwd=graph_api_bridge_dir,
        output='screen',
        name='habitat_metrics_collector',
        condition=IfCondition(collect_metrics),
    )

    # ------------------------------------------------------------
    # 4) rviz2
    # ------------------------------------------------------------
    # GA-479. RVIZ_CONFIG names a .rviz file to open with. EMPTY BY DEFAULT, so a plain
    # `ros2 launch lost3dsg habitat_launch.py` still opens rviz with its own defaults and this
    # file behaves exactly as before. live_stack_container.sh sets it to
    # /graph_api/lost3dsg/test/live.rviz, which carries the map, the clouds, the object markers
    # and the exploration-schedule display on /schedule_markers.
    #
    # Read from the environment rather than as a launch argument because the value is a path
    # decided by the launcher, and an `arguments` list has to be built here, once. A missing file
    # is IGNORED WITH A MESSAGE: rviz2 exits immediately on a `-d` it cannot open, and losing the
    # whole viewer over a mistyped path is worse than losing the layout.
    _rviz_args = ['--fixed-frame', 'map']
    _rviz_cfg = os.environ.get('RVIZ_CONFIG', '').strip()
    if _rviz_cfg and os.path.isfile(_rviz_cfg):
        _rviz_args += ['-d', _rviz_cfg]
    elif _rviz_cfg:
        print(f"[habitat_launch] RVIZ_CONFIG={_rviz_cfg} does not exist; "
              "starting rviz2 with its own defaults")

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        # La vista resta nel riferimento globale anche quando SLAM corregge map->odom.
        arguments=_rviz_args,
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
        actions=[perception_node, object_manager_node, wall_detector_node],
    )

    delayed_bridge = TimerAction(
        period=bridge_delay,
        actions=[graph_api_bridge],
    )

    return LaunchDescription([
        localization_mode_arg,
        odom_args_arg,
        graph_api_bridge_dir_arg,
        use_rviz_arg,
        use_wall_detector_arg,
        collect_metrics_arg,
        metrics_output_dir_arg,
        perception_delay_arg,
        bridge_delay_arg,
        rtabmap_output_arg,
        SetEnvironmentVariable('GRAPH_API_OUTPUT_DIR', metrics_output_dir),
        metrics_collector,
        rtabmap_launch,
        delayed_perception,
        delayed_bridge,
        rviz_node,
    ])
