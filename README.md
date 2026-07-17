# GRAPH API
Per far partire tutto, in 4 diversi terminali: 

ros2 launch simulation.launch.py se Gazebo
ros2 run lost3dsg habitat_camera_node.py

ros2 run lost3dsg perception.py --ros-args -p use_sim_time:=True

ros2 run lost3dsg object_manager_6.py

cd lost3dsg/src/perception_module
python3 graph_api_bridge.py
