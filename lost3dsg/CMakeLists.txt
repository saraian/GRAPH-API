cmake_minimum_required(VERSION 3.8)
project(lost3dsg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# find dependencies
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Centroid.msg"
  "msg/CentroidArray.msg"
  "msg/Bbox3d.msg"
  "msg/Bbox3dArray.msg"
  "msg/ObjectDescription.msg"
  "msg/ObjectDescriptionArray.msg"
  "srv/ObjectTrackingService.srv"
  "srv/AddObject.srv"              
  "srv/RemoveObject.srv"
  "srv/UpdateObject.srv"
  "srv/MergeObjects.srv"
  "srv/DeleteObjects.srv"
  "srv/QueryObjects.srv"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

# Install Python scripts
install(PROGRAMS
  src/perception_module/perception_2.py
  src/perception_module/cv_utils.py
  src/perception_module/utils.py
  src/perception_module/detection_pipeline.py
  src/perception_module/detection_types.py
  src/perception_module/input_output.py
  src/perception_module/perception_utils.py
  src/perception_module/object_manager_6.py
  src/perception_module/object_services.py
  src/perception_module/room_manager.py
  src/perception_module/habitat_camera_node.py
  src/perception_module/habitat_feed_node.py
  src/perception_module/walls_rooms.py
  src/perception_module/vlm_call.py
  # sibling modules the nodes import at runtime — without these the installed
  # copies die with ImportError under `ros2 run`
  src/perception_module/config.py
  src/perception_module/box_view.py
  src/perception_module/hooks.py
  src/perception_module/nlp_utils.py
  src/perception_module/world_model.py
  src/perception_module/object_info.py
  src/perception_module/map_database.py
  src/perception_module/models.py
  src/perception_module/query_map.py
  src/perception_module/graph_api_bridge.py
  src/perception_module/wall_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install data files (text prompts, runtime config, web viewer)
install(FILES
  src/perception_module/config.yaml
  DESTINATION lib/${PROJECT_NAME}
)
install(FILES
  src/perception_module/prompts/object_identification_prompt.txt
  src/perception_module/prompts/visual_prompt.txt
  DESTINATION lib/${PROJECT_NAME}/prompts
)
install(DIRECTORY src/perception_module/viewer
  DESTINATION lib/${PROJECT_NAME}
)
# vendored library imported by models.py (VitSam preprocessing)
install(DIRECTORY src/perception_module/efficientvit
  DESTINATION lib/${PROJECT_NAME}
)


# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
