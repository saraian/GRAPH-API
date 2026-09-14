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
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Centroid.msg"
  "msg/CentroidArray.msg"
  "msg/ObservationRef.msg"
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
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs builtin_interfaces
)

# Install Python scripts
install(PROGRAMS
  # The offline evaluator. run_hm3d_metrics.py imports all four; without them installed the
  # node dies at startup with ModuleNotFoundError (GA-128).
  src/perception_module/build_hm3d_eval_manifest.py
  src/perception_module/object_metrics.py
  src/perception_module/metrics_eval_visualize.py
  src/perception_module/extract_hovsg_object_embeddings.py
  src/perception_module/perception_2.py
  src/perception_module/cv_utils.py
  src/perception_module/utils.py
  src/perception_module/detection_pipeline.py
  src/perception_module/detection_types.py
  src/perception_module/input_output.py
  src/perception_module/perception_utils.py
  src/perception_module/object_manager_6.py
  src/perception_module/object_services.py
  src/perception_module/scan_hook.py
  src/perception_module/room_manager.py
  # GA-299. habitat_camera_node.py is NOT installed: it is a HOST-side node. It imports
  # habitat_sim, which is not in graphapi-run:humble and is not meant to be -- habitat renders on
  # the host in the conda env and reaches the container over the TCP feed. Installing it put a
  # module in the image that cannot be imported there, and smoke_test.sh step 4, which derives its
  # list FROM THE TREE rather than by hand (GA-84), correctly failed on it.
  #
  # The install list is the declaration of what belongs in the ROS image, so the fix belongs here,
  # not in the test: a name list in the test would be the hand-maintained inventory GA-84 removed.
  # Nothing in the container imports or runs it -- habitat_feed_node.py only NAMES it in comments,
  # one of which already said "that module also imports habitat_sim".
  src/perception_module/habitat_feed_node.py
  src/perception_module/graph_api_bridge.py
  src/perception_module/check_mapping_tf.py
  src/perception_module/ros_bev.py
  src/perception_module/live_overlay.py
  src/perception_module/vlm_call.py
  src/perception_module/scene_analysis.py
  src/perception_module/label_norm.py
  # sibling modules the nodes import at runtime — without these the installed
  # copies die with ImportError under `ros2 run`
  src/perception_module/config.py
  src/perception_module/bbox_fusion.py
  src/perception_module/bbox_fusion_parallel.py
  src/perception_module/box_view.py
  src/perception_module/hooks.py
  src/perception_module/nlp_utils.py
  src/perception_module/world_model.py
  src/perception_module/object_info.py
  src/perception_module/map_database.py
  src/perception_module/models.py
  src/perception_module/clip_embedder.py
  src/perception_module/image_transport.py
  src/perception_module/query_map.py
  # Imported by habitat_metrics_collector.py / openai_habitat_assistant.py / run_habitat_script.py
  # but never declared, so an installed copy dies with ModuleNotFoundError at startup. Found by
  # test_perception_smoke.py's GA-128 check, which derives its list FROM THE TREE.
  src/perception_module/habitat_run_metrics.py
  src/perception_module/script_runner.py
  src/perception_module/graph_api_bridge.py
  src/perception_module/wall_detector.py
  # GA-128. These four are imported by perception_2 / each other and MUST be installed, or
  # the node dies at startup with ModuleNotFoundError after a gate that passed -- this list
  # is a hand-maintained inventory and nothing checks it against the imports.
  src/perception_module/detection_archive.py
  src/perception_module/ga493_replay_capture.py
  src/perception_module/crop_context.py
  src/perception_module/gt_codec.py
  src/perception_module/visual_reid.py
  src/perception_module/association.py
  src/perception_module/detection_index.py
  # GA-209's grid describer, imported by perception_2 and missing from this list — caught by
  # check_install_list.py in the smoke suite, which is what that check exists for.
  # Imported by build_hm3d_eval_manifest.py and metrics_eval_visualize.py and missing from this
  # list, so an installed copy died at startup. Found by check_install_list.py, which is what
  # that check exists for.
  src/perception_module/metrics_eval.py
  # Size check against the measured envelope corpus, keyed on the detector label with no
  # ontology step. Its data file is installed by the install(DIRECTORY ... data) rule below;
  # the module without that directory raises FileNotFoundError on its first lookup.
  src/perception_module/envelope_size.py
  src/perception_module/crop_grid.py
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
  src/perception_module/prompts/scene_analysis_prompt.txt
  DESTINATION lib/${PROJECT_NAME}/prompts
)
install(DIRECTORY src/perception_module/data
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY src/perception_module/viewer
  DESTINATION lib/${PROJECT_NAME}
)
# `cloud` perception backend package imported by perception_2.py
install(DIRECTORY src/perception_module/cloud
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
  find_package(ament_cmake_pytest REQUIRED)
  ament_lint_auto_find_test_dependencies()
  ament_add_pytest_test(scan_merge_test test/test_scan_merge.py)
  ament_add_pytest_test(detection_index_test test/test_detection_index.py)
  ament_add_pytest_test(bounding_box_path_test test/test_bounding_box_path.py)
endif()

ament_package()
