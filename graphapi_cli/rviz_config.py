"""Use the acquisition's frames and camera topics in every RViz launcher."""
import copy


def settings(mode, cfg, env):
    tiago = mode in ("tiago", "bag")
    return {
        "rgb": env.get("TIAGO_RGB_TOPIC", "/head_front_camera/rgb/image_raw") if tiago else "/camera/rgb",
        "depth": env.get("TIAGO_DEPTH_TOPIC", "/head_front_camera/depth/image_raw") if tiago else "/camera/depth",
        "camera_info": env.get("TIAGO_CAMERA_INFO_TOPIC", "/head_front_camera/rgb/camera_info") if tiago else "/camera/camera_info",
        "camera_frame": env.get("TIAGO_CAMERA_FRAME", cfg.get("frames", {}).get("camera", "head_front_camera_color_optical_frame")) if tiago else cfg.get("frames", {}).get("camera", "habitat_camera_optical"),
        "robot_frame": "base_footprint" if tiago else "base_link",
        "use_sim_time": tiago and env.get("FOUND_USE_SIM_TIME", "true" if mode == "bag" else "false").lower() in ("1", "true"),
        "RMW_IMPLEMENTATION": env.get("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp" if tiago else "rmw_fastrtps_cpp"),
        "ROS_LOCALHOST_ONLY": "1" if mode == "bag" else "0",
        "CYCLONEDDS_URI": "" if mode == "bag" else env.get("CYCLONEDDS_URI", ""),
    }


def configure(layout, cfg, mode, view):
    layout = copy.deepcopy(layout)
    manager = layout["Visualization Manager"]
    world = cfg.get("tf", {}).get("world_frame", "map")
    bev = cfg.get("bev", {})
    map_topic = bev.get("map_topic", "/rtabmap/map")
    cloud_topic = bev.get("cloud_map_topic", "/rtabmap/cloud_map")
    manager["Global Options"]["Fixed Frame"] = world
    manager["Views"]["Current"]["Target Frame"] = view["robot_frame"]
    for display in manager["Displays"]:
        kind = display.get("Class", "")
        topic = display.get("Topic", {})
        if kind == "rviz_default_plugins/TF":
            display["Frames"] = {"All Enabled": False, **{frame: {"Value": True}
                                 for frame in (world, "odom", view["robot_frame"], view["camera_frame"])}}
        elif kind == "rviz_default_plugins/Map":
            topic["Value"] = map_topic
            display["Name"] = f"Map ({map_topic})"
            display["Enabled"] = display["Value"] = bool(map_topic)
            display["Update Topic"]["Value"] = map_topic + "_updates" if map_topic else ""
        elif kind == "rviz_default_plugins/PointCloud2" and topic.get("Value") == "/rtabmap/cloud_map":
            topic["Value"] = cloud_topic
            display["Enabled"] = display["Value"] = bool(cloud_topic)
        elif kind == "rviz_default_plugins/Image":
            if topic.get("Value") == "/camera/rgb":
                topic["Value"] = view["rgb"]
            topic["Reliability Policy"] = "Best Effort"
        elif kind == "rviz_default_plugins/DepthCloud":
            display["Depth Map Topic"] = view["depth"]
            display["Color Image Topic"] = view["rgb"]
        elif kind == "rviz_default_plugins/Axes":
            display["Reference Frame"] = view["robot_frame"]
    return layout
