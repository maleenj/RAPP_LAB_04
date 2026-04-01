"""Launch ZED2i camera with RAPP Lab body tracking overrides.

Usage (in rapp_hw container):
    ros2 launch vam_inference zed2i_camera.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = get_package_share_directory("vam_inference")
    zed_wrapper_share = get_package_share_directory("zed_wrapper")

    override_config = os.path.join(pkg_share, "config", "zed2i_override.yaml")

    zed_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(zed_wrapper_share, "launch", "zed_camera.launch.py")
        ),
        launch_arguments={
            "camera_model": "zed2i",
            "ros_params_override_path": override_config,
        }.items(),
    )

    return LaunchDescription([zed_launch])
