"""Launch the vam_viz_bridge WebSocket bridge.

    ros2 launch vam_viz_bridge viz_bridge.launch.py
    ros2 launch vam_viz_bridge viz_bridge.launch.py config:=/abs/path/bridge_channels.yaml

By default it loads the bridge_channels.yaml shipped with the package. The bridge
taps existing ROS topics, so it works against the live robot, headless inference,
or a rosbag replay with no extra setup.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("vam_viz_bridge")
    default_config = str(Path(pkg_share) / "config" / "bridge_channels.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=default_config,
                description="Path to bridge_channels.yaml (topic -> channel map).",
            ),
            Node(
                package="vam_viz_bridge",
                executable="viz_bridge_node",
                name="vam_viz_bridge",
                output="screen",
                parameters=[{"config": LaunchConfiguration("config")}],
            ),
        ]
    )
