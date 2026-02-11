"""Launch VAM inference node with robot_state_publisher and RViz2.

Usage:
    # Live skeleton tracking
    ros2 launch vam_inference vam_rviz.launch.py

    # Rosbag playback
    ros2 bag play /data/rosbags/<name> --clock
    ros2 launch vam_inference vam_rviz.launch.py use_sim_time:=true
"""

import os
import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


VAM_PREFIX = "vam/"


def _prefix_urdf(urdf_path: str, prefix: str) -> str:
    """Read a URDF, remove the world link/joint, and prefix all remaining
    link frame names.

    This makes the prefixed base_link the URDF root so we can attach it
    directly to the real robot's base_link via a static identity transform,
    guaranteeing the bases overlap regardless of bag TF.
    """
    urdf = Path(urdf_path).read_text()

    # Remove <link name="world"/> (self-closing or empty)
    urdf = re.sub(r'<link\s+name="world"\s*/>', '', urdf)
    urdf = re.sub(r'<link\s+name="world"\s*>\s*</link>', '', urdf)

    # Remove the single-line comment before base_joint (if present)
    urdf = re.sub(r'<!--[^\n]*base_joint[^\n]*-->\s*', '', urdf)

    # Remove the base_joint element that connects world → base_link
    urdf = re.sub(
        r'<joint\s+name="base_joint"\s+type="fixed">.*?</joint>',
        '',
        urdf,
        flags=re.DOTALL,
    )

    # Collect remaining link names
    link_names = re.findall(r'<link\s+name="([^"]+)"', urdf)

    # Prefix every link name used as a frame reference
    for name in link_names:
        prefixed = prefix + name
        urdf = urdf.replace(f'<link name="{name}"', f'<link name="{prefixed}"')
        urdf = urdf.replace(f'link="{name}"', f'link="{prefixed}"')

    return urdf


def generate_launch_description():
    pkg_share = get_package_share_directory("vam_inference")
    rviz_config = os.path.join(pkg_share, "config", "vam.rviz")
    urdf_path = "/data/processed/ur10.urdf"

    # Build the prefixed URDF (no world link, root = vam/base_link)
    vam_urdf = _prefix_urdf(urdf_path, VAM_PREFIX)

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("mode", default_value="rviz"),
            DeclareLaunchArgument("rviz_config", default_value=rviz_config),

            # --- Robot state publisher (actual robot URDF, driven by /joint_states) ---
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                parameters=[
                    {
                        "robot_description": open(urdf_path).read(),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),

            # --- Second robot state publisher for VAM predictions ---
            # Prefixed URDF with world link removed (root = vam/base_link).
            # Attached to real base_link via static transform below.
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="vam_robot_state_publisher",
                namespace="vam",
                parameters=[
                    {
                        "robot_description": vam_urdf,
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
                remappings=[
                    ("joint_states", "/vam/joint_states"),
                ],
            ),

            # --- Static transforms ---
            # map → world: global root for RViz fixed frame
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_world",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "map",
                    "--child-frame-id", "world",
                ],
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}
                ],
            ),
            # base_link → vam/base_link: pins the VAM robot to the exact
            # same base position as the real robot, regardless of any
            # world→base_link offset in the bag's TF.
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_vam_base",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "base_link",
                    "--child-frame-id", "vam/base_link",
                ],
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}
                ],
            ),

            # --- VAM inference node ---
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_share, "launch", "vam_inference.launch.py")
                ),
                launch_arguments={
                    "mode": LaunchConfiguration("mode"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                }.items(),
            ),

            # --- RViz2 ---
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", LaunchConfiguration("rviz_config")],
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}
                ],
            ),
        ]
    )
