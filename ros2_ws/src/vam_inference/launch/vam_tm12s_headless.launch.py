"""Launch VAM inference for TM12S in headless mode (no RViz).

Starts: robot_state_publishers, static transforms, and the inference node.
Run RViz separately from the ZED container where meshes are available.

Usage:
    # In VAM container — play rosbag + launch inference
    ros2 bag play /data/rosbags/<name> --clock
    ros2 launch vam_inference vam_tm12s_headless.launch.py use_sim_time:=true

    # In ZED container — open RViz with the config from shared volume
    rviz2 -d /data/processed/vam_tm12s.rviz
"""

import re
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


VAM_PREFIX = "vam/"
TM12S_PLANNING_FRAME = "base"


def _find_latest_model(prefix: str = "vam_skelonly_") -> str | None:
    """Find the most recent model directory matching a prefix."""
    models_dir = Path("/data/models")
    if not models_dir.exists():
        return None
    matches = sorted(models_dir.glob(f"{prefix}*"))
    if matches and (matches[-1] / "best.pt").exists():
        return str(matches[-1])
    return None


def _prefix_urdf_tm12s(urdf_path: str, prefix: str) -> str:
    """Read a TM12S URDF, remove world link/joint, prefix link frames."""
    urdf = Path(urdf_path).read_text()

    urdf = re.sub(r'<link\s+name="world"\s*/>', '', urdf)
    urdf = re.sub(r'<link\s+name="world"\s*>\s*</link>', '', urdf)

    for joint_name in ["base_joint", "world_joint", "world_fixed"]:
        urdf = re.sub(
            rf'<joint\s+name="{joint_name}"\s+type="fixed">.*?</joint>',
            '', urdf, flags=re.DOTALL,
        )

    link_names = re.findall(r'<link\s+name="([^"]+)"', urdf)
    for name in link_names:
        prefixed = prefix + name
        urdf = urdf.replace(f'<link name="{name}"', f'<link name="{prefixed}"')
        urdf = urdf.replace(f'link="{name}"', f'link="{prefixed}"')

    return urdf


def generate_launch_description():
    # TM12S URDF — update path once exported
    tm12s_urdf_path = "/data/processed/tm12s.urdf"
    vam_urdf = _prefix_urdf_tm12s(tm12s_urdf_path, VAM_PREFIX)

    # Auto-detect latest skeleton-only model, fall back to original
    latest_skelonly = _find_latest_model("vam_skelonly_")
    if latest_skelonly:
        default_model_dir = latest_skelonly
    else:
        default_model_dir = "/data/models/vam_skelonly_tm12_20260404_0631"

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value=f"{default_model_dir}/best.pt",
            ),
            DeclareLaunchArgument(
                "model_config_path",
                default_value=f"{default_model_dir}/model_config.json",
            ),
            DeclareLaunchArgument(
                "norm_stats_path",
                default_value="/data/processed/tensors/2026_04_04_tm12/norm_stats.pt",
            ),
            DeclareLaunchArgument("device", default_value="cuda"),
            DeclareLaunchArgument(
                "max_joint_velocity_rad_s",
                default_value="1.0",
                description="SafetyChecker velocity limit (rad/s). "
                "TM12S hardware limits are 2.27–7.85 rad/s per joint.",
            ),
            DeclareLaunchArgument(
                "max_joint_acceleration_rad_s2",
                default_value="5.0",
            ),

            # --- TM12S robot state publisher (real robot URDF, from /joint_states) ---
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                parameters=[
                    {
                        "robot_description": open(tm12s_urdf_path).read(),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),

            # --- VAM prediction robot state publisher (prefixed TM12S URDF) ---
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
            # The rosbag already publishes (from /tf_static):
            #   map → base          (calibration: x=3.6, y=-0.27, z=-0.25, yaw=π)
            #   base → link_0       (identity)
            #   flange → link_6     (identity)
            #   zed_camera_link → zed_camera_center → zed_left_camera_frame
            #
            # We only need to connect the camera tree to map:
            #   map → zed_camera_link (identity, camera is at map origin)
            #
            # TF chain for skeleton transform:
            #   base ← map → zed_camera_link → ... → zed_left_camera_frame
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_camera",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "map",
                    "--child-frame-id", "zed_camera_link",
                ],
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_vam_base",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", TM12S_PLANNING_FRAME,
                    "--child-frame-id", f"{VAM_PREFIX}{TM12S_PLANNING_FRAME}",
                ],
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}
                ],
            ),

            # --- VAM TM12S inference node (rviz mode) ---
            Node(
                package="vam_inference",
                executable="vam_tm12s_node",
                name="vam_tm12s_inference_node",
                output="screen",
                parameters=[
                    {
                        "mode": "rviz",
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "model_config_path": LaunchConfiguration(
                            "model_config_path"
                        ),
                        "norm_stats_path": LaunchConfiguration("norm_stats_path"),
                        "device": LaunchConfiguration("device"),
                        "max_joint_velocity_rad_s": LaunchConfiguration(
                            "max_joint_velocity_rad_s"
                        ),
                        "max_joint_acceleration_rad_s2": LaunchConfiguration(
                            "max_joint_acceleration_rad_s2"
                        ),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),
        ]
    )
