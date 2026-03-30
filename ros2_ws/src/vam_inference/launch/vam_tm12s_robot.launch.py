"""Launch VAM inference for real TM12S robot operation via direct PVT streaming.

VAM publishes normalized joint targets; the PVT streamer sends them directly
to the TM12S via PVT mode with collision checking via MoveIt's planning scene.
No MoveIt Servo needed.

Prerequisites (run in separate terminals in rapp_hw container):

    # 1. MoveIt + TM12S driver + RViz (no fake ros2_control)
    ros2 launch vam_inference tm12s_moveit_hw.launch.py robot_ip:=192.168.10.2

    # 2. Rosbag skeleton data (from rapp_vam container)
    ros2 bag play /data/rosbags/<name> \\
        --topics /zed/zed_node/body_trk/skeletons --loop

Usage:
    ros2 launch vam_inference vam_tm12s_robot.launch.py
"""

import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
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
    """Read a TM12S URDF, remove world link/joint if present, prefix link frames.

    TM12S URDF root link is typically 'base' (not 'base_link').
    We prefix all links so the ghost robot doesn't conflict with the real one.
    """
    urdf = Path(urdf_path).read_text()

    # Remove world link if present
    urdf = re.sub(r'<link\s+name="world"\s*/>', '', urdf)
    urdf = re.sub(r'<link\s+name="world"\s*>\s*</link>', '', urdf)

    # Remove fixed joint connecting world to base (various naming conventions)
    for joint_name in ["base_joint", "world_joint", "world_fixed"]:
        urdf = re.sub(
            rf'<joint\s+name="{joint_name}"\s+type="fixed">.*?</joint>',
            '', urdf, flags=re.DOTALL,
        )

    # Collect remaining link names and prefix them
    link_names = re.findall(r'<link\s+name="([^"]+)"', urdf)
    for name in link_names:
        prefixed = prefix + name
        urdf = urdf.replace(f'<link name="{name}"', f'<link name="{prefixed}"')
        urdf = urdf.replace(f'link="{name}"', f'link="{prefixed}"')

    return urdf


def generate_launch_description():
    pkg_share = get_package_share_directory("vam_inference")

    # TM12S URDF — exported from tm_description or placed manually
    # TODO: Update this path once you export the TM12S URDF
    tm12s_urdf_path = "/data/processed/tm12s.urdf"
    vam_urdf = _prefix_urdf_tm12s(tm12s_urdf_path, VAM_PREFIX)

    # Auto-detect latest skeleton-only model
    latest_skelonly = _find_latest_model("vam_skelonly_")
    if latest_skelonly:
        default_model_dir = latest_skelonly
    else:
        default_model_dir = "/data/models/vam_20260210_2342"

    return LaunchDescription(
        [
            # --- Arguments with CONSERVATIVE defaults ---
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
                default_value="/data/processed/tensors/2026_02_10_tin10_tout10/norm_stats.pt",
            ),
            DeclareLaunchArgument("device", default_value="cuda"),
            DeclareLaunchArgument("prediction_stride_K", default_value="1"),
            DeclareLaunchArgument("ensemble_decay_weight", default_value="0.5"),
            DeclareLaunchArgument(
                "max_joint_velocity_rad_s",
                default_value="2.0",
                description="SafetyChecker pre-filter velocity limit (rad/s). "
                "TM12S joints have varying limits (2.27–7.85 rad/s).",
            ),
            DeclareLaunchArgument(
                "max_joint_acceleration_rad_s2",
                default_value="5.0",
            ),
            DeclareLaunchArgument("target_skeleton_id", default_value="-1"),
            DeclareLaunchArgument("tracking_timeout_sec", default_value="0.5"),
            DeclareLaunchArgument("trajectory_lookahead_frames", default_value="5"),
            DeclareLaunchArgument(
                "pvt_rate_hz", default_value="10.0",
                description="PVT streaming rate in Hz",
            ),
            DeclareLaunchArgument(
                "velocity_scale", default_value="0.7",
                description="Fraction of TM12S hardware velocity limits to use",
            ),
            DeclareLaunchArgument(
                "catch_up_threshold_rad", default_value="0.3",
                description="Position gap (rad) that triggers MoveIt catch-up trajectory",
            ),
            DeclareLaunchArgument(
                "catch_up_velocity_scale", default_value="1.0",
                description="MoveIt velocity scaling during catch-up (0.0-1.0)",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),

            LogInfo(
                msg="=== VAM TM12S Robot Launch: mode=robot via direct PVT streaming ==="
            ),

            # --- VAM prediction robot_state_publisher (prefixed TM12S URDF) ---
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

            # --- Static transform: map → world (camera-to-robot calibration) ---
            # Must match the headless launch. Uses --roll (X-axis rotation),
            # NOT yaw. The old manual prerequisite used positional args
            # (x y z yaw pitch roll) which applied 1.5708 as yaw incorrectly.
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_world",
                arguments=[
                    "--x", "4.4", "--y", "0.1", "--z", "-0.25",
                    "--roll", "1.5708", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "map",
                    "--child-frame-id", "world",
                ],
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}
                ],
            ),

            # --- Static transform: base → vam/base ---
            # Pins VAM ghost robot to same base as real TM12S robot.
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

            # --- Static transform: map → zed_left_camera_frame (identity) ---
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_camera_frame",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "map",
                    "--child-frame-id", "zed_left_camera_frame",
                ],
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}
                ],
            ),

            # --- VAM TM12S inference node (mode=robot) ---
            Node(
                package="vam_inference",
                executable="vam_tm12s_node",
                name="vam_tm12s_inference_node",
                output="screen",
                parameters=[
                    {
                        "mode": "robot",
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "model_config_path": LaunchConfiguration(
                            "model_config_path"
                        ),
                        "norm_stats_path": LaunchConfiguration("norm_stats_path"),
                        "device": LaunchConfiguration("device"),
                        "prediction_stride_K": LaunchConfiguration(
                            "prediction_stride_K"
                        ),
                        "ensemble_decay_weight": LaunchConfiguration(
                            "ensemble_decay_weight"
                        ),
                        "max_joint_velocity_rad_s": LaunchConfiguration(
                            "max_joint_velocity_rad_s"
                        ),
                        "max_joint_acceleration_rad_s2": LaunchConfiguration(
                            "max_joint_acceleration_rad_s2"
                        ),
                        "target_skeleton_id": LaunchConfiguration(
                            "target_skeleton_id"
                        ),
                        "tracking_timeout_sec": LaunchConfiguration(
                            "tracking_timeout_sec"
                        ),
                        "trajectory_lookahead_frames": LaunchConfiguration(
                            "trajectory_lookahead_frames"
                        ),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),

            # --- PVT Streamer runs in hw-container (needs moveit_msgs) ---
            # Start separately:
            #   ros2 run vam_inference vam_pvt_streamer
            # Or with parameters:
            #   ros2 run vam_inference vam_pvt_streamer --ros-args \
            #       -p pvt_rate_hz:=10.0 -p velocity_scale:=0.7 \
            #       -p catch_up_threshold_rad:=0.3 -p catch_up_velocity_scale:=1.0
        ]
    )
