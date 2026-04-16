"""Launch VAM inference (DIAGNOSTIC) for TM12S robot with CSV logging.

Same as vam_tm12s_robot.launch.py but uses the diagnostic inference node
that logs every pipeline stage to CSV.

The PVT streamer is NOT included — run vam_pvt_streamer_diag separately
in the hw container (same as the normal workflow).

Usage:
    # Inference node (in rapp_vam container):
    ros2 launch vam_inference vam_tm12s_robot_diag.launch.py \\
        feedback_gain:=25.0 feedback_max_vel:=50.0 \\
        max_joint_velocity_rad_s:=20.0 max_joint_acceleration_rad_s2:=50.0 \\
        control_rate_hz:=100.0 diag_label:=vel02_acc01

    # PVT streamer (in rapp_hw container):
    ros2 run vam_inference vam_pvt_streamer_diag --ros-args \\
        -p velocity_scale:=0.2 -p accel_scale:=0.1 \\
        -p filter_type:=none -p catch_up_threshold_rad:=0.1 \\
        -p catch_up_velocity_scale:=0.2 -p diag_label:=vel02_acc01
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


def _prefix_urdf_tm12s(urdf_path: str, prefix: str) -> str:
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
    pkg_share = get_package_share_directory("vam_inference")

    tm12s_urdf_path = "/data/processed/tm12s.urdf"
    vam_urdf = _prefix_urdf_tm12s(tm12s_urdf_path, VAM_PREFIX)

    default_models_config = str(
        Path(pkg_share) / "config" / "vam_models.yaml"
    )

    return LaunchDescription(
        [
            # --- Arguments (same as original + diagnostic args) ---
            DeclareLaunchArgument(
                "models_config",
                default_value=default_models_config,
                description="Path to vam_models.yaml model registry.",
            ),
            DeclareLaunchArgument(
                "active_model",
                default_value="1",
                description="Model ID to load from registry (1, 2, 3, ...).",
            ),
            DeclareLaunchArgument("device", default_value="cuda"),
            DeclareLaunchArgument("prediction_stride_K", default_value="1"),
            DeclareLaunchArgument("ensemble_decay_weight", default_value="0.5"),
            DeclareLaunchArgument(
                "control_rate_hz", default_value="30.0",
                description="VAM node control loop rate (Hz).",
            ),
            DeclareLaunchArgument(
                "max_joint_velocity_rad_s",
                default_value="2.0",
                description="SafetyChecker velocity limit (rad/s).",
            ),
            DeclareLaunchArgument(
                "max_joint_acceleration_rad_s2",
                default_value="5.0",
            ),
            DeclareLaunchArgument("target_skeleton_id", default_value="-1"),
            DeclareLaunchArgument("tracking_timeout_sec", default_value="0.5"),
            DeclareLaunchArgument("trajectory_lookahead_frames", default_value="5"),
            DeclareLaunchArgument(
                "filter_type", default_value="feedback",
                description="Smoothing filter: 'feedback', 'one_euro', or 'ema'.",
            ),
            DeclareLaunchArgument("feedback_gain", default_value="3.0"),
            DeclareLaunchArgument("feedback_max_vel", default_value="1.4"),
            DeclareLaunchArgument("smoothing_alpha", default_value="0.3"),
            DeclareLaunchArgument("one_euro_min_cutoff", default_value="0.4"),
            DeclareLaunchArgument("one_euro_beta", default_value="0.1"),
            DeclareLaunchArgument("one_euro_d_cutoff", default_value="1.0"),
            DeclareLaunchArgument("deadzone_rad", default_value="0.12"),
            DeclareLaunchArgument("still_frames_threshold", default_value="2"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),

            # --- Diagnostic-specific arguments ---
            DeclareLaunchArgument(
                "diag_output_dir",
                default_value="/data/processed/diagnostics",
                description="Directory for diagnostic CSV output.",
            ),
            DeclareLaunchArgument(
                "diag_label",
                default_value="",
                description="Label appended to CSV filenames (e.g. vel02_acc01).",
            ),

            LogInfo(
                msg="=== VAM TM12S Robot Launch (DIAGNOSTIC): CSV logging enabled ==="
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

            # --- Static transforms (identical to original) ---
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_world",
                arguments=[
                    "--x", "3.6", "--y", "-0.27", "--z", "-0.25",
                    "--roll", "0", "--pitch", "0", "--yaw", "3.1416",
                    "--frame-id", "map",
                    "--child-frame-id", "world",
                ],
                parameters=[
                    {"use_sim_time": LaunchConfiguration("use_sim_time")}
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="world_to_base",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "0",
                    "--roll", "0", "--pitch", "0", "--yaw", "0",
                    "--frame-id", "world",
                    "--child-frame-id", TM12S_PLANNING_FRAME,
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

            # --- VAM TM12S inference node (DIAGNOSTIC clone) ---
            Node(
                package="vam_inference",
                executable="vam_tm12s_node_diag",
                name="vam_tm12s_inference_node",
                output="screen",
                parameters=[
                    {
                        "mode": "robot",
                        "models_config": LaunchConfiguration("models_config"),
                        "active_model": LaunchConfiguration("active_model"),
                        "device": LaunchConfiguration("device"),
                        "prediction_stride_K": LaunchConfiguration(
                            "prediction_stride_K"
                        ),
                        "ensemble_decay_weight": LaunchConfiguration(
                            "ensemble_decay_weight"
                        ),
                        "control_rate_hz": LaunchConfiguration(
                            "control_rate_hz"
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
                        "filter_type": LaunchConfiguration("filter_type"),
                        "feedback_gain": LaunchConfiguration("feedback_gain"),
                        "feedback_max_vel": LaunchConfiguration("feedback_max_vel"),
                        "smoothing_alpha": LaunchConfiguration("smoothing_alpha"),
                        "one_euro_min_cutoff": LaunchConfiguration(
                            "one_euro_min_cutoff"
                        ),
                        "one_euro_beta": LaunchConfiguration("one_euro_beta"),
                        "one_euro_d_cutoff": LaunchConfiguration(
                            "one_euro_d_cutoff"
                        ),
                        "deadzone_rad": LaunchConfiguration("deadzone_rad"),
                        "still_frames_threshold": LaunchConfiguration(
                            "still_frames_threshold"
                        ),
                        "diag_output_dir": LaunchConfiguration("diag_output_dir"),
                        "diag_label": LaunchConfiguration("diag_label"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),

            # --- PVT Streamer: run separately in hw-container ---
            # ros2 run vam_inference vam_pvt_streamer_diag --ros-args \
            #     -p velocity_scale:=0.2 -p accel_scale:=0.1 \
            #     -p filter_type:=none -p diag_label:=vel02_acc01
        ]
    )
