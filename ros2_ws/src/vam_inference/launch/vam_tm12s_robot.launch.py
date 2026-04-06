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
    ros2 launch vam_inference vam_tm12s_robot.launch.py active_model:=2

Switch model at runtime:
    ros2 service call /vam/switch_model vam_interfaces/srv/SwitchModel "{model_id: 2}"
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

    # Model registry YAML — defines all available models.
    # Select which model to load via active_model argument.
    default_models_config = str(
        Path(pkg_share) / "config" / "vam_models.yaml"
    )

    return LaunchDescription(
        [
            # --- Arguments with CONSERVATIVE defaults ---
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
                description="VAM node control loop rate (Hz). Higher = smoother output. "
                "Skeleton input is 15Hz; node reuses latest between frames.",
            ),
            DeclareLaunchArgument(
                "max_joint_velocity_rad_s",
                default_value="2.0",
                description="SafetyChecker velocity limit (rad/s). Must stay in sync "
                "with PVT velocity_scale to prevent drift/CPERR 241.",
            ),
            DeclareLaunchArgument(
                "max_joint_acceleration_rad_s2",
                default_value="5.0",
            ),
            DeclareLaunchArgument("target_skeleton_id", default_value="-1"),
            DeclareLaunchArgument("tracking_timeout_sec", default_value="0.5"),
            DeclareLaunchArgument("trajectory_lookahead_frames", default_value="5"),
            DeclareLaunchArgument(
                "pvt_rate_hz", default_value="15.0",
                description="PVT streaming rate in Hz",
            ),
            DeclareLaunchArgument(
                "velocity_scale", default_value="0.1",
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
            DeclareLaunchArgument(
                "filter_type", default_value="feedback",
                description="Smoothing filter: 'feedback' (P-controller, like UR10), 'one_euro', or 'ema'.",
            ),
            DeclareLaunchArgument(
                "feedback_gain", default_value="3.0",
                description="Feedback smoother Kp. Higher = more responsive. Lower = smoother.",
            ),
            DeclareLaunchArgument(
                "feedback_max_vel", default_value="1.4",
                description="Feedback smoother max velocity (rad/s). Rule: ≈ velocity_scale × 2.0. "
                "1.4 matches velocity_scale=0.7.",
            ),
            DeclareLaunchArgument(
                "smoothing_alpha", default_value="0.3",
                description="EMA smoothing factor (only used if filter_type=ema).",
            ),
            DeclareLaunchArgument(
                "one_euro_min_cutoff", default_value="0.4",
                description="One-Euro min cutoff Hz. Lower = smoother when slow (0.15=heavy, 0.4=moderate, 1.0=light).",
            ),
            DeclareLaunchArgument(
                "one_euro_beta", default_value="0.1",
                description="One-Euro speed coefficient. Higher = more responsive when fast (0.01=smooth, 0.05=balanced, 0.1=fast).",
            ),
            DeclareLaunchArgument(
                "one_euro_d_cutoff", default_value="1.0",
                description="One-Euro derivative cutoff Hz. Usually leave at 1.0.",
            ),
            DeclareLaunchArgument(
                "max_drift_deg", default_value="1.5",
                description="Max drift (deg) between sent and actual position before re-anchoring. Prevents CPERR 241.",
            ),
            DeclareLaunchArgument(
                "deadzone_rad", default_value="0.12",
                description="Stillness deadzone (rad). Target changes below this are suppressed. 0.12 rad ≈ 6.9°.",
            ),
            DeclareLaunchArgument(
                "still_frames_threshold", default_value="2",
                description="Consecutive frames within deadzone before locking position.",
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
            # Values from TM12S rosbag /tf_static (26_04_03 recordings).
            # world→base is identity in the TM12S MoveIt config, so
            # map→world effectively equals map→base.
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

            # --- Static transform: world → base (identity) ---
            # URDF root is 'base', not 'world'. This bridges the
            # map→world tree to the robot_state_publisher tree.
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
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),

            # --- PVT Streamer runs in hw-container (needs moveit_msgs) ---
            # Start separately:
            #   ros2 run vam_inference vam_pvt_streamer
            # Or with parameters:
            #   ros2 run vam_inference vam_pvt_streamer --ros-args \
            #       -p pvt_rate_hz:=15.0 -p velocity_scale:=0.1 \
            #       -p catch_up_threshold_rad:=0.3 -p catch_up_velocity_scale:=0.2 \
            #       -p max_drift_deg:=8.0
        ]
    )
