"""Launch VAM inference for real UR10 robot operation via MoveIt Servo.

VAM publishes JointJog velocity commands to MoveIt Servo, which handles
250Hz interpolation, self-collision avoidance, and streaming to
forward_position_controller.

Prerequisites (run in separate terminals):

    # 1. UR10 driver
    ros2 launch ur_robot_driver ur_control.launch.py \\
        ur_type:=ur10 robot_ip:=10.0.0.89 reverse_port:=5002 \\
        launch_rviz:=true

    # 2. Switch controller (once, after UR driver is ready)
    ros2 service call /controller_manager/switch_controller \\
        controller_manager_msgs/srv/SwitchController \\
        "{activate_controllers: ['forward_position_controller'], \\
          deactivate_controllers: ['scaled_joint_trajectory_controller'], \\
          strictness: 2}"

    # 3. MoveIt Servo (from ZED container — has moveit packages)
    #    First, copy our tuned servo config (less Butterworth smoothing):
    docker cp ros2_ws/src/vam_inference/config/ur_servo_vam.yaml \\
        <zed_container>:/opt/ros/humble/share/ur_moveit_config/config/ur_servo.yaml
    #    Then launch Servo:
    ros2 launch ur_moveit_config ur_moveit.launch.py \\
        ur_type:=ur10 launch_servo:=true launch_rviz:=false

    # 4. Static transform — map → world (camera-to-robot, adjust for lab)
    #    UR driver publishes world → base_link from URDF.
    ros2 run tf2_ros static_transform_publisher \\
        4.4 0.1 -0.25 1.5708 0 0 map world

    # 5. Rosbag skeleton data (filtered, no --clock)
    ros2 bag play /data/rosbags/<name> \\
        --topics /zed/zed_node/body_trk/skeletons --loop

Usage:
    ros2 launch vam_inference vam_robot.launch.py

    # Ultra-conservative first test (low P-gain = slow tracking):
    ros2 launch vam_inference vam_robot.launch.py \\
        servo_proportional_gain:=2.0
"""

import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


VAM_PREFIX = "vam/"


def _prefix_urdf(urdf_path: str, prefix: str) -> str:
    """Read a URDF, remove world link/joint, prefix remaining link frames."""
    urdf = Path(urdf_path).read_text()
    urdf = re.sub(r'<link\s+name="world"\s*/>', '', urdf)
    urdf = re.sub(r'<link\s+name="world"\s*>\s*</link>', '', urdf)
    urdf = re.sub(r'<!--[^\n]*base_joint[^\n]*-->\s*', '', urdf)
    urdf = re.sub(
        r'<joint\s+name="base_joint"\s+type="fixed">.*?</joint>',
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
    urdf_path = "/data/processed/ur10.urdf"
    vam_urdf = _prefix_urdf(urdf_path, VAM_PREFIX)

    return LaunchDescription(
        [
            # --- Arguments with CONSERVATIVE defaults for real robot ---
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value="/data/models/vam_20260210_2342/best.pt",
            ),
            DeclareLaunchArgument(
                "model_config_path",
                default_value="/data/models/vam_20260210_2342/model_config.json",
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
                "MoveIt Servo is the primary safety layer; this catches outliers. "
                "UR10 hardware max is ~2.09 rad/s (large joints).",
            ),
            DeclareLaunchArgument(
                "max_joint_acceleration_rad_s2",
                default_value="8.0",
                description="SafetyChecker pre-filter acceleration limit (rad/s^2). "
                "MoveIt Servo handles actual smoothing.",
            ),
            DeclareLaunchArgument("target_skeleton_id", default_value="-1"),
            DeclareLaunchArgument("tracking_timeout_sec", default_value="0.5"),
            DeclareLaunchArgument("trajectory_lookahead_frames", default_value="5"),
            DeclareLaunchArgument(
                "servo_proportional_gain",
                default_value="12.0",
                description="P-controller gain for JointJog velocity computation. "
                "v = Kp * (target - current). Lower = slower tracking. "
                "12.0 gives ~2Hz tracking bandwidth.",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="false"),

            LogInfo(
                msg="=== VAM Robot Launch: mode=robot via MoveIt Servo, "
                "CONSERVATIVE safety limits ==="
            ),

            # --- VAM prediction robot_state_publisher (prefixed URDF) ---
            # Visualizes model predictions as a ghost robot in RViz.
            # Does NOT conflict with UR driver's robot_state_publisher
            # because all links are prefixed with "vam/".
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

            # --- Static transform: base_link → vam/base_link ---
            # Pins VAM ghost robot to same base as real robot.
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

            # --- Static transform: map → zed_left_camera_frame (identity) ---
            # The ZED camera is at the map frame origin.
            # Skeleton data from rosbag arrives in zed_left_camera_frame.
            # TF2 chain: base_link ← world ← map ← zed_left_camera_frame
            # (user publishes map → world, UR driver publishes world → base_link)
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

            # --- VAM inference node (mode=robot hardcoded) ---
            Node(
                package="vam_inference",
                executable="vam_inference_node",
                name="vam_inference_node",
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
                        "servo_proportional_gain": LaunchConfiguration(
                            "servo_proportional_gain"
                        ),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),
        ]
    )
