"""Launch the VAM inference node with configurable parameters."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # --- Arguments ---
            DeclareLaunchArgument("mode", default_value="rviz",
                                  description="rviz or robot"),
            DeclareLaunchArgument("checkpoint_path",
                                  default_value="/data/models/vam_20260210_2342/best.pt"),
            DeclareLaunchArgument("model_config_path",
                                  default_value="/data/models/vam_20260210_2342/model_config.json"),
            DeclareLaunchArgument("norm_stats_path",
                                  default_value="/data/processed/tensors/2026_02_10_tin10_tout10/norm_stats.pt"),
            DeclareLaunchArgument("device", default_value="cuda"),
            DeclareLaunchArgument("prediction_stride_K", default_value="1"),
            DeclareLaunchArgument("ensemble_decay_weight", default_value="0.5"),
            DeclareLaunchArgument("max_joint_velocity_rad_s", default_value="1.0"),
            DeclareLaunchArgument("max_joint_acceleration_rad_s2", default_value="5.0"),
            DeclareLaunchArgument("target_skeleton_id", default_value="-1"),
            DeclareLaunchArgument("tracking_timeout_sec", default_value="0.5"),
            DeclareLaunchArgument("trajectory_lookahead_frames", default_value="5"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),

            # --- VAM inference node ---
            Node(
                package="vam_inference",
                executable="vam_inference_node",
                name="vam_inference_node",
                output="screen",
                parameters=[
                    {
                        "mode": LaunchConfiguration("mode"),
                        "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                        "model_config_path": LaunchConfiguration("model_config_path"),
                        "norm_stats_path": LaunchConfiguration("norm_stats_path"),
                        "device": LaunchConfiguration("device"),
                        "prediction_stride_K": LaunchConfiguration("prediction_stride_K"),
                        "ensemble_decay_weight": LaunchConfiguration("ensemble_decay_weight"),
                        "max_joint_velocity_rad_s": LaunchConfiguration("max_joint_velocity_rad_s"),
                        "max_joint_acceleration_rad_s2": LaunchConfiguration("max_joint_acceleration_rad_s2"),
                        "target_skeleton_id": LaunchConfiguration("target_skeleton_id"),
                        "tracking_timeout_sec": LaunchConfiguration("tracking_timeout_sec"),
                        "trajectory_lookahead_frames": LaunchConfiguration("trajectory_lookahead_frames"),
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    }
                ],
            ),
        ]
    )
