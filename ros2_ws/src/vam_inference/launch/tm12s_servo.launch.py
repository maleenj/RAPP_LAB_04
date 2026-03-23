"""Launch MoveIt Servo node for TM12S.

This is a standalone servo launcher because tm12s_moveit_config does not
include MoveIt Servo support (unlike ur_moveit_config).

The servo node needs the full MoveIt config (URDF, SRDF, kinematics) plus
our custom servo parameters from tm12s_servo_vam.yaml.

MoveIt Servo in Humble expects parameters under a "moveit_servo" namespace
(see ServoParameters::makeServoParameters default ns="moveit_servo").
The config YAML is flat (no node-name nesting), and we wrap it under
{"moveit_servo": <yaml>} — same pattern as the official panda example.

Prerequisites:
    ros2 launch vam_inference tm12s_moveit_hw.launch.py robot_ip:=192.168.10.2

Usage:
    ros2 launch vam_inference tm12s_servo.launch.py
"""

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml(package_name, file_path):
    """Load a YAML file from a ROS2 package."""
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    with open(absolute_file_path, "r") as f:
        return yaml.safe_load(f)


def generate_launch_description():
    # Build MoveIt config (same pattern as tm12s_moveit.launch.py)
    moveit_config = (
        MoveItConfigsBuilder("tm12s", package_name="tm12s_moveit_config")
        .robot_description(file_path="config/tm12s.urdf.xacro")
        .robot_description_semantic(file_path="config/tm12s.srdf")
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    # Load servo config and wrap under "moveit_servo" namespace.
    # MoveIt Servo Humble looks up params as "moveit_servo.<param_name>"
    # (ServoParameters::makeServoParameters default ns="moveit_servo").
    # The YAML has "servo_node: ros__parameters: ..." nesting from when it
    # was designed as a --params-file. Extract the inner params and re-wrap.
    servo_yaml = load_yaml(
        "tm12s_moveit_config", "config/tm12s_servo_vam.yaml"
    )
    # Handle both flat YAML and nested "servo_node: ros__parameters:" format
    if "servo_node" in servo_yaml and "ros__parameters" in servo_yaml["servo_node"]:
        servo_inner = servo_yaml["servo_node"]["ros__parameters"]
    else:
        servo_inner = servo_yaml
    servo_params = {"moveit_servo": servo_inner}

    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            servo_params,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
    )

    return LaunchDescription([servo_node])
