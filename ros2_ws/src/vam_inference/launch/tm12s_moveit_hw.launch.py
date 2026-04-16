"""Launch MoveIt + TM12S driver for real hardware — NO fake ros2_control.

The upstream tm12s_moveit.launch.py starts both the tm_driver AND a fake
ros2_control_node + joint_state_broadcaster, which creates two publishers
on /joint_states and causes the "blinking robot" problem in RViz.

This launch file is a corrected version that only starts:
  - tm_driver (real hardware — publishes /joint_states)
  - move_group (MoveIt planning, collision checking)
  - robot_state_publisher (URDF -> TF from /joint_states)
  - rviz2
  - static TF: world -> base

Usage:
    ros2 launch vam_inference tm12s_moveit_hw.launch.py robot_ip:=192.168.10.2
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("tm12s", package_name="tm12s_moveit_config")
        .robot_description(file_path="config/tm12s.urdf.xacro")
        .robot_description_semantic(file_path="config/tm12s.srdf")
        .trajectory_execution(file_path="config/moveit2_controllers.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .to_moveit_configs()
    )

    rviz_config_file = "/data/rosbags/rapp_rvizconfig.rviz"
    # moveit_config_path = "tm12s_moveit_config"
    # rviz_config_file = os.path.join(
    #     get_package_share_directory(moveit_config_path), "rviz", "moveit.rviz"
    # )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_ip",
                default_value="192.168.10.2",
                description="TM12S robot IP address",
            ),

            # TM12S driver (real hardware)
            Node(
                package="tm_driver",
                executable="tm_driver",
                output="screen",
                arguments=[["robot_ip:=", LaunchConfiguration("robot_ip")]],
            ),

            # MoveIt move_group (planning, collision checking)
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=[
                    moveit_config.to_dict(),
                    {"use_sim_time": False},
                ],
            ),

            # Robot state publisher (URDF -> TF, reads /joint_states from tm_driver)
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                output="both",
                parameters=[
                    moveit_config.robot_description,
                    {"use_sim_time": False},
                ],
            ),

            # RViz
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="log",
                arguments=["-d", rviz_config_file],
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.planning_pipelines,
                    moveit_config.robot_description_kinematics,
                    moveit_config.joint_limits,
                    {"use_sim_time": False},
                ],
            ),

            # Static TF: world -> base
            # Node(
            #     package="tf2_ros",
            #     executable="static_transform_publisher",
            #     name="static_transform_publisher",
            #     output="log",
            #     arguments=[
            #         "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "world", "base"
            #     ],
            # ),
        ]
    )
