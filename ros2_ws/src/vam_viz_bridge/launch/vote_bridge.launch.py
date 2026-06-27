"""Launch the vote_robot_bridge (audience vote -> robot mirror/contrast mode).

    ros2 launch vam_viz_bridge vote_bridge.launch.py \
         state_url:=https://yourdomain.com/fringe/api/tally.php

Override any threshold the same way, e.g.:
    ros2 launch vam_viz_bridge vote_bridge.launch.py state_url:=... hold_seconds:=4.0

Freeze auto-switching at any time (manual control), then resume:
    ros2 param set /vote_robot_bridge auto_switch false
    ros2 param set /vote_robot_bridge auto_switch true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# name, default, type — launch args forwarded as typed node params. The type coercion
# is required: launch substitutions are strings, but the node declares int/float/bool.
_ARGS = [
    ("state_url", "http://localhost/fringe/api/tally.php", str),
    ("state_token", "", str),
    ("poll_interval", "1.0", float),
    ("http_timeout", "2.0", float),
    ("active_window", "25", int),
    ("enter_contrast", "0.55", float),
    ("enter_mirror", "0.45", float),
    ("hold_seconds", "3.0", float),
    ("min_switch_interval", "12.0", float),
    ("mirror_model_id", "1", int),
    ("contrast_model_id", "2", int),
    ("auto_switch", "true", bool),
]


def generate_launch_description():
    decls = [DeclareLaunchArgument(name, default_value=default) for name, default, _ in _ARGS]
    params = {
        name: ParameterValue(LaunchConfiguration(name), value_type=typ)
        for name, _, typ in _ARGS
    }
    return LaunchDescription(
        decls
        + [
            Node(
                package="vam_viz_bridge",
                executable="vote_robot_bridge",
                name="vote_robot_bridge",
                output="screen",
                parameters=[params],
            )
        ]
    )
