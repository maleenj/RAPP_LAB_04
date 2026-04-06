from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to your custom URDF
    ur_trajectory_publisher_path = get_package_share_directory('ur_trajectory_publisher')
    custom_urdf_path = os.path.join(ur_trajectory_publisher_path, 'urdf', 'ur10_with_board.urdf.xacro')

    # Declare arguments
    declared_arguments = []
    
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_fake_hardware',
            default_value='false',
            description='Start robot with fake hardware mirroring command to its states.'
        )
    )

    launch_rviz = LaunchConfiguration('launch_rviz')
    declared_arguments.append(
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Launch RViz2 automatically with this launch file.'
        )
    )

    # Get the launch directory
    moveit_config_path = get_package_share_directory('ur_moveit_config')

    # Include the main UR MoveIt launch file
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_config_path, 'launch', 'ur_moveit.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur10',
            'use_fake_hardware': use_fake_hardware,
            'launch_rviz': launch_rviz,
            'urdf_path': custom_urdf_path,  # Use our custom URDF
        }.items()
    )

    return LaunchDescription(declared_arguments + [moveit_launch])