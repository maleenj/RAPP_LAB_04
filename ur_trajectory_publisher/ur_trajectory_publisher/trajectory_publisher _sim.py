import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import os

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
        
        # UR10 Joint limits - keeping these as safety checks
        self.joint_limits = {
            'position': {
                'min': [-2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi],
                'max': [2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]
            }
        }
        
        # Blender to ROS joint mapping
        self.joint_mapping = {
            0: {'index': 0, 'factor': 1.0, 'offset': 0.0},      # shoulder_pan
            1: {'index': 1, 'factor': 1.0, 'offset': -np.pi/2}, # shoulder_lift
            2: {'index': 2, 'factor': 1.0, 'offset': 0.0},      # elbow
            3: {'index': 3, 'factor': 1.0, 'offset': -np.pi/2}, # wrist_1
            4: {'index': 4, 'factor': 1.0, 'offset': 0.0},      # wrist_2
            5: {'index': 5, 'factor': 1.0, 'offset': 0.0}       # wrist_3
        }
        
        # Initialize node
        self.declare_parameter('joint_angles_file', '')
        self.joint_angles_file = self.get_parameter('joint_angles_file').get_parameter_value().string_value
        
        if not self.joint_angles_file or not os.path.exists(self.joint_angles_file):
            self.get_logger().error('Invalid joint angles file path!')
            return
        
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10)
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Load and convert joint angles
        self.raw_joint_angles = self.load_joint_angles(self.joint_angles_file)
        if not self.raw_joint_angles:
            return
            
        self.joint_angles = self.convert_joint_angles(self.raw_joint_angles)
        
        # Configuration - much faster now
        self.frame_rate = 24.0  # Original Blender frame rate
        self.current_point = 0
        
        # Create timer that matches Blender's frame rate
        self.timer = self.create_timer(1.0/self.frame_rate, self.timer_callback)
        
        self.get_logger().info('Trajectory publisher initialized')
        self.get_logger().info(f'Publishing at {self.frame_rate} Hz')
        
    def convert_joint_angles(self, raw_angles):
        """Convert Blender joint angles to ROS joint angles"""
        converted_angles = []
        for angles in raw_angles:
            converted = []
            for i, angle in enumerate(angles):
                mapping = self.joint_mapping[i]
                converted_angle = angle * mapping['factor'] + mapping['offset']
                # Normalize angle to [-pi, pi]
                converted_angle = ((converted_angle + np.pi) % (2 * np.pi)) - np.pi
                converted.append(converted_angle)
            converted_angles.append(converted)
        return converted_angles

    def load_joint_angles(self, filename):
        try:
            joint_data = []
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines[3:]:  # Skip header lines
                    angles = eval(line.strip())
                    joint_data.append(angles)
            return joint_data
        except Exception as e:
            self.get_logger().error(f'Error loading joint angles: {str(e)}')
            return []

    def timer_callback(self):
        if not self.joint_angles or self.current_point >= len(self.joint_angles):
            self.current_point = 0
            return

        # Create trajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names

        # Create single point trajectory
        point = JointTrajectoryPoint()
        point.positions = self.joint_angles[self.current_point]
        
        # Set time from start (small duration to ensure continuous motion)
        point.time_from_start = Duration(sec=0, nanosec=int(0.5e9/self.frame_rate))
        
        trajectory_msg.points = [point]
        
        # Publish trajectory
        self.trajectory_publisher.publish(trajectory_msg)
        
        if self.current_point % int(self.frame_rate) == 0:  # Log every second
            self.get_logger().info(f'Publishing frame {self.current_point}/{len(self.joint_angles)}')
        
        self.current_point += 1

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()