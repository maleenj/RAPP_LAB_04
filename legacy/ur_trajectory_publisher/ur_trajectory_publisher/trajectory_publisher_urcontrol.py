import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
import os
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from rclpy.callback_groups import ReentrantCallbackGroup
import asyncio
import threading

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
        
        # Initialize parameters
        self.declare_parameter('joint_angles_file', '')
        self.joint_angles_file = self.get_parameter('joint_angles_file').get_parameter_value().string_value
        
        # Blender to ROS joint mapping
        self.joint_mapping = {
            0: {'index': 0, 'factor': 1.0, 'offset': 0.0},      # shoulder_pan
            1: {'index': 1, 'factor': 1.0, 'offset': -np.pi/2}, # shoulder_lift
            2: {'index': 2, 'factor': 1.0, 'offset': 0.0},      # elbow
            3: {'index': 3, 'factor': 1.0, 'offset': -np.pi/2}, # wrist_1
            4: {'index': 4, 'factor': 1.0, 'offset': 0.0},      # wrist_2
            5: {'index': 5, 'factor': 1.0, 'offset': 0.0}       # wrist_3
        }
        
        # Initialize action client
        self.callback_group = ReentrantCallbackGroup()
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.callback_group
        )
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Initialize and start execution
        self.initialize()
        
    def initialize(self):
        """Initialize the node and start execution"""
        self.get_logger().info('Initializing trajectory publisher...')
        
        # Check file
        if not self.joint_angles_file or not os.path.exists(self.joint_angles_file):
            self.get_logger().error('Invalid joint angles file path!')
            return
            
        # Load waypoints
        self.waypoints = self.load_waypoints(self.joint_angles_file)
        if not self.waypoints:
            return
            
        # Wait for action server
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available!')
            return
            
        self.get_logger().info('Trajectory publisher initialized')
        
        # Start execution in a separate thread
        self.execution_thread = threading.Thread(target=self.start_execution)
        self.execution_thread.start()

    def start_execution(self):
        """Start the execution loop in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.execute_waypoints())
        loop.close()

    async def execute_waypoints(self):
        """Execute each waypoint in sequence"""
        try:
            for i, waypoint in enumerate(self.waypoints):
                self.get_logger().info(f'Executing waypoint {i + 1}/{len(self.waypoints)}')
                
                # Extract waypoint data
                positions = waypoint['positions']
                movement_time = waypoint['movement_time']
                dwell_time = waypoint['dwell_time']
                
                # Execute movement to waypoint
                success = await self.move_to_position(positions, movement_time)
                if not success:
                    self.get_logger().error(f'Failed to execute waypoint {i + 1}')
                    return
                
                # Dwell at position if specified
                if dwell_time > 0:
                    self.get_logger().info(f'Dwelling for {dwell_time} seconds')
                    await asyncio.sleep(dwell_time)
                
            self.get_logger().info('Waypoint execution completed')
            
        except Exception as e:
            self.get_logger().error(f'Error in waypoint execution: {str(e)}')

    async def move_to_position(self, positions, movement_time):
        """Move to a position with specified movement time"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * 6  # Zero velocity at goal
        point.accelerations = [0.0] * 6  # Zero acceleration at goal
        point.time_from_start = Duration(sec=movement_time, nanosec=0)
        
        trajectory_msg.points.append(point)
        
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = trajectory_msg
        
        try:
            goal_future = await self.trajectory_client.send_goal_async(goal_msg)
            
            if goal_future.accepted:
                self.get_logger().info(f'Moving to position (time: {movement_time}s)')
                result_future = await goal_future.get_result_async()
                return result_future.result.error_code == 0
            else:
                self.get_logger().error('Goal rejected')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error moving to position: {str(e)}')
            return False

    def load_waypoints(self, filename):
        """Load waypoints from file"""
        try:
            waypoints = []
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines[3:]:  # Skip header lines
                    data = eval(line.strip())
                    waypoint = {
                        'positions': self.convert_joint_angles(data[:6]),  # First 6 values are joint angles
                        'movement_time': data[6],  # 7th value is movement time
                        'dwell_time': data[7]      # 8th value is dwell time
                    }
                    waypoints.append(waypoint)
            return waypoints
        except Exception as e:
            self.get_logger().error(f'Error loading waypoints: {str(e)}')
            return []

    def convert_joint_angles(self, angles):
        """Convert single set of Blender joint angles to ROS joint angles"""
        converted = []
        for i, angle in enumerate(angles):
            mapping = self.joint_mapping[i]
            converted_angle = angle * mapping['factor'] + mapping['offset']
            # Normalize angle to [-pi, pi]
            converted_angle = ((converted_angle + np.pi) % (2 * np.pi)) - np.pi
            converted.append(converted_angle)
        return converted

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryPublisher()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()