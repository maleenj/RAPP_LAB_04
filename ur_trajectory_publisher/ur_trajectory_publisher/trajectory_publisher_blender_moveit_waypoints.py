import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit_msgs.msg import DisplayTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from moveit_msgs.msg import RobotState, MoveItErrorCodes
from moveit_msgs.action import MoveGroup
from rclpy.action import ActionClient
import numpy as np
import os
import asyncio
import threading

class TrajectoryPublisher(Node):
    def __init__(self):
        super().__init__('trajectory_publisher')
        
        # Initialize parameters
        self.declare_parameter('joint_angles_file', '')
        self.joint_angles_file = self.get_parameter('joint_angles_file').get_parameter_value().string_value
        
        # Initialize action clients
        self.callback_group = ReentrantCallbackGroup()
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            'move_action',
            callback_group=self.callback_group
        )
        
        # Blender to ROS joint mapping
        self.joint_mapping = {
            0: {'index': 0, 'factor': 1.0, 'offset': 0.0},      # shoulder_pan
            1: {'index': 1, 'factor': 1.0, 'offset': -np.pi/2}, # shoulder_lift
            2: {'index': 2, 'factor': 1.0, 'offset': 0.0},      # elbow
            3: {'index': 3, 'factor': 1.0, 'offset': -np.pi/2}, # wrist_1
            4: {'index': 4, 'factor': 1.0, 'offset': 0.0},      # wrist_2
            5: {'index': 5, 'factor': 1.0, 'offset': 0.0}       # wrist_3
        }
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Publisher for visualizing trajectories
        self.trajectory_publisher = self.create_publisher(
            DisplayTrajectory,
            '/display_planned_path',
            10)
            
        # Initialize and start execution
        self.initialize()
        
    def initialize(self):
        """Initialize the node and start execution"""
        self.get_logger().info('Initializing trajectory publisher with MoveIt2...')
        
        # Check file
        if not self.joint_angles_file or not os.path.exists(self.joint_angles_file):
            self.get_logger().error('Invalid joint angles file path!')
            return
            
        # Load waypoints
        self.waypoints = self.load_waypoints(self.joint_angles_file)
        if not self.waypoints:
            return
            
        # Wait for action server
        if not self.move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Move group action server not available!')
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
        """Execute each waypoint in sequence using MoveIt2"""
        try:
            for i, waypoint in enumerate(self.waypoints):
                self.get_logger().info(f'Planning and executing waypoint {i + 1}/{len(self.waypoints)}')
                
                # Extract waypoint data
                positions = waypoint['positions']
                movement_time = waypoint['movement_time']
                dwell_time = waypoint['dwell_time']
                
                # Plan and execute movement to waypoint
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
        """Plan and execute movement to a position using MoveIt2"""
        try:
            # Create move group goal
            goal_msg = MoveGroup.Goal()
            
            # Set the planning group
            goal_msg.request.group_name = "ur_manipulator"
            
            # Set planning parameters
            goal_msg.request.allowed_planning_time = 5.0
            goal_msg.request.num_planning_attempts = 10
            goal_msg.request.max_velocity_scaling_factor = 1.0 / movement_time
            goal_msg.request.max_acceleration_scaling_factor = 1.0 / movement_time
            
            # Create the joint constraints
            from moveit_msgs.msg import Constraints, JointConstraint
            
            joint_constraints = []
            for name, position in zip(self.joint_names, positions):
                constraint = JointConstraint()
                constraint.joint_name = name
                constraint.position = position
                constraint.tolerance_above = 0.0001
                constraint.tolerance_below = 0.0001
                constraint.weight = 1.0
                joint_constraints.append(constraint)
            
            # Create the goal constraints
            goal_constraints = Constraints()
            goal_constraints.joint_constraints = joint_constraints
            goal_msg.request.goal_constraints = [goal_constraints]
            
            # Send goal and wait for result
            self.get_logger().info('Sending goal to move group action server')
            goal_future = await self.move_group_client.send_goal_async(goal_msg)
            
            if not goal_future.accepted:
                self.get_logger().error('Goal rejected!')
                return False
                
            result_future = await goal_future.get_result_async()
            result = result_future.result
            
            # Check result
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.get_logger().info('Successfully planned and executed motion')
                return True
            else:
                self.get_logger().error(f'Failed to plan/execute motion. Error code: {result.error_code.val}')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error in motion planning/execution: {str(e)}')
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