import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import readline  # For better console input handling

class JointStateCapture(Node):
    def __init__(self):
        super().__init__('joint_state_capture')
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        self.current_angles = None
        self.captured_points = []
        
        # Subscribe to joint states with a larger queue size
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            100)  # Increased queue size
            
        # Create a timer for the interactive prompt
        self.timer = self.create_timer(0.1, self.interactive_prompt)
        self.waiting_for_input = False

    def joint_state_callback(self, msg):
        """Store current joint angles whenever we receive them"""
        # Create a mapping of joint name to position
        positions = dict(zip(msg.name, msg.position))
        
        # Extract our desired joints in the correct order
        self.current_angles = [positions[name] for name in self.joint_names]

    def capture_current_position(self, movement_time, dwell_time):
        """Capture current position and add specified times"""
        if self.current_angles is None:
            self.get_logger().warn('No joint states received yet!')
            return False
            
        # Create waypoint entry
        waypoint = self.current_angles + [movement_time, dwell_time]
        self.captured_points.append(waypoint)
        
        # Print the captured position
        angles_str = ', '.join([f'{angle:.3f}' for angle in self.current_angles])
        self.get_logger().info(f'Captured position: [{angles_str}, {movement_time}, {dwell_time}]')
        return True

    def save_to_file(self, filename):
        """Save captured points to waypoints file"""
        try:
            with open(filename, 'w') as f:
                # Write header
                f.write("ROS2 JointTrajectoryPoint Message\n")
                f.write("Frame rate: 24\n")
                f.write("Joint names: ['Bone.001', 'Bone.002', 'Bone.003', 'Bone.004', 'Bone.005', 'Bone.006']\n")
                
                # Write waypoints
                for point in self.captured_points:
                    point_str = '[' + ', '.join([f'{x:.3f}' for x in point]) + ']'
                    f.write(point_str + '\n')
                    
            self.get_logger().info(f'Saved {len(self.captured_points)} waypoints to {filename}')
            return True
        except Exception as e:
            self.get_logger().error(f'Error saving file: {str(e)}')
            return False

    def interactive_prompt(self):
        """Handle interactive console input"""
        if self.waiting_for_input:
            return
            
        self.waiting_for_input = True
        try:
            command = input("\nEnter command (c = capture, s = save, q = quit): ").strip().lower()
            
            if command == 'c':
                try:
                    move_time = float(input("Enter movement time (seconds): "))
                    dwell_time = float(input("Enter dwell time (seconds): "))
                    self.capture_current_position(move_time, dwell_time)
                except ValueError:
                    self.get_logger().error('Please enter valid numbers for times')
                    
            elif command == 's':
                filename = input("Enter filename to save (default: waypoints.txt): ").strip()
                if not filename:
                    filename = 'waypoints.txt'
                self.save_to_file(filename)
                
            elif command == 'q':
                self.get_logger().info('Shutting down...')
                rclpy.shutdown()
                
        finally:
            self.waiting_for_input = False

def main(args=None):
    rclpy.init(args=args)
    node = JointStateCapture()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()