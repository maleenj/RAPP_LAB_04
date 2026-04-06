import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import yaml
import os
from datetime import datetime


JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
DEFAULT_SAVE_DIR = '/data/rosbags'


class RecordWaypoints(Node):
    def __init__(self):
        super().__init__('record_waypoints')
        self.current_angles = None
        self.waypoints = []

        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 100)

        self.timer = self.create_timer(0.1, self.interactive_prompt)
        self.waiting = False

    def joint_state_cb(self, msg):
        positions = dict(zip(msg.name, msg.position))
        try:
            self.current_angles = [positions[name] for name in JOINT_NAMES]
        except KeyError:
            pass  # not all joints published yet

    def interactive_prompt(self):
        if self.waiting:
            return
        self.waiting = True
        try:
            self._prompt()
        finally:
            self.waiting = False

    def _prompt(self):
        print(f'\n--- {len(self.waypoints)} waypoints captured ---')
        if self.current_angles:
            angles_str = ', '.join(f'{a:.3f}' for a in self.current_angles)
            print(f'Current: [{angles_str}]')
        else:
            print('Waiting for /joint_states...')

        cmd = input('[c]apture  [l]ist  [d]elete last  [s]ave  [q]uit: ').strip().lower()

        if cmd == 'c':
            self._capture()
        elif cmd == 'l':
            self._list()
        elif cmd == 'd':
            self._delete_last()
        elif cmd == 's':
            self._save()
        elif cmd == 'q':
            self.get_logger().info('Shutting down.')
            rclpy.shutdown()

    def _capture(self):
        if self.current_angles is None:
            self.get_logger().warn('No joint states received yet!')
            return
        try:
            move_time = input('Move time (s) [3.0]: ').strip()
            move_time = float(move_time) if move_time else 3.0
            wait_time = input('Wait time (s) [0.5]: ').strip()
            wait_time = float(wait_time) if wait_time else 0.5
        except ValueError:
            self.get_logger().error('Invalid number.')
            return

        wp = {
            'joints': [round(a, 4) for a in self.current_angles],
            'move_time': move_time,
            'wait_time': wait_time,
        }
        self.waypoints.append(wp)
        angles_str = ', '.join(f'{a:.4f}' for a in self.current_angles)
        self.get_logger().info(
            f'Captured #{len(self.waypoints)}: [{angles_str}] '
            f'move={move_time}s wait={wait_time}s')

    def _list(self):
        if not self.waypoints:
            print('No waypoints captured yet.')
            return
        for i, wp in enumerate(self.waypoints):
            joints_str = ', '.join(f'{j:.4f}' for j in wp['joints'])
            print(f'  {i+1}: [{joints_str}]  move={wp["move_time"]}s  wait={wp["wait_time"]}s')

    def _delete_last(self):
        if not self.waypoints:
            print('Nothing to delete.')
            return
        removed = self.waypoints.pop()
        self.get_logger().info(f'Deleted waypoint #{len(self.waypoints)+1}')

    def _save(self):
        if not self.waypoints:
            self.get_logger().warn('No waypoints to save.')
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_name = f'waypoints_{timestamp}.yaml'
        default_path = os.path.join(DEFAULT_SAVE_DIR, default_name)

        path = input(f'Save path [{default_path}]: ').strip()
        if not path:
            path = default_path

        data = {
            'joint_names': JOINT_NAMES,
            'waypoints': self.waypoints,
        }

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f'Saved {len(self.waypoints)} waypoints to {path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RecordWaypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
