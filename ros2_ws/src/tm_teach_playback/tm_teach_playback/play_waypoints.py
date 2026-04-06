import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
import yaml
import asyncio
import threading


JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']


class PlayWaypoints(Node):
    def __init__(self):
        super().__init__('play_waypoints')

        self.declare_parameter('waypoints_file', '')
        self.declare_parameter('loop', False)

        self.waypoints_file = self.get_parameter('waypoints_file').value
        self.loop = self.get_parameter('loop').value

        if not self.waypoints_file:
            self.get_logger().error('No waypoints_file parameter set!')
            return

        self.waypoints = self._load_waypoints(self.waypoints_file)
        if not self.waypoints:
            return

        self.cb_group = ReentrantCallbackGroup()
        self.move_client = ActionClient(
            self, MoveGroup, 'move_action', callback_group=self.cb_group)

        if not self.move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('MoveGroup action server not available!')
            return

        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints. Loop={self.loop}')

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _load_waypoints(self, path):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            waypoints = data.get('waypoints', [])
            if not waypoints:
                self.get_logger().error('No waypoints found in file.')
            return waypoints
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints: {e}')
            return []

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._execute())
        loop.close()
        self.get_logger().info('Playback finished.')

    async def _execute(self):
        iteration = 0
        while True:
            iteration += 1
            if self.loop:
                self.get_logger().info(f'--- Loop iteration {iteration} ---')

            for i, wp in enumerate(self.waypoints):
                joints = wp['joints']
                move_time = wp.get('move_time', 3.0)
                wait_time = wp.get('wait_time', 0.0)

                self.get_logger().info(
                    f'Waypoint {i+1}/{len(self.waypoints)} '
                    f'(move={move_time}s, wait={wait_time}s)')

                success = await self._move_to(joints, move_time)
                if not success:
                    self.get_logger().error(
                        f'Failed at waypoint {i+1}. Stopping.')
                    return

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            if not self.loop:
                break

    async def _move_to(self, joint_positions, move_time):
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = 'tmr_arm'
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.num_planning_attempts = 10

        # Scale velocity/acceleration: shorter move_time = faster
        vel_scale = min(1.0, 1.0 / move_time) if move_time > 0 else 1.0
        goal_msg.request.max_velocity_scaling_factor = vel_scale
        goal_msg.request.max_acceleration_scaling_factor = vel_scale

        constraints = Constraints()
        for name, pos in zip(JOINT_NAMES, joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.0001
            jc.tolerance_below = 0.0001
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints = [constraints]

        try:
            goal_handle = await self.move_client.send_goal_async(goal_msg)
            if not goal_handle.accepted:
                self.get_logger().error('Goal rejected by MoveGroup.')
                return False

            result = await goal_handle.get_result_async()
            if result.result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.get_logger().info('Motion succeeded.')
                return True
            else:
                self.get_logger().error(
                    f'Motion failed (error code {result.result.error_code.val})')
                return False
        except Exception as e:
            self.get_logger().error(f'Motion error: {e}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = PlayWaypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
