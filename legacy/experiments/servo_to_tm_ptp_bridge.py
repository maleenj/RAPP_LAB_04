"""Plan C Bridge: MoveIt Servo → TM12S via blended PTP_J (set_positions).

Sends MoveIt Servo's position output as rapid PTP_J commands with
blend_percentage=100 and fine_goal=false. The robot blends consecutive
motions together instead of stopping at each target, producing reasonably
smooth motion. Requires ZERO driver code changes.

Why blended PTP:
  - Works with the driver exactly as shipped (no C++ modifications)
  - blend_percentage=100 + fine_goal=false = smooth transitions
  - Quick to validate (5-minute test)

Trade-offs:
  - Max effective command rate ~8-10Hz (blocking service call overhead)
  - Robot "cuts corners" — position accuracy is approximate
  - Queue management needed to prevent lag from stale commands
  - Less reactive than velocity mode (Plan A) or PVT (Plan B)

Data flow:
  MoveIt Servo (250Hz Float64MultiArray on /forward_position_controller/commands)
    → This bridge (latest-only at send_rate Hz)
    → /set_positions service (PTP_J, blend=100%, fine_goal=false)
    → TM12S controller (queues and blends consecutive PTP commands)

Safety features:
  - Convergence gate: no commands until servo output matches robot position
  - Step clamping: max position change per command (max_step_rad)
  - Queue flush: StopAndClearBuffer() if position lag exceeds threshold
  - Watchdog: stops sending if no servo data
  - Clean exit: StopAndClearBuffer() on shutdown

Usage:
    ros2 run vam_inference servo_to_tm_ptp_bridge

    # Conservative:
    ros2 run vam_inference servo_to_tm_ptp_bridge --ros-args \\
        -p send_rate_hz:=5.0 -p velocity_rad_s:=0.5 -p max_step_rad:=0.05
"""

import math
import threading
import time
import signal
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tm_msgs.srv import SetPositions, SendScript

N_JOINTS = 6


class ServoToTmPtpBridge(Node):
    """Bridge: MoveIt Servo positions → TM12S blended PTP_J commands."""

    def __init__(self):
        super().__init__("servo_to_tm_ptp_bridge")

        # Parameters
        self.declare_parameter("command_topic", "/forward_position_controller/commands")
        self.declare_parameter("send_rate_hz", 8.0)
        self.declare_parameter("velocity_rad_s", 1.0)
        self.declare_parameter("acc_time_ms", 150.0)
        self.declare_parameter("blend_percentage", 100)
        self.declare_parameter("max_step_rad", 0.08)
        self.declare_parameter("convergence_threshold_rad", 0.05)
        self.declare_parameter("watchdog_timeout_sec", 0.5)
        self.declare_parameter("queue_clear_threshold_rad", 0.3)

        command_topic = self.get_parameter("command_topic").value
        self._send_rate = self.get_parameter("send_rate_hz").value
        self._velocity = self.get_parameter("velocity_rad_s").value
        self._acc_time = self.get_parameter("acc_time_ms").value
        self._blend_pct = self.get_parameter("blend_percentage").value
        self._max_step = self.get_parameter("max_step_rad").value
        self._convergence_thresh = self.get_parameter("convergence_threshold_rad").value
        self._watchdog_timeout = self.get_parameter("watchdog_timeout_sec").value
        self._queue_clear_thresh = self.get_parameter("queue_clear_threshold_rad").value

        self._cb_group = ReentrantCallbackGroup()

        # State
        self._converged = False
        self._current_positions = None
        self._latest_servo_pos = None
        self._servo_lock = threading.Lock()
        self._last_servo_time = None
        self._last_sent_pos = None
        self._cmds_sent = 0

        # Subscriptions
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 1,
            callback_group=self._cb_group,
        )
        self._cmd_sub = self.create_subscription(
            Float64MultiArray, command_topic, self._on_command, 1,
            callback_group=self._cb_group,
        )

        # Service clients
        self._set_pos_cli = self.create_client(
            SetPositions, "/set_positions",
            callback_group=self._cb_group,
        )
        self._send_script_cli = self.create_client(
            SendScript, "/send_script",
            callback_group=self._cb_group,
        )

        # Send timer
        self._timer = self.create_timer(
            1.0 / self._send_rate, self._send_waypoint,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"Plan C PTP bridge: {command_topic} → /set_positions (blended PTP)\n"
            f"  rate={self._send_rate:.0f}Hz, vel={self._velocity:.1f} rad/s, "
            f"acc={self._acc_time:.0f}ms, blend={self._blend_pct}%, "
            f"max_step={self._max_step:.3f} rad, "
            f"queue_clear={self._queue_clear_thresh:.2f} rad"
        )
        self.get_logger().info("Waiting for services...")

        while not self._set_pos_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Waiting for /set_positions service...")

        self.get_logger().info("Services ready. Waiting for convergence...")

    def _on_joint_state(self, msg: JointState):
        if len(msg.position) >= N_JOINTS:
            self._current_positions = list(msg.position[:N_JOINTS])

    def _on_command(self, msg: Float64MultiArray):
        if len(msg.data) != N_JOINTS:
            return
        self._last_servo_time = self.get_clock().now()
        with self._servo_lock:
            self._latest_servo_pos = list(msg.data)

    def _stop_and_clear(self):
        """Send StopAndClearBuffer() to flush the command queue."""
        if not self._send_script_cli.service_is_ready():
            return
        req = SendScript.Request()
        req.id = "Flush"
        req.script = "StopAndClearBuffer()"
        future = self._send_script_cli.call_async(req)
        deadline = time.monotonic() + 2.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.get_logger().warn("Sent StopAndClearBuffer()")

    def _send_waypoint(self):
        """Timer callback: send latest servo position as blended PTP_J."""
        if self._current_positions is None:
            return

        with self._servo_lock:
            servo_pos = self._latest_servo_pos

        if servo_pos is None:
            return

        # Convergence gate
        if not self._converged:
            max_err = max(
                abs(servo_pos[j] - self._current_positions[j])
                for j in range(N_JOINTS)
            )
            if max_err >= self._convergence_thresh:
                self.get_logger().info(
                    f"Waiting for convergence: max_err={max_err:.4f} rad "
                    f"(need <{self._convergence_thresh:.3f})",
                    throttle_duration_sec=1.0,
                )
                return
            self._converged = True
            self._last_sent_pos = list(self._current_positions)
            self.get_logger().info(f"Converged (err={max_err:.4f}). Sending PTP commands.")

        # Watchdog
        if self._last_servo_time is not None:
            elapsed = (
                self.get_clock().now() - self._last_servo_time
            ).nanoseconds / 1e9
            if elapsed > self._watchdog_timeout:
                self.get_logger().warn(
                    "No servo data — skipping",
                    throttle_duration_sec=2.0,
                )
                return

        # Queue lag detection: if robot is far from where we last commanded,
        # flush the queue to prevent following stale targets
        if self._last_sent_pos is not None:
            pos_error = max(
                abs(self._current_positions[j] - self._last_sent_pos[j])
                for j in range(N_JOINTS)
            )
            if pos_error > self._queue_clear_thresh:
                self._stop_and_clear()
                self._last_sent_pos = list(self._current_positions)

        # Compute step-limited target
        target = np.array(servo_pos)
        prev = np.array(self._last_sent_pos)
        step = target - prev
        step = np.clip(step, -self._max_step, self._max_step)
        send_pos = prev + step

        # Send PTP_J with blending
        if not self._set_pos_cli.service_is_ready():
            self.get_logger().warn(
                "set_positions not ready", throttle_duration_sec=2.0
            )
            return

        req = SetPositions.Request()
        req.motion_type = SetPositions.Request.PTP_J
        req.positions = send_pos.tolist()
        req.velocity = self._velocity
        req.acc_time = self._acc_time
        req.blend_percentage = self._blend_pct
        req.fine_goal = False

        # Fire async — don't block the control loop
        self._set_pos_cli.call_async(req)

        self._last_sent_pos = send_pos.tolist()
        self._cmds_sent += 1

        if self._cmds_sent <= 3 or self._cmds_sent % 50 == 0:
            max_err = max(
                abs(servo_pos[j] - self._current_positions[j])
                for j in range(N_JOINTS)
            )
            self.get_logger().info(
                f"PTP #{self._cmds_sent}: max_err={math.degrees(max_err):.3f} deg, "
                f"step={math.degrees(np.max(np.abs(step))):.3f} deg"
            )

    def shutdown(self):
        """Clean shutdown — flush command queue."""
        self.get_logger().info("Shutting down — clearing command buffer...")
        self._stop_and_clear()


def main(args=None):
    rclpy.init(args=args)
    node = ServoToTmPtpBridge()

    def signal_handler(sig, frame):
        node.get_logger().warn("Ctrl+C — shutting down...")
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except Exception:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
