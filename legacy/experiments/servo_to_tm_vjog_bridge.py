"""Plan A Bridge: MoveIt Servo → TM12S via ContinueVJog (velocity mode).

Converts MoveIt Servo's 250Hz position output into joint velocity commands
by finite-differencing consecutive positions, then streams them to the TM12S
via ContinueVJog TMScript commands through the /send_script service.

Why finite difference instead of a P-controller:
  The VAM node already runs a P-controller (positions → velocities) to feed
  MoveIt Servo, which integrates those velocities back into positions. Using
  a second P-controller here would be a double conversion with two gains to
  tune. Instead, we differentiate Servo's smooth position output to recover
  the velocity that Servo already computed — one clean conversion, no extra
  Kp, no cascaded control lag.

  vel = (servo_pos_new - servo_pos_old) / dt

  Servo's Butterworth smoothing ensures the differentiated velocity is smooth.

Data flow:
  VAM model (target positions)
    → P-controller in vam_tm12s_node (positions → velocities)
    → MoveIt Servo (collision check + Butterworth smoothing, 250Hz)
    → Float64MultiArray positions on /forward_position_controller/commands
    → This bridge (finite difference → velocities at send_rate Hz)
    → /send_script → SetContinueVJog TMScript (fire-and-forget)
    → TM12S motor controller (direct velocity drive)

Safety features:
  - Convergence gate: no commands until servo output matches robot position
  - Per-joint velocity clamping (max_velocity_rad_s)
  - Watchdog: sends zero velocity if no servo data
  - Clean exit: StopContinueVmode() on shutdown

Usage:
    ros2 run vam_inference servo_to_tm_vjog_bridge

    # Conservative:
    ros2 run vam_inference servo_to_tm_vjog_bridge --ros-args \\
        -p max_velocity_rad_s:=0.3 -p send_rate_hz:=20.0
"""

import math
import threading
import time
import signal
import sys

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tm_msgs.srv import SendScript

N_JOINTS = 6


class ServoToTmVJogBridge(Node):
    """Bridge: MoveIt Servo positions → TM12S ContinueVJog velocity commands."""

    def __init__(self):
        super().__init__("servo_to_tm_vjog_bridge")

        # Parameters
        self.declare_parameter("command_topic", "/forward_position_controller/commands")
        self.declare_parameter("send_rate_hz", 30.0)
        self.declare_parameter("max_velocity_rad_s", 0.5)
        self.declare_parameter("convergence_threshold_rad", 0.05)
        self.declare_parameter("watchdog_timeout_sec", 0.5)

        command_topic = self.get_parameter("command_topic").value
        self._rate = self.get_parameter("send_rate_hz").value
        self._max_vel = self.get_parameter("max_velocity_rad_s").value
        self._convergence_thresh = self.get_parameter("convergence_threshold_rad").value
        self._watchdog_timeout = self.get_parameter("watchdog_timeout_sec").value

        self._cb_group = ReentrantCallbackGroup()

        # State
        self._converged = False
        self._vel_mode_active = False
        self._current_positions = None
        self._latest_servo_pos = None
        self._prev_servo_pos = None
        self._servo_lock = threading.Lock()
        self._last_servo_time = None
        self._prev_servo_stamp = None
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

        # Service client for sending TMScript
        self._send_script_cli = self.create_client(
            SendScript, "/send_script",
            callback_group=self._cb_group,
        )

        # Send loop timer
        self._timer = self.create_timer(
            1.0 / self._rate, self._send_loop,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"Plan A VJog bridge: {command_topic} → ContinueVJog\n"
            f"  rate={self._rate:.0f}Hz, "
            f"max_vel={self._max_vel:.2f} rad/s, "
            f"convergence={self._convergence_thresh:.3f} rad"
        )
        self.get_logger().info("Waiting for /send_script service...")

        while not self._send_script_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Still waiting for /send_script service...")

        self.get_logger().info("Service ready. Waiting for convergence...")

    def _on_joint_state(self, msg: JointState):
        if len(msg.position) >= N_JOINTS:
            self._current_positions = list(msg.position[:N_JOINTS])

    def _on_command(self, msg: Float64MultiArray):
        if len(msg.data) != N_JOINTS:
            return
        now = self.get_clock().now()
        with self._servo_lock:
            self._prev_servo_pos = self._latest_servo_pos
            self._prev_servo_stamp = self._last_servo_time
            self._latest_servo_pos = list(msg.data)
            self._last_servo_time = now

    def _send_script_fire_and_forget(self, script_id: str, script: str):
        """Send TMScript without waiting for response (non-blocking)."""
        req = SendScript.Request()
        req.id = script_id
        req.script = script
        self._send_script_cli.call_async(req)

    def _send_script_blocking(self, script_id: str, script: str) -> bool:
        """Send TMScript and wait for response."""
        req = SendScript.Request()
        req.id = script_id
        req.script = script
        future = self._send_script_cli.call_async(req)
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if future.done() and future.result() is not None:
            return future.result().ok
        return False

    def _enter_vel_mode(self) -> bool:
        """Enter ContinueVJog mode (blocking, one-time call)."""
        ok = self._send_script_blocking("VStart", "ContinueVJog()")
        if ok:
            self._vel_mode_active = True
            self.get_logger().info("Entered velocity mode (ContinueVJog)")
        else:
            self.get_logger().error("Failed to enter velocity mode!")
        return ok

    def _exit_vel_mode(self):
        """Exit velocity mode (blocking, one-time call)."""
        self._send_vel_target([0.0] * N_JOINTS)
        time.sleep(0.05)
        self._send_script_blocking("VStop", "StopContinueVmode()")
        self._vel_mode_active = False
        self.get_logger().info("Exited velocity mode (StopContinueVmode)")

    def _send_vel_target(self, velocities_deg_s: list[float]):
        """Send SetContinueVJog with 6 joint velocities in deg/s."""
        vel_str = ",".join(f"{v:.4f}" for v in velocities_deg_s)
        self._send_script_fire_and_forget("VTrgt", f"SetContinueVJog({vel_str})")

    def _send_loop(self):
        """Finite-difference Servo's position output → velocity → ContinueVJog."""
        if self._current_positions is None:
            return

        with self._servo_lock:
            servo_pos = self._latest_servo_pos
            prev_pos = self._prev_servo_pos
            servo_stamp = self._last_servo_time
            prev_stamp = self._prev_servo_stamp

        if servo_pos is None:
            return

        # Convergence gate: wait until Servo output matches robot position
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
            self.get_logger().info(
                f"Converged (err={max_err:.4f}). Entering velocity mode..."
            )
            if not self._enter_vel_mode():
                self._converged = False
                return

        # Watchdog: zero velocity if no servo data recently
        if servo_stamp is not None:
            elapsed = (
                self.get_clock().now() - servo_stamp
            ).nanoseconds / 1e9
            if elapsed > self._watchdog_timeout:
                self._send_vel_target([0.0] * N_JOINTS)
                self.get_logger().warn(
                    "No servo data — sending zero velocity",
                    throttle_duration_sec=2.0,
                )
                return

        # Need two consecutive servo positions to differentiate
        if prev_pos is None or prev_stamp is None:
            self._send_vel_target([0.0] * N_JOINTS)
            return

        # Compute dt between consecutive servo messages
        dt = (servo_stamp - prev_stamp).nanoseconds / 1e9
        if dt < 1e-6:
            # Avoid division by zero for duplicate timestamps
            self._send_vel_target([0.0] * N_JOINTS)
            return

        # Finite difference: velocity = (pos_new - pos_old) / dt
        vel_cmd_deg = []
        for j in range(N_JOINTS):
            vel_rad = (servo_pos[j] - prev_pos[j]) / dt
            # Clamp to safety limit
            vel_rad = max(-self._max_vel, min(self._max_vel, vel_rad))
            vel_cmd_deg.append(math.degrees(vel_rad))

        self._send_vel_target(vel_cmd_deg)
        self._cmds_sent += 1

        if self._cmds_sent <= 3 or self._cmds_sent % 100 == 0:
            max_vel = max(abs(v) for v in vel_cmd_deg)
            max_err = max(
                abs(servo_pos[j] - self._current_positions[j])
                for j in range(N_JOINTS)
            )
            self.get_logger().info(
                f"VJog #{self._cmds_sent}: max_vel={max_vel:.2f} deg/s, "
                f"servo-robot err={math.degrees(max_err):.3f} deg, dt={dt*1000:.1f}ms"
            )

    def shutdown(self):
        """Clean shutdown — exit velocity mode."""
        if self._vel_mode_active:
            self.get_logger().info("Shutting down — exiting velocity mode...")
            self._exit_vel_mode()


def main(args=None):
    rclpy.init(args=args)
    node = ServoToTmVJogBridge()

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
