"""Plan B Bridge: MoveIt Servo → TM12S via PVT (Position-Velocity-Time) streaming.

Forwards MoveIt Servo's collision-checked, smoothed position output directly
to the TM12S as PVT points. The robot's firmware performs cubic spline
interpolation between consecutive PVT points for smooth motion.

Key design: Servo's output is ALREADY safe (collision-checked, Butterworth-
smoothed, velocity-limited). This bridge does NOT add step clamping or
conservative velocity limits on top — that was causing the robot to crawl.
The only safety here is a per-joint velocity clamp at the TM12S hardware
limits as a last-resort guardrail.

Data flow:
  MoveIt Servo (250Hz, collision-checked + smoothed)
    → This bridge (latest position at pvt_rate Hz, finite-diff velocity)
    → PVTPoint(pos, vel, time) via /send_script
    → TM12S (cubic spline interpolation)

Usage:
    ros2 run vam_inference servo_to_tm_pvt_bridge

    # Adjust rate if PVT points get rejected:
    ros2 run vam_inference servo_to_tm_pvt_bridge --ros-args \\
        -p pvt_rate_hz:=15.0
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
from tm_msgs.srv import SendScript

N_JOINTS = 6

# TM12S hardware velocity limits (rad/s) — absolute maximum, last-resort clamp
TM12S_HW_VEL_LIMITS = np.array([2.27, 2.27, 3.67, 3.93, 3.93, 7.85])


class ServoToTmPvtBridge(Node):
    """Bridge: MoveIt Servo positions → TM12S PVT point streaming."""

    def __init__(self):
        super().__init__("servo_to_tm_pvt_bridge")

        # Parameters
        self.declare_parameter("command_topic", "/forward_position_controller/commands")
        self.declare_parameter("pvt_rate_hz", 10.0)
        self.declare_parameter("velocity_scale", 0.8)
        self.declare_parameter("convergence_threshold_rad", 0.05)
        self.declare_parameter("watchdog_timeout_sec", 0.5)

        command_topic = self.get_parameter("command_topic").value
        self._pvt_rate = self.get_parameter("pvt_rate_hz").value
        self._vel_scale = self.get_parameter("velocity_scale").value
        self._convergence_thresh = self.get_parameter("convergence_threshold_rad").value
        self._watchdog_timeout = self.get_parameter("watchdog_timeout_sec").value

        self._dt = 1.0 / self._pvt_rate
        # Per-joint velocity clamp = hardware limit * scale (safety margin)
        self._max_vel = TM12S_HW_VEL_LIMITS * self._vel_scale

        self._cb_group = ReentrantCallbackGroup()

        # State
        self._converged = False
        self._pvt_active = False
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

        # Service client
        self._send_script_cli = self.create_client(
            SendScript, "/send_script",
            callback_group=self._cb_group,
        )

        # PVT send timer
        self._timer = self.create_timer(
            self._dt, self._send_pvt_point,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"Plan B PVT bridge: {command_topic} → PVT streaming\n"
            f"  rate={self._pvt_rate:.0f}Hz, vel_scale={self._vel_scale:.0%} of HW limits, "
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
        self._last_servo_time = self.get_clock().now()
        with self._servo_lock:
            self._latest_servo_pos = list(msg.data)

    def _send_script_blocking(self, script_id: str, script: str) -> bool:
        """Send TMScript and wait for response."""
        req = SendScript.Request()
        req.id = script_id
        req.script = script
        future = self._send_script_cli.call_async(req)
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.005)
        if future.done() and future.result() is not None:
            return future.result().ok
        return False

    def _pvt_enter(self) -> bool:
        """Enter joint PVT mode."""
        ok = self._send_script_blocking("PvtEnter", "PVTEnter(0)")
        if ok:
            self._pvt_active = True
            self.get_logger().info("Entered PVT mode")
        else:
            self.get_logger().error("Failed to enter PVT mode!")
        return ok

    def _pvt_exit(self):
        """Exit PVT mode."""
        self._send_script_blocking("PvtExit", "PVTExit()")
        self._pvt_active = False
        self.get_logger().info("Exited PVT mode")

    def _send_pvt_script(self, positions_rad: list[float],
                         velocities_rad_s: list[float], time_sec: float):
        """Send a single PVTPoint TMScript command (blocking).

        PVT points MUST use blocking sends — the robot needs to acknowledge
        each point before accepting the next. Fire-and-forget floods the SCT
        channel and causes CPERR 241 + connection drop.
        """
        pos_deg = [math.degrees(p) for p in positions_rad]
        vel_deg = [math.degrees(v) for v in velocities_rad_s]

        parts = ([f"{p:.4f}" for p in pos_deg]
                 + [f"{v:.4f}" for v in vel_deg]
                 + [f"{time_sec:.4f}"])
        script = f"PVTPoint({','.join(parts)})"
        self._send_script_blocking("PvtPt", script)

    def _send_pvt_point(self):
        """Timer callback: forward Servo's position directly as PVT point."""
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
            self._last_sent_pos = np.array(self._current_positions)
            self.get_logger().info(
                f"Converged (err={max_err:.4f}). Entering PVT mode..."
            )
            if not self._pvt_enter():
                self._converged = False
                return

        # Watchdog
        if self._last_servo_time is not None:
            elapsed = (
                self.get_clock().now() - self._last_servo_time
            ).nanoseconds / 1e9
            if elapsed > self._watchdog_timeout:
                if self._last_sent_pos is not None:
                    self._send_pvt_script(
                        self._last_sent_pos.tolist(), [0.0] * N_JOINTS, self._dt
                    )
                self.get_logger().warn(
                    "No servo data — sending hold-position PVT point",
                    throttle_duration_sec=2.0,
                )
                return

        # Pass Servo's position directly — it's already safe
        target = np.array(servo_pos)
        prev = self._last_sent_pos

        # Finite-difference velocity
        velocity = (target - prev) / self._dt

        # Only clamp at hardware limits (last-resort safety, not performance limiter)
        velocity = np.clip(velocity, -self._max_vel, self._max_vel)

        # If velocity was clamped, adjust position to match
        # (so the PVT point is physically consistent)
        clamped_step = velocity * self._dt
        next_pos = prev + clamped_step

        self._send_pvt_script(next_pos.tolist(), velocity.tolist(), self._dt)
        self._last_sent_pos = next_pos.copy()
        self._cmds_sent += 1

        if self._cmds_sent <= 5 or self._cmds_sent % 200 == 0:
            max_vel_deg = math.degrees(np.max(np.abs(velocity)))
            max_err = max(
                abs(servo_pos[j] - self._current_positions[j])
                for j in range(N_JOINTS)
            )
            self.get_logger().info(
                f"PVT #{self._cmds_sent}: max_vel={max_vel_deg:.1f} deg/s, "
                f"servo-robot err={math.degrees(max_err):.2f} deg"
            )

    def shutdown(self):
        """Clean shutdown — exit PVT mode."""
        if self._pvt_active:
            self.get_logger().info("Shutting down — exiting PVT mode...")
            self._pvt_exit()


def main(args=None):
    rclpy.init(args=args)
    node = ServoToTmPvtBridge()

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
