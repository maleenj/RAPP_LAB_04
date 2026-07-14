"""Direct PVT Streamer: VAM joint targets → TM12S via PVT streaming.

Bypasses MoveIt Servo entirely. Receives normalized joint targets from the
VAM node and sends them directly to the TM12S as PVT (Position-Velocity-Time)
points. The robot's firmware performs cubic spline interpolation between
consecutive PVT points for smooth motion at its native 1kHz+ servo rate.

State machine:
    IDLE → CATCHING_UP → STREAMING ↔ HOLDING
    - CATCHING_UP: Robot is far from target, use MoveIt trajectory planning
    - STREAMING:   Robot is near target, stream PVT points at 10Hz
    - HOLDING:     No VAM target received, hold current position

Safety:
    - Collision checking via MoveIt's /check_state_validity service
    - When target is in collision, find alternative safe position nearby
    - Per-joint velocity clamping at 70% of TM12S hardware limits
    - Watchdog: hold position if no target for 500ms
    - PVT points use blocking sends (prevents CPERR 241)

Data flow:
    /vam/joint_targets (Float64MultiArray, 15Hz from VAM)
      → This streamer (collision check + velocity limit, 10Hz)
      → PVTPoint(pos_deg, vel_deg, time) via /send_script
      → TM12S (cubic spline interpolation)

Usage:
    ros2 run vam_inference vam_pvt_streamer

    # Adjust parameters:
    ros2 run vam_inference vam_pvt_streamer --ros-args \\
        -p pvt_rate_hz:=10.0 -p catch_up_threshold_rad:=0.3
"""

import enum
import math
import signal
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    RobotState,
)
from moveit_msgs.srv import GetStateValidity
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tm_msgs.srv import SendScript

N_JOINTS = 6
TM12S_JOINT_NAMES = [
    "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
]
MOVE_GROUP_NAME = "tmr_arm"
PLANNING_FRAME = "base"

# TM12S hardware velocity limits (rad/s)
TM12S_HW_VEL_LIMITS = np.array([2.27, 2.27, 3.67, 3.93, 3.93, 7.85])


class State(enum.Enum):
    IDLE = "IDLE"
    CATCHING_UP = "CATCHING_UP"
    STREAMING = "STREAMING"
    HOLDING = "HOLDING"


class VamPvtStreamer(Node):
    """Direct PVT streamer: VAM targets → TM12S robot."""

    def __init__(self):
        super().__init__("vam_pvt_streamer")

        # Parameters
        self.declare_parameter("pvt_rate_hz", 15.0)
        self.declare_parameter("velocity_scale", 0.3)
        self.declare_parameter("accel_scale", 0.3)
        self.declare_parameter("catch_up_threshold_rad", 0.3)
        self.declare_parameter("catch_up_velocity_scale", 1.0)
        self.declare_parameter("watchdog_timeout_sec", 0.5)
        self.declare_parameter("holding_timeout_sec", 2.0)
        self.declare_parameter("collision_perturbation_rad", 0.08)
        self.declare_parameter("collision_candidates", 8)
        self.declare_parameter("filter_cutoff_hz", 2.0)

        self._pvt_rate = self.get_parameter("pvt_rate_hz").value
        self._vel_scale = self.get_parameter("velocity_scale").value
        self._accel_scale = self.get_parameter("accel_scale").value
        self._catch_up_thresh = self.get_parameter("catch_up_threshold_rad").value
        self._catch_up_vel_scale = self.get_parameter("catch_up_velocity_scale").value
        self._watchdog_timeout = self.get_parameter("watchdog_timeout_sec").value
        self._holding_timeout = self.get_parameter("holding_timeout_sec").value
        self._collision_perturb = self.get_parameter("collision_perturbation_rad").value
        self._collision_candidates = self.get_parameter("collision_candidates").value
        self._filter_cutoff = self.get_parameter("filter_cutoff_hz").value

        # 2nd-order Butterworth low-pass (biquad) — manual coefficients
        # Avoids scipy dependency
        omega = 2.0 * math.pi * self._filter_cutoff / self._pvt_rate
        omega_w = math.tan(omega / 2.0)  # pre-warped frequency
        k = omega_w * omega_w
        q = math.sqrt(2.0)  # Q for Butterworth
        norm = 1.0 / (1.0 + omega_w / q + k)
        self._b0 = k * norm
        self._b1 = 2.0 * self._b0
        self._b2 = self._b0
        self._a1 = 2.0 * (k - 1.0) * norm
        self._a2 = (1.0 - omega_w / q + k) * norm
        # Per-joint filter state: [x1, x2, y1, y2]
        self._filter_state = np.zeros((N_JOINTS, 4))
        self._filter_initialized = False

        self._dt = 1.0 / self._pvt_rate
        self._max_vel = TM12S_HW_VEL_LIMITS * self._vel_scale
        # Max velocity change per tick — ramp to max_vel over ~1 second (10 ticks)
        # accel_scale=0.3 means each tick can add 30% of max_vel / 10 ticks
        self._max_dv_per_tick = self._max_vel * 0.1  # 10 ticks to reach max_vel

        self._cb_group = ReentrantCallbackGroup()

        # State
        self._state = State.IDLE
        self._pvt_active = False
        self._current_positions = None
        self._latest_target = None
        self._target_lock = threading.Lock()
        self._tick_lock = threading.Lock()  # prevent concurrent timer ticks
        self._last_target_time = None
        self._last_sent_pos = None
        self._last_sent_vel = np.zeros(N_JOINTS)
        self._cmds_sent = 0
        self._consecutive_collisions = 0
        self._catching_up = False

        # Subscriptions
        self._js_sub = self.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 1,
            callback_group=self._cb_group,
        )
        self._target_sub = self.create_subscription(
            Float64MultiArray, "/vam/joint_targets", self._on_target, 1,
            callback_group=self._cb_group,
        )

        # Service clients
        self._send_script_cli = self.create_client(
            SendScript, "/send_script",
            callback_group=self._cb_group,
        )
        self._validity_cli = self.create_client(
            GetStateValidity, "/check_state_validity",
            callback_group=self._cb_group,
        )

        # MoveGroup action client for catch-up trajectories
        self._move_group_cli = ActionClient(
            self, MoveGroup, "/move_group",
            callback_group=self._cb_group,
        )

        # Main timer
        self._timer = self.create_timer(
            self._dt, self._tick,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"VAM PVT Streamer: /vam/joint_targets → PVT streaming\n"
            f"  rate={self._pvt_rate:.0f}Hz, vel_scale={self._vel_scale:.0%}, "
            f"accel_scale={self._accel_scale:.0%} of HW limits\n"
            f"  max joint vel: [{', '.join(f'{math.degrees(v):.0f}' for v in self._max_vel)}] deg/s\n"
            f"  catch_up_threshold={math.degrees(self._catch_up_thresh):.1f} deg"
        )

        # Wait for essential services (send_script is required for PVT)
        self.get_logger().info("Waiting for /send_script service...")
        while not self._send_script_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Still waiting for /send_script...")

        # Collision checking and MoveGroup are optional at startup
        self._validity_available = self._validity_cli.wait_for_service(
            timeout_sec=2.0
        )
        if self._validity_available:
            self.get_logger().info("/check_state_validity service available")
        else:
            self.get_logger().warn(
                "/check_state_validity NOT available — "
                "collision checking disabled, will retry in background"
            )

        self._move_group_available = self._move_group_cli.wait_for_server(
            timeout_sec=10.0
        )
        if self._move_group_available:
            self.get_logger().info("/move_action action server available")
        else:
            self.get_logger().warn(
                "/move_action NOT available — "
                "catch-up will use direct PVT instead of MoveIt planning"
            )

        self.get_logger().info("Ready. State: IDLE")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _on_joint_state(self, msg: JointState):
        if len(msg.position) >= N_JOINTS:
            # Reorder to match TM12S_JOINT_NAMES order
            positions = [0.0] * N_JOINTS
            for i, name in enumerate(TM12S_JOINT_NAMES):
                for j, msg_name in enumerate(msg.name):
                    if msg_name == name:
                        positions[i] = msg.position[j]
                        break
            self._current_positions = np.array(positions)

    def _on_target(self, msg: Float64MultiArray):
        if len(msg.data) != N_JOINTS:
            return
        with self._target_lock:
            self._latest_target = np.array(msg.data)
            self._last_target_time = self.get_clock().now()

    # ------------------------------------------------------------------
    # TMScript helpers (from servo_to_tm_pvt_bridge.py)
    # ------------------------------------------------------------------

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
        """Enter joint PVT mode and send seed point at current position.

        The seed point tells the firmware "the robot is HERE, with ZERO velocity".
        Without it, the first real PVT point might be far from the robot's actual
        position, causing CPERR 241 (the firmware can't reach it in time).
        """
        # Clean up any stale PVT state
        self._send_script_blocking("PvtClean", "PVTExit()")
        time.sleep(0.2)
        ok = self._send_script_blocking("PvtEnter", "PVTEnter(0)")
        if not ok:
            self.get_logger().error("Failed to enter PVT mode!")
            return False
        self._pvt_active = True

        # Send seed point: current position, zero velocity
        if self._current_positions is not None:
            seed_pos = self._current_positions.copy()
            self._send_pvt_point(seed_pos.tolist(), [0.0] * N_JOINTS, self._dt)
            self._last_sent_pos = seed_pos
            self._last_sent_vel = np.zeros(N_JOINTS)
            self.get_logger().info(
                f"Entered PVT mode — seeded at current position"
            )
        else:
            self.get_logger().info("Entered PVT mode (no seed — no joint data)")
        return True

    def _pvt_exit(self):
        """Exit PVT mode."""
        if not self._pvt_active:
            return
        self._send_script_blocking("PvtExit", "PVTExit()")
        self._pvt_active = False
        self.get_logger().info("Exited PVT mode")

    def _send_pvt_point(self, positions_rad, velocities_rad_s, time_sec):
        """Send a single PVTPoint TMScript command (blocking). Returns True on success."""
        pos_deg = [math.degrees(p) for p in positions_rad]
        vel_deg = [math.degrees(v) for v in velocities_rad_s]
        parts = ([f"{p:.4f}" for p in pos_deg]
                 + [f"{v:.4f}" for v in vel_deg]
                 + [f"{time_sec:.4f}"])
        script = f"PVTPoint({','.join(parts)})"
        return self._send_script_blocking("PvtPt", script)

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def _check_state_valid(self, joint_positions) -> bool:
        """Check if a joint configuration is collision-free."""
        # Retry service discovery if not yet available
        if not self._validity_available:
            self._validity_available = self._validity_cli.wait_for_service(
                timeout_sec=0.01
            )
            if not self._validity_available:
                return True  # skip check if service unavailable

        req = GetStateValidity.Request()
        req.robot_state = RobotState()
        req.robot_state.joint_state = JointState()
        req.robot_state.joint_state.name = list(TM12S_JOINT_NAMES)
        req.robot_state.joint_state.position = [float(p) for p in joint_positions]
        req.group_name = MOVE_GROUP_NAME

        future = self._validity_cli.call_async(req)
        deadline = time.monotonic() + 0.05
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.002)
        if future.done() and future.result() is not None:
            return future.result().valid
        return True  # assume safe if service times out (don't block motion)

    def _find_safe_target(self, target, current):
        """Find a collision-free position close to the target.

        Strategy:
        1. Try the target itself
        2. Try random perturbations near the target
        3. Fall back to binary search between current and target
        """
        # 1. Target is safe — use it directly
        if self._check_state_valid(target):
            self._consecutive_collisions = 0
            return target

        self._consecutive_collisions += 1

        # 2. Perturbation search: try nearby configurations
        best_candidate = None
        best_dist = float("inf")
        rng = np.random.default_rng()
        for _ in range(self._collision_candidates):
            perturbation = rng.uniform(
                -self._collision_perturb, self._collision_perturb, N_JOINTS
            )
            candidate = target + perturbation
            if self._check_state_valid(candidate):
                dist = np.max(np.abs(candidate - target))
                if dist < best_dist:
                    best_dist = dist
                    best_candidate = candidate

        if best_candidate is not None:
            self.get_logger().warn(
                f"Target in collision — using perturbation "
                f"(max_offset={math.degrees(best_dist):.1f} deg)",
                throttle_duration_sec=1.0,
            )
            return best_candidate

        # 3. Binary search: find closest safe point along current→target line
        safe = current.copy()
        unsafe = target.copy()
        for _ in range(3):
            mid = (safe + unsafe) / 2.0
            if self._check_state_valid(mid):
                safe = mid
            else:
                unsafe = mid

        self.get_logger().warn(
            f"Target in collision — interpolating to boundary "
            f"({self._consecutive_collisions} consecutive)",
            throttle_duration_sec=1.0,
        )
        return safe

    # ------------------------------------------------------------------
    # Catch-up mode (MoveIt trajectory planning)
    # ------------------------------------------------------------------

    def _execute_catch_up(self, target):
        """Plan and execute a MoveIt trajectory to the target position."""
        # Retry action server discovery
        if not self._move_group_available:
            self._move_group_available = self._move_group_cli.wait_for_server(
                timeout_sec=0.5
            )
        if not self._move_group_available:
            self.get_logger().warn(
                "CATCH-UP: /move_action not available — skipping, "
                "will stream directly (robot may be far from target)"
            )
            return True  # proceed to streaming anyway

        self._catching_up = True
        self.get_logger().info(
            f"CATCH-UP: Planning trajectory to target "
            f"(vel_scale={self._catch_up_vel_scale:.0%})"
        )

        # Build MoveGroup goal
        goal = MoveGroup.Goal()

        # Motion plan request
        goal.request = MotionPlanRequest()
        goal.request.group_name = MOVE_GROUP_NAME
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = self._catch_up_vel_scale
        goal.request.max_acceleration_scaling_factor = self._catch_up_vel_scale

        # Joint constraints for target
        constraints = Constraints()
        for i, name in enumerate(TM12S_JOINT_NAMES):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(target[i])
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        # Planning options
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False  # plan AND execute
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        # Send goal
        send_future = self._move_group_cli.send_goal_async(goal)
        deadline = time.monotonic() + 30.0  # 30s max for planning + execution
        while not send_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)

        if not send_future.done():
            self.get_logger().error("CATCH-UP: Timed out waiting for goal acceptance")
            self._catching_up = False
            return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("CATCH-UP: Goal rejected by MoveGroup")
            self._catching_up = False
            return False

        self.get_logger().info("CATCH-UP: Goal accepted, executing trajectory...")

        # Wait for result
        result_future = goal_handle.get_result_async()
        while not result_future.done() and time.monotonic() < deadline:
            time.sleep(0.05)

        if not result_future.done():
            self.get_logger().error("CATCH-UP: Timed out waiting for execution")
            self._catching_up = False
            return False

        result = result_future.result()
        error_code = result.result.error_code.val
        if error_code == 1:  # SUCCESS
            self.get_logger().info("CATCH-UP: Trajectory execution complete")
            self._catching_up = False
            return True
        else:
            self.get_logger().error(
                f"CATCH-UP: Failed with error code {error_code}"
            )
            self._catching_up = False
            return False

    # ------------------------------------------------------------------
    # Main tick (timer callback)
    # ------------------------------------------------------------------

    def _tick(self):
        """Main control loop at pvt_rate_hz."""
        # Prevent concurrent ticks (ReentrantCallbackGroup can re-enter)
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            self._tick_impl()
        finally:
            self._tick_lock.release()

    def _tick_impl(self):
        """Actual tick logic (called under lock)."""
        if self._current_positions is None:
            return

        # Skip ticks while catch-up is executing
        if self._catching_up:
            return

        with self._target_lock:
            target = self._latest_target
            target_time = self._last_target_time

        # ---- State: IDLE ----
        if self._state == State.IDLE:
            if target is None:
                return
            # First target received — check if we need catch-up
            gap = np.max(np.abs(target - self._current_positions))
            if gap >= self._catch_up_thresh:
                self._state = State.CATCHING_UP
                self.get_logger().info(
                    f"State: IDLE → CATCHING_UP (gap={math.degrees(gap):.1f} deg)"
                )
                self._execute_catch_up(target)
                # After catch-up, transition to streaming
                self._state = State.STREAMING
                self._last_sent_pos = self._current_positions.copy()
                self._last_sent_vel = np.zeros(N_JOINTS)
                self._cmds_sent = 0
                if not self._pvt_enter():
                    self._state = State.IDLE
                    return
                self.get_logger().info("State: CATCHING_UP → STREAMING")
            else:
                # Close enough — go straight to streaming
                self._state = State.STREAMING
                self._last_sent_pos = self._current_positions.copy()
                self._last_sent_vel = np.zeros(N_JOINTS)
                self._cmds_sent = 0
                if not self._pvt_enter():
                    self._state = State.IDLE
                    return
                self.get_logger().info(
                    f"State: IDLE → STREAMING (gap={math.degrees(gap):.1f} deg)"
                )
            return

        # ---- Watchdog: check for stale target ----
        if target_time is not None:
            elapsed = (
                self.get_clock().now() - target_time
            ).nanoseconds / 1e9
        else:
            elapsed = float("inf")

        # ---- State: STREAMING ----
        if self._state == State.STREAMING:
            if target is None or elapsed > self._watchdog_timeout:
                # No fresh target — transition to HOLDING
                self._state = State.HOLDING
                self.get_logger().info("State: STREAMING → HOLDING (no target)")
                if self._pvt_active and self._last_sent_pos is not None:
                    self._send_pvt_point(
                        self._last_sent_pos.tolist(), [0.0] * N_JOINTS, self._dt
                    )
                return

            # Check for large gap — only use MoveIt catch-up if available,
            # otherwise just keep streaming (velocity clamping ensures safety)
            gap = np.max(np.abs(target - self._current_positions))
            if gap >= self._catch_up_thresh and self._move_group_available:
                self.get_logger().info(
                    f"State: STREAMING → CATCHING_UP (gap={math.degrees(gap):.1f} deg)"
                )
                if self._pvt_active:
                    self._pvt_exit()
                self._state = State.CATCHING_UP
                self._execute_catch_up(target)
                self._state = State.STREAMING
                self._last_sent_pos = self._current_positions.copy()
                self._last_sent_vel = np.zeros(N_JOINTS)
                self._cmds_sent = 0
                if not self._pvt_enter():
                    self._state = State.IDLE
                    return
                self.get_logger().info("State: CATCHING_UP → STREAMING")
                return

            # Normal streaming: collision check + PVT
            self._stream_pvt(target)
            return

        # ---- State: HOLDING ----
        if self._state == State.HOLDING:
            if elapsed > self._holding_timeout:
                # Too long without target — go idle
                self.get_logger().info("State: HOLDING → IDLE (timeout)")
                if self._pvt_active:
                    self._pvt_exit()
                self._state = State.IDLE
                return

            if target is not None and elapsed <= self._watchdog_timeout:
                # Target resumed
                gap = np.max(np.abs(target - self._current_positions))
                if gap >= self._catch_up_thresh and self._move_group_available:
                    if self._pvt_active:
                        self._pvt_exit()
                    self._state = State.CATCHING_UP
                    self.get_logger().info(
                        f"State: HOLDING → CATCHING_UP (gap={math.degrees(gap):.1f} deg)"
                    )
                    self._execute_catch_up(target)
                    self._state = State.STREAMING
                    self._last_sent_pos = self._current_positions.copy()
                    self._cmds_sent = 0
                    if not self._pvt_enter():
                        self._state = State.IDLE
                        return
                    self.get_logger().info("State: CATCHING_UP → STREAMING")
                else:
                    self._state = State.STREAMING
                    self.get_logger().info("State: HOLDING → STREAMING")
                return

            # Still holding — send hold-position PVT
            if self._pvt_active and self._last_sent_pos is not None:
                self._send_pvt_point(
                    self._last_sent_pos.tolist(), [0.0] * N_JOINTS, self._dt
                )

    # ------------------------------------------------------------------
    # Target filtering
    # ------------------------------------------------------------------

    def _filter_target(self, target):
        """Apply 2nd-order Butterworth low-pass filter per joint to remove jitter."""
        if not self._filter_initialized:
            # Initialize filter state to the first target value
            # (avoids a transient spike from zero → first target)
            for j in range(N_JOINTS):
                self._filter_state[j] = [target[j], target[j], target[j], target[j]]
            self._filter_initialized = True

        filtered = np.empty(N_JOINTS)
        for j in range(N_JOINTS):
            x0 = target[j]
            x1, x2, y1, y2 = self._filter_state[j]
            # Direct Form I biquad
            y0 = self._b0 * x0 + self._b1 * x1 + self._b2 * x2 - self._a1 * y1 - self._a2 * y2
            self._filter_state[j] = [x0, x1, y0, y1]
            filtered[j] = y0
        return filtered

    # ------------------------------------------------------------------
    # PVT streaming core
    # ------------------------------------------------------------------

    def _stream_pvt(self, target):
        """Collision-check target, compute velocity, send PVT point.

        Critical: PVT points must be reachable from the previous point.
        The first point after PVTEnter is seeded from _last_sent_pos (= current
        robot position). Each subsequent point is a small step from the last,
        with velocity ramping up gradually over ~1 second.
        """
        current = self._current_positions
        prev = self._last_sent_pos
        if prev is None:
            # Seed from actual robot position — this is critical!
            prev = current.copy()
            self._last_sent_pos = prev

        # Collision check with alternative path finding
        safe_target = self._find_safe_target(target, current)

        # Persistent collision — hold position
        if self._consecutive_collisions >= 5:
            self._send_pvt_point(prev.tolist(), [0.0] * N_JOINTS, self._dt)
            self._last_sent_vel = np.zeros(N_JOINTS)
            self.get_logger().warn(
                "Persistent collision — holding position",
                throttle_duration_sec=2.0,
            )
            return

        # Low-pass filter the target to remove high-frequency jitter
        safe_target = self._filter_target(safe_target)

        # Desired velocity toward target (from last SENT position, not current)
        # Normalize to [-π, π] to take shortest angular path
        error = (safe_target - prev + np.pi) % (2 * np.pi) - np.pi
        desired_vel = error / self._dt

        # 1. Clamp velocity magnitude
        desired_vel = np.clip(desired_vel, -self._max_vel, self._max_vel)

        # 2. Acceleration ramping — max 10% of max_vel change per tick
        #    Takes ~1 second (10 ticks) to reach full speed from standstill
        dv = desired_vel - self._last_sent_vel
        dv = np.clip(dv, -self._max_dv_per_tick, self._max_dv_per_tick)
        velocity = self._last_sent_vel + dv
        velocity = np.clip(velocity, -self._max_vel, self._max_vel)

        # Position = previous + velocity * dt (small step from last sent)
        next_pos = prev + velocity * self._dt

        # Send PVT point
        ok = self._send_pvt_point(next_pos.tolist(), velocity.tolist(), self._dt)
        if not ok:
            self.get_logger().error("PVT send failed — re-entering PVT mode")
            self._pvt_exit()
            time.sleep(0.5)
            # Re-seed from current robot position
            self._last_sent_pos = self._current_positions.copy()
            self._last_sent_vel = np.zeros(N_JOINTS)
            if not self._pvt_enter():
                self._state = State.IDLE
            return

        self._last_sent_pos = next_pos.copy()
        self._last_sent_vel = velocity.copy()
        self._cmds_sent += 1

        drift_deg = math.degrees(np.max(np.abs(next_pos - current)))
        gap_deg = math.degrees(np.max(np.abs(target - current)))
        max_vel_deg = math.degrees(np.max(np.abs(velocity)))

        if self._cmds_sent <= 10 or self._cmds_sent % 50 == 0:
            self.get_logger().info(
                f"PVT #{self._cmds_sent}: "
                f"vel={max_vel_deg:.1f}°/s, "
                f"gap={gap_deg:.1f}°, "
                f"drift={drift_deg:.1f}°"
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Clean shutdown — exit PVT mode."""
        if self._pvt_active:
            self.get_logger().info("Shutting down — exiting PVT mode...")
            self._pvt_exit()


def main(args=None):
    rclpy.init(args=args)
    node = VamPvtStreamer()

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
