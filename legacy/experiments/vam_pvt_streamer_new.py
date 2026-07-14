"""Direct PVT Streamer (v2): VAM joint targets → TM12S via PVT streaming.

Improvements over vam_pvt_streamer.py:
- Fixed accel_scale parameter (was declared but unused)
- Selectable filter: One-Euro (adaptive, low-latency), Butterworth, or none
- One-Euro filter adapts cutoff based on signal speed: smooth when slow,
  responsive when fast — less phase lag than fixed Butterworth at high speeds

Bypasses MoveIt Servo entirely. Receives normalized joint targets from the
VAM node and sends them directly to the TM12S as PVT (Position-Velocity-Time)
points. The robot's firmware performs cubic spline interpolation between
consecutive PVT points for smooth motion at its native 1kHz+ servo rate.

State machine:
    IDLE → CATCHING_UP → STREAMING ↔ HOLDING
    - CATCHING_UP: Robot is far from target, use MoveIt trajectory planning
    - STREAMING:   Robot is near target, stream PVT points at 15Hz
    - HOLDING:     No VAM target received, hold current position

Safety:
    - Collision checking via MoveIt's /check_state_validity service
    - When target is in collision, find alternative safe position nearby
    - Per-joint velocity clamping at fraction of TM12S hardware limits
    - Watchdog: hold position if no target for 500ms
    - PVT points use blocking sends (prevents CPERR 241)

Data flow:
    /vam/joint_targets (Float64MultiArray, 15Hz from VAM)
      → This streamer (collision check + velocity limit, 15Hz)
      → PVTPoint(pos_deg, vel_deg, time) via /send_script
      → TM12S (cubic spline interpolation)

Usage:
    ros2 run vam_inference vam_pvt_streamer_new

    # Adjust parameters:
    ros2 run vam_inference vam_pvt_streamer_new --ros-args \\
        -p velocity_scale:=0.25 -p accel_scale:=0.15 -p filter_type:=one_euro
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

# ---------------------------------------------------------------------------
# Inline One-Euro filter (avoids vam_utils dependency in hw-container)
# ---------------------------------------------------------------------------

def _smoothing_factor(rate: float, cutoff: float) -> float:
    """Compute alpha for a simple low-pass filter given rate and cutoff freq."""
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * rate)


class OneEuroFilter:
    """Vectorised 1-Euro filter for N signals sampled at a fixed rate.

    Adapts cutoff based on signal speed: smooth when slow, responsive when fast.
    """

    def __init__(self, n_signals, rate, min_cutoff=0.3, beta=0.01, d_cutoff=1.0):
        self.n = n_signals
        self.rate = rate
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros(self.n, dtype=np.float64)
            return x.copy()
        dx = (x - self._x_prev) * self.rate
        a_d = _smoothing_factor(self.rate, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = np.array([_smoothing_factor(self.rate, c) for c in cutoff])
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        return x_hat


# ---------------------------------------------------------------------------

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


class VamPvtStreamerNew(Node):
    """Direct PVT streamer v2: VAM targets → TM12S robot."""

    def __init__(self):
        super().__init__("vam_pvt_streamer_new")

        # Parameters
        self.declare_parameter("pvt_rate_hz", 15.0)
        self.declare_parameter("velocity_scale", 0.3)
        self.declare_parameter("accel_scale", 0.1)
        self.declare_parameter("catch_up_threshold_rad", 0.3)
        self.declare_parameter("catch_up_velocity_scale", 1.0)
        self.declare_parameter("watchdog_timeout_sec", 0.5)
        self.declare_parameter("holding_timeout_sec", 2.0)
        self.declare_parameter("collision_perturbation_rad", 0.08)
        self.declare_parameter("collision_candidates", 8)
        # Filter parameters
        self.declare_parameter("filter_type", "one_euro")
        self.declare_parameter("filter_cutoff_hz", 2.0)
        self.declare_parameter("one_euro_min_cutoff", 1.0)
        self.declare_parameter("one_euro_beta", 0.05)
        self.declare_parameter("one_euro_d_cutoff", 1.0)

        self._pvt_rate = self.get_parameter("pvt_rate_hz").value
        self._vel_scale = self.get_parameter("velocity_scale").value
        self._accel_scale = self.get_parameter("accel_scale").value
        self._catch_up_thresh = self.get_parameter("catch_up_threshold_rad").value
        self._catch_up_vel_scale = self.get_parameter("catch_up_velocity_scale").value
        self._watchdog_timeout = self.get_parameter("watchdog_timeout_sec").value
        self._holding_timeout = self.get_parameter("holding_timeout_sec").value
        self._collision_perturb = self.get_parameter("collision_perturbation_rad").value
        self._collision_candidates = self.get_parameter("collision_candidates").value
        self._filter_type = self.get_parameter("filter_type").value
        self._filter_cutoff = self.get_parameter("filter_cutoff_hz").value

        # --- Filter setup ---
        if self._filter_type == "butterworth":
            # 2nd-order Butterworth low-pass (biquad) — manual coefficients
            omega = 2.0 * math.pi * self._filter_cutoff / self._pvt_rate
            omega_w = math.tan(omega / 2.0)
            k = omega_w * omega_w
            q = math.sqrt(2.0)
            norm = 1.0 / (1.0 + omega_w / q + k)
            self._b0 = k * norm
            self._b1 = 2.0 * self._b0
            self._b2 = self._b0
            self._a1 = 2.0 * (k - 1.0) * norm
            self._a2 = (1.0 - omega_w / q + k) * norm
            self._bw_filter_state = np.zeros((N_JOINTS, 4))
            self._bw_filter_initialized = False
        elif self._filter_type == "one_euro":
            self._one_euro = OneEuroFilter(
                n_signals=N_JOINTS,
                rate=self._pvt_rate,
                min_cutoff=self.get_parameter("one_euro_min_cutoff").value,
                beta=self.get_parameter("one_euro_beta").value,
                d_cutoff=self.get_parameter("one_euro_d_cutoff").value,
            )
        # filter_type == "none" — no filter object needed

        self._dt = 1.0 / self._pvt_rate
        self._max_vel = TM12S_HW_VEL_LIMITS * self._vel_scale
        # Max velocity change per tick: accel_scale controls ramp speed.
        # accel_scale=0.1 → reach max_vel in ~10 ticks (matches old hardcoded behavior)
        # accel_scale=0.2 → reach max_vel in ~5 ticks (faster ramp)
        self._max_dv_per_tick = self._max_vel * self._accel_scale

        self._cb_group = ReentrantCallbackGroup()

        # State
        self._state = State.IDLE
        self._pvt_active = False
        self._current_positions = None
        self._latest_target = None
        self._target_lock = threading.Lock()
        self._tick_lock = threading.Lock()
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
            f"VAM PVT Streamer v2: /vam/joint_targets → PVT streaming\n"
            f"  rate={self._pvt_rate:.0f}Hz, vel_scale={self._vel_scale:.0%}, "
            f"accel_scale={self._accel_scale:.0%}\n"
            f"  filter={self._filter_type}"
            + (f" (cutoff={self._filter_cutoff:.1f}Hz)"
               if self._filter_type == "butterworth" else
               f" (min_cutoff={self.get_parameter('one_euro_min_cutoff').value}, "
               f"beta={self.get_parameter('one_euro_beta').value})"
               if self._filter_type == "one_euro" else "") + "\n"
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
    # TMScript helpers
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
        """Enter joint PVT mode and send seed point at current position."""
        self._send_script_blocking("PvtClean", "PVTExit()")
        time.sleep(0.2)
        ok = self._send_script_blocking("PvtEnter", "PVTEnter(0)")
        if not ok:
            self.get_logger().error("Failed to enter PVT mode!")
            return False
        self._pvt_active = True

        # Reset filter state on PVT re-entry
        self._reset_filter()

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
        if not self._validity_available:
            self._validity_available = self._validity_cli.wait_for_service(
                timeout_sec=0.01
            )
            if not self._validity_available:
                return True

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
        return True

    def _find_safe_target(self, target, current):
        """Find a collision-free position close to the target."""
        if self._check_state_valid(target):
            self._consecutive_collisions = 0
            return target

        self._consecutive_collisions += 1

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
        if not self._move_group_available:
            self._move_group_available = self._move_group_cli.wait_for_server(
                timeout_sec=0.5
            )
        if not self._move_group_available:
            self.get_logger().warn(
                "CATCH-UP: /move_action not available — skipping, "
                "will stream directly (robot may be far from target)"
            )
            return True

        self._catching_up = True
        self.get_logger().info(
            f"CATCH-UP: Planning trajectory to target "
            f"(vel_scale={self._catch_up_vel_scale:.0%})"
        )

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest()
        goal.request.group_name = MOVE_GROUP_NAME
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = self._catch_up_vel_scale
        goal.request.max_acceleration_scaling_factor = self._catch_up_vel_scale

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

        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 3

        send_future = self._move_group_cli.send_goal_async(goal)
        deadline = time.monotonic() + 30.0
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

        if self._catching_up:
            return

        with self._target_lock:
            target = self._latest_target
            target_time = self._last_target_time

        # ---- State: IDLE ----
        if self._state == State.IDLE:
            if target is None:
                return
            gap = np.max(np.abs(target - self._current_positions))
            if gap >= self._catch_up_thresh:
                self._state = State.CATCHING_UP
                self.get_logger().info(
                    f"State: IDLE → CATCHING_UP (gap={math.degrees(gap):.1f} deg)"
                )
                self._execute_catch_up(target)
                self._state = State.STREAMING
                self._last_sent_pos = self._current_positions.copy()
                self._last_sent_vel = np.zeros(N_JOINTS)
                self._cmds_sent = 0
                if not self._pvt_enter():
                    self._state = State.IDLE
                    return
                self.get_logger().info("State: CATCHING_UP → STREAMING")
            else:
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
                self._state = State.HOLDING
                self.get_logger().info("State: STREAMING → HOLDING (no target)")
                if self._pvt_active and self._last_sent_pos is not None:
                    self._send_pvt_point(
                        self._last_sent_pos.tolist(), [0.0] * N_JOINTS, self._dt
                    )
                return

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

            self._stream_pvt(target)
            return

        # ---- State: HOLDING ----
        if self._state == State.HOLDING:
            if elapsed > self._holding_timeout:
                self.get_logger().info("State: HOLDING → IDLE (timeout)")
                if self._pvt_active:
                    self._pvt_exit()
                self._state = State.IDLE
                return

            if target is not None and elapsed <= self._watchdog_timeout:
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

            if self._pvt_active and self._last_sent_pos is not None:
                self._send_pvt_point(
                    self._last_sent_pos.tolist(), [0.0] * N_JOINTS, self._dt
                )

    # ------------------------------------------------------------------
    # Target filtering
    # ------------------------------------------------------------------

    def _reset_filter(self):
        """Reset filter state (called on PVT re-entry)."""
        if self._filter_type == "one_euro":
            self._one_euro.reset()
        elif self._filter_type == "butterworth":
            self._bw_filter_state = np.zeros((N_JOINTS, 4))
            self._bw_filter_initialized = False

    def _filter_target(self, target):
        """Apply selected filter to remove jitter from targets."""
        if self._filter_type == "none":
            return target

        if self._filter_type == "one_euro":
            return self._one_euro(target)

        # Butterworth fallback
        if not self._bw_filter_initialized:
            for j in range(N_JOINTS):
                self._bw_filter_state[j] = [target[j], target[j], target[j], target[j]]
            self._bw_filter_initialized = True

        filtered = np.empty(N_JOINTS)
        for j in range(N_JOINTS):
            x0 = target[j]
            x1, x2, y1, y2 = self._bw_filter_state[j]
            y0 = self._b0 * x0 + self._b1 * x1 + self._b2 * x2 - self._a1 * y1 - self._a2 * y2
            self._bw_filter_state[j] = [x0, x1, y0, y1]
            filtered[j] = y0
        return filtered

    # ------------------------------------------------------------------
    # PVT streaming core
    # ------------------------------------------------------------------

    def _stream_pvt(self, target):
        """Collision-check target, compute velocity, send PVT point."""
        current = self._current_positions
        prev = self._last_sent_pos
        if prev is None:
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

        # Filter the target to remove high-frequency jitter
        safe_target = self._filter_target(safe_target)

        # Desired velocity toward target (from last SENT position, not current)
        # Normalize to [-π, π] to take shortest angular path
        error = (safe_target - prev + np.pi) % (2 * np.pi) - np.pi
        desired_vel = error / self._dt

        # 1. Clamp velocity magnitude
        desired_vel = np.clip(desired_vel, -self._max_vel, self._max_vel)

        # 2. Acceleration ramping
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
    node = VamPvtStreamerNew()

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
