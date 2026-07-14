"""Direct PVT Streamer — DIAGNOSTIC CLONE with CSV logging.

Identical to vam_pvt_streamer_new.py but logs every pipeline stage to CSV:
  input → collision-safe → filtered → desired_vel → clamped_vel →
  ramped_vel → sent_pos/vel → actual

Usage:
    ros2 run vam_inference vam_pvt_streamer_diag --ros-args \
        -p velocity_scale:=0.2 -p accel_scale:=0.1 \
        -p diag_label:=vel02_acc01

CSV output: ~/csvdata/rapplab04/diagnostics/streamer_diag_<label>_<timestamp>.csv
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

from vam_inference.diag_csv_logger import DiagCSVLogger

# ---------------------------------------------------------------------------
# Inline One-Euro filter (avoids vam_utils dependency in hw-container)
# ---------------------------------------------------------------------------

def _smoothing_factor(rate: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * rate)


class OneEuroFilter:
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

TM12S_HW_VEL_LIMITS = np.array([2.27, 2.27, 3.67, 3.93, 3.93, 7.85])

# CSV column definitions
_JN = ["j0", "j1", "j2", "j3", "j4", "j5"]

DIAG_COLUMNS = (
    ["timestamp", "wall_time", "tick", "dt_actual", "state"]
    + [f"input_{j}" for j in _JN]
    + [f"safe_{j}" for j in _JN]
    + [f"filtered_{j}" for j in _JN]
    + [f"desired_vel_{j}" for j in _JN]
    + [f"clamped_vel_{j}" for j in _JN]
    + [f"ramped_vel_{j}" for j in _JN]
    + [f"sent_pos_{j}" for j in _JN]
    + [f"sent_vel_{j}" for j in _JN]
    + [f"actual_{j}" for j in _JN]
    + ["collision_detected", "consecutive_collisions", "pvt_send_duration_ms",
       "velocity_scale", "accel_scale", "filter_type", "pvt_rate_hz"]
)


class State(enum.Enum):
    IDLE = "IDLE"
    CATCHING_UP = "CATCHING_UP"
    STREAMING = "STREAMING"
    HOLDING = "HOLDING"


class VamPvtStreamerDiag(Node):
    """Direct PVT streamer v2 — DIAGNOSTIC CLONE with CSV logging."""

    def __init__(self):
        super().__init__("vam_pvt_streamer_new")

        # Parameters (identical to original)
        self.declare_parameter("pvt_rate_hz", 15.0)
        self.declare_parameter("velocity_scale", 0.3)
        self.declare_parameter("accel_scale", 0.1)
        self.declare_parameter("catch_up_threshold_rad", 0.3)
        self.declare_parameter("catch_up_velocity_scale", 1.0)
        self.declare_parameter("watchdog_timeout_sec", 0.5)
        self.declare_parameter("holding_timeout_sec", 2.0)
        self.declare_parameter("collision_perturbation_rad", 0.08)
        self.declare_parameter("collision_candidates", 8)
        self.declare_parameter("filter_type", "one_euro")
        self.declare_parameter("filter_cutoff_hz", 2.0)
        self.declare_parameter("one_euro_min_cutoff", 1.0)
        self.declare_parameter("one_euro_beta", 0.05)
        self.declare_parameter("one_euro_d_cutoff", 1.0)
        # Drift correction: closes the loop between sent and actual position
        self.declare_parameter("drift_correction_gain", 0.3)
        self.declare_parameter("max_drift_rad", 0.12)

        # Diagnostic parameters
        self.declare_parameter("diag_output_dir", "/data/processed/diagnostics")
        self.declare_parameter("diag_label", "")

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
        self._drift_gain = self.get_parameter("drift_correction_gain").value
        self._max_drift_rad = self.get_parameter("max_drift_rad").value

        # Filter setup
        if self._filter_type == "butterworth":
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

        self._dt = 1.0 / self._pvt_rate
        self._max_vel = TM12S_HW_VEL_LIMITS * self._vel_scale
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

        # MoveGroup action client
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
            f"VAM PVT Streamer v2 (DIAG): /vam/joint_targets -> PVT streaming\n"
            f"  rate={self._pvt_rate:.0f}Hz, vel_scale={self._vel_scale:.0%}, "
            f"accel_scale={self._accel_scale:.0%}\n"
            f"  filter={self._filter_type}"
            + (f" (cutoff={self._filter_cutoff:.1f}Hz)"
               if self._filter_type == "butterworth" else
               f" (min_cutoff={self.get_parameter('one_euro_min_cutoff').value}, "
               f"beta={self.get_parameter('one_euro_beta').value})"
               if self._filter_type == "one_euro" else "") + "\n"
            f"  max joint vel: [{', '.join(f'{math.degrees(v):.0f}' for v in self._max_vel)}] deg/s\n"
            f"  drift_correction_gain={self._drift_gain:.2f}, "
            f"max_drift={math.degrees(self._max_drift_rad):.1f} deg\n"
            f"  catch_up_threshold={math.degrees(self._catch_up_thresh):.1f} deg"
        )

        # --- Diagnostic CSV logger ---
        self._diag_logger = DiagCSVLogger(
            prefix="streamer_diag",
            columns=DIAG_COLUMNS,
            output_dir=self.get_parameter("diag_output_dir").value,
            label=self.get_parameter("diag_label").value,
        )
        self._diag_tick_count = 0
        self._diag_last_tick_time = time.monotonic()
        self.get_logger().info(
            f"[DIAG] CSV logging to: {self._diag_logger.path}"
        )

        # Wait for essential services
        self.get_logger().info("Waiting for /send_script service...")
        while not self._send_script_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("Still waiting for /send_script...")

        self._validity_available = self._validity_cli.wait_for_service(
            timeout_sec=2.0
        )
        if self._validity_available:
            self.get_logger().info("/check_state_validity service available")
        else:
            self.get_logger().warn(
                "/check_state_validity NOT available -- "
                "collision checking disabled, will retry in background"
            )

        self._move_group_available = self._move_group_cli.wait_for_server(
            timeout_sec=10.0
        )
        if self._move_group_available:
            self.get_logger().info("/move_action action server available")
        else:
            self.get_logger().warn(
                "/move_action NOT available -- "
                "catch-up will use direct PVT instead of MoveIt planning"
            )

        self.get_logger().info("Ready. State: IDLE")

    # ------------------------------------------------------------------
    # Subscriber callbacks (identical to original)
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
    # TMScript helpers (identical to original)
    # ------------------------------------------------------------------

    def _send_script_blocking(self, script_id: str, script: str) -> bool:
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
        self._send_script_blocking("PvtClean", "PVTExit()")
        time.sleep(0.2)
        ok = self._send_script_blocking("PvtEnter", "PVTEnter(0)")
        if not ok:
            self.get_logger().error("Failed to enter PVT mode!")
            return False
        self._pvt_active = True
        self._reset_filter()
        if self._current_positions is not None:
            seed_pos = self._current_positions.copy()
            self._send_pvt_point(seed_pos.tolist(), [0.0] * N_JOINTS, self._dt)
            self._last_sent_pos = seed_pos
            self._last_sent_vel = np.zeros(N_JOINTS)
            self.get_logger().info(
                f"Entered PVT mode -- seeded at current position"
            )
        else:
            self.get_logger().info("Entered PVT mode (no seed -- no joint data)")
        return True

    def _pvt_exit(self):
        if not self._pvt_active:
            return
        self._send_script_blocking("PvtExit", "PVTExit()")
        self._pvt_active = False
        self.get_logger().info("Exited PVT mode")

    def _send_pvt_point(self, positions_rad, velocities_rad_s, time_sec):
        pos_deg = [math.degrees(p) for p in positions_rad]
        vel_deg = [math.degrees(v) for v in velocities_rad_s]
        parts = ([f"{p:.4f}" for p in pos_deg]
                 + [f"{v:.4f}" for v in vel_deg]
                 + [f"{time_sec:.4f}"])
        script = f"PVTPoint({','.join(parts)})"
        return self._send_script_blocking("PvtPt", script)

    # ------------------------------------------------------------------
    # Collision checking (identical to original)
    # ------------------------------------------------------------------

    def _check_state_valid(self, joint_positions) -> bool:
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
                f"Target in collision -- using perturbation "
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
            f"Target in collision -- interpolating to boundary "
            f"({self._consecutive_collisions} consecutive)",
            throttle_duration_sec=1.0,
        )
        return safe

    # ------------------------------------------------------------------
    # Catch-up mode (identical to original)
    # ------------------------------------------------------------------

    def _execute_catch_up(self, target):
        if not self._move_group_available:
            self._move_group_available = self._move_group_cli.wait_for_server(
                timeout_sec=0.5
            )
        if not self._move_group_available:
            self.get_logger().warn(
                "CATCH-UP: /move_action not available -- skipping, "
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
        if error_code == 1:
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
    # Main tick (with diagnostic logging)
    # ------------------------------------------------------------------

    def _tick(self):
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            self._tick_impl()
        finally:
            self._tick_lock.release()

    def _tick_impl(self):
        if self._current_positions is None:
            return

        if self._catching_up:
            return

        wall_now = time.monotonic()
        dt_actual = wall_now - self._diag_last_tick_time
        self._diag_last_tick_time = wall_now
        self._diag_tick_count += 1

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
                    f"State: IDLE -> CATCHING_UP (gap={math.degrees(gap):.1f} deg)"
                )
                self._execute_catch_up(target)
                self._state = State.STREAMING
                self._last_sent_pos = self._current_positions.copy()
                self._last_sent_vel = np.zeros(N_JOINTS)
                self._cmds_sent = 0
                if not self._pvt_enter():
                    self._state = State.IDLE
                    return
                self.get_logger().info("State: CATCHING_UP -> STREAMING")
            else:
                self._state = State.STREAMING
                self._last_sent_pos = self._current_positions.copy()
                self._last_sent_vel = np.zeros(N_JOINTS)
                self._cmds_sent = 0
                if not self._pvt_enter():
                    self._state = State.IDLE
                    return
                self.get_logger().info(
                    f"State: IDLE -> STREAMING (gap={math.degrees(gap):.1f} deg)"
                )
            return

        # ---- Watchdog ----
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
                self.get_logger().info("State: STREAMING -> HOLDING (no target)")
                if self._pvt_active and self._last_sent_pos is not None:
                    self._send_pvt_point(
                        self._last_sent_pos.tolist(), [0.0] * N_JOINTS, self._dt
                    )
                return

            gap = np.max(np.abs(target - self._current_positions))
            if gap >= self._catch_up_thresh and self._move_group_available:
                self.get_logger().info(
                    f"State: STREAMING -> CATCHING_UP (gap={math.degrees(gap):.1f} deg)"
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
                self.get_logger().info("State: CATCHING_UP -> STREAMING")
                return

            self._stream_pvt(target, dt_actual)
            return

        # ---- State: HOLDING ----
        if self._state == State.HOLDING:
            if elapsed > self._holding_timeout:
                self.get_logger().info("State: HOLDING -> IDLE (timeout)")
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
                        f"State: HOLDING -> CATCHING_UP (gap={math.degrees(gap):.1f} deg)"
                    )
                    self._execute_catch_up(target)
                    self._state = State.STREAMING
                    self._last_sent_pos = self._current_positions.copy()
                    self._cmds_sent = 0
                    if not self._pvt_enter():
                        self._state = State.IDLE
                        return
                    self.get_logger().info("State: CATCHING_UP -> STREAMING")
                else:
                    self._state = State.STREAMING
                    self.get_logger().info("State: HOLDING -> STREAMING")
                return

            # [DIAG] Log holding state
            if self._pvt_active and self._last_sent_pos is not None:
                self._send_pvt_point(
                    self._last_sent_pos.tolist(), [0.0] * N_JOINTS, self._dt
                )
                ros_ts = self.get_clock().now().nanoseconds / 1e9
                actual = self._current_positions
                hold = self._last_sent_pos
                zeros = np.zeros(N_JOINTS)
                self._diag_logger.log(
                    [ros_ts, time.monotonic(), self._diag_tick_count,
                     f"{dt_actual:.6f}", "HOLDING"]
                    + [f"{v:.6f}" for v in hold]      # input (last known)
                    + [f"{v:.6f}" for v in hold]      # safe
                    + [f"{v:.6f}" for v in hold]      # filtered
                    + [f"{v:.6f}" for v in zeros]     # desired_vel
                    + [f"{v:.6f}" for v in zeros]     # clamped_vel
                    + [f"{v:.6f}" for v in zeros]     # ramped_vel
                    + [f"{v:.6f}" for v in hold]      # sent_pos
                    + [f"{v:.6f}" for v in zeros]     # sent_vel
                    + [f"{v:.6f}" for v in actual]    # actual
                    + [0, 0, 0.0,                     # collision, consec, send_ms
                       self._vel_scale, self._accel_scale,
                       self._filter_type, self._pvt_rate]
                )

    # ------------------------------------------------------------------
    # Target filtering (identical to original)
    # ------------------------------------------------------------------

    def _reset_filter(self):
        if self._filter_type == "one_euro":
            self._one_euro.reset()
        elif self._filter_type == "butterworth":
            self._bw_filter_state = np.zeros((N_JOINTS, 4))
            self._bw_filter_initialized = False

    def _filter_target(self, target):
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
    # PVT streaming core (with CSV logging)
    # ------------------------------------------------------------------

    def _stream_pvt(self, target, dt_actual):
        """Collision-check target, compute velocity, send PVT point — with logging."""
        current = self._current_positions
        prev = self._last_sent_pos
        if prev is None:
            prev = current.copy()
            self._last_sent_pos = prev

        input_target = target.copy()  # [DIAG]

        # Collision check
        safe_target = self._find_safe_target(target, current)
        collision_detected = self._consecutive_collisions > 0

        # Persistent collision — hold position
        if self._consecutive_collisions >= 5:
            self._send_pvt_point(prev.tolist(), [0.0] * N_JOINTS, self._dt)
            self._last_sent_vel = np.zeros(N_JOINTS)
            self.get_logger().warn(
                "Persistent collision -- holding position",
                throttle_duration_sec=2.0,
            )
            # [DIAG] Log collision hold
            ros_ts = self.get_clock().now().nanoseconds / 1e9
            zeros = np.zeros(N_JOINTS)
            self._diag_logger.log(
                [ros_ts, time.monotonic(), self._diag_tick_count,
                 f"{dt_actual:.6f}", "STREAMING"]
                + [f"{v:.6f}" for v in input_target]
                + [f"{v:.6f}" for v in safe_target]
                + [f"{v:.6f}" for v in safe_target]
                + [f"{v:.6f}" for v in zeros]
                + [f"{v:.6f}" for v in zeros]
                + [f"{v:.6f}" for v in zeros]
                + [f"{v:.6f}" for v in prev]
                + [f"{v:.6f}" for v in zeros]
                + [f"{v:.6f}" for v in current]
                + [1, self._consecutive_collisions, 0.0,
                   self._vel_scale, self._accel_scale,
                   self._filter_type, self._pvt_rate]
            )
            return

        safe_pre_filter = safe_target.copy()  # [DIAG]

        # Filter
        safe_target = self._filter_target(safe_target)
        filtered_target = safe_target.copy()  # [DIAG]

        # --- Drift correction: bias velocity toward actual robot position ---
        drift = prev - current
        drift_mag = np.max(np.abs(drift))

        # Nonlinear gain: gentle at small drift, aggressive near limit
        drift_urgency = 1.0 + 2.0 * min(drift_mag / self._max_drift_rad, 1.0)
        effective_gain = self._drift_gain * drift_urgency
        drift_correction = -effective_gain * drift / self._dt

        if drift_mag > self._max_drift_rad * 0.8:
            self.get_logger().warn(
                f"High drift {math.degrees(drift_mag):.1f}° "
                f"(limit {math.degrees(self._max_drift_rad):.1f}°) — "
                f"prioritizing drift reduction (gain={effective_gain:.2f})",
                throttle_duration_sec=0.5,
            )

        # Desired velocity
        error = (safe_target - prev + np.pi) % (2 * np.pi) - np.pi
        desired_vel = error / self._dt + drift_correction
        desired_vel_capture = desired_vel.copy()  # [DIAG]

        # Clamp velocity
        desired_vel = np.clip(desired_vel, -self._max_vel, self._max_vel)
        clamped_vel = desired_vel.copy()  # [DIAG]

        # Acceleration ramping
        dv = desired_vel - self._last_sent_vel
        dv = np.clip(dv, -self._max_dv_per_tick, self._max_dv_per_tick)
        velocity = self._last_sent_vel + dv
        velocity = np.clip(velocity, -self._max_vel, self._max_vel)
        ramped_vel = velocity.copy()  # [DIAG]

        # Position step (smooth chain, no jumps)
        next_pos = prev + velocity * self._dt

        # Send PVT point (timed)
        t0 = time.monotonic()
        ok = self._send_pvt_point(next_pos.tolist(), velocity.tolist(), self._dt)
        pvt_send_ms = (time.monotonic() - t0) * 1000.0

        if not ok:
            self.get_logger().error("PVT send failed -- re-entering PVT mode")
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

        # [DIAG] Log full pipeline state
        ros_ts = self.get_clock().now().nanoseconds / 1e9
        self._diag_logger.log(
            [ros_ts, time.monotonic(), self._diag_tick_count,
             f"{dt_actual:.6f}", "STREAMING"]
            + [f"{v:.6f}" for v in input_target]
            + [f"{v:.6f}" for v in safe_pre_filter]
            + [f"{v:.6f}" for v in filtered_target]
            + [f"{v:.6f}" for v in desired_vel_capture]
            + [f"{v:.6f}" for v in clamped_vel]
            + [f"{v:.6f}" for v in ramped_vel]
            + [f"{v:.6f}" for v in next_pos]
            + [f"{v:.6f}" for v in velocity]
            + [f"{v:.6f}" for v in current]
            + [int(collision_detected), self._consecutive_collisions,
               f"{pvt_send_ms:.2f}",
               self._vel_scale, self._accel_scale,
               self._filter_type, self._pvt_rate]
        )

        # Text logging (sampled, same as original)
        drift_deg = math.degrees(np.max(np.abs(next_pos - current)))
        gap_deg = math.degrees(np.max(np.abs(target - current)))
        max_vel_deg = math.degrees(np.max(np.abs(velocity)))

        if self._cmds_sent <= 10 or self._cmds_sent % 50 == 0:
            self.get_logger().info(
                f"PVT #{self._cmds_sent}: "
                f"vel={max_vel_deg:.1f} deg/s, "
                f"gap={gap_deg:.1f} deg, "
                f"drift={drift_deg:.1f} deg"
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        if self._pvt_active:
            self.get_logger().info("Shutting down -- exiting PVT mode...")
            self._pvt_exit()
        self._diag_logger.close()
        self.get_logger().info(
            f"[DIAG] CSV saved: {self._diag_logger.path} "
            f"({self._diag_tick_count} ticks)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = VamPvtStreamerDiag()

    def signal_handler(sig, frame):
        node.get_logger().warn("Ctrl+C -- shutting down...")
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
