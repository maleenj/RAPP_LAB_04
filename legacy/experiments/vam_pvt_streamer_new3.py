"""Direct PVT Streamer (v4): VAM joint targets → TM12S via PVT streaming.

Same as vam_pvt_streamer_new2.py PLUS the following-error (lag) throttle —
FIX 5 — which is the change that actually addresses the remaining STOs.

  FIX 5 (v4) — close the loop on ACTUAL position (following-error throttle).
    Diagnosis: with the Performance Safety joint speeds confirmed at MAX
    (J3 240, J4/J5 260 °/s), trips at only 37–44 °/s are NOT raw overspeed.
    The fingerprint in every failure log is `drift` (commanded position
    leading the actual arm) GROWING just before the trip — a following-error
    runaway. The streamer integrates next_pos from its own last COMMANDED
    position with no feedback from where the arm actually is, so on a fast
    ramp the arm falls behind, drift accumulates, and the TM firmware ends up
    rejecting a PVT segment as "speed command too large" because the buffered
    command sits too far ahead of the lagging arm to reach in one segment.
    FIX 5 throttles the per-joint velocity ceiling DOWN as that joint's lag
    (|commanded − actual|) grows: full speed below following_error_soft_rad,
    linearly to zero at following_error_hard_rad. The command can no longer
    run away from the arm; it is self-correcting (lag shrinks → speed
    returns) and per-joint (only the lagging joint slows), so smooth tracking
    stays responsive.

Inherited fixes from v2/v3:

  FIX 1 (the important one) — trapezoidal position integration.
    The TM firmware interpolates a *cubic Hermite* between consecutive PVT
    points. For a segment (p0,v0)→(p1,v1) over T, the velocity profile is:
        v(s) = (6d/T)·s(1−s) + v0·(3s²−4s+1) + v1·(3s²−2s),  d = p1−p0
    The old code set d = v1·dt (forward-Euler). That leaves the s² terms
    uncancelled, so the spline *bulges above v1* mid-segment — the bulge is
    what clipped the J4/J5 safety ceiling and tripped the STO.
    Setting d = ½(v0+v1)·dt (trapezoidal) gives 6d/T = 3(v0+v1), which
    collapses the profile to v(s) = v0 + (v1−v0)·s — perfectly LINEAR, peak
    velocity = max(v0,v1), ZERO overshoot. This is a mathematical guarantee,
    not a heuristic.

  FIX 1 (the important one) — trapezoidal position integration.
    The TM firmware interpolates a *cubic Hermite* between consecutive PVT
    points. For a segment (p0,v0)→(p1,v1) over T, the velocity profile is:
        v(s) = (6d/T)·s(1−s) + v0·(3s²−4s+1) + v1·(3s²−2s),  d = p1−p0
    The old code set d = v1·dt (forward-Euler). That leaves the s² terms
    uncancelled, so the spline *bulges above v1* mid-segment — the bulge is
    what clipped the J4/J5 safety ceiling and tripped the STO.
    Setting d = ½(v0+v1)·dt (trapezoidal) gives 6d/T = 3(v0+v1), which
    collapses the profile to v(s) = v0 + (v1−v0)·s — perfectly LINEAR, peak
    velocity = max(v0,v1), ZERO overshoot. This is a mathematical guarantee,
    not a heuristic.

  FIX 2 — velocity clamp baselined on the real Performance Safety ceilings.
    The clamp now derives from the actual safety-controller joint-speed
    limits (160/160/240/260/260/520 °/s) with a configurable headroom
    (default 85%) so transient command values can never reach the ceiling.

  FIX 3 — no re-entry fail-loop. On a PVT send failure the old code reset
    velocity to 0 and immediately streamed again, producing a v0=0→v1=high
    step that re-tripped the STO (the "re-entering PVT mode" log spam).
    Re-entry now goes through a short cool-down and the normal acceleration
    ramp, so the first post-seed segment starts from rest smoothly. With
    FIX 1, even that first segment can't overshoot.

  FIX 4 — per-joint diagnostics. Logs now report WHICH joint is at peak
    velocity, so a future trip is immediately attributable.

  FIX 3 — no re-entry fail-loop (cool-down + rest ramp on PVT send failure).
  FIX 4 — per-joint diagnostics (logs WHICH joint is at peak velocity).
  Sudden-jump guard (v3) — debounce transient jumps (actor switch, walk-on,
    misdetection); confirmed jumps run as a slow approach. Smooth fast
    tracking stays full-speed (slow-approach is jump-triggered, not gap-based).

Everything else (state machine, collision checking, MoveIt catch-up,
filters) is unchanged from vam_pvt_streamer_new2.py.

Usage:
    ros2 run vam_inference vam_pvt_streamer_new3

    ros2 run vam_inference vam_pvt_streamer_new3 --ros-args \\
        -p velocity_scale:=0.2 -p accel_scale:=0.1 -p filter_type:=none \\
        -p catch_up_threshold_rad:=0.1 -p catch_up_velocity_scale:=0.2
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

# TM12S Performance Safety joint-speed ceilings — the values the safety
# controller actually enforces in Auto mode (deg/s → rad/s). Exceeding any of
# these for even one command cycle trips xxF061 → 0x35 Joint Drivers Alarm.
#   J1/J2: 160 °/s, J3: 240 °/s, J4/J5: 260 °/s, J6: 520 °/s
TM12S_SAFETY_VEL_LIMITS = np.radians(
    np.array([160.0, 160.0, 240.0, 260.0, 260.0, 520.0])
)


class State(enum.Enum):
    IDLE = "IDLE"
    CATCHING_UP = "CATCHING_UP"
    STREAMING = "STREAMING"
    HOLDING = "HOLDING"


class VamPvtStreamerNew3(Node):
    """Direct PVT streamer v4: VAM targets → TM12S robot (following-error safe)."""

    def __init__(self):
        super().__init__("vam_pvt_streamer_new3")

        # Parameters
        self.declare_parameter("pvt_rate_hz", 15.0)
        self.declare_parameter("velocity_scale", 0.3)
        self.declare_parameter("accel_scale", 0.1)
        # Fraction of the Performance Safety ceiling the streamer is allowed to
        # reach. Leaves margin so no transient command value can clip the limit.
        self.declare_parameter("safety_headroom", 0.85)
        self.declare_parameter("catch_up_threshold_rad", 0.3)
        self.declare_parameter("catch_up_velocity_scale", 1.0)
        # Acceleration scaling for the MoveIt catch-up trajectory. Decoupled
        # from velocity so the trajectory can ramp DOWN gently at its end,
        # minimizing residual motion when PVT seeds at handoff. <=0 → follow
        # catch_up_velocity_scale (old coupled behavior).
        self.declare_parameter("catch_up_accel_scale", 0.0)
        # Settle dwell (sec) after a catch-up trajectory completes, before
        # re-entering PVT. Lets the arm come to rest so the v=0 seed point is
        # taken at a stationary position — avoids a moving-seed discontinuity.
        self.declare_parameter("catch_up_settle_sec", 0.3)
        self.declare_parameter("watchdog_timeout_sec", 0.5)
        self.declare_parameter("holding_timeout_sec", 2.0)
        self.declare_parameter("collision_perturbation_rad", 0.08)
        self.declare_parameter("collision_candidates", 8)
        # PVT re-entry cool-down (sec) — lets the driver alarm clear before
        # streaming resumes, preventing the trip→re-enter→trip fail-loop.
        self.declare_parameter("reentry_cooldown_sec", 0.5)
        # --- Following-error (lag) throttle [FIX 5] ---
        # As a joint's commanded position leads its ACTUAL position (the arm
        # lagging the stream), scale that joint's velocity ceiling down: full
        # speed below following_error_soft_rad, linearly to zero at
        # following_error_hard_rad. Stops the drift runaway that trips xxF061;
        # self-correcting and per-joint, so it only bites on the lagging joint
        # during fast accelerations and releases as the arm catches up.
        self.declare_parameter("following_error_soft_rad", 0.07)   # ~4°
        self.declare_parameter("following_error_hard_rad", 0.18)   # ~10°
        # --- Sudden-jump guard (general) ---
        # Protects against ALL sudden target jumps, whatever the cause: an actor
        # switching between two people on stage, someone walking onto the space
        # while the robot is idle, a skeleton misdetection, or a tracking glitch.
        # They all look identical to the robot — the target teleports — and a raw
        # chase saturates velocity at the ceiling for a sustained slew → trips the
        # safety limit (xxF061 → 0x35 STO). Two layers:
        #   1) DEBOUNCE — a target that jumps more than jump_threshold_rad from
        #      what we're tracking (or, when idle, the first target seen) is held
        #      as a candidate and ignored until it persists jump_confirm_frames.
        #      Rejects flicker, walk-throughs, and one-frame misdetections.
        #   2) SLOW APPROACH — once (and only once) a jump is CONFIRMED, the
        #      move to the new target is capped at traverse_velocity_scale, well
        #      under the safety limit, so the (discontinuous) move is a slow slew
        #      — never a lurch. Released below traverse_done_rad of the target.
        #      NOT triggered by the raw robot-to-target gap: smoothly tracking a
        #      fast actor produces a large gap too and must stay fully responsive.
        self.declare_parameter("jump_threshold_rad", 0.4)
        self.declare_parameter("jump_confirm_frames", 5)
        self.declare_parameter("traverse_velocity_scale", 0.1)
        self.declare_parameter("traverse_done_rad", 0.15)
        # Filter parameters
        self.declare_parameter("filter_type", "one_euro")
        self.declare_parameter("filter_cutoff_hz", 2.0)
        self.declare_parameter("one_euro_min_cutoff", 1.0)
        self.declare_parameter("one_euro_beta", 0.05)
        self.declare_parameter("one_euro_d_cutoff", 1.0)

        self._pvt_rate = self.get_parameter("pvt_rate_hz").value
        self._vel_scale = self.get_parameter("velocity_scale").value
        self._accel_scale = self.get_parameter("accel_scale").value
        self._safety_headroom = self.get_parameter("safety_headroom").value
        self._catch_up_thresh = self.get_parameter("catch_up_threshold_rad").value
        self._catch_up_vel_scale = self.get_parameter("catch_up_velocity_scale").value
        _ca = self.get_parameter("catch_up_accel_scale").value
        self._catch_up_accel_scale = _ca if _ca > 0.0 else self._catch_up_vel_scale
        self._catch_up_settle = self.get_parameter("catch_up_settle_sec").value
        self._watchdog_timeout = self.get_parameter("watchdog_timeout_sec").value
        self._holding_timeout = self.get_parameter("holding_timeout_sec").value
        self._collision_perturb = self.get_parameter("collision_perturbation_rad").value
        self._collision_candidates = self.get_parameter("collision_candidates").value
        self._reentry_cooldown = self.get_parameter("reentry_cooldown_sec").value
        self._fe_soft = self.get_parameter("following_error_soft_rad").value
        self._fe_hard = self.get_parameter("following_error_hard_rad").value
        self._jump_thresh = self.get_parameter("jump_threshold_rad").value
        self._jump_confirm_frames = self.get_parameter("jump_confirm_frames").value
        self._traverse_vel_scale = self.get_parameter("traverse_velocity_scale").value
        self._traverse_done = self.get_parameter("traverse_done_rad").value
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
        # Streamer velocity ceiling = safety limit × headroom × user scale.
        # The headroom term guarantees we stay strictly below the safety
        # controller's trip threshold even after FIX 1 removes spline overshoot.
        self._streamer_ceiling = TM12S_SAFETY_VEL_LIMITS * self._safety_headroom
        self._max_vel = self._streamer_ceiling * self._vel_scale
        # Max velocity change per tick: accel_scale controls ramp speed.
        # accel_scale=0.1 → reach max_vel in ~10 ticks (matches old behavior)
        # accel_scale=0.2 → reach max_vel in ~5 ticks (faster ramp)
        self._max_dv_per_tick = self._max_vel * self._accel_scale
        # Reduced ceiling used during a confirmed actor-switch traverse.
        self._traverse_max_vel = self._streamer_ceiling * self._traverse_vel_scale

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
        self._reentry_count = 0

        # Sudden-jump guard state
        self._accepted_target = None   # stable target we're tracking (None=unacquired)
        self._pending_target = None    # candidate target after a sudden jump
        self._pending_count = 0        # frames the candidate has persisted
        self._traversing = False       # mid slow approach to a far target

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
            f"VAM PVT Streamer v4 (following-error safe): /vam/joint_targets → PVT\n"
            f"  rate={self._pvt_rate:.0f}Hz, vel_scale={self._vel_scale:.0%}, "
            f"accel_scale={self._accel_scale:.0%}, "
            f"headroom={self._safety_headroom:.0%}\n"
            f"  filter={self._filter_type}"
            + (f" (cutoff={self._filter_cutoff:.1f}Hz)"
               if self._filter_type == "butterworth" else
               f" (min_cutoff={self.get_parameter('one_euro_min_cutoff').value}, "
               f"beta={self.get_parameter('one_euro_beta').value})"
               if self._filter_type == "one_euro" else "") + "\n"
            f"  streamer vel ceiling: "
            f"[{', '.join(f'{math.degrees(v):.0f}' for v in self._max_vel)}] deg/s\n"
            f"  safety ceiling:       "
            f"[{', '.join(f'{math.degrees(v):.0f}' for v in TM12S_SAFETY_VEL_LIMITS)}] deg/s\n"
            f"  catch_up_threshold={math.degrees(self._catch_up_thresh):.1f} deg, "
            f"catch_up vel={self._catch_up_vel_scale:.0%}/accel={self._catch_up_accel_scale:.0%}, "
            f"settle={self._catch_up_settle:.2f}s\n"
            f"  sudden-jump guard: jump>{math.degrees(self._jump_thresh):.0f}° "
            f"held {self._jump_confirm_frames} frames → slow approach @ "
            f"{self._traverse_vel_scale:.0%} ceiling "
            f"([{', '.join(f'{math.degrees(v):.0f}' for v in self._traverse_max_vel)}] deg/s); "
            f"smooth tracking stays full-speed\n"
            f"  following-error throttle: full speed <{math.degrees(self._fe_soft):.0f}° lag, "
            f"→0 at {math.degrees(self._fe_hard):.0f}° lag (per-joint, self-correcting)\n"
            f"  position integration = TRAPEZOIDAL (cubic-overshoot-safe)"
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
        raw = np.array(msg.data)
        with self._target_lock:
            # Timestamp always refreshes: we ARE receiving data, even while a
            # sudden jump is being debounced — keeps the watchdog from firing.
            self._last_target_time = self.get_clock().now()
            self._latest_target = self._debounce_target(raw)

    @staticmethod
    def _wrap(x):
        """Wrap angle differences to [-π, π] for shortest-path comparison."""
        return (x + np.pi) % (2 * np.pi) - np.pi

    def _debounce_target(self, raw):
        """Reject transient sudden jumps in the target (any cause).

        Returns the stable target the streamer should track, or None while no
        target has been acquired yet (so the robot stays put). A target that
        jumps more than jump_threshold_rad from what we're tracking — or, when
        unacquired, the first target seen — is held as a candidate and ignored
        until it persists jump_confirm_frames. This rejects skeleton flicker,
        a person walking across/through, and one-frame misdetections, both
        mid-stream AND at first acquisition from idle.
        Must be called under _target_lock.
        """
        # --- Acquisition: nothing tracked yet (robot idle / just started) ---
        if self._accepted_target is None:
            if (self._pending_target is None
                    or np.max(np.abs(self._wrap(raw - self._pending_target)))
                    > self._jump_thresh):
                # New / moved candidate — (re)start the persistence counter.
                self._pending_target = raw.copy()
                self._pending_count = 1
            else:
                self._pending_count += 1
                self._pending_target = raw.copy()
            if self._pending_count >= self._jump_confirm_frames:
                self._accepted_target = self._pending_target.copy()
                self._pending_target = None
                self._pending_count = 0
                self._traversing = True  # slow-approach the first target
                self.get_logger().info("Target acquired (stable) — slow approach")
                return self._accepted_target
            return None  # still acquiring — robot holds

        # --- Tracking: already locked onto a target ---
        jump = np.max(np.abs(self._wrap(raw - self._accepted_target)))
        if jump <= self._jump_thresh:
            # Normal small motion — follow it, drop any pending candidate.
            self._accepted_target = raw.copy()
            self._pending_target = None
            self._pending_count = 0
            return self._accepted_target

        # Sudden jump → candidate new target. Don't commit until it persists.
        if (self._pending_target is not None
                and np.max(np.abs(self._wrap(raw - self._pending_target)))
                <= self._jump_thresh):
            self._pending_count += 1
            self._pending_target = raw.copy()
        else:
            self._pending_target = raw.copy()
            self._pending_count = 1

        if self._pending_count >= self._jump_confirm_frames:
            # Sustained → genuine new target. Commit and arm the slow approach
            # so the (now large, discontinuous) move is a gentle slew.
            self._accepted_target = self._pending_target.copy()
            self._pending_target = None
            self._pending_count = 0
            self._traversing = True
            self.get_logger().warn(
                f"Sudden target jump confirmed ({self._jump_confirm_frames} "
                f"frames) — slow approach"
            )
            return self._accepted_target

        # Not yet confirmed — ignore the jump, keep tracking current target.
        return self._accepted_target

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

        # Send seed point: current position, zero velocity.
        # With trapezoidal integration the first streamed segment ramps from
        # this v=0 seed linearly — no overshoot, so this is now safe.
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
        goal.request.max_acceleration_scaling_factor = self._catch_up_accel_scale

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
            # Settle dwell: let the arm come fully to rest before the caller
            # seeds PVT at v=0, so the seed position isn't taken mid-decel.
            if self._catch_up_settle > 0.0:
                time.sleep(self._catch_up_settle)
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
                    self._send_hold_point()
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
                    self._last_sent_vel = np.zeros(N_JOINTS)
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
                self._send_hold_point()

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

    def _send_hold_point(self):
        """Send a zero-velocity hold at the last sent position.

        Velocity is held at exactly 0 and _last_sent_vel zeroed so a resumed
        stream ramps from rest (no v0=high → v1 step that would overshoot).
        """
        self._send_pvt_point(self._last_sent_pos.tolist(), [0.0] * N_JOINTS, self._dt)
        self._last_sent_vel = np.zeros(N_JOINTS)

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
            self._send_hold_point()
            self.get_logger().warn(
                "Persistent collision — holding position",
                throttle_duration_sec=2.0,
            )
            return

        # Filter the target to remove high-frequency jitter
        safe_target = self._filter_target(safe_target)

        # Slow-approach guard: armed by the debounce ONLY when a sudden target
        # discontinuity (or first acquisition) is confirmed — never by the raw
        # robot-to-target gap. This is the key distinction: smoothly tracking a
        # fast-moving actor produces a large gap too, but small per-frame deltas,
        # so it must stay FULLY responsive. Only a true jump trips the slow slew.
        # Here we just release it once the arm has reached the new target.
        if self._traversing:
            gap_to_target = np.max(np.abs(self._wrap(safe_target - current)))
            if gap_to_target <= self._traverse_done:
                self._traversing = False
                self.get_logger().info("Reached target — resuming normal tracking")
        active_max_vel = self._traverse_max_vel if self._traversing else self._max_vel

        # === FIX 5: following-error (lag) throttle ===
        # Scale each joint's velocity ceiling DOWN by how far the COMMANDED
        # position (prev) currently leads the ACTUAL arm (current). This is the
        # only feedback from the real robot in the loop: full speed while the
        # arm keeps up (lag ≤ fe_soft), linearly to zero by fe_hard, so the
        # commanded stream can never run away from the arm. Prevents the drift
        # runaway that the firmware rejects as "speed command too large"
        # (xxF061). Per-joint and self-correcting — as the arm catches up the
        # lag shrinks and full speed returns, so smooth tracking stays snappy.
        lag = np.abs(self._wrap(prev - current))
        span = max(self._fe_hard - self._fe_soft, 1e-6)
        lag_scale = np.clip(1.0 - (lag - self._fe_soft) / span, 0.0, 1.0)
        active_max_vel = active_max_vel * lag_scale

        # Desired velocity toward target (from last SENT position, not current)
        # Normalize to [-π, π] to take shortest angular path
        error = (safe_target - prev + np.pi) % (2 * np.pi) - np.pi
        desired_vel = error / self._dt

        # 1. Clamp velocity magnitude (per-joint, against active ceiling)
        desired_vel = np.clip(desired_vel, -active_max_vel, active_max_vel)

        # 2. Acceleration ramping
        dv = desired_vel - self._last_sent_vel
        dv = np.clip(dv, -self._max_dv_per_tick, self._max_dv_per_tick)
        velocity = self._last_sent_vel + dv
        velocity = np.clip(velocity, -active_max_vel, active_max_vel)

        # === FIX 1: trapezoidal position integration ===
        # The TM firmware fits a cubic Hermite between consecutive PVT points.
        # Using next_pos = prev + velocity*dt (forward-Euler) makes that spline
        # bulge ABOVE the endpoint velocity mid-segment — the overshoot that
        # trips the J4/J5 safety ceiling (xxF061 → 0x35 STO).
        # Setting the position delta to the trapezoidal integral of the endpoint
        # velocities makes the interpolated velocity profile exactly LINEAR
        # (v(s) = v0 + (v1-v0)s), so peak velocity == max(v0,v1) with zero
        # overshoot. This is the primary fix.
        next_pos = prev + 0.5 * (self._last_sent_vel + velocity) * self._dt

        # Send PVT point
        ok = self._send_pvt_point(next_pos.tolist(), velocity.tolist(), self._dt)
        if not ok:
            # === FIX 3: re-enter through a cool-down + rest ramp, not a hot
            # re-stream. Reset velocity to 0 so the resumed stream ramps from
            # rest; the cool-down lets any driver alarm clear before we resume.
            self._reentry_count += 1
            self.get_logger().error(
                f"PVT send failed — re-entering PVT mode "
                f"(re-entry #{self._reentry_count}, "
                f"cooldown={self._reentry_cooldown:.2f}s)"
            )
            self._pvt_exit()
            time.sleep(self._reentry_cooldown)
            self._last_sent_pos = self._current_positions.copy()
            self._last_sent_vel = np.zeros(N_JOINTS)
            if not self._pvt_enter():
                self._state = State.IDLE
            return

        self._last_sent_pos = next_pos.copy()
        self._last_sent_vel = velocity.copy()
        self._cmds_sent += 1

        # === FIX 4: per-joint diagnostics ===
        drift_deg = math.degrees(np.max(np.abs(next_pos - current)))
        gap_deg = math.degrees(np.max(np.abs(target - current)))
        peak_j = int(np.argmax(np.abs(velocity)))
        peak_vel_deg = math.degrees(abs(velocity[peak_j]))
        active_ceiling = self._traverse_max_vel if self._traversing else self._max_vel
        peak_ceiling_deg = math.degrees(active_ceiling[peak_j])
        mode = "TRAVERSE" if self._traversing else "track"
        # Following-error: max lag and whether the throttle is biting.
        lag_deg = math.degrees(np.max(lag))
        throttled = "" if np.min(lag_scale) > 0.99 else f", THROTTLED×{np.min(lag_scale):.2f}"

        if self._cmds_sent <= 10 or self._cmds_sent % 50 == 0 or self._traversing:
            self.get_logger().info(
                f"PVT #{self._cmds_sent} [{mode}]: "
                f"peak J{peak_j + 1}={peak_vel_deg:.1f}/{peak_ceiling_deg:.0f}°/s, "
                f"gap={gap_deg:.1f}°, "
                f"drift={drift_deg:.1f}°, "
                f"lag={lag_deg:.1f}°{throttled}",
                throttle_duration_sec=0.5,
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
    node = VamPvtStreamerNew3()

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
