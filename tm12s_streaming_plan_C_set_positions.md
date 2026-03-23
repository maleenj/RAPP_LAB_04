# Plan C: TM12S Smooth Joint Streaming via set_positions (PTP with Blending)

## Problem

Same as Plans A and B: stream continuous joint angles from a predictive transformer vision-action model to a TM12S with smooth motion. This plan uses **no driver modifications at all** — only the existing `/set_positions` service with careful parameter tuning to achieve smooth-enough motion via PTP blending.

If you haven't read Plan A, read the "Background: TM12S Robot Driver Architecture" section there first — it explains how the driver communicates with the robot (TMSCT/TMSVR sockets, Listen Node, etc.) and lists all the prerequisites.

---

## Solution Overview (Plain English)

### What We're Building

We are building a **blended PTP command pipeline** that takes the vision-action model's joint position outputs and sends them as rapid-fire PTP (Point-to-Point) motion commands with maximum blending enabled. The robot blends consecutive motions together instead of stopping at each target, producing reasonably smooth continuous motion. This requires **zero driver code changes** — it uses only the existing `/set_positions` ROS2 service.

### What is PTP Motion and Why Does It Usually Cause Jerky Motion?

PTP (Point-to-Point) is the TM robot's basic motion command. You tell the robot "go to this joint configuration" and it:

1. Computes a trapezoidal velocity profile (accelerate → cruise → decelerate)
2. Executes the motion
3. Stops at the target position
4. Waits for the next command

When you send rapid successive PTP commands (like from a vision model at 10-30Hz), the robot receives each one, fully stops at the target, then starts moving to the next target. This acceleration-stop-acceleration cycle is what produces the violent jerky motion you're experiencing.

### The Trick: Blend Percentage and fine_goal=false

The PTP command has two parameters that change this stop-start behavior:

**`blend_percentage` (0-100%)**:
- At 0%: The robot must reach the exact target before starting the next motion (full stop between commands)
- At 100%: The robot begins transitioning to the next queued target as early as possible, creating a smooth arc through the waypoint rather than stopping at it
- The "blend zone" is a region near the target. At 100% blend, the robot enters the blend zone very early and starts curving toward the next target. The robot may never actually reach the commanded position — it "cuts the corner"

**`fine_goal` (true/false)**:
- `true`: Robot must converge to within a tight tolerance of the exact target before considering the motion complete (incompatible with blending)
- `false`: Robot considers the motion complete when it enters the blend zone, allowing immediate transition to the next command

With `blend_percentage=100` and `fine_goal=false`, consecutive PTP commands blend smoothly together. The robot traces a continuous path through the waypoints without stopping.

### The Queue Management Challenge

The TM robot has an internal **command queue** for PTP motions. When you send PTP commands faster than the robot executes them, they accumulate in this queue. This creates a critical challenge:

- **Queue too full (many stale commands)**: The robot follows an outdated sequence of targets. If the model changes direction, the robot is still executing commands from 500ms ago. This manifests as the robot "lagging behind" the model's intent.
- **Queue empty (gap between commands)**: The robot finishes the current command, has nothing queued, and stops. When the next command arrives, it starts from rest — producing a jerk.
- **Sweet spot (2-3 commands buffered)**: The robot always has the next command ready when it finishes the current one, but the buffer is shallow enough that lag stays manageable.

Our servo node manages this by:

1. Sending commands at a fixed rate (~8Hz) to keep the queue fed
2. Monitoring position error between actual joint positions and last sent target
3. If error exceeds a threshold (robot is lagging badly), sending `StopAndClearBuffer()` to flush the queue and start fresh

### How `StopAndClearBuffer()` Works

`StopAndClearBuffer()` is a TMScript command that:

1. Immediately stops robot motion
2. Clears all queued PTP/Line commands
3. The robot is ready for new commands instantly

This is our "escape hatch" for when the queue gets stale. It causes a brief motion interruption (the robot stops and restarts), but it's better than following an outdated trajectory. In practice, with good rate tuning, you should rarely need to flush.

### Why This is the Weakest Option (But Still Useful)

**Compared to Plan A (velocity mode)**:
- PTP commands are queued, velocity commands override instantly
- PTP blending "cuts corners" — position accuracy is approximate
- PTP uses blocking service calls, velocity uses non-blocking topic publish
- Max effective command rate ~8-10Hz vs 50Hz for velocity mode

**Compared to Plan B (PVT mode)**:
- PTP blending uses trapezoidal velocity profiles, PVT uses cubic spline interpolation
- PTP blending has no velocity specification — robot picks its own speed profile; PVT specifies exact velocities at each point
- PTP queue depth is opaque; PVT timing is explicit

**Why it's still useful**:
- Zero driver code changes — works with the driver exactly as shipped
- 5-minute validation test — confirms the approach immediately
- Good enough for low-frequency model outputs (<10Hz) with moderate smoothness requirements
- Excellent for quick prototyping before committing to Plan A or B

### End-to-End Data Flow

```
Vision-Action Model (Python, 10-30 Hz)
    |
    | Publishes sensor_msgs/JointState to /target_joint_positions
    | (6 joint angles in radians — the model's predicted next pose)
    v
tm_ptp_servo_node.py (Python ROS2 node, 8 Hz command loop)
    |
    | Reads current joint positions from /joint_states
    | Computes step: clamp(target - last_sent, max_step_rad)
    | Sends step-limited position via /set_positions service:
    |   motion_type=PTP_J, velocity=1.0 rad/s (~32% max),
    |   acc_time=150ms, blend_percentage=100, fine_goal=false
    | Monitors position error — flushes queue if lag exceeds threshold
    v
/set_positions (existing ROS2 service in tm_driver)
    |
    | TmSctRos2::set_positions() handler
    | Converts velocity from rad/s to percentage of max (pi rad/s)
    | Calls iface_.set_joint_pos_PTP(positions, vel%, acc, blend%, false)
    | Which calls sct.send_script_str() — BLOCKING
    | Sends PTP("JPP", j1_deg,...,j6_deg, vel%, acc_ms, blend%, false)
    v
TM12S Robot Controller (Listen Node on TMflow)
    |
    | Receives PTP command via TMSCT socket
    | Queues the motion command
    | Executes with trapezoidal velocity profile
    | Blends into the next queued command (blend=100%)
    | Never fully stops between consecutive commands
    v
Reasonably smooth continuous motion!
```

### Safety Mechanisms

1. **Step clamping**: Each command's position change is clamped to `max_step_rad` (default 0.08 rad = ~4.6 deg). At 8Hz this limits max speed to ~0.64 rad/s.
2. **Velocity parameter**: PTP velocity is set to 1.0 rad/s (~32% of max). Adjustable.
3. **Queue flush**: If actual-vs-commanded position error exceeds `queue_clear_threshold` (default 0.3 rad = ~17 deg), `StopAndClearBuffer()` is sent to purge stale commands.
4. **Target timeout**: If no new target arrives for `target_timeout` seconds, the node stops sending commands (robot decelerates to stop on its own).
5. **Shutdown cleanup**: Signal handlers always send `StopAndClearBuffer()` on exit.

---

## Why This Approach

This is the **zero-modification fallback**. If velocity mode (Plan A) fails due to firmware issues and PVT streaming (Plan B) has timing problems, this approach works with the driver exactly as shipped. The key insight is that PTP commands support a **blend_percentage** parameter — when set high, the robot does NOT stop between consecutive motions but smoothly blends them together.

### How PTP Blending Works

The TM robot's PTP command:
```
PTP("JPP", j1_deg, j2_deg, ..., j6_deg, velocity%, acc_time_ms, blend%, fine_goal)
```

- **`blend_percentage`** (0-100%): Controls how much the robot blends between consecutive motions.
  - `0%` = robot fully stops at each target before starting the next (this is what causes the jerky motion)
  - `100%` = robot starts moving to the next target as early as possible, creating a smooth path through waypoints
  - The blending zone is a percentage of the motion distance — at 100%, the robot may never actually reach the exact target position before moving on

- **`fine_goal`** (`true`/`false`): When `false`, the robot considers the motion complete when it enters the blend zone, not when it reaches the exact target. **Must be `false` for smooth streaming.**

- **`velocity`** (% of max): Controls motion speed. For streaming, we want moderate speed so the robot can keep up with incoming targets.

- **`acc_time`** (ms): Acceleration/deceleration time. Lower = snappier response, higher = smoother acceleration profiles.

### The Trick: Command Queue Management

The TM robot queues incoming PTP commands. With `blend_percentage=100%` and `fine_goal=false`, the robot smoothly transitions from one queued target to the next without stopping. The challenge is **queue management**:

- **Too many commands queued** → robot lags behind model output, following stale targets
- **Too few commands** → robot finishes current command and stops before the next arrives (gap = jerk)
- **Sweet spot** → maintain 2-3 commands in the queue at all times

### Trade-offs vs Plans A and B

| Aspect | Plan C (set_positions) | Plan A (velocity) | Plan B (PVT) |
|--------|----------------------|-------------------|--------------|
| Driver changes | None | C++ subscriber + services | C++ subscriber + services |
| Smoothness | Good (blended PTP) | Good (raw velocity) | Best (cubic interpolation) |
| Reactivity | Low (queued motions) | Highest (immediate override) | Moderate (queued points) |
| Complexity | Simplest | Moderate | Moderate |
| Position accuracy | Approximate (blending skips targets) | Approximate (velocity tracking) | Good (robot interpolates to positions) |
| Maximum rate | ~5-10 Hz (blocking service) | 50 Hz (non-blocking topic) | ~20 Hz (blocking send) |
| Best for | Quick validation, low-frequency output | High-frequency, reactive control | Smooth pre-planned paths |

---

## Existing Driver Interface Used

### `/set_positions` Service (from `tm_msgs/srv/SetPositions.srv`)

```
# Request
int8 PTP_J = 1        # Joint-space PTP
int8 PTP_T = 2        # Tool-space PTP
int8 LINE_T = 4       # Linear tool motion

int8 motion_type
float64[] positions    # Joint angles in radians (for PTP_J)
float64 velocity       # rad/s (converted to % of max internally — max is pi rad/s)
float64 acc_time       # Acceleration time in milliseconds
int32 blend_percentage # 0-100%
bool fine_goal         # Must be false for blending to work
---
bool ok
```

### How It Works Internally (from `tm_driver/src/tm_driver.cpp:128-136`)

```cpp
bool TmDriver::set_joint_pos_PTP(const std::vector<double> &angs,
    double vel, double acc_time, int blend_percent, bool fine_goal, const std::string &id)
{
    int vel_pa = int(100.0 * (vel / _max_velocity));  // _max_velocity = M_PI
    if (vel_pa >= 100) vel_pa = 100;
    return (sct.send_script_str(
        id, TmCommand::set_joint_pos_PTP(angs, vel_pa, acc_time, blend_percent, fine_goal)
    ) == RC_OK);
}
```

- `velocity` in rad/s is converted to a percentage of `_max_velocity` (pi rad/s = 180 deg/s)
- A velocity of 1.0 rad/s = ~32% of max
- Uses `send_script_str` (blocking) — each call waits for the packet to be sent

### `/send_script` Service (from `tm_msgs/srv/SendScript.srv`)

```
string id
string script
---
bool ok
```

Can send any TMScript — useful for `StopAndClearBuffer()` to flush the command queue.

---

## Phase 0: Quick Validation Test (NO CODE CHANGES)

### Prerequisites

1. TMflow project with Listen Node active, robot in Auto Mode
2. TM driver launched: `ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>`
3. Verify: `ros2 topic echo /joint_states --once`

### Test Script: `test_set_positions_smooth.py`

Tests smooth motion by sending a sequence of PTP targets with high blending.

```python
#!/usr/bin/env python3
"""
Quick validation test for smooth motion via set_positions with PTP blending.
Uses ONLY existing /set_positions and /send_script services — no driver modifications.

Moves joint 1 through a series of waypoints with blend_percentage=100%.
Compare the smoothness against blend_percentage=0% to see the difference.

WARNING: Ensure workspace is clear.

Usage:
    1. Launch tm_driver: ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<IP>
    2. Run: python3 test_set_positions_smooth.py
    3. Press Ctrl+C to stop
"""

import rclpy
from rclpy.node import Node
from tm_msgs.srv import SetPositions, SendScript
from sensor_msgs.msg import JointState
import math
import time
import signal
import sys
import numpy as np


class SetPositionsSmoothTest(Node):
    def __init__(self):
        super().__init__('set_positions_smooth_test')

        self.set_pos_cli = self.create_client(SetPositions, 'set_positions')
        self.send_script_cli = self.create_client(SendScript, 'send_script')

        while not self.set_pos_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Waiting for /set_positions service...')
        while not self.send_script_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Waiting for /send_script service...')

        self.current_joints = None
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_cb, 10)

        self.get_logger().info('SetPositions smooth test node ready.')

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position[:6])

    def send_ptp(self, positions_rad, velocity_rad_s=0.8, acc_time_ms=200.0,
                 blend_percent=100, fine_goal=False):
        """
        Send a PTP_J command via /set_positions service.

        Args:
            positions_rad: 6 joint angles in radians
            velocity_rad_s: Joint velocity in rad/s (max = pi ≈ 3.14)
            acc_time_ms: Acceleration time in milliseconds
            blend_percent: Blending percentage (0=stop at target, 100=max blending)
            fine_goal: If True, waits for precise position (must be False for blending)
        """
        req = SetPositions.Request()
        req.motion_type = SetPositions.Request.PTP_J
        req.positions = list(positions_rad)
        req.velocity = velocity_rad_s
        req.acc_time = acc_time_ms
        req.blend_percentage = blend_percent
        req.fine_goal = fine_goal

        future = self.set_pos_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            return future.result().ok
        return False

    def stop_and_clear(self):
        """Send StopAndClearBuffer to flush the command queue."""
        req = SendScript.Request()
        req.id = 'Stop'
        req.script = 'StopAndClearBuffer()'
        future = self.send_script_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        self.get_logger().info('Sent StopAndClearBuffer()')

    def run_waypoint_test(self, blend_percent=100):
        """
        Send joint 1 through a series of waypoints with the given blend percentage.
        Run once with blend=100 (smooth) and once with blend=0 (jerky) to compare.
        """
        self.get_logger().info('Waiting for /joint_states...')
        while self.current_joints is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        start_pos = self.current_joints.copy()
        self.get_logger().info(
            f'Start position (deg): {[f"{math.degrees(j):.2f}" for j in start_pos]}'
        )
        self.get_logger().info(f'Running waypoint test with blend_percentage={blend_percent}%')

        # Generate sine wave waypoints for joint 1
        num_waypoints = 40
        amplitude_rad = 0.15  # ~8.6 degrees
        waypoints = []
        for i in range(num_waypoints):
            t = i / num_waypoints * 4.0 * math.pi  # Two full cycles
            pos = start_pos.copy()
            pos[0] = start_pos[0] + amplitude_rad * math.sin(t)
            waypoints.append(pos)

        # Return to start
        waypoints.append(start_pos.copy())

        self.get_logger().info(f'Sending {len(waypoints)} waypoints...')

        for i, wp in enumerate(waypoints):
            ok = self.send_ptp(
                wp,
                velocity_rad_s=0.8,       # ~25% of max speed
                acc_time_ms=200.0,         # 200ms acceleration
                blend_percent=blend_percent,
                fine_goal=False
            )
            if not ok:
                self.get_logger().error(f'PTP command failed at waypoint {i}')
                break

            # Pace the commands — don't flood the queue
            # With blend=100%, the robot processes these quickly
            # We want 2-3 commands buffered ahead
            time.sleep(0.15)  # ~6.7 Hz command rate

            # Process callbacks
            rclpy.spin_once(self, timeout_sec=0)

            if i % 10 == 0:
                actual = self.current_joints
                self.get_logger().info(
                    f'  Waypoint {i}/{len(waypoints)} | '
                    f'target_j1={math.degrees(wp[0]):.2f} deg | '
                    f'actual_j1={math.degrees(actual[0]):.2f} deg'
                )

        self.get_logger().info('Waypoint test complete.')

    def run_tracking_test(self):
        """
        Track /target_joint_positions using set_positions with blending.
        This is the actual streaming use case.
        """
        self.get_logger().info('Waiting for /joint_states...')
        while self.current_joints is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        self.target_joints = None
        self.target_sub = self.create_subscription(
            JointState, 'target_joint_positions', self._target_cb, 10)

        self.get_logger().info('Tracking mode: publishing to /target_joint_positions')
        self.get_logger().info('Waiting for first target...')

        while self.target_joints is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        # Configuration
        send_rate = 8.0       # Hz — how often we send PTP commands
        velocity = 1.0        # rad/s — ~32% of max
        acc_time = 150.0      # ms
        blend = 100           # percent
        dt = 1.0 / send_rate

        self.get_logger().info(
            f'Tracking with: rate={send_rate}Hz, vel={velocity}rad/s, '
            f'acc={acc_time}ms, blend={blend}%'
        )

        prev_target = self.current_joints.copy()
        max_step = 0.08  # Safety: max position change per command (rad)

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0)

                if self.target_joints is None:
                    time.sleep(dt)
                    continue

                target = self.target_joints.copy()

                # Safety: limit step size
                step = target - prev_target
                step = np.clip(step, -max_step, max_step)
                clamped_target = prev_target + step

                self.send_ptp(
                    clamped_target,
                    velocity_rad_s=velocity,
                    acc_time_ms=acc_time,
                    blend_percent=blend,
                    fine_goal=False
                )

                prev_target = clamped_target.copy()
                time.sleep(dt)

        except KeyboardInterrupt:
            pass
        finally:
            self.get_logger().info('Stopping — clearing command buffer...')
            self.stop_and_clear()

    def _target_cb(self, msg):
        if len(msg.position) >= 6:
            self.target_joints = np.array(msg.position[:6])


def main():
    rclpy.init()
    node = SetPositionsSmoothTest()

    def signal_handler(sig, frame):
        node.get_logger().warn('Ctrl+C — clearing buffer and stopping...')
        node.stop_and_clear()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Test 1: Smooth (blend=100%)
        node.get_logger().info('=== TEST 1: blend=100% (should be smooth) ===')
        node.run_waypoint_test(blend_percent=100)

        time.sleep(2.0)

        # Test 2: Jerky (blend=0%) for comparison
        node.get_logger().info('=== TEST 2: blend=0% (should be jerky — for comparison) ===')
        node.run_waypoint_test(blend_percent=0)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### What to Observe

Run the test and compare:
- **Test 1 (blend=100%)**: Joint 1 should trace the sine wave with reasonably smooth motion. There may be slight speed variations but NO full stops between waypoints.
- **Test 2 (blend=0%)**: Joint 1 should move in a jerky stop-start pattern. This is what you're seeing with MoveIt currently.

The difference should be immediately visible. If blend=100% is smooth enough for your use case, this approach requires zero driver changes.

---

## Phase 1: Python Streaming Node (NO DRIVER CHANGES)

Since this approach uses only existing services, there's no C++ driver modification. The entire solution is a single Python node.

### File: `tm_driver/tm_driver/tm_ptp_servo_node.py`

```python
#!/usr/bin/env python3
"""
TM12S PTP Servo Node (Plan C — no driver modifications)

Achieves smooth-ish motion by sending PTP_J commands with high blend_percentage
and careful queue management. Uses only the existing /set_positions service.

This is the simplest approach but has limitations:
- Max effective command rate ~8-10 Hz (service call overhead)
- Slight position inaccuracy from blending (robot cuts corners)
- Less reactive to sudden direction changes than velocity mode

Subscribes:
    /target_joint_positions (sensor_msgs/JointState) — target from vision-action model
    /joint_states (sensor_msgs/JointState) — current robot state

Services called:
    /set_positions — send PTP_J commands with blending
    /send_script — for StopAndClearBuffer on shutdown

Parameters:
    command_rate (double, default=8.0): PTP command send rate in Hz
    velocity (double, default=1.0): Joint velocity in rad/s (max pi)
    acc_time (double, default=150.0): Acceleration time in milliseconds
    blend_percentage (int, default=100): Blending between motions (0-100%)
    max_step_rad (double, default=0.08): Max position change per command (safety)
    target_timeout (double, default=0.5): Hold position if no target for this long
    queue_clear_threshold (double, default=0.3): If position error exceeds this (rad),
        flush queue and send fresh target (prevents lag from stale commands)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tm_msgs.srv import SetPositions, SendScript
import numpy as np
import math
import time
import signal
import sys
import threading


class TmPtpServoNode(Node):
    def __init__(self):
        super().__init__('tm_ptp_servo')

        # Parameters
        self.declare_parameter('command_rate', 8.0)
        self.declare_parameter('velocity', 1.0)            # rad/s
        self.declare_parameter('acc_time', 150.0)           # ms
        self.declare_parameter('blend_percentage', 100)     # 0-100
        self.declare_parameter('max_step_rad', 0.08)        # rad per command
        self.declare_parameter('target_timeout', 0.5)       # seconds
        self.declare_parameter('queue_clear_threshold', 0.3)  # rad

        self.rate = self.get_parameter('command_rate').value
        self.velocity = self.get_parameter('velocity').value
        self.acc_time = self.get_parameter('acc_time').value
        self.blend = self.get_parameter('blend_percentage').value
        self.max_step = self.get_parameter('max_step_rad').value
        self.target_timeout = self.get_parameter('target_timeout').value
        self.queue_clear_threshold = self.get_parameter('queue_clear_threshold').value

        self.dt = 1.0 / self.rate

        # State
        self.current_joints = None
        self.target_joints = None
        self.last_target_time = None
        self.last_sent_target = None
        self.running = True
        self.send_lock = threading.Lock()

        # Service clients
        self.set_pos_cli = self.create_client(SetPositions, 'set_positions')
        self.send_script_cli = self.create_client(SendScript, 'send_script')

        while not self.set_pos_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Waiting for /set_positions service...')

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self._joint_cb, 10)
        self.target_sub = self.create_subscription(
            JointState, 'target_joint_positions', self._target_cb, 10)

        # Control loop timer
        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f'TM PTP Servo Node started: rate={self.rate}Hz, vel={self.velocity}rad/s, '
            f'acc={self.acc_time}ms, blend={self.blend}%, max_step={self.max_step}rad'
        )
        self.get_logger().info('Publish targets to /target_joint_positions (sensor_msgs/JointState)')

    def _joint_cb(self, msg):
        self.current_joints = np.array(msg.position[:6])

    def _target_cb(self, msg):
        if len(msg.position) >= 6:
            new_target = np.array(msg.position[:6])

            # Check if target has changed significantly — if so, consider flushing queue
            if (self.current_joints is not None and self.last_sent_target is not None):
                # Error between current actual position and what we last sent
                pos_error = np.max(np.abs(self.current_joints - self.last_sent_target))
                # If error is large, robot is lagging — flush queue on next control tick
                if pos_error > self.queue_clear_threshold:
                    self._flush_queue()

            self.target_joints = new_target
            self.last_target_time = time.time()

    def _send_ptp(self, positions_rad):
        """Send PTP_J with blending via /set_positions service (async, non-waiting)."""
        req = SetPositions.Request()
        req.motion_type = SetPositions.Request.PTP_J
        req.positions = list(positions_rad)
        req.velocity = self.velocity
        req.acc_time = self.acc_time
        req.blend_percentage = self.blend
        req.fine_goal = False

        # Fire async — don't wait for response (keeps control loop fast)
        self.set_pos_cli.call_async(req)

    def _flush_queue(self):
        """Send StopAndClearBuffer to flush stale commands from robot queue."""
        req = SendScript.Request()
        req.id = 'Flush'
        req.script = 'StopAndClearBuffer()'
        future = self.send_script_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.get_logger().warn('Flushed command queue (target lag detected)')

    def _control_loop(self):
        """Main control loop — sends PTP commands at command_rate Hz."""
        if self.current_joints is None:
            return

        if self.target_joints is None:
            return

        # Check target timeout
        if self.last_target_time and (time.time() - self.last_target_time) > self.target_timeout:
            return

        target = self.target_joints.copy()

        # Compute step from current position to target
        if self.last_sent_target is not None:
            step = target - self.last_sent_target
        else:
            step = target - self.current_joints

        # Clamp step size for safety
        step = np.clip(step, -self.max_step, self.max_step)

        if self.last_sent_target is not None:
            send_pos = self.last_sent_target + step
        else:
            send_pos = self.current_joints + step

        self._send_ptp(send_pos)
        self.last_sent_target = send_pos.copy()

    def stop(self):
        """Clean shutdown — flush queue."""
        self.running = False
        self._flush_queue()
        self.get_logger().info('PTP servo stopped.')


def main(args=None):
    rclpy.init(args=args)
    node = TmPtpServoNode()

    def shutdown_handler(sig, frame):
        node.get_logger().warn('Shutting down PTP servo...')
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        if node.running:
            node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Install the Python Script

Add to `tm_driver/CMakeLists.txt` before `ament_package()`:

```cmake
install(PROGRAMS
  tm_driver/tm_ptp_servo_node.py
  DESTINATION lib/${PROJECT_NAME}
)
```

Create directory and place the script:

```bash
mkdir -p ~/git/tm2_ros2/tm_driver/tm_driver
# Place tm_ptp_servo_node.py here
chmod +x ~/git/tm2_ros2/tm_driver/tm_driver/tm_ptp_servo_node.py
```

Rebuild:

```bash
cd ~/git/tm2_ros2
colcon build --packages-select tm_driver
source install/setup.bash
```

---

## Phase 2: Integration

### Data Flow

```
Vision-Action Model (10-30Hz)
    | publishes sensor_msgs/JointState
    v
/target_joint_positions
    |
    v
tm_ptp_servo_node.py (8Hz PTP command loop)
    | computes step-limited target position
    | calls /set_positions service (PTP_J, blend=100%, fine_goal=false)
    v
/set_positions (existing ROS2 service)
    |
    v
TmSctRos2::set_positions() handler
    | calls iface_.set_joint_pos_PTP(pos, vel%, acc, blend%, false)
    | sends PTP("JPP", ..., 100, false) via send_script_str
    v
TM12S Robot (blends between consecutive PTP commands)
```

### Launch Sequence

```bash
# Terminal 1: TM Driver (unmodified)
ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>

# Terminal 2: PTP Servo Node
ros2 run tm_driver tm_ptp_servo_node.py --ros-args \
  -p command_rate:=8.0 \
  -p velocity:=1.0 \
  -p acc_time:=150.0 \
  -p blend_percentage:=100 \
  -p max_step_rad:=0.08

# Terminal 3: Vision-Action Model
python3 your_model_inference.py
```

### Tuning Guide

| Parameter | Effect | Start With | Increase For | Decrease For |
|-----------|--------|------------|-------------|-------------|
| `command_rate` | PTP commands per second | 8.0 Hz | More waypoints, smoother path | Less network load, safer |
| `velocity` | Joint speed | 1.0 rad/s | Faster tracking, keep up with model | Slower, safer motion |
| `acc_time` | Acceleration ramp | 150.0 ms | Smoother accel (but sluggish start) | Snappier response (but jerkier accel) |
| `blend_percentage` | Motion blending | 100% | Smoother transitions (skip targets) | More accurate positions (more stops) |
| `max_step_rad` | Safety clamp per command | 0.08 rad | Faster large motions | Safer, slower max speed |
| `queue_clear_threshold` | Flush stale commands | 0.3 rad | Tolerate more lag | Flush sooner, more reactive |

### Critical Tuning Tips

1. **`command_rate` vs `velocity` balance**: If `velocity` is too low relative to `command_rate`, the robot can't reach each target before the next arrives, creating a growing lag. Rule of thumb: `velocity` should be at least `max_step_rad * command_rate`.

2. **Queue lag detection**: The node monitors `current_joints` vs `last_sent_target`. If the error exceeds `queue_clear_threshold`, it sends `StopAndClearBuffer()` to flush stale commands and sends a fresh target. This prevents the robot from following an outdated trajectory when the model changes direction.

3. **`acc_time` trade-off**: Lower acceleration time means the robot reaches target velocity faster, improving responsiveness. But very low values (~50ms) can cause mechanical vibration. Start at 150ms and tune from there.

4. **The fundamental limitation**: PTP blending creates smooth paths but the robot "cuts corners" — it never precisely reaches intermediate targets. For a vision-action model this is usually acceptable (the model keeps correcting), but if you need precise waypoint following, use Plan B (PVT).

---

## Comparison: When to Use Which Plan

| Scenario | Best Plan |
|----------|-----------|
| Model outputs at 10-30Hz, needs most reactive control | **Plan A** (velocity mode) |
| Model outputs at 10-30Hz, needs smoothest interpolation | **Plan B** (PVT mode) |
| Quick test with zero driver changes | **Plan C** (set_positions) |
| Model outputs at < 10Hz, moderate smoothness OK | **Plan C** (set_positions) |
| Velocity mode firmware not supported on your TM version | **Plan B** or **Plan C** |
| PVT timing issues or point rejection errors | **Plan C** (set_positions) |
| Robot firmware is old / limited TMScript support | **Plan C** (set_positions) |

### Recommended Test Order

1. **Plan C Phase 0** — 5 minutes, zero changes, validates that blended PTP is smooth enough
2. **Plan A Phase 0** — 5 minutes, zero changes, validates velocity mode works on your firmware
3. **Plan B Phase 0** — 5 minutes, zero changes, validates PVT streaming works
4. Pick whichever gave the best result and proceed to its Phase 1/2

---

## Verification Checklist

1. **Phase 0 (no code changes):**
   - [ ] `test_set_positions_smooth.py` Test 1 (blend=100%) is visibly smoother than Test 2 (blend=0%)
   - [ ] Position tracking test tracks `/target_joint_positions` without violent jerks
   - [ ] No TMScript errors on `/sct_response`
   - [ ] Robot stops cleanly on Ctrl+C (`StopAndClearBuffer` sent)
   - [ ] Queue doesn't grow unbounded (check robot lag stays < 200ms)

2. **Phase 1 (Python node):**
   - [ ] `tm_ptp_servo_node.py` launches and responds to targets
   - [ ] Multiple rapid target changes tracked without full stops
   - [ ] Queue flush triggers when lag exceeds threshold
   - [ ] Node handles Ctrl+C with clean buffer flush

3. **Integration:**
   - [ ] Vision-action model publishes targets
   - [ ] Robot follows model output with acceptable smoothness
   - [ ] If smoothness is insufficient → fall back to Plan A or Plan B
