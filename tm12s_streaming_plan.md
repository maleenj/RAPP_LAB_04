# Plan A: TM12S Smooth Joint Streaming via Velocity Mode

## Problem

We need to stream continuous joint angles from a predictive transformer vision-action model to a TM12S robot arm. The model publishes `sensor_msgs/JointState` messages with target joint positions at ~10-30Hz. Using MoveIt's plan-then-execute approach causes violent jerky motion because each new target triggers a full trajectory plan, deceleration-stop, replan cycle. We need the equivalent of MoveIt Servo — smooth, continuous real-time joint control without any MoveIt planning in the loop.

This is Plan A (preferred). If velocity mode doesn't work on your firmware, see Plan B (PVT mode) or Plan C (set_positions with blending).

## Background: TM12S Robot Driver Architecture

### How the TM Driver Communicates with the Robot

The TM12S ROS2 driver (`tm_driver` package) does NOT use the standard ros2_control framework. Instead, it communicates with the robot via two direct TCP socket connections:

1. **TMSCT (port 5890)** — "TM Script Communication". This is the command channel. The driver sends TMScript expressions (the robot's native scripting language) through this socket. These commands tell the robot to move, set IO, enter special modes, etc. The robot's "Listen Node" (a task block in TMflow, the robot's visual programming environment) must be active and listening on this port. The driver connects as a client.

2. **TMSVR (port 5891)** — "TM Server Communication" / "Ethernet Slave". This is the state feedback channel. The robot streams binary telemetry data (joint positions, velocities, torques, TCP pose, IO states, error codes) at ~25-50Hz. The driver parses these packets and publishes them as ROS2 topics.

### Packet Format

All TMSCT packets follow this structure:
```
$TMSCT,<LENGTH>,<DATA>,*<CHECKSUM>\r\n
```
Where `<DATA>` contains a transaction ID and a TMScript expression. The checksum is XOR of all bytes between `$` and `*`.

### Key ROS2 Interfaces Already Provided by the Driver

- **`/joint_states`** (sensor_msgs/JointState) — Current joint positions, velocities, efforts. Published from TMSVR telemetry.
- **`/feedback_states`** (tm_msgs/FeedbackState) — Comprehensive robot state including IO, errors, TCP pose/force.
- **`/tool_pose`** (geometry_msgs/PoseStamped) — Current TCP pose.
- **`/send_script`** (tm_msgs/srv/SendScript) — Service that sends arbitrary TMScript to the robot via TMSCT. Accepts `{id: string, script: string}`, returns `{ok: bool}`. Internally calls `sct.send_script_str(id, script)` which is **blocking** (waits for robot acknowledgment).
- **`/set_positions`** (tm_msgs/srv/SetPositions) — Service for PTP/Line motions. Accepts motion type, positions (rad), velocity (rad/s), acceleration time (ms), blend percentage, fine_goal flag.
- **`/set_event`** (tm_msgs/srv/SetEvent) — Stop, Pause, Resume, ScriptExit.
- **`/sct_response`** (tm_msgs/SctResponse) — Echoes TMScript execution responses from the robot.

### Why MoveIt Servo Won't Work Here

MoveIt Servo requires a ros2_control hardware interface with velocity command support. The TM12S driver does NOT use ros2_control — it communicates directly with the robot via Ethernet sockets. The ros2_control xacro at `tm12s_moveit_config/config/tm12s.ros2_control.xacro` uses `mock_components/GenericSystem` (simulation only). Building a full ros2_control hardware interface wrapper would be a massive undertaking and is not necessary — the driver already has velocity mode support at the C++ level, it just isn't exposed via ROS2.

### Prerequisites for ALL Plans

1. **TMflow project**: The TM12S must have a TMflow project running with a **Listen Node** task block active. The Listen Node is what opens the TMSCT socket on port 5890 and allows external script execution.
2. **Auto Mode**: The robot must be in Auto Mode (not Manual), with the project playing.
3. **TM Driver launched**: `ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>` — this connects both TMSCT and TMSVR sockets.
4. **Verify connection**: `ros2 topic echo /joint_states --once` should print current joint positions.

---

---

## Solution Overview (Plain English)

### What We're Building

We are building a **real-time velocity streaming pipeline** that converts the vision-action model's joint position outputs into smooth joint velocity commands, which are sent directly to the TM12S robot's motor controller via its native "ContinueVJog" mode. This completely bypasses MoveIt and its plan-execute-stop cycle.

### Why Velocity Mode is the Best Approach

The TM12S robot's firmware supports a special mode called **ContinueVJog** (continuous velocity jog). When activated, the robot's motor controller expects a stream of velocity commands — 6 joint velocities in degrees/sec — and drives the joints at those velocities continuously. Each new velocity command **immediately overrides** the previous one. If no command arrives within a configurable timeout, the robot safely decelerates to a stop.

This is fundamentally different from position-based commands (PTP, PVT):
- **Position commands are queued**: The robot receives a target, plans a trapezoidal velocity profile to reach it, executes, then moves to the next queued target. Multiple consecutive position commands stack up in a buffer, creating lag.
- **Velocity commands are instantaneous**: Each command directly sets the motor velocities. There is no queue, no buffer, no planning. The robot simply moves at the commanded speed. This makes velocity mode the most reactive option possible.

The trade-off is that we must do position tracking ourselves — we convert desired positions to velocities using a proportional (P) controller: `velocity = Kp * (target_position - current_position)`. This is a well-understood control law that produces smooth exponential convergence to targets.

### Why This Isn't Exposed Yet

The TM driver's C++ code already has the full velocity mode API implemented at the `TmDriver` class level (`set_vel_mode_start`, `set_vel_mode_target`, `set_vel_mode_stop`). However, the ROS2 layer (services and topics) only exposes position-based commands (`/set_positions`, `/send_script`, `FollowJointTrajectory` action). Nobody added a ROS2 topic subscriber for velocity commands. Our job is to add that thin ROS2 wrapper and write a Python node that converts positions to velocities.

### The Critical Implementation Detail: Blocking vs Non-Blocking Sends

The TM driver has two ways to send TMScript commands through the TMSCT socket:

1. **`send_script_str(id, script)`** — Sends the script and **blocks** until the robot acknowledges receipt. Typical round-trip is 5-20ms. Fine for one-time commands (enter/exit velocity mode, PTP motions) but would limit streaming to ~50-100Hz at best.

2. **`send_script_str_silent(id, script)`** — Sends the script and **returns immediately** without waiting for acknowledgment. This is fire-and-forget. It can handle much higher command rates because there's no round-trip delay.

The velocity mode target function (`set_vel_mode_target`) uses `send_script_str_silent` — this is what makes 30-50Hz velocity streaming feasible. The enter/exit functions use the blocking version because they're one-time calls where we want confirmation.

### End-to-End Data Flow

```
Vision-Action Model (Python, 10-30 Hz)
    |
    | Publishes sensor_msgs/JointState to /target_joint_positions
    | (6 joint angles in radians — the model's predicted next pose)
    v
tm_joint_servo_node.py (Python ROS2 node, 50 Hz control loop)
    |
    | Reads current joint positions from /joint_states
    | Computes error: target - current (radians)
    | Computes velocity: Kp * error (rad/s)
    | Clamps to max_joint_vel (safety limit)
    | Applies deadband (ignore tiny errors to prevent jitter)
    | Publishes std_msgs/Float64MultiArray to /servo_cmd_vel_joint
    v
tm_driver C++ node (vel_cmd_callback subscriber)
    |
    | Calls iface_.set_vel_mode_target(VelMode::Joint, vel)
    | Which calls sct.send_script_str_silent(id, script)
    | Which sends "SetContinueVJog(v1_deg,v2_deg,...,v6_deg)" over TCP to port 5890
    | (velocities converted from rad/s to deg/s internally by TmCommand::deg())
    v
TM12S Robot Controller (Listen Node on TMflow)
    |
    | Receives TMScript via TMSCT socket
    | Sets motor velocities directly — no trajectory planning
    | Robot joints move at commanded velocities
    | Built-in safety: auto-stops if no command within timeout
    v
Smooth continuous motion!
```

### Safety Mechanisms

1. **Velocity clamping**: The Python servo node clamps all velocity commands to `max_joint_vel` (default 0.5 rad/s = ~28.6 deg/s). The robot's absolute max is pi rad/s = 180 deg/s.
2. **Target timeout**: If the vision model stops publishing, the servo node sends zero velocities after `target_timeout` seconds (default 0.5s).
3. **Robot-side timeout**: ContinueVJog mode has its own timeout — if no SetContinueVJog command arrives within `timeout_stop` seconds (default 1.0s), the robot decelerates to a full stop automatically.
4. **Deadband**: Errors smaller than `deadband` (default 0.001 rad = ~0.06 deg) produce zero velocity, preventing jitter when holding a position.
5. **Destructor cleanup**: The C++ driver automatically calls `set_vel_mode_stop()` if the node shuts down while velocity mode is active.

---

## Key Discovery: Velocity Mode Already Exists in the Driver

The TM driver C++ API already has a **velocity mode** that is perfect for this use case but is **NOT exposed via any ROS2 topic or service**:

### Velocity Mode API (in `tm_driver/src/tm_driver.cpp:303-316`)

```cpp
// Enter velocity mode — sends "ContinueVJog()" via TMSCT (blocking call, one-time)
bool TmDriver::set_vel_mode_start(VelMode mode, double timeout_zero_vel, double timeout_stop, const std::string &id)
{
    return (sct.send_script_str(id, TmCommand::set_vel_mode_start(mode, timeout_zero_vel, timeout_stop)) == RC_OK);
}

// Exit velocity mode — sends "StopContinueVmode()" (blocking call, one-time)
bool TmDriver::set_vel_mode_stop(const std::string &id)
{
    return (sct.send_script_str(id, TmCommand::set_vel_mode_stop()) == RC_OK);
}

// Send velocity target — sends "SetContinueVJog(v1,...,v6)" via send_script_str_silent (NON-BLOCKING!)
bool TmDriver::set_vel_mode_target(VelMode mode, const std::vector<double> &vel, const std::string &id)
{
    return (sct.send_script_str_silent(id, TmCommand::set_vel_mode_target(mode, vel)) == RC_OK);
}
```

Critical detail: `set_vel_mode_target` uses `send_script_str_silent()` which is **fire-and-forget** — it sends the packet without waiting for a response. This is the key to high-frequency streaming.

### TMScript Commands Generated (in `tm_driver/src/tm_command.cpp:130-172`)

- `ContinueVJog()` — enters joint velocity mode
- `SetContinueVJog(v1_deg, v2_deg, v3_deg, v4_deg, v5_deg, v6_deg)` — sets joint velocities in **degrees/sec** (the API converts rad/s to deg/s internally via `deg()`)
- `StopContinueVmode()` — exits velocity mode

### Existing ROS2 Interface That Can Be Used for Quick Testing

The `/send_script` service (`tm_msgs/srv/SendScript`) can send arbitrary TMScript to the robot's Listen Node. Its definition at `tm_msgs/srv/SendScript.srv`:

```
string id
string script
---
bool ok
```

It calls `sct_.send_script_str(id, script)` internally — this is the blocking version, but for testing and for enter/exit velocity mode (one-time calls), this is fine.

---

## Phase 0: Quick Validation Test (NO CODE CHANGES — use existing driver as-is)

**Purpose**: Validate that velocity mode streaming works for smooth motion on your TM12S before writing any new code. This uses only the existing `/send_script` ROS2 service.

### Prerequisites

1. TMflow project with Listen Node is active on the TM12S
2. Robot is in Auto Mode, project is running
3. TM driver is launched and connected:
   ```bash
   ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>
   ```
4. Verify connection:
   ```bash
   ros2 topic echo /joint_states --once
   # Should print current joint positions
   ```

### Test Script: `test_vel_mode.py`

Create this standalone Python script (NOT a ROS2 package — just a script). It uses the existing `/send_script` service to test velocity mode with a gentle oscillation on joint 1.

```python
#!/usr/bin/env python3
"""
Quick validation test for TM12S velocity mode streaming.
Uses ONLY the existing /send_script service — no driver modifications needed.

WARNING: This will move joint 1 of the robot. Ensure the workspace is clear.
Run with the robot at a safe position with clearance in all directions.

Usage:
    1. Launch tm_driver: ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<IP>
    2. Run this script: python3 test_vel_mode.py
    3. Press Ctrl+C to stop
"""

import rclpy
from rclpy.node import Node
from tm_msgs.srv import SendScript
from sensor_msgs.msg import JointState
import math
import time
import signal
import sys


class VelModeTest(Node):
    def __init__(self):
        super().__init__('vel_mode_test')

        # Service client for sending scripts
        self.send_script_cli = self.create_client(SendScript, 'send_script')
        while not self.send_script_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Waiting for /send_script service...')

        # Subscribe to joint states for monitoring
        self.current_joints = None
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_cb, 10)

        self.vel_mode_active = False
        self.get_logger().info('VelModeTest node ready.')

    def joint_cb(self, msg):
        self.current_joints = list(msg.position)

    def send_script_sync(self, script_id, script):
        """Send a TMScript command via the /send_script service (blocking)."""
        req = SendScript.Request()
        req.id = script_id
        req.script = script
        future = self.send_script_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            self.get_logger().info(f'Script [{script_id}]: {script} -> ok={future.result().ok}')
            return future.result().ok
        else:
            self.get_logger().error(f'Script [{script_id}] call failed')
            return False

    def enter_vel_mode(self):
        """Enter joint velocity mode via ContinueVJog()."""
        ok = self.send_script_sync('VStart', 'ContinueVJog()')
        if ok:
            self.vel_mode_active = True
            self.get_logger().info('Entered velocity mode')
        return ok

    def exit_vel_mode(self):
        """Exit velocity mode via StopContinueVmode()."""
        ok = self.send_script_sync('VStop', 'StopContinueVmode()')
        self.vel_mode_active = False
        self.get_logger().info('Exited velocity mode')
        return ok

    def send_vel_target(self, velocities_deg_per_sec):
        """
        Send velocity target. Velocities are in DEGREES/SEC (TMScript native unit).
        Uses the /send_script service — slightly higher latency than direct API,
        but works without driver modifications.
        """
        vel_str = ','.join(f'{v:.4f}' for v in velocities_deg_per_sec)
        script = f'SetContinueVJog({vel_str})'
        # For velocity commands we still use send_script service.
        # In production, the driver modification will use the non-blocking path.
        req = SendScript.Request()
        req.id = 'VTrgt'
        req.script = script
        # Fire and don't wait — we want to be fast
        self.send_script_cli.call_async(req)

    def run_oscillation_test(self, duration_sec=10.0, max_vel_deg=5.0, freq_hz=0.5, rate_hz=30.0):
        """
        Oscillate joint 1 with a sine wave velocity profile.
        All other joints held at zero velocity.

        Args:
            duration_sec: Total test duration
            max_vel_deg: Peak velocity in degrees/sec (KEEP THIS LOW for safety: 5 deg/s)
            freq_hz: Oscillation frequency
            rate_hz: Velocity command update rate
        """
        self.get_logger().info(
            f'Starting oscillation test: duration={duration_sec}s, '
            f'max_vel={max_vel_deg} deg/s, freq={freq_hz} Hz, rate={rate_hz} Hz'
        )

        if self.current_joints:
            self.get_logger().info(
                f'Current joints (rad): {[f"{j:.4f}" for j in self.current_joints]}'
            )

        if not self.enter_vel_mode():
            self.get_logger().error('Failed to enter velocity mode. Aborting.')
            return

        start_time = time.time()
        dt = 1.0 / rate_hz

        try:
            while (time.time() - start_time) < duration_sec:
                t = time.time() - start_time

                # Sine wave velocity on joint 1 only
                v1 = max_vel_deg * math.sin(2.0 * math.pi * freq_hz * t)
                velocities = [v1, 0.0, 0.0, 0.0, 0.0, 0.0]

                self.send_vel_target(velocities)

                # Log periodically
                if int(t * rate_hz) % int(rate_hz) == 0:
                    pos_str = ''
                    if self.current_joints:
                        pos_str = f' | pos(deg): {[f"{math.degrees(j):.2f}" for j in self.current_joints]}'
                    self.get_logger().info(f't={t:.1f}s | vel_cmd=[{v1:.2f}, 0, 0, 0, 0, 0] deg/s{pos_str}')

                # Allow ROS callbacks to process
                rclpy.spin_once(self, timeout_sec=0)
                time.sleep(dt)

        finally:
            # ALWAYS stop velocity mode
            self.get_logger().info('Stopping velocity mode...')
            # Send zero velocities first
            self.send_vel_target([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            time.sleep(0.1)
            self.exit_vel_mode()


def main():
    rclpy.init()
    node = VelModeTest()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        node.get_logger().warn('Ctrl+C detected — stopping velocity mode...')
        if node.vel_mode_active:
            node.send_vel_target([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            time.sleep(0.1)
            node.exit_vel_mode()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        node.run_oscillation_test(
            duration_sec=10.0,   # 10 second test
            max_vel_deg=5.0,     # VERY slow — 5 deg/sec peak
            freq_hz=0.25,        # Slow oscillation — one full cycle every 4 seconds
            rate_hz=30.0         # 30 Hz update rate
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### What to Observe

**Success indicators:**
- Joint 1 oscillates smoothly back and forth in a sine wave pattern
- Motion is fluid with no jerks, pauses, or stuttering
- Robot responds immediately to velocity commands
- Clean stop when the test ends or Ctrl+C is pressed

**Failure indicators:**
- Robot doesn't move → Listen Node not active, or connection issue
- Jerky motion → update rate too low, or service overhead too high (motivates the driver modification in Phase 1)
- Robot error/E-stop → velocity too high, or robot hit a limit

**What to check in the logs:**
- `/sct_response` topic will echo TMScript responses — watch for errors
- Driver terminal will print `TM_ROS: (TM_SCT): MSG:` for each script sent

### Test Script 2: Position Tracking Test

Once the oscillation test works, test position tracking with a proportional controller (this validates the full approach):

```python
#!/usr/bin/env python3
"""
Test position tracking via velocity mode.
Subscribes to /target_joint_positions and tracks them using a P controller.

Usage:
    1. Launch tm_driver: ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<IP>
    2. Run this script: python3 test_position_tracking.py
    3. In another terminal, publish a target:
       ros2 topic pub /target_joint_positions sensor_msgs/JointState \
         "{position: [0.1, -0.2, 0.3, 0.0, 0.5, 0.0]}" --once
    4. Watch the robot smoothly move to the target
    5. Press Ctrl+C to stop
"""

import rclpy
from rclpy.node import Node
from tm_msgs.srv import SendScript
from sensor_msgs.msg import JointState
import math
import time
import signal
import sys
import numpy as np


class PositionTrackingTest(Node):
    def __init__(self):
        super().__init__('position_tracking_test')

        # --- Parameters ---
        self.Kp = 2.0                    # Proportional gain (rad/s per rad of error)
        self.max_joint_vel_rad = 0.5     # Max velocity per joint (rad/s) — ~28.6 deg/s
        self.control_rate = 30.0         # Hz
        self.deadband_rad = 0.001        # Ignore errors below this (rad)
        self.target_timeout = 1.0        # Stop if no target received for this many seconds

        # --- State ---
        self.current_joints = None
        self.target_joints = None
        self.last_target_time = None
        self.vel_mode_active = False

        # --- ROS interfaces ---
        self.send_script_cli = self.create_client(SendScript, 'send_script')
        while not self.send_script_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Waiting for /send_script service...')

        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_cb, 10)

        self.target_sub = self.create_subscription(
            JointState, 'target_joint_positions', self.target_cb, 10)

        self.get_logger().info('PositionTrackingTest ready.')
        self.get_logger().info(f'  Kp={self.Kp}, max_vel={self.max_joint_vel_rad} rad/s, '
                               f'rate={self.control_rate} Hz')
        self.get_logger().info('Publish targets to /target_joint_positions (sensor_msgs/JointState)')

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position)

    def target_cb(self, msg):
        if len(msg.position) >= 6:
            self.target_joints = np.array(msg.position[:6])
            self.last_target_time = time.time()
            self.get_logger().info(
                f'New target (deg): {[f"{math.degrees(j):.2f}" for j in self.target_joints]}'
            )

    def send_script_sync(self, script_id, script):
        req = SendScript.Request()
        req.id = script_id
        req.script = script
        future = self.send_script_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            return future.result().ok
        return False

    def enter_vel_mode(self):
        ok = self.send_script_sync('VStart', 'ContinueVJog()')
        if ok:
            self.vel_mode_active = True
            self.get_logger().info('Entered velocity mode')
        return ok

    def exit_vel_mode(self):
        ok = self.send_script_sync('VStop', 'StopContinueVmode()')
        self.vel_mode_active = False
        self.get_logger().info('Exited velocity mode')
        return ok

    def send_vel_target(self, velocities_deg_per_sec):
        vel_str = ','.join(f'{v:.4f}' for v in velocities_deg_per_sec)
        script = f'SetContinueVJog({vel_str})'
        req = SendScript.Request()
        req.id = 'VTrgt'
        req.script = script
        self.send_script_cli.call_async(req)

    def run_control_loop(self):
        self.get_logger().info('Waiting for first /joint_states message...')
        while self.current_joints is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        if not self.enter_vel_mode():
            self.get_logger().error('Failed to enter velocity mode')
            return

        dt = 1.0 / self.control_rate
        max_vel_deg = math.degrees(self.max_joint_vel_rad)
        log_counter = 0

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0)

                if self.target_joints is None or self.current_joints is None:
                    # No target yet — send zero velocity
                    self.send_vel_target([0.0] * 6)
                    time.sleep(dt)
                    continue

                # Check target timeout
                if self.last_target_time and (time.time() - self.last_target_time) > self.target_timeout:
                    self.send_vel_target([0.0] * 6)
                    time.sleep(dt)
                    continue

                # P controller: velocity = Kp * (target - current)
                error = self.target_joints - self.current_joints
                vel_cmd_rad = self.Kp * error

                # Apply deadband
                vel_cmd_rad[np.abs(error) < self.deadband_rad] = 0.0

                # Clamp to velocity limits
                vel_cmd_rad = np.clip(vel_cmd_rad, -self.max_joint_vel_rad, self.max_joint_vel_rad)

                # Convert to degrees/sec for TMScript
                vel_cmd_deg = [math.degrees(v) for v in vel_cmd_rad]

                self.send_vel_target(vel_cmd_deg)

                # Log every second
                log_counter += 1
                if log_counter % int(self.control_rate) == 0:
                    err_deg = [f'{math.degrees(e):.2f}' for e in error]
                    vel_deg = [f'{v:.2f}' for v in vel_cmd_deg]
                    self.get_logger().info(f'err(deg): {err_deg} | vel_cmd(deg/s): {vel_deg}')

                time.sleep(dt)

        finally:
            self.get_logger().info('Stopping...')
            self.send_vel_target([0.0] * 6)
            time.sleep(0.1)
            self.exit_vel_mode()


def main():
    rclpy.init()
    node = PositionTrackingTest()

    def signal_handler(sig, frame):
        node.get_logger().warn('Ctrl+C — stopping...')
        if node.vel_mode_active:
            node.send_vel_target([0.0] * 6)
            time.sleep(0.1)
            node.exit_vel_mode()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        node.run_control_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Phase 0 Test Procedure

1. **Start the driver:**
   ```bash
   ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>
   ```

2. **Verify connectivity:**
   ```bash
   ros2 topic echo /joint_states --once
   ros2 service list | grep send_script
   ```

3. **Run oscillation test (test_vel_mode.py):**
   ```bash
   python3 test_vel_mode.py
   ```
   - Observe joint 1 oscillating smoothly for 10 seconds
   - Confirm motion is fluid, not jerky

4. **Run position tracking test (test_position_tracking.py):**
   ```bash
   python3 test_position_tracking.py
   ```
   Then in another terminal:
   ```bash
   # Move joint 1 to ~5.7 degrees (0.1 rad), all others near zero
   ros2 topic pub /target_joint_positions sensor_msgs/JointState \
     "{position: [0.1, -0.2, 0.3, 0.0, 0.5, 0.0]}" --once
   ```
   - Observe all joints smoothly converge to the target
   - Publish several different targets in sequence and observe smooth transitions

5. **Evaluate results:**
   - If both tests show smooth motion → proceed to Phase 1 (driver modification for production quality)
   - If latency is noticeable but motion is smooth → Phase 1 will fix this (non-blocking path)
   - If motion is still jerky → investigate `/sct_response` for errors, reduce rate/velocity

---

## Phase 1: Add Velocity Command Subscriber to the TM Driver (C++)

**Purpose**: Expose velocity mode via a ROS2 topic subscriber that calls `set_vel_mode_target()` directly (the non-blocking `send_script_str_silent` path), plus services for entering/exiting velocity mode. This eliminates the overhead of routing every velocity command through the `/send_script` service.

### File: `tm_driver/include/tm_driver/tm_ros2_sct.h`

Add these includes at the top (after existing includes):

```cpp
#include "std_msgs/msg/float64_multi_array.hpp"
```

Add these members to the `TmSctRos2` class, in the `public:` section alongside the existing service members (after line 50 `ask_sta_srv_`):

```cpp
    // Velocity mode streaming
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr vel_cmd_sub_;
    rclcpp::Service<tm_msgs::srv::SendScript>::SharedPtr vel_mode_start_srv_;
    rclcpp::Service<tm_msgs::srv::SendScript>::SharedPtr vel_mode_stop_srv_;
    bool vel_mode_active_ = false;
```

Add these method declarations in the `public:` section (after `ask_sta` declaration, before the closing `};`):

```cpp
    void vel_cmd_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    bool vel_mode_start(
        const std::shared_ptr<tm_msgs::srv::SendScript::Request> req,
        std::shared_ptr<tm_msgs::srv::SendScript::Response> res);
    bool vel_mode_stop(
        const std::shared_ptr<tm_msgs::srv::SendScript::Request> req,
        std::shared_ptr<tm_msgs::srv::SendScript::Response> res);
```

### File: `tm_driver/src/tm_ros2_sct.cpp`

Add to the constructor (after the `ask_sta_srv_` setup on line 48, before the closing `}`):

```cpp
    // Velocity mode streaming
    vel_cmd_sub_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
        "servo_cmd_vel_joint", 1,
        std::bind(&TmSctRos2::vel_cmd_callback, this, std::placeholders::_1));

    vel_mode_start_srv_ = node->create_service<tm_msgs::srv::SendScript>(
        "vel_mode_start", std::bind(&TmSctRos2::vel_mode_start, this,
        std::placeholders::_1, std::placeholders::_2));

    vel_mode_stop_srv_ = node->create_service<tm_msgs::srv::SendScript>(
        "vel_mode_stop", std::bind(&TmSctRos2::vel_mode_stop, this,
        std::placeholders::_1, std::placeholders::_2));
```

Add these method implementations at the end of the file (after `ask_sta`):

```cpp
void TmSctRos2::vel_cmd_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (!vel_mode_active_) {
        return;
    }
    if (msg->data.size() < 6) {
        print_error("TM_ROS: vel_cmd requires 6 joint velocities, got %zu", msg->data.size());
        return;
    }
    // Extract first 6 values as joint velocities (rad/s)
    std::vector<double> vel(msg->data.begin(), msg->data.begin() + 6);
    // This calls send_script_str_silent internally — non-blocking
    iface_.set_vel_mode_target(VelMode::Joint, vel);
}

bool TmSctRos2::vel_mode_start(
    const std::shared_ptr<tm_msgs::srv::SendScript::Request> req,
    std::shared_ptr<tm_msgs::srv::SendScript::Response> res)
{
    // Default timeouts: 200ms zero-vel timeout, 1000ms stop timeout
    // The user can override by passing custom TMScript via the script field
    // but for the standard case we use sensible defaults.
    double timeout_zero_vel = 0.2;   // seconds — robot stops if zero vel for this long
    double timeout_stop = 1.0;       // seconds — robot stops if no command for this long
    (void)req;  // unused — we use fixed defaults

    bool ok = iface_.set_vel_mode_start(VelMode::Joint, timeout_zero_vel, timeout_stop);
    if (ok) {
        vel_mode_active_ = true;
        print_info("TM_ROS: Velocity mode STARTED");
    } else {
        print_error("TM_ROS: Failed to start velocity mode");
    }
    res->ok = ok;
    return ok;
}

bool TmSctRos2::vel_mode_stop(
    const std::shared_ptr<tm_msgs::srv::SendScript::Request> req,
    std::shared_ptr<tm_msgs::srv::SendScript::Response> res)
{
    (void)req;
    bool ok = iface_.set_vel_mode_stop();
    vel_mode_active_ = false;
    print_info("TM_ROS: Velocity mode STOPPED");
    res->ok = ok;
    return ok;
}
```

Also add cleanup in the destructor `TmSctRos2::~TmSctRos2()` (at line 52, before `print_info`):

```cpp
    // Stop velocity mode if active when shutting down
    if (vel_mode_active_) {
        iface_.set_vel_mode_stop();
        vel_mode_active_ = false;
    }
```

### Build Verification

No changes to `CMakeLists.txt` needed — `std_msgs` is already a dependency. No changes to `package.xml` needed — `std_msgs` is already listed.

```bash
cd ~/git/tm2_ros2
colcon build --packages-select tm_driver
source install/setup.bash
```

### Manual Test After Phase 1

```bash
# Terminal 1: Launch driver
ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>

# Terminal 2: Enter velocity mode
ros2 service call /vel_mode_start tm_msgs/srv/SendScript "{id: 'test', script: ''}"

# Terminal 3: Send a small velocity command (joint 1 at ~5 deg/s = 0.087 rad/s)
ros2 topic pub /servo_cmd_vel_joint std_msgs/msg/Float64MultiArray \
  "{data: [0.087, 0.0, 0.0, 0.0, 0.0, 0.0]}" --rate 30

# Observe joint 1 moving smoothly
# Ctrl+C the publisher, then:

# Terminal 2: Exit velocity mode
ros2 service call /vel_mode_stop tm_msgs/srv/SendScript "{id: 'test', script: ''}"
```

---

## Phase 2: Python Servo Node

**Purpose**: A standalone ROS2 Python node that subscribes to target joint positions from the vision-action model, computes velocity commands via a proportional controller, and publishes them to the driver's velocity command topic.

### File: `tm_driver/tm_driver/tm_joint_servo_node.py`

Note: `tm_driver` is a C++ ament_cmake package, so this Python script is placed as a standalone executable, not as a Python module. It will be installed via the `install()` directive in CMakeLists.txt.

```python
#!/usr/bin/env python3
"""
TM12S Joint Servo Node

Converts continuous target joint positions into smooth velocity commands
for the TM12S robot arm using the driver's velocity mode.

Subscribes:
    /target_joint_positions (sensor_msgs/JointState) — target from vision-action model
    /joint_states (sensor_msgs/JointState) — current robot state

Publishes:
    /servo_cmd_vel_joint (std_msgs/Float64MultiArray) — velocity commands to driver

Services called:
    /vel_mode_start — enter velocity mode on startup
    /vel_mode_stop — exit velocity mode on shutdown

Parameters:
    kp (double, default=2.0): Proportional gain
    max_joint_vel (double, default=0.5): Max joint velocity in rad/s
    control_rate (double, default=50.0): Control loop rate in Hz
    target_timeout (double, default=0.5): Seconds before ramping to zero if no new target
    deadband (double, default=0.001): Ignore position errors below this (rad)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from tm_msgs.srv import SendScript
import numpy as np
import math
import time
import signal
import sys


class TmJointServoNode(Node):
    def __init__(self):
        super().__init__('tm_joint_servo')

        # Declare parameters
        self.declare_parameter('kp', 2.0)
        self.declare_parameter('max_joint_vel', 0.5)       # rad/s
        self.declare_parameter('control_rate', 50.0)        # Hz
        self.declare_parameter('target_timeout', 0.5)       # seconds
        self.declare_parameter('deadband', 0.001)           # rad

        self.kp = self.get_parameter('kp').value
        self.max_vel = self.get_parameter('max_joint_vel').value
        self.rate_hz = self.get_parameter('control_rate').value
        self.target_timeout = self.get_parameter('target_timeout').value
        self.deadband = self.get_parameter('deadband').value

        # State
        self.current_joints = None
        self.target_joints = None
        self.last_target_time = None
        self.vel_mode_active = False

        # Publisher for velocity commands
        self.vel_pub = self.create_publisher(Float64MultiArray, 'servo_cmd_vel_joint', 1)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self._joint_cb, 10)
        self.target_sub = self.create_subscription(
            JointState, 'target_joint_positions', self._target_cb, 10)

        # Service clients
        self.vel_start_cli = self.create_client(SendScript, 'vel_mode_start')
        self.vel_stop_cli = self.create_client(SendScript, 'vel_mode_stop')

        # Control loop timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self._control_loop)

        self.get_logger().info(
            f'TM Joint Servo Node started: kp={self.kp}, max_vel={self.max_vel} rad/s, '
            f'rate={self.rate_hz} Hz, timeout={self.target_timeout}s, deadband={self.deadband} rad'
        )

    def _joint_cb(self, msg):
        self.current_joints = np.array(msg.position[:6])

    def _target_cb(self, msg):
        if len(msg.position) >= 6:
            self.target_joints = np.array(msg.position[:6])
            self.last_target_time = time.time()

    def start_vel_mode(self):
        """Call /vel_mode_start service to enter velocity mode."""
        if not self.vel_start_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/vel_mode_start service not available')
            return False
        req = SendScript.Request()
        req.id = 'servo_start'
        req.script = ''
        future = self.vel_start_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() and future.result().ok:
            self.vel_mode_active = True
            self.get_logger().info('Velocity mode started')
            return True
        self.get_logger().error('Failed to start velocity mode')
        return False

    def stop_vel_mode(self):
        """Call /vel_mode_stop service to exit velocity mode."""
        # Send zero velocity first
        self._publish_vel([0.0] * 6)

        if not self.vel_stop_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/vel_mode_stop service not available')
            return
        req = SendScript.Request()
        req.id = 'servo_stop'
        req.script = ''
        future = self.vel_stop_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        self.vel_mode_active = False
        self.get_logger().info('Velocity mode stopped')

    def _publish_vel(self, vel_list):
        msg = Float64MultiArray()
        msg.data = vel_list
        self.vel_pub.publish(msg)

    def _control_loop(self):
        """Main control loop — runs at control_rate Hz."""
        if not self.vel_mode_active:
            return

        # Need both current and target to compute velocity
        if self.current_joints is None:
            return

        if self.target_joints is None:
            self._publish_vel([0.0] * 6)
            return

        # Check target timeout
        if self.last_target_time and (time.time() - self.last_target_time) > self.target_timeout:
            self._publish_vel([0.0] * 6)
            return

        # Proportional controller
        error = self.target_joints - self.current_joints
        vel_cmd = self.kp * error

        # Apply deadband
        vel_cmd[np.abs(error) < self.deadband] = 0.0

        # Clamp to velocity limits
        vel_cmd = np.clip(vel_cmd, -self.max_vel, self.max_vel)

        self._publish_vel(vel_cmd.tolist())


def main(args=None):
    rclpy.init(args=args)
    node = TmJointServoNode()

    # Wait for joint states before entering velocity mode
    node.get_logger().info('Waiting for /joint_states...')
    while node.current_joints is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)

    if not node.start_vel_mode():
        node.get_logger().error('Cannot start — exiting')
        node.destroy_node()
        rclpy.shutdown()
        return

    def shutdown_handler(sig, frame):
        node.get_logger().warn('Shutting down servo node...')
        node.stop_vel_mode()
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
        if node.vel_mode_active:
            node.stop_vel_mode()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Install the Python Script

Add to `tm_driver/CMakeLists.txt` before `ament_package()` (line 158):

```cmake
install(PROGRAMS
  tm_driver/tm_joint_servo_node.py
  DESTINATION lib/${PROJECT_NAME}
)
```

Create the directory and place the script:

```bash
mkdir -p ~/git/tm2_ros2/tm_driver/tm_driver
# Place tm_joint_servo_node.py in this directory
chmod +x ~/git/tm2_ros2/tm_driver/tm_driver/tm_joint_servo_node.py
```

After building, run with:

```bash
ros2 run tm_driver tm_joint_servo_node.py
# Or with parameters:
ros2 run tm_driver tm_joint_servo_node.py --ros-args -p kp:=3.0 -p max_joint_vel:=0.8
```

---

## Phase 3: Integration with Vision-Action Model

Once Phase 1 and Phase 2 are implemented and tested:

### Launch Sequence

```bash
# Terminal 1: TM Driver
ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>

# Terminal 2: Servo Node
ros2 run tm_driver tm_joint_servo_node.py --ros-args \
  -p kp:=2.0 \
  -p max_joint_vel:=0.5 \
  -p control_rate:=50.0

# Terminal 3: Vision-Action Model
# Your model publishes sensor_msgs/JointState to /target_joint_positions
python3 your_model_inference.py
```

### Data Flow

```
Vision-Action Model (10-30Hz)
    | publishes sensor_msgs/JointState
    v
/target_joint_positions
    |
    v
tm_joint_servo_node.py (50Hz control loop)
    | reads /joint_states for current position
    | computes: vel = Kp * (target - current)
    | clamps to max_joint_vel
    | publishes std_msgs/Float64MultiArray
    v
/servo_cmd_vel_joint
    |
    v
tm_driver C++ subscriber callback (vel_cmd_callback)
    | calls iface_.set_vel_mode_target(VelMode::Joint, vel)
    | which calls sct.send_script_str_silent() — NON-BLOCKING
    | sends "SetContinueVJog(v1,...,v6)" via TMSCT socket
    v
TM12S Robot — smooth continuous motion
```

### Tuning Guide

| Parameter | Effect | Start With | Notes |
|-----------|--------|------------|-------|
| `kp` | Responsiveness. Higher = faster tracking, more overshoot | 2.0 | If oscillating, reduce to 1.0. If sluggish, increase to 3.0-5.0 |
| `max_joint_vel` | Safety limit. Clamps velocity commands | 0.5 rad/s | Increase to 1.0 for faster motions. Never exceed pi rad/s |
| `control_rate` | Command frequency. Higher = smoother but more CPU/network | 50 Hz | 30-50 Hz is sweet spot for TM robot ethernet communication |
| `target_timeout` | How quickly robot stops when model stops publishing | 0.5 s | Reduce for safety, increase if model has variable latency |
| `deadband` | Prevents jitter at target. Ignores small errors | 0.001 rad | Increase if you see jitter when the robot should be still |

---

## Architecture Summary

### Files Modified (Phase 1)

| File | Change |
|------|--------|
| `tm_driver/include/tm_driver/tm_ros2_sct.h` | Add `#include "std_msgs/msg/float64_multi_array.hpp"`, add `vel_cmd_sub_`, `vel_mode_start_srv_`, `vel_mode_stop_srv_`, `vel_mode_active_` members, add `vel_cmd_callback`, `vel_mode_start`, `vel_mode_stop` method declarations |
| `tm_driver/src/tm_ros2_sct.cpp` | Add subscriber + service setup in constructor, add `vel_cmd_callback`, `vel_mode_start`, `vel_mode_stop` implementations, add cleanup in destructor |

### Files Created (Phase 2)

| File | Purpose |
|------|---------|
| `tm_driver/tm_driver/tm_joint_servo_node.py` | Python servo node — position-to-velocity converter |

### Files Modified (Phase 2)

| File | Change |
|------|--------|
| `tm_driver/CMakeLists.txt` | Add `install(PROGRAMS ...)` for the Python script |

### New ROS2 Interfaces

| Interface | Type | Purpose |
|-----------|------|---------|
| `/servo_cmd_vel_joint` | Topic (sub) `std_msgs/Float64MultiArray` | 6 joint velocities in rad/s |
| `/vel_mode_start` | Service `tm_msgs/srv/SendScript` | Enter velocity mode |
| `/vel_mode_stop` | Service `tm_msgs/srv/SendScript` | Exit velocity mode |

### Important Notes

- **Unit conversion**: The C++ driver converts rad/s to deg/s internally. The Python servo node and the `/servo_cmd_vel_joint` topic use **rad/s**. The TMScript `SetContinueVJog()` uses **deg/s** — this conversion happens in `TmCommand::set_vel_mode_target()` via the `deg()` function.
- **`ContinueVJog()` vs `ContinueVLine()`**: We use `ContinueVJog()` (joint velocity mode) because the model outputs joint angles. `ContinueVLine()` is for Cartesian velocity control and takes timeouts as parameters — not needed for our use case.
- **Safety**: Velocity mode has built-in safety — the robot automatically stops if no velocity command is received within the timeout (default 1 second). The servo node also sends zero velocities on shutdown and on target timeout.
- **Listen Node**: The TMflow project MUST have a Listen Node active and the robot must be in Auto Mode for any of this to work. The driver connects to the Listen Node on port 5890.
- **`send_script_str_silent` vs `send_script_str`**: The `_silent` variant sends the TMSCT packet without waiting for a response — critical for high-frequency streaming. The regular version waits for an acknowledgment. Velocity targets use `_silent`; enter/exit commands use the regular version (they're one-time calls where we want confirmation).
