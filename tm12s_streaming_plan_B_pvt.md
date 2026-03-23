# Plan B: TM12S Smooth Joint Streaming via PVT Mode

## Problem

Same as Plan A: stream continuous joint angles from a predictive transformer vision-action model to a TM12S robot arm with smooth motion. This plan uses **PVT (Position-Velocity-Time) mode** instead of velocity mode.

If you haven't read Plan A, read the "Background: TM12S Robot Driver Architecture" section there first — it explains how the driver communicates with the robot (TMSCT/TMSVR sockets, Listen Node, etc.) and lists all the prerequisites.

---

## Solution Overview (Plain English)

### What We're Building

We are building a **PVT point streaming pipeline** that takes the vision-action model's joint position outputs and feeds them directly to the TM12S robot's PVT (Position-Velocity-Time) trajectory engine. The robot's firmware performs **cubic spline interpolation** between consecutive PVT points, producing mathematically smooth motion with continuous acceleration.

### What is PVT Mode?

PVT stands for **Position-Velocity-Time**. It is the TM robot's native trajectory execution mechanism. Each PVT point tells the robot:
- **Position**: Where the joints should be (6 angles in degrees)
- **Velocity**: How fast the joints should be moving when they reach that position (6 velocities in degrees/sec)
- **Time**: How long (in seconds) the robot has to travel from the previous point to this one

Given two consecutive PVT points, the robot's controller computes a **cubic polynomial** for each joint that satisfies both the position and velocity constraints at both endpoints. This produces motion that is smooth in position, velocity, AND acceleration — no sudden jerks or discontinuities.

This is fundamentally different from PTP (Point-to-Point) commands where the robot plans its own trapezoidal velocity profile and you have no control over the intermediate motion shape.

### How This Differs from Plan A (Velocity Mode)

In Plan A, we send **velocity commands** and the robot just drives motors at those speeds. We compute the velocities ourselves using a P controller. The robot does no interpolation — it just obeys the latest velocity.

In Plan B, we send **position + velocity + time** points and the robot does sophisticated cubic interpolation between them. We still need to compute velocities (via finite differencing of successive targets), but the robot's interpolation produces smoother motion for a given update rate.

The key trade-off: PVT points are **queued and executed in sequence**, meaning there's inherent latency. If the model changes direction, the robot must finish executing any buffered points before it responds. Velocity mode overrides instantly.

### How PVT is Already Used in the Driver

MoveIt's `FollowJointTrajectory` action server (implemented in `tm_driver/src/tm_ros2_movit_sct.cpp`) already converts MoveIt trajectories into PVT format internally. The function `get_pvt_traj()` takes MoveIt's trajectory points and converts them to `TmPvtTraj` objects, which are then sent to the robot via `run_pvt_traj()`. The jerkiness you see with MoveIt is NOT from PVT — it's from MoveIt's plan-stop-replan cycle between successive goal positions. By streaming PVT points directly, we keep the robot in PVT mode continuously.

### The Blocking Problem and How We Handle It

All PVT commands go through `send_script_str()` (the **blocking** variant), not `send_script_str_silent()`. This means each `set_pvt_point()` call waits for the robot to acknowledge receipt before returning. Typical round-trip: 5-20ms. This limits the practical PVT point rate to ~20-40Hz.

For our use case (model outputs at 10-30Hz), this is fine — we send PVT points at 10Hz (one every 100ms), and the robot smoothly interpolates between them. The robot's cubic spline engine does the heavy lifting of making the motion smooth, not our update rate.

### End-to-End Data Flow

```
Vision-Action Model (Python, 10-30 Hz)
    |
    | Publishes sensor_msgs/JointState to /target_joint_positions
    | (6 joint angles in radians — the model's predicted next pose)
    v
tm_pvt_servo_node.py (Python ROS2 node, 10 Hz PVT point generation)
    |
    | Reads current joint positions from /joint_states
    | Computes step: clamp(target - last_sent, max_step_rad)
    | Computes next_pos: last_sent + step
    | Computes velocity via finite difference: (next_pos - last_sent) / dt
    | Clamps velocity to max_joint_vel
    | Publishes Float64MultiArray [p1..p6, v1..v6, time] to /servo_pvt_point
    v
tm_driver C++ node (pvt_point_callback subscriber)
    |
    | Calls iface_.set_pvt_point(TmPvtMode::Joint, t, pos, vel)
    | Which calls sct.send_script_str(id, script) — BLOCKING
    | Which sends "PVTPoint(p1_deg,...,v1_deg,...,time)" over TCP to port 5890
    | (positions/velocities converted from rad to deg internally)
    v
TM12S Robot Controller (Listen Node on TMflow)
    |
    | Receives PVT point via TMSCT socket
    | Computes cubic spline from previous point to this point
    | Executes the interpolated motion over the specified time duration
    | Smooth continuous acceleration profile — no jerk
    v
Smooth continuous motion!
```

### Why PVT is Better Than set_positions (Plan C) But Worse Than Velocity Mode (Plan A)

**Better than Plan C** because:
- Robot-side cubic interpolation means continuous acceleration (no sudden speed changes)
- You specify both position AND velocity at each point, giving the robot full information for smooth curves
- Explicit timing means deterministic execution — no mystery queue lag

**Worse than Plan A** because:
- PVT points are queued, so there's 1-2 point lag (~100-200ms at 10Hz) before direction changes take effect
- Uses blocking `send_script_str` (not the non-blocking silent variant), limiting practical rate
- Requires computing velocities (finite differences) rather than just using a simple P controller

### Safety Mechanisms

1. **Step clamping**: Each PVT point's position change is clamped to `max_step_rad` (default 0.05 rad = ~2.9 deg) per point. At 10Hz this limits max speed to 0.5 rad/s.
2. **Velocity clamping**: Computed velocities are clamped to `max_joint_vel` (default 0.5 rad/s).
3. **Target timeout**: If no new target arrives for `target_timeout` seconds (default 0.5s), zero-velocity hold-position PVT points are sent.
4. **PVTExit on shutdown**: The destructor and signal handlers always send `PVTExit()` to cleanly exit PVT mode.
5. **Robot-side**: PVT mode will error if time is too short or velocities are physically unreachable.

---

## Why PVT Mode

PVT mode is the TM robot's native trajectory streaming mechanism. You send the robot a sequence of points, each specifying **target position + velocity + time duration**, and the robot internally performs **cubic spline interpolation** between them. This is what MoveIt already uses under the hood via the `FollowJointTrajectory` action — but the jerkiness comes from MoveIt's plan-execute-stop-replan cycle, not from PVT itself.

By streaming PVT points directly without MoveIt, we skip the planning overhead and get smooth continuous interpolation.

### PVT vs Velocity Mode Trade-offs

| Aspect | PVT Mode (this plan) | Velocity Mode (Plan A) |
|--------|---------------------|----------------------|
| Input type | Positions (natural for model output) | Velocities (derived from positions) |
| Interpolation | Robot does cubic spline internally | None — raw velocity commands |
| Reactivity | Moderate — points are queued | High — each command overrides previous |
| Direction changes | Lag from queued points | Immediate |
| Smoothness | Excellent — cubic interpolation | Good — depends on controller tuning |
| Blocking | `send_script_str` (blocking per point) | `send_script_str_silent` (non-blocking) |
| Complexity | Must compute velocities + timing | Simple P controller |

**Use PVT when**: model output is relatively predictable (no sudden reversals), you want the smoothest possible interpolation, and slight latency (50-100ms) is acceptable.

---

## How PVT Works in the TM Driver

### TMScript Commands (from `tm_driver/src/tm_command.cpp:83-128`)

```
PVTEnter(0)                                          -- Enter joint PVT mode (0=Joint, 1=Tool)
PVTPoint(p1,p2,p3,p4,p5,p6,v1,v2,v3,v4,v5,v6,t)   -- Position(deg), velocity(deg/s), time(sec)
PVTExit()                                            -- Exit PVT mode
```

### C++ Driver API (from `tm_driver/include/tm_driver/tm_driver.h:84-91`)

```cpp
bool set_pvt_enter(TmPvtMode mode, const std::string &id = "PvtEnter");
bool set_pvt_exit(const std::string &id = "PvtExit");

// Send a single point: time (sec), positions (rad), velocities (rad/s)
bool set_pvt_point(TmPvtMode mode,
    double t, const std::vector<double> &pos, const std::vector<double> &vel,
    const std::string &id = "PvtPt");

// Send an entire trajectory as one script (PVTEnter + all points + PVTExit)
bool set_pvt_traj(const TmPvtTraj &pvts, const std::string &id = "PvtTraj");
```

### Key Details

- **Unit conversion**: The API accepts rad and rad/s but converts to degrees internally via `TmCommand::deg()` before sending TMScript.
- **Minimum time between points**: 0.025 seconds (from MoveIt integration at `tm_driver/src/tm_ros2_movit_sct.cpp:96` — `get_pvt_traj(traj_points, 0.025)`).
- **All PVT calls use `send_script_str`** (blocking) — unlike velocity mode which has a silent variant.
- **`set_pvt_traj`** sends the entire trajectory (PVTEnter + all points + PVTExit) as a single script string. This is how MoveIt uses it — but it requires the full trajectory upfront.
- **`set_pvt_point`** sends individual points — this is what we need for streaming.
- **Cubic interpolation** happens on the robot controller between consecutive points (verified in `tm_driver/src/tm_driver.cpp:225-240`).

### Existing ROS2 Interface

The `/send_script` service (`tm_msgs/srv/SendScript`) can send arbitrary TMScript:

```
# tm_msgs/srv/SendScript.srv
string id
string script
---
bool ok
```

This calls `sct_.send_script_str(id, script)` internally (blocking).

---

## Phase 0: Quick Validation Test (NO CODE CHANGES)

### Prerequisites

1. TMflow project with Listen Node active, robot in Auto Mode
2. TM driver launched: `ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>`
3. Verify: `ros2 topic echo /joint_states --once`

### Test Script: `test_pvt_mode.py`

Streams a smooth sine wave trajectory via PVT to validate the approach works.

```python
#!/usr/bin/env python3
"""
Quick validation test for TM12S PVT mode streaming.
Uses ONLY the existing /send_script service — no driver modifications needed.

Moves joint 1 in a smooth sine wave using PVT points.
WARNING: Ensure workspace is clear.

Usage:
    1. Launch tm_driver: ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<IP>
    2. Run: python3 test_pvt_mode.py
    3. Press Ctrl+C to stop (sends PVTExit)
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


class PvtModeTest(Node):
    def __init__(self):
        super().__init__('pvt_mode_test')

        self.send_script_cli = self.create_client(SendScript, 'send_script')
        while not self.send_script_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Waiting for /send_script service...')

        self.current_joints = None
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_cb, 10)

        self.pvt_active = False
        self.get_logger().info('PVT mode test node ready.')

    def joint_cb(self, msg):
        self.current_joints = np.array(msg.position[:6])

    def send_script_sync(self, script_id, script):
        """Send TMScript via /send_script service (blocking)."""
        req = SendScript.Request()
        req.id = script_id
        req.script = script
        future = self.send_script_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() is not None:
            return future.result().ok
        self.get_logger().error(f'Script [{script_id}] call failed')
        return False

    def pvt_enter(self):
        """Enter joint PVT mode."""
        ok = self.send_script_sync('PvtEnter', 'PVTEnter(0)')
        if ok:
            self.pvt_active = True
            self.get_logger().info('Entered PVT mode')
        return ok

    def pvt_exit(self):
        """Exit PVT mode."""
        ok = self.send_script_sync('PvtExit', 'PVTExit()')
        self.pvt_active = False
        self.get_logger().info('Exited PVT mode')
        return ok

    def pvt_point(self, positions_rad, velocities_rad_s, time_sec):
        """
        Send a single PVT point.
        positions: 6 joint angles in RADIANS (converted to degrees for TMScript)
        velocities: 6 joint velocities in RAD/S (converted to deg/s for TMScript)
        time_sec: duration for this segment in seconds
        """
        pos_deg = [math.degrees(p) for p in positions_rad]
        vel_deg = [math.degrees(v) for v in velocities_rad_s]

        parts = [f'{p:.5f}' for p in pos_deg] + [f'{v:.5f}' for v in vel_deg] + [f'{time_sec:.4f}']
        script = f'PVTPoint({",".join(parts)})'
        return self.send_script_sync('PvtPt', script)

    def run_sine_wave_test(self, duration_sec=12.0, amplitude_rad=0.15, freq_hz=0.25, dt=0.1):
        """
        Generate a sine wave on joint 1 and stream it as PVT points.

        Args:
            duration_sec: Total test duration
            amplitude_rad: Sine wave amplitude in radians (~8.6 degrees)
            freq_hz: Oscillation frequency
            dt: Time between PVT points (seconds). Must be >= 0.025.
        """
        self.get_logger().info('Waiting for /joint_states...')
        while self.current_joints is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        start_pos = self.current_joints.copy()
        self.get_logger().info(
            f'Start position (deg): {[f"{math.degrees(j):.2f}" for j in start_pos]}'
        )
        self.get_logger().info(
            f'Sine wave: amplitude={math.degrees(amplitude_rad):.1f} deg, '
            f'freq={freq_hz} Hz, dt={dt} s, duration={duration_sec} s'
        )

        if not self.pvt_enter():
            self.get_logger().error('Failed to enter PVT mode')
            return

        t = 0.0
        omega = 2.0 * math.pi * freq_hz
        point_count = 0

        try:
            while t < duration_sec:
                # Target position: sine wave on joint 1, others stay at start
                target_pos = start_pos.copy()
                target_pos[0] = start_pos[0] + amplitude_rad * math.sin(omega * t)

                # Analytical velocity (derivative of sine)
                target_vel = np.zeros(6)
                target_vel[0] = amplitude_rad * omega * math.cos(omega * t)

                ok = self.pvt_point(target_pos, target_vel, dt)
                if not ok:
                    self.get_logger().error(f'PVT point failed at t={t:.2f}')
                    break

                point_count += 1
                t += dt

                # Log every ~1 second
                if point_count % int(1.0 / dt) == 0:
                    actual = self.current_joints if self.current_joints is not None else start_pos
                    err = math.degrees(target_pos[0] - actual[0])
                    self.get_logger().info(
                        f't={t:.1f}s | target_j1={math.degrees(target_pos[0]):.2f} deg | '
                        f'actual_j1={math.degrees(actual[0]):.2f} deg | err={err:.3f} deg'
                    )

                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0)

                # Pace ourselves — PVT points should be sent roughly in real-time
                # The robot buffers a few points ahead, so slight timing jitter is OK
                time.sleep(dt * 0.8)  # Send slightly ahead of real-time

        finally:
            self.get_logger().info(f'Sent {point_count} PVT points in {t:.1f}s')
            self.pvt_exit()

    def run_position_tracking_test(self, dt=0.1):
        """
        Track /target_joint_positions using PVT mode.
        Publishes targets, computes finite-difference velocities, streams PVT points.

        Press Ctrl+C to stop.
        """
        self.get_logger().info('Waiting for /joint_states...')
        while self.current_joints is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        self.target_joints = None
        self.target_sub = self.create_subscription(
            JointState, 'target_joint_positions', self._target_cb, 10)

        self.get_logger().info('PVT position tracking mode.')
        self.get_logger().info(f'Publish targets to /target_joint_positions, dt={dt}s')
        self.get_logger().info('Waiting for first target...')

        while self.target_joints is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        if not self.pvt_enter():
            self.get_logger().error('Failed to enter PVT mode')
            return

        prev_target = self.current_joints.copy()
        max_vel_rad = 0.5  # Safety clamp

        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0)

                if self.target_joints is None:
                    time.sleep(dt)
                    continue

                target = self.target_joints.copy()

                # Compute velocity via finite difference
                vel = (target - prev_target) / dt
                vel = np.clip(vel, -max_vel_rad, max_vel_rad)

                self.pvt_point(target, vel, dt)
                prev_target = target.copy()

                time.sleep(dt * 0.8)

        finally:
            self.pvt_exit()

    def _target_cb(self, msg):
        if len(msg.position) >= 6:
            self.target_joints = np.array(msg.position[:6])


def main():
    rclpy.init()
    node = PvtModeTest()

    def signal_handler(sig, frame):
        node.get_logger().warn('Ctrl+C — exiting PVT mode...')
        if node.pvt_active:
            node.pvt_exit()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Run sine wave test first
        node.run_sine_wave_test(
            duration_sec=12.0,
            amplitude_rad=0.15,   # ~8.6 degrees
            freq_hz=0.25,         # One cycle every 4 seconds
            dt=0.1                # 10 Hz PVT points
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### What to Observe

**Success**: Joint 1 traces a smooth sine wave. Motion should be visibly smoother than discrete PTP commands — the cubic interpolation produces continuous acceleration profiles.

**Potential issue — command queue buildup**: If points are sent faster than the robot executes them, a lag develops. The test script uses `time.sleep(dt * 0.8)` to stay slightly ahead. If you see the robot lagging behind the commanded trajectory, increase `dt` (e.g., 0.15 or 0.2).

**Potential issue — PVTPoint rejected**: The robot may reject points if the time is too short or velocities are unreachable. Check `/sct_response` for error messages.

---

## Phase 1: Add PVT Streaming Topic to the TM Driver (C++)

### Why Modify the Driver

The `/send_script` service uses `send_script_str` which is blocking. For PVT streaming at 10-20Hz this is likely fine, but adding a dedicated topic gives cleaner integration and avoids service call overhead.

### File: `tm_driver/include/tm_driver/tm_ros2_sct.h`

Add this include at the top (after existing includes):

```cpp
#include "std_msgs/msg/float64_multi_array.hpp"
```

Add these members to the `TmSctRos2` class (after `ask_sta_srv_`, around line 50):

```cpp
    // PVT streaming
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr pvt_point_sub_;
    rclcpp::Service<tm_msgs::srv::SendScript>::SharedPtr pvt_enter_srv_;
    rclcpp::Service<tm_msgs::srv::SendScript>::SharedPtr pvt_exit_srv_;
    bool pvt_mode_active_ = false;
```

Add these method declarations (after `ask_sta` declaration):

```cpp
    void pvt_point_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);

    bool pvt_enter(
        const std::shared_ptr<tm_msgs::srv::SendScript::Request> req,
        std::shared_ptr<tm_msgs::srv::SendScript::Response> res);
    bool pvt_exit(
        const std::shared_ptr<tm_msgs::srv::SendScript::Request> req,
        std::shared_ptr<tm_msgs::srv::SendScript::Response> res);
```

### File: `tm_driver/src/tm_ros2_sct.cpp`

Add to the constructor (after `ask_sta_srv_` setup, before closing `}`):

```cpp
    // PVT streaming
    pvt_point_sub_ = node->create_subscription<std_msgs::msg::Float64MultiArray>(
        "servo_pvt_point", 10,
        std::bind(&TmSctRos2::pvt_point_callback, this, std::placeholders::_1));

    pvt_enter_srv_ = node->create_service<tm_msgs::srv::SendScript>(
        "pvt_enter", std::bind(&TmSctRos2::pvt_enter, this,
        std::placeholders::_1, std::placeholders::_2));

    pvt_exit_srv_ = node->create_service<tm_msgs::srv::SendScript>(
        "pvt_exit", std::bind(&TmSctRos2::pvt_exit, this,
        std::placeholders::_1, std::placeholders::_2));
```

Add these implementations at the end of the file:

```cpp
void TmSctRos2::pvt_point_callback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
    if (!pvt_mode_active_) {
        return;
    }
    // Expected layout: [p1, p2, p3, p4, p5, p6, v1, v2, v3, v4, v5, v6, time]
    // Positions in rad, velocities in rad/s, time in seconds
    if (msg->data.size() < 13) {
        print_error("TM_ROS: pvt_point requires 13 values (6 pos + 6 vel + 1 time), got %zu",
                     msg->data.size());
        return;
    }
    std::vector<double> pos(msg->data.begin(), msg->data.begin() + 6);
    std::vector<double> vel(msg->data.begin() + 6, msg->data.begin() + 12);
    double t = msg->data[12];

    iface_.set_pvt_point(TmPvtMode::Joint, t, pos, vel);
}

bool TmSctRos2::pvt_enter(
    const std::shared_ptr<tm_msgs::srv::SendScript::Request> req,
    std::shared_ptr<tm_msgs::srv::SendScript::Response> res)
{
    (void)req;
    bool ok = iface_.set_pvt_enter(TmPvtMode::Joint);
    if (ok) {
        pvt_mode_active_ = true;
        print_info("TM_ROS: PVT mode ENTERED");
    } else {
        print_error("TM_ROS: Failed to enter PVT mode");
    }
    res->ok = ok;
    return ok;
}

bool TmSctRos2::pvt_exit(
    const std::shared_ptr<tm_msgs::srv::SendScript::Request> req,
    std::shared_ptr<tm_msgs::srv::SendScript::Response> res)
{
    (void)req;
    bool ok = iface_.set_pvt_exit();
    pvt_mode_active_ = false;
    print_info("TM_ROS: PVT mode EXITED");
    res->ok = ok;
    return ok;
}
```

Add cleanup in the destructor `TmSctRos2::~TmSctRos2()` (before `print_info`):

```cpp
    if (pvt_mode_active_) {
        iface_.set_pvt_exit();
        pvt_mode_active_ = false;
    }
```

### No changes needed to `CMakeLists.txt` or `package.xml`

`std_msgs` is already a dependency.

### Build

```bash
cd ~/git/tm2_ros2
colcon build --packages-select tm_driver
source install/setup.bash
```

### Manual Test After Phase 1

```bash
# Terminal 1: Launch driver
ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>

# Terminal 2: Enter PVT mode
ros2 service call /pvt_enter tm_msgs/srv/SendScript "{id: 'test', script: ''}"

# Terminal 3: Send a single PVT point — move joint 1 by ~5.7 deg over 1 second
# Format: [p1..p6 (rad), v1..v6 (rad/s), time (sec)]
# Get current position first, then offset joint 1 by 0.1 rad
ros2 topic pub /servo_pvt_point std_msgs/msg/Float64MultiArray \
  "{data: [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}" --once

# Terminal 2: Exit PVT mode
ros2 service call /pvt_exit tm_msgs/srv/SendScript "{id: 'test', script: ''}"
```

---

## Phase 2: Python PVT Servo Node

### File: `tm_driver/tm_driver/tm_pvt_servo_node.py`

```python
#!/usr/bin/env python3
"""
TM12S PVT Servo Node

Converts continuous target joint positions into PVT (Position-Velocity-Time)
points for the TM12S robot arm. The robot internally performs cubic spline
interpolation between points for smooth motion.

Subscribes:
    /target_joint_positions (sensor_msgs/JointState) — target from vision-action model
    /joint_states (sensor_msgs/JointState) — current robot state

Publishes:
    /servo_pvt_point (std_msgs/Float64MultiArray) — PVT points to driver
        Layout: [p1..p6 (rad), v1..v6 (rad/s), time (sec)]

Services called:
    /pvt_enter — enter PVT mode on startup
    /pvt_exit — exit PVT mode on shutdown

Parameters:
    point_rate (double, default=10.0): PVT point publish rate in Hz
    max_joint_vel (double, default=0.5): Max velocity per joint in rad/s
    max_step_rad (double, default=0.05): Max position change per point in rad (safety)
    lookahead_gain (double, default=1.0): Velocity computation gain (1.0 = exact finite diff)
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


class TmPvtServoNode(Node):
    def __init__(self):
        super().__init__('tm_pvt_servo')

        # Parameters
        self.declare_parameter('point_rate', 10.0)          # Hz — PVT point rate
        self.declare_parameter('max_joint_vel', 0.5)        # rad/s
        self.declare_parameter('max_step_rad', 0.05)        # rad per point — safety
        self.declare_parameter('target_timeout', 0.5)       # seconds
        self.declare_parameter('lookahead_gain', 1.0)       # velocity scaling

        self.point_rate = self.get_parameter('point_rate').value
        self.max_vel = self.get_parameter('max_joint_vel').value
        self.max_step = self.get_parameter('max_step_rad').value
        self.target_timeout = self.get_parameter('target_timeout').value
        self.lookahead_gain = self.get_parameter('lookahead_gain').value

        self.dt = 1.0 / self.point_rate  # Time between PVT points (seconds)

        # State
        self.current_joints = None
        self.target_joints = None
        self.last_target_time = None
        self.prev_target = None          # For finite-difference velocity
        self.pvt_active = False
        self.last_sent_pos = None        # Track what we last sent to avoid drift

        # Publisher
        self.pvt_pub = self.create_publisher(Float64MultiArray, 'servo_pvt_point', 10)

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self._joint_cb, 10)
        self.target_sub = self.create_subscription(
            JointState, 'target_joint_positions', self._target_cb, 10)

        # Service clients
        self.pvt_enter_cli = self.create_client(SendScript, 'pvt_enter')
        self.pvt_exit_cli = self.create_client(SendScript, 'pvt_exit')

        # Control loop timer
        self.timer = self.create_timer(self.dt, self._control_loop)

        self.get_logger().info(
            f'TM PVT Servo Node started: rate={self.point_rate} Hz (dt={self.dt:.3f}s), '
            f'max_vel={self.max_vel} rad/s, max_step={self.max_step} rad'
        )

    def _joint_cb(self, msg):
        self.current_joints = np.array(msg.position[:6])

    def _target_cb(self, msg):
        if len(msg.position) >= 6:
            self.target_joints = np.array(msg.position[:6])
            self.last_target_time = time.time()

    def start_pvt_mode(self):
        """Call /pvt_enter service."""
        if not self.pvt_enter_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/pvt_enter service not available')
            return False
        req = SendScript.Request()
        req.id = 'servo_pvt_enter'
        req.script = ''
        future = self.pvt_enter_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result() and future.result().ok:
            self.pvt_active = True
            self.last_sent_pos = self.current_joints.copy()
            self.prev_target = self.current_joints.copy()
            self.get_logger().info('PVT mode started')
            return True
        self.get_logger().error('Failed to start PVT mode')
        return False

    def stop_pvt_mode(self):
        """Call /pvt_exit service."""
        if not self.pvt_exit_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('/pvt_exit service not available')
            return
        req = SendScript.Request()
        req.id = 'servo_pvt_exit'
        req.script = ''
        future = self.pvt_exit_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        self.pvt_active = False
        self.get_logger().info('PVT mode stopped')

    def _publish_pvt(self, positions_rad, velocities_rad_s, time_sec):
        """Publish a PVT point as Float64MultiArray [p1..p6, v1..v6, time]."""
        msg = Float64MultiArray()
        msg.data = list(positions_rad) + list(velocities_rad_s) + [time_sec]
        self.pvt_pub.publish(msg)

    def _control_loop(self):
        """Runs at point_rate Hz — computes and sends next PVT point."""
        if not self.pvt_active:
            return
        if self.current_joints is None or self.last_sent_pos is None:
            return

        # If no target or target timed out, hold current position
        if self.target_joints is None:
            self._publish_pvt(self.last_sent_pos, np.zeros(6), self.dt)
            return

        if self.last_target_time and (time.time() - self.last_target_time) > self.target_timeout:
            self._publish_pvt(self.last_sent_pos, np.zeros(6), self.dt)
            return

        target = self.target_joints.copy()

        # Compute step from last sent position to target
        step = target - self.last_sent_pos

        # Clamp step to max_step_rad for safety
        step_magnitude = np.abs(step)
        scale = np.where(step_magnitude > self.max_step,
                         self.max_step / step_magnitude, 1.0)
        step = step * scale

        # Compute the actual position to send
        next_pos = self.last_sent_pos + step

        # Compute velocity via finite difference from previous target
        vel = self.lookahead_gain * (next_pos - self.last_sent_pos) / self.dt
        vel = np.clip(vel, -self.max_vel, self.max_vel)

        # Clamp velocity-implied speed
        max_step_from_vel = np.abs(vel) * self.dt
        too_fast = max_step_from_vel > self.max_step
        if np.any(too_fast):
            vel[too_fast] = np.sign(vel[too_fast]) * self.max_step / self.dt

        self._publish_pvt(next_pos, vel, self.dt)
        self.last_sent_pos = next_pos.copy()
        self.prev_target = target.copy()


def main(args=None):
    rclpy.init(args=args)
    node = TmPvtServoNode()

    # Wait for joint states
    node.get_logger().info('Waiting for /joint_states...')
    while node.current_joints is None and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.5)

    if not node.start_pvt_mode():
        node.get_logger().error('Cannot start — exiting')
        node.destroy_node()
        rclpy.shutdown()
        return

    def shutdown_handler(sig, frame):
        node.get_logger().warn('Shutting down PVT servo node...')
        node.stop_pvt_mode()
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
        if node.pvt_active:
            node.stop_pvt_mode()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Install the Python Script

Add to `tm_driver/CMakeLists.txt` before `ament_package()`:

```cmake
install(PROGRAMS
  tm_driver/tm_pvt_servo_node.py
  DESTINATION lib/${PROJECT_NAME}
)
```

---

## Phase 3: Integration

### Data Flow

```
Vision-Action Model (10-30Hz)
    | publishes sensor_msgs/JointState
    v
/target_joint_positions
    |
    v
tm_pvt_servo_node.py (10Hz PVT point generation)
    | computes step-limited position + finite-difference velocity
    | publishes Float64MultiArray [p1..p6, v1..v6, time]
    v
/servo_pvt_point
    |
    v
tm_driver C++ subscriber (pvt_point_callback)
    | calls iface_.set_pvt_point(Joint, t, pos, vel)
    | sends "PVTPoint(p1_deg,...,v1_deg,...,time)" via TMSCT
    v
TM12S Robot (cubic spline interpolation between points)
```

### Launch Sequence

```bash
# Terminal 1: TM Driver
ros2 launch tm_driver tm_bringup.launch.py robot_ip:=<ROBOT_IP>

# Terminal 2: PVT Servo Node
ros2 run tm_driver tm_pvt_servo_node.py --ros-args \
  -p point_rate:=10.0 \
  -p max_joint_vel:=0.5 \
  -p max_step_rad:=0.05

# Terminal 3: Vision-Action Model
python3 your_model_inference.py
```

### Tuning Guide

| Parameter | Effect | Start With | Notes |
|-----------|--------|------------|-------|
| `point_rate` | How often PVT points are sent | 10.0 Hz | Higher = smoother but more network load. Max practical ~20Hz due to blocking `send_script_str`. Lower = more robot-side interpolation. |
| `max_joint_vel` | Velocity clamp per joint | 0.5 rad/s | Increase for faster tracking. Robot max is pi rad/s. |
| `max_step_rad` | Max position change per PVT point | 0.05 rad | Safety limit. At 10Hz and 0.05 rad/point = 0.5 rad/s effective max speed. |
| `target_timeout` | Hold position if no new target | 0.5 s | Robot holds last position via zero-velocity PVT points. |
| `lookahead_gain` | Scales computed velocity | 1.0 | >1.0 = overshoot slightly (smooths transitions). <1.0 = conservative. |

### Key Differences from Plan A (Velocity Mode)

1. **PVT is position-based** — you send where the robot should BE, not how fast it should MOVE. More natural for position-output models.
2. **Robot-side interpolation** — cubic splines between points mean the motion between points is mathematically smooth (continuous acceleration).
3. **Blocking sends** — `set_pvt_point` uses `send_script_str` (blocking), limiting practical rate to ~20Hz. Plan A's velocity mode uses `send_script_str_silent` (non-blocking) allowing 50Hz+.
4. **Queued execution** — PVT points are buffered and executed in order. If the model changes direction rapidly, there's a 1-2 point lag (~100-200ms at 10Hz). Plan A overrides immediately.
5. **Smoothness advantage** — for predictable, relatively slow trajectories, PVT produces the smoothest possible motion because the robot controller handles interpolation.

---

## Verification Checklist

1. **Phase 0 (no code changes):**
   - [ ] `test_pvt_mode.py` sine wave test shows smooth joint 1 oscillation
   - [ ] Position tracking test tracks published targets smoothly
   - [ ] No TMScript errors on `/sct_response`
   - [ ] Robot stops cleanly on Ctrl+C (PVTExit sent)

2. **Phase 1 (driver build):**
   - [ ] `colcon build --packages-select tm_driver` succeeds
   - [ ] `/pvt_enter`, `/pvt_exit` services appear in `ros2 service list`
   - [ ] `/servo_pvt_point` topic appears in `ros2 topic list`
   - [ ] Manual PVT point via `ros2 topic pub` moves the robot

3. **Phase 2 (servo node):**
   - [ ] `tm_pvt_servo_node.py` launches and connects to PVT mode
   - [ ] Publishing to `/target_joint_positions` causes smooth tracking
   - [ ] Multiple rapid target changes track without violent jerks
   - [ ] Node handles Ctrl+C with clean PVT exit

4. **Integration:**
   - [ ] Vision-action model publishes targets
   - [ ] Robot follows model output smoothly
   - [ ] Acceptable latency (< 200ms for PVT at 10Hz)
