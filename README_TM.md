# RAPP Lab 04: TM12S Robot Operation Guide

TM12S deployment of the Vision-Action Model pipeline. Uses the same trained model as UR10 with direct joint angle pass-through (both are 6-DOF serial manipulators with matching joint layouts).

**Important:** Only run one robot at a time. Stop UR10 containers before starting TM12S, and vice versa. All services use ROS_DOMAIN_ID=0.

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with driver 535+
- Docker and Docker Compose
- NVIDIA Container Toolkit

## Architecture

Two containers, both using `network_mode: host` so ROS2 topics are visible across both:

| Container | Compose File | Name | What's inside |
|---|---|---|---|
| **Hardware** | `docker/docker-compose.hw.yml` | `rapp_hw` | ZED SDK, TM2 driver, MoveIt, RViz, TM12S meshes |
| **VAM** | `docker/docker-compose.yml` | `rapp_vam` | PyTorch, model inference, rosbag playback |

Both containers share the same volumes:

| Host path | Container path | Purpose |
|---|---|---|
| `ros2_ws/` | `/workspace/ros2_ws` | VAM inference ROS2 package |
| `vam_utils/` | `/workspace/vam_utils` | Shared utilities |
| `notebooks/` | `/workspace/notebooks` | Jupyter notebooks |
| `scripts/` | `/workspace/scripts` | Helper scripts |
| `config/` | `/config` | Configuration files |
| `~/rosbags/rapplab04` | `/data/rosbags` | Rosbag recordings |
| `~/csvdata/rapplab04` | `/data/processed` | Processed data, URDFs, RViz configs |
| `~/models/rapplab04` | `/data/models` | Trained models |

The hardware container also auto-mounts the servo and controller configs into the TM2 install tree — no manual copying needed.

## Setup

### 1. Start Containers

```bash
# Enable X11 for GUI (RViz, matplotlib)
xhost +local:docker

# Start hardware container
cd docker
docker compose -f docker-compose.hw.yml up -d

# Start VAM container
docker compose up -d
```

### 2. Export TM12S URDF (one-time)

```bash
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
source /tm2_ws/install/setup.bash

ros2 run xacro xacro \
    /tm2_ws/install/tm_description/share/tm_description/xacro/tm12s.urdf.xacro \
    > /data/processed/tm12s.urdf
```

### 3. Build VAM inference package

```bash
docker exec -it rapp_vam bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Key Differences from UR10

| Property | UR10 | TM12S |
|---|---|---|
| Joint names | `shoulder_pan_joint` .. `wrist_3_joint` | `joint_1` .. `joint_6` |
| MoveIt group | `ur_manipulator` | `tmr_arm` |
| Planning frame | `base_link` | `base` |
| EE frame | `tool0` | `flange` |
| Robot IP | 10.0.0.89 | 192.168.10.2 |
| Driver | `ur_robot_driver` | `tm_driver` (tm2_ros2) |

## Running the Inference Node

### Mode 1: RViz Visualization (rosbag replay)

Visualize model predictions on a TM12S ghost robot — no real robot needed.
VAM inference runs in the VAM container; RViz runs in the hardware container (where meshes are available).

```bash
# Terminal 1 (rapp_vam): Play rosbag with clock
docker exec -it rapp_vam bash
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 bag play /data/rosbags/<name> --clock

# Terminal 2 (rapp_vam): Launch inference + robot_state_publishers (no RViz)
docker exec -it rapp_vam bash
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch vam_inference vam_tm12s_headless.launch.py use_sim_time:=true

# Terminal 3 (rapp_hw): Open RViz with TM12S config
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
source /tm2_ws/install/setup.bash
rviz2 -d /data/processed/vam_tm12s.rviz
```

Check that each joint bends in the expected direction. If a joint moves opposite, see [Sign Conventions](#sign-conventions) below.

### Mode 2: Real Robot with Rosbag Skeleton Data

Control the real TM12S using skeleton data from a rosbag.

Unlike UR10 (where `ur_moveit_config` has built-in servo support via `launch_servo:=true`),
the TM12S `tm12s_moveit_config` package has **no** MoveIt Servo support, and its launch
files start a fake `ros2_control_node` that conflicts with the real driver. We use custom
launch files from `vam_inference` that fix both issues.

**Important:** Do NOT use `tm12s_moveit.launch.py` or `demo.launch.py` — both start a
fake ros2_control stack that publishes competing zero joint states, causing a "blinking
robot" in RViz. Use `tm12s_moveit_hw.launch.py` instead.

```bash
# Terminal 1 (rapp_hw): Launch MoveIt + TM12S driver + RViz (no fake ros2_control)
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
source /tm2_ws/install/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch vam_inference tm12s_moveit_hw.launch.py robot_ip:=192.168.10.2

# Terminal 2 (rapp_hw): Launch MoveIt Servo node
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
source /tm2_ws/install/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch vam_inference tm12s_servo.launch.py

# Terminal 3 (rapp_hw): Start servo + velocity streaming bridge
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
source /tm2_ws/install/setup.bash
source /workspace/ros2_ws/install/setup.bash

ros2 service call /servo_node/start_servo std_srvs/srv/Trigger {}

#   VJog bridge: differentiates servo positions into velocities, then
#   streams them to the TM12S via ContinueVJog (native velocity mode).
#   Waits for convergence before sending any commands to the robot.
ros2 run vam_inference servo_to_tm_vjog_bridge

# Terminal 4 (rapp_hw): Static transform — map -> world (adjust for lab)
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
ros2 run tf2_ros static_transform_publisher \
    4.4 0.1 -0.25 1.5708 0 0 map world

# Terminal 5 (rapp_vam): Play rosbag — skeleton topic ONLY (no --clock!)
docker exec -it rapp_vam bash
source /opt/ros/humble/setup.bash
ros2 bag play /data/rosbags/<name> \
    --topics /zed/zed_node/body_trk/skeletons --loop

# Terminal 6 (rapp_vam): VAM inference
docker exec -it rapp_vam bash
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch vam_inference vam_tm12s_robot.launch.py
```

The `--topics` filter is critical — it prevents the rosbag from publishing
`/joint_states`, `/tf`, `/tf_static` which would conflict with the real robot.
No `--clock` because the real robot operates in wall-clock time.

**First-time safety:** Start with very low P-gain:

```bash
ros2 launch vam_inference vam_tm12s_robot.launch.py servo_proportional_gain:=1.0
```

### Mode 3: Live Performance (camera + robot)

```bash
# Terminals 1-4 (rapp_hw): Same as Mode 2

# Terminal 5 (rapp_hw): ZED camera (replaces rosbag)
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i

# Terminal 6 (rapp_vam): VAM inference
docker exec -it rapp_vam bash
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch vam_inference vam_tm12s_robot.launch.py
```

## Launch Files

| Launch File | Mode | Use Case |
|---|---|---|
| `vam_tm12s_headless.launch.py` | headless | Inference + robot_state_publishers, no RViz (run RViz from rapp_hw) |
| `vam_tm12s_robot.launch.py` | robot | Real robot control via MoveIt Servo |
| `tm12s_moveit_hw.launch.py` | robot | MoveIt + TM driver + RViz for real hardware (replaces upstream `tm12s_moveit.launch.py`) |
| `tm12s_servo.launch.py` | robot | MoveIt Servo node for TM12S (launched separately from move_group) |

**Bridge nodes** (required for real robot — TM driver has no ros2_control hardware plugin):

| Node | Approach | Purpose |
|---|---|---|
| `servo_to_tm_vjog_bridge` | **ContinueVJog** (recommended) | Differentiates servo positions → streams joint velocities via TM's native velocity mode. Smooth, no batch boundaries. |
| `servo_to_tm_bridge` | FollowJointTrajectory (fallback) | Batches servo positions into PVT trajectory goals. May be jerky at batch boundaries. |

### Key Parameters

| Parameter | Default (rviz) | Default (robot) | Description |
|---|---|---|---|
| `servo_proportional_gain` | — | 2.0 | P-controller gain for MoveIt Servo (lower = slower) |
| `max_joint_velocity_rad_s` | 1.0 | 2.0 | SafetyChecker pre-filter velocity limit (rad/s) |
| `max_joint_acceleration_rad_s2` | 5.0 | 5.0 | SafetyChecker pre-filter acceleration limit (rad/s²) |
| `prediction_stride_K` | 1 | 1 | Re-predict every K frames |
| `ensemble_decay_weight` | 0.5 | 0.5 | Temporal smoothing (lower = smoother) |
| `tracking_timeout_sec` | 0.5 | 0.5 | Skeleton loss detection threshold |
| `target_skeleton_id` | -1 | -1 | Skeleton to track (-1 = auto) |
| `joint_sign_multipliers` | [1,1,1,1,1,1] | [1,1,1,1,1,1] | Per-joint sign flip (-1 reverses direction) |

**VJog bridge parameters** (`servo_to_tm_vjog_bridge`):

| Parameter | Default | Description |
|---|---|---|
| `max_velocity_deg_s` | 30.0 | Per-joint velocity clamp (deg/s). Start low, increase after verification. |
| `ema_alpha` | 0.3 | Velocity smoothing (0.1 = very smooth, 0.5 = responsive) |
| `send_rate_hz` | 30.0 | Rate for sending velocity commands to TM |
| `convergence_threshold_rad` | 0.05 | Servo must be within this of robot before streaming starts |
| `watchdog_timeout_sec` | 0.3 | Stop robot if no servo data for this long |

## Sign Conventions

Both UR10 and TM12S are 6-DOF serial manipulators with the same joint layout. The model outputs are passed through 1:1. If a joint bends the wrong way, flip its sign multiplier:

```bash
# Example: if joint_2 (shoulder) is inverted
ros2 launch vam_inference vam_tm12s_robot.launch.py \
    joint_sign_multipliers:="[1.0, -1.0, 1.0, 1.0, 1.0, 1.0]"
```

| Index | Joint | Function |
|---|---|---|
| 0 | joint_1 | Base rotation |
| 1 | joint_2 | Shoulder |
| 2 | joint_3 | Elbow (narrower range: ±162° vs UR10 ±180°) |
| 3 | joint_4 | Wrist 1 |
| 4 | joint_5 | Wrist 2 |
| 5 | joint_6 | Wrist 3 |

## Safety Architecture

Robot mode uses **MoveIt Servo** as the primary safety layer, with TM12SSafetyChecker as defense-in-depth:

**MoveIt Servo (primary — 250Hz):**

1. **Self-collision avoidance** — Full URDF collision model, scales/stops before collision
2. **Joint limit enforcement** — 0.1 rad margin from limits
3. **Singularity detection** — Scales velocity near singular configurations
4. **Butterworth smoothing** — Low-pass filter eliminates high-frequency jitter
5. **Auto-halt on timeout** — Stops robot if commands stop for >0.1s

**TM12SSafetyChecker (pre-filter — 15Hz, joint_limits_only in robot mode):**

1. **Joint limit clamping** — TM12S URDF limits (joint_3 narrower: ±162°)
2. **Per-joint velocity limits** — [2.27, 2.27, 3.67, 3.93, 3.93, 7.85] rad/s
3. Velocity/acceleration limiting disabled in robot mode (MoveIt Servo handles this)

Additional protections:

- **SafetyChecker seeded** with robot's actual position on startup (prevents first-frame jump)
- **Hold-position on tracking loss** — Zero-velocity JointJog on skeleton timeout
- **Graceful shutdown** — Ctrl+C sends hold-position before exiting
- **TM12S built-in safety** — The robot controller rejects commands violating its internal limits

## TM12S-Specific Configuration Files

| File | Purpose |
|---|---|
| `ros2_ws/src/vam_inference/config/tm12s_servo_vam.yaml` | MoveIt Servo config (ForwardCommandController) |
| `ros2_ws/src/vam_inference/config/tm12s_servo_vam_traj.yaml` | Fallback Servo config (JointTrajectory output) |
| `ros2_ws/src/vam_inference/config/tm12s_ros2_controllers.yaml` | ros2_control config with forward_position_controller |
| `vam_utils/data/robot_configs.py` | TM12S joint names, limits, velocities |
| `docker/zed/Dockerfile.desktop-humble` | Hardware container Dockerfile (ZED + TM2 driver) |
| `docker/docker-compose.hw.yml` | Hardware container compose file |
| `docker/docker-compose.yml` | VAM container compose file |

## Controller Discovery

If unsure which controller setup works:

```bash
# Inside rapp_hw (with driver running)
ros2 control list_controllers
ros2 control list_hardware_interfaces
```

Position command interfaces (`joint_N/position [available]`) confirm ForwardCommandController will work. Use `tm12s_servo_vam.yaml`.

If only trajectory controller is available, use `tm12s_servo_vam_traj.yaml` instead.

## Container Management

```bash
# Start containers
cd docker
docker compose -f docker-compose.hw.yml up -d   # hardware
docker compose up -d                              # VAM

# Shell access
docker exec -it rapp_hw bash
docker exec -it rapp_vam bash

# Stop containers
docker compose -f docker-compose.hw.yml down
docker compose down

# Verify ROS2 communication between containers
ros2 topic list    # should show topics from both containers
```

## Troubleshooting

### TF lookup fails for 'base' frame

The TM12S uses `base` as its root frame (not `base_link` like UR10). Verify:

```bash
# Inside rapp_hw with driver running
ros2 run tf2_ros tf2_echo base link_0
```

### forward_position_controller won't load

The servo and controller configs are auto-mounted by docker-compose.hw.yml into the tm2_ws install tree. If something is wrong, verify the mounts:

```bash
docker exec rapp_hw cat /tm2_ws/install/tm12s_moveit_config/share/tm12s_moveit_config/config/ros2_controllers.yaml
```

Then restart MoveIt.

### Joint moves in wrong direction

Flip that joint's sign multiplier. See [Sign Conventions](#sign-conventions).

### ROS2 topics not visible between containers

Both containers must use the same ROS_DOMAIN_ID and RMW implementation:

```bash
# Check in both containers
echo $ROS_DOMAIN_ID        # should be 0
echo $RMW_IMPLEMENTATION   # should be rmw_cyclonedds_cpp
```

### Elbow range clamping warnings

TM12S joint_3 has narrower range (±162°) than UR10 elbow (±180°). The safety checker clamps predictions automatically. Occasional "Joint limit clamp on joints [2]" warnings are expected and harmless.

---

**Status:** Pipeline configured. Pending real-robot verification and sign convention calibration.
