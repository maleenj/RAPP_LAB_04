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
# Run from anywhere inside the container
cd /workspace/ros2_ws && colcon build --symlink-install && source /workspace/ros2_ws/install/setup.bash
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

Control the real TM12S using skeleton data from a rosbag. Uses **direct PVT streaming** — VAM joint targets are sent straight to the TM12S firmware as PVT (Position-Velocity-Time) points, bypassing MoveIt Servo entirely. The firmware performs cubic spline interpolation between points for smooth 1kHz+ motion.

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

# Terminal 2 (rapp_hw): PVT Streamer (bridges VAM targets to TM12S PVT mode)
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
source /tm2_ws/install/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 run vam_inference vam_pvt_streamer

# Terminal 3 (rapp_hw): Static transform — map -> world (adjust for lab)
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
ros2 run tf2_ros static_transform_publisher \
    4.4 0.1 -0.25 1.5708 0 0 map world

ros2 run tf2_ros static_transform_publisher   4.4 0.1 -0.25 1.5708 0 0 map base


# Terminal 4 (rapp_vam): Play rosbag — skeleton topic ONLY (no --clock!)
docker exec -it rapp_vam bash
source /opt/ros/humble/setup.bash
ros2 bag play /data/rosbags/<name> \
    --topics /zed/zed_node/body_trk/skeletons --loop

# Terminal 5 (rapp_vam): VAM inference
docker exec -it rapp_vam bash
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch vam_inference vam_tm12s_robot.launch.py
```

The `--topics` filter is critical — it prevents the rosbag from publishing
`/joint_states`, `/tf`, `/tf_static` which would conflict with the real robot.
No `--clock` because the real robot operates in wall-clock time.

**First-time safety:** Start with low velocity scale:

```bash
ros2 run vam_inference vam_pvt_streamer --ros-args -p velocity_scale:=0.3
```

### Mode 3: Live Performance (camera + robot)

```bash
# Terminals 1-3 (rapp_hw): Same as Mode 2

# Terminal 4 (rapp_hw): ZED camera (replaces rosbag)
docker exec -it rapp_hw bash
source /opt/ros/humble/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i

# Terminal 5 (rapp_vam): VAM inference
docker exec -it rapp_vam bash
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
ros2 launch vam_inference vam_tm12s_robot.launch.py
```

## Launch Files

| Launch File | Mode | Use Case |
|---|---|---|
| `vam_tm12s_headless.launch.py` | headless | Inference + robot_state_publishers, no RViz (run RViz from rapp_hw) |
| `vam_tm12s_robot.launch.py` | robot | VAM inference + ghost robot state publisher (no MoveIt Servo) |
| `tm12s_moveit_hw.launch.py` | robot | MoveIt + TM driver + RViz for real hardware (replaces upstream `tm12s_moveit.launch.py`) |

**Standalone node** (run separately in rapp_hw — needs `tm_msgs` and MoveIt planning scene):

| Node | Purpose |
|---|---|
| `vam_pvt_streamer` | Receives `/vam/joint_targets` → collision-checks via MoveIt → streams PVT points to TM12S at 10Hz. Seeds from current robot position and ramps velocity over ~1 second. |

### Key Parameters

**VAM inference** (`vam_tm12s_robot.launch.py`):

| Parameter | Default | Description |
|---|---|---|
| `prediction_stride_K` | 1 | Re-predict every K frames |
| `ensemble_decay_weight` | 0.5 | Temporal smoothing (lower = smoother) |
| `max_joint_velocity_rad_s` | 2.0 | SafetyChecker pre-filter velocity limit (rad/s) |
| `max_joint_acceleration_rad_s2` | 5.0 | SafetyChecker pre-filter acceleration limit (rad/s²) |
| `tracking_timeout_sec` | 0.5 | Skeleton loss detection threshold |
| `target_skeleton_id` | -1 | Skeleton to track (-1 = auto) |
| `trajectory_lookahead_frames` | 5 | Frames to look ahead in trajectory |

**PVT streamer** (`vam_pvt_streamer`):

| Parameter | Default | Description |
|---|---|---|
| `pvt_rate_hz` | 10.0 | PVT point send rate (Hz). Each point = 1/rate seconds. |
| `velocity_scale` | 0.3 | Fraction of TM12S hardware velocity limits (0.0–1.0). Start low! |
| `catch_up_threshold_rad` | 0.3 | Position gap (rad) that triggers MoveIt catch-up trajectory |
| `catch_up_velocity_scale` | 1.0 | MoveIt velocity scaling during catch-up (0.0–1.0) |
| `watchdog_timeout_sec` | 0.5 | Hold position if no target for this long |
| `holding_timeout_sec` | 2.0 | Return to IDLE after holding this long |
| `collision_perturbation_rad` | 0.08 | Perturbation radius for finding collision-free alternatives |
| `collision_candidates` | 8 | Number of random alternatives to try when target is in collision |

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

Robot mode uses a **layered safety approach** — the PVT streamer handles motion safety, MoveIt provides collision checking, and the TM12S firmware enforces hardware limits:

**PVT Streamer (motion control — 10Hz):**

1. **Velocity clamping** — Per-joint limits at configurable fraction of hardware max (default 30%)
2. **Acceleration ramping** — Velocity changes by max 10% of max_vel per tick (~1 second to reach full speed from standstill)
3. **Position seeding** — First PVT point always matches robot's current position with zero velocity (prevents CPERR 241 jumps)
4. **Watchdog** — Holds position if no VAM target for 500ms
5. **Catch-up via MoveIt** — If robot is far from target (>0.3 rad), uses MoveIt trajectory planning instead of PVT streaming

**MoveIt Planning Scene (collision checking):**

1. **Self-collision checking** — Every PVT target is validated via `/check_state_validity`
2. **Alternative path finding** — When target is in collision, tries nearby collision-free positions
3. **Persistent collision hold** — After 5 consecutive collisions, holds position

**TM12SSafetyChecker (pre-filter in VAM node — 15Hz):**

1. **Joint limit clamping** — TM12S URDF limits (joint_3 narrower: ±162°)
2. **Per-joint velocity limits** — [2.27, 2.27, 3.67, 3.93, 3.93, 7.85] rad/s

**TM12S firmware (final layer):**

- Rejects commands violating hardware limits (CPERR errors)
- Cubic spline interpolation between PVT points at 1kHz+ servo rate

## TM12S-Specific Configuration Files

| File | Purpose |
|---|---|
| `ros2_ws/src/vam_inference/vam_inference/vam_pvt_streamer.py` | PVT streamer node (VAM targets → TM12S PVT mode) |
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

**Status:** Pipeline operational. Direct PVT streaming verified on real TM12S hardware.
