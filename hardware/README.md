# RAPP Lab 04: TM12S Robot Operation Guide

TM12S deployment of the Vision-Action Model (VAM) pipeline. Uses skeleton-only models trained on native TM12S data with direct PVT streaming to the robot firmware.

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
| `vam/vam_utils/` | `/workspace/vam_utils` | Shared utilities |
| `notebooks/` | `/workspace/notebooks` | Jupyter notebooks |
| `scripts/` | `/workspace/scripts` | Helper scripts |
| `config/` | `/config` | Configuration files |
| `~/rosbags/rapplab04` | `/data/rosbags` | Rosbag recordings |
| `~/csvdata/rapplab04` | `/data/processed` | Processed data, URDFs, RViz configs |
| `~/models/rapplab04` | `/data/models` | Trained models |

The hardware container also auto-mounts the servo and controller configs into the TM2 install tree.

## Setup

### 1. Start Containers

```bash
xhost +local:docker

cd docker
docker compose -f docker-compose.hw.yml up -d   # hardware
docker compose up -d                              # VAM
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

### 3. Build Packages

```bash
cd /workspace/ros2_ws && \
colcon build --symlink-install --packages-select vam_interfaces vam_inference && \
source /workspace/ros2_ws/install/setup.bash
```

## Running the Pipeline

### Mode 1: RViz Visualization (rosbag replay)

Visualize model predictions on a ghost robot — no real robot needed.

```bash
# Terminal 1 (rapp_vam): Play rosbag with clock
ros2 bag play /data/rosbags/<name> --clock

# Terminal 2 (rapp_vam): Launch inference (headless, no RViz)
ros2 launch vam_inference vam_tm12s_headless.launch.py use_sim_time:=true

# Terminal 3 (rapp_hw): Open RViz with TM12S config
rviz2 -d /data/processed/vam_tm12s.rviz
```

Pick a specific model with `active_model:=N` or hot-swap at runtime — same registry + service as the robot launch (see [Model Management](#model-management)).

### Mode 2: Real Robot with Rosbag Skeleton Data

Control the real TM12S using skeleton data from a rosbag. Uses **direct PVT streaming** — VAM joint targets are sent to the TM12S firmware as PVT (Position-Velocity-Time) points. The firmware performs cubic spline interpolation between points at 1kHz+.

**Important:** Use `tm12s_moveit_hw.launch.py`, NOT `tm12s_moveit.launch.py` or `demo.launch.py` (those start a fake ros2_control stack that publishes competing zero joint states).

```bash
# Terminal 1 (rapp_hw): MoveIt + TM12S driver + RViz
ros2 launch vam_inference tm12s_moveit_hw.launch.py robot_ip:=192.168.10.2

# Terminal 2 (rapp_hw): PVT Streamer
ros2 run vam_inference vam_pvt_streamer --ros-args \
    -p velocity_scale:=0.15 \
    -p catch_up_threshold_rad:=0.3 \
    -p catch_up_velocity_scale:=0.1 \
    -p filter_cutoff_hz:=2.0

# Terminal 3 (rapp_vam): Play rosbag — skeleton topic ONLY (no --clock!)
ros2 bag play /data/rosbags/<name> \
    --topics /zed/zed_node/body_trk/skeletons --loop

# Terminal 4 (rapp_vam): VAM inference
ros2 launch vam_inference vam_tm12s_robot.launch.py \
    feedback_gain:=25.0 \
    feedback_max_vel:=50.0 \
    max_joint_velocity_rad_s:=20.0 \
    max_joint_acceleration_rad_s2:=50.0 \
    control_rate_hz:=100.0
```

The `--topics` filter is critical — it prevents the rosbag from publishing `/joint_states`, `/tf`, `/tf_static` which would conflict with the real robot. No `--clock` because the real robot operates in wall-clock time.

The static transforms (map->world, world->base, map->camera) are published by the launch file automatically.

### Mode 3: Live Performance (camera + robot)

Same as Mode 2, but replace the rosbag terminal with the ZED camera:

```bash
# Terminal 3 (rapp_hw): ZED camera (replaces rosbag)
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

### Mode 4: Live Camera Preview (virtual robot)

Live ZED camera driving a **ghost** TM12S in RViz — no real robot, no PVT streamer. Useful for verifying a model against a live performer before running it on hardware, or for demos without the arm.

```bash
# Terminal 1 (rapp_hw): Real ZED camera with body tracking
ros2 launch vam_inference zed2i_camera.launch.py

# Terminal 2 (rapp_vam): VAM inference + ghost robot (wall-clock, no rosbag)
ros2 launch vam_inference vam_tm12s_headless.launch.py use_sim_time:=false

# Terminal 3 (rapp_hw): RViz with TM12S config
rviz2 -d /data/processed/vam_tm12s.rviz
```

Notes:

- `use_sim_time:=false` is required — live ZED runs on wall-clock, unlike the rosbag-replay path in Mode 1.
- The launch publishes a static `map → zed_camera_link` identity transform. Fine for ghost-only sanity-checking; **do not** reuse this transform when running the real arm — calibrate it first.
- No PVT streamer, no collision check, no velocity/acceleration limits are enforced by the streamer (none of those nodes are running). The only safety in this mode is the VAM node's joint-limit clamping.
- Pick or hot-swap the model the same way as Mode 1 — `active_model:=N` at launch or `/vam/switch_model` at runtime.
- A second non-prefixed "real" robot model may appear in RViz at the zero pose because nothing publishes `/joint_states`. Hide that RobotModel display if it's distracting.

## Model Management

Models are defined in a YAML registry at `ros2_ws/src/vam_inference/config/vam_models.yaml`. Each entry specifies a model directory (containing `best.pt` and `model_config.json`) and the corresponding `norm_stats.pt` path.

### Registry Format

```yaml
models:
  1:
    name: "mirror_pom"
    description: "Mirror actor - POM trained 2026-04-05"
    model_dir: "/data/models/vam_skelonly_tm12_mirror_pom20260405_0314"
    norm_stats_path: "/data/processed/tensors/2026_04_05_tm12_mirror_pom/norm_stats.pt"
  2:
    name: "mirror_basic"
    description: "Mirror actor - basic training"
    model_dir: "/data/models/vam_skelonly_tm12_20260404_0631"
    norm_stats_path: "/data/processed/tensors/2026_04_04_tm12/norm_stats.pt"
```

To add a new model, add an entry with the next ID and rebuild (`colcon build`).

### Selecting a Model at Launch

```bash
# Default (model 1):
ros2 launch vam_inference vam_tm12s_robot.launch.py

# Specific model:
ros2 launch vam_inference vam_tm12s_robot.launch.py active_model:=2
```

### Switching Models at Runtime

Switch models mid-operation without restarting the node:

```bash
ros2 service call /vam/switch_model vam_interfaces/srv/SwitchModel "{model_id: 2}"
```

The switch is safe for the real robot:
1. Inference pauses (no targets published)
2. PVT streamer detects the gap and holds the robot's current position
3. New model, normalization stats, and pipeline state are loaded
4. Inference resumes — the feedback filter ramps smoothly from the current position to the new model's predictions

The terminal running the inference node logs which model is active:

```
[vam_tm12s_inference_node] === Active model: #1 "mirror_pom" - Mirror actor - POM trained 2026-04-05 ===
[vam_tm12s_inference_node] Model switch requested: #1 "mirror_pom" -> #2 "mirror_basic"
[vam_tm12s_inference_node] Model swap complete, resuming inference.
```

## Tuning Parameters

### VAM Inference Node (`vam_tm12s_robot.launch.py`)

These control how the model's raw predictions are processed before being sent to the PVT streamer.

| Parameter | Default | Tested Good | Effect |
|---|---|---|---|
| `control_rate_hz` | 30.0 | **100.0** | Main loop frequency. Higher = smoother output, finer feedback control. Skeleton input is 15Hz; the node reuses the latest frame between updates. |
| `feedback_gain` | 3.0 | **25.0** | P-controller gain (Kp). Higher = predictions tracked more tightly. Lower = sluggish but smoother. Too high can cause oscillation. |
| `feedback_max_vel` | 1.4 | **50.0** | Max output velocity of the feedback filter (rad/s). Caps how fast the filtered target can move per tick. Set high to avoid being the bottleneck when PVT velocity_scale is also set. |
| `max_joint_velocity_rad_s` | 2.0 | **20.0** | SafetyChecker velocity limit (rad/s). Pre-filter clamp applied before the feedback smoother. Set high so the safety checker doesn't fight the feedback filter. |
| `max_joint_acceleration_rad_s2` | 5.0 | **50.0** | SafetyChecker acceleration limit. Same idea — set high to let the feedback filter and PVT streamer be the actual rate limiters. |
| `filter_type` | `feedback` | `feedback` | Smoothing filter type: `feedback` (P-controller from actual joints, recommended), `one_euro`, or `ema`. |
| `prediction_stride_K` | 1 | 1 | Re-predict every K frames. K=1 is smoothest (every frame). Higher K = less GPU usage but chunkier output. |
| `ensemble_decay_weight` | 0.5 | 0.5 | Temporal ensemble blending. Lower = smoother (more averaging of overlapping chunks). Higher = more responsive. |
| `target_skeleton_id` | -1 | -1 | Which skeleton to track. -1 = first detected. Set to a specific ID (e.g., 5, 22) if multiple people are visible. |
| `tracking_timeout_sec` | 0.5 | 0.5 | Seconds without a skeleton before pausing inference. Robot holds position during the gap. |

**Smoothing filter variants** (only relevant if you change `filter_type`):

| Parameter | Default | Used with | Effect |
|---|---|---|---|
| `one_euro_min_cutoff` | 0.4 | `one_euro` | Min cutoff Hz. Lower = heavier smoothing when slow. |
| `one_euro_beta` | 0.1 | `one_euro` | Speed coefficient. Higher = more responsive to fast motion. |
| `one_euro_d_cutoff` | 1.0 | `one_euro` | Derivative cutoff Hz. Usually leave at 1.0. |
| `smoothing_alpha` | 0.3 | `ema` | EMA factor. Lower = smoother but more lag. |
| `deadzone_rad` | 0.12 | `one_euro`, `ema` | Stillness deadzone (rad). Changes below this are suppressed. Not used with `feedback` filter. |
| `still_frames_threshold` | 2 | `one_euro`, `ema` | Frames within deadzone before locking position. |

### PVT Streamer (`vam_pvt_streamer`)

These control how the robot actually moves. The PVT streamer is the last software layer before the TM12S firmware.

| Parameter | Default | Tested Good | Effect |
|---|---|---|---|
| `velocity_scale` | 0.3 | **0.15** | Fraction of TM12S hardware velocity limits (0.0-1.0). This is the primary speed knob. 0.15 = 15% of max speed. Start low, increase gradually. |
| `filter_cutoff_hz` | 2.0 | **2.0** | Butterworth low-pass filter cutoff (Hz). Removes high-frequency jitter from VAM targets before computing velocity commands. Lower = smoother but more lag. |
| `catch_up_threshold_rad` | 0.3 | **0.3** | Position gap (rad, ~17deg) that triggers MoveIt catch-up trajectory instead of PVT streaming. Prevents large jumps. |
| `catch_up_velocity_scale` | 1.0 | **0.1** | MoveIt velocity scaling during catch-up (0.0-1.0). Lower = slower, safer catch-up motion. |
| `pvt_rate_hz` | 15.0 | 15.0 | PVT point streaming rate. Higher = smoother robot motion but more communication overhead. |
| `accel_scale` | 0.3 | 0.3 | Fraction of hardware acceleration limits. Limits how quickly velocity can change between PVT points. |
| `watchdog_timeout_sec` | 0.5 | 0.5 | Seconds without a fresh target before holding position. Protects against node crashes or network issues. |
| `holding_timeout_sec` | 2.0 | 2.0 | Seconds in HOLDING state before returning to IDLE and exiting PVT mode. |
| `collision_perturbation_rad` | 0.08 | 0.08 | Perturbation radius for finding collision-free alternatives when target is in self-collision. |
| `collision_candidates` | 8 | 8 | Number of random nearby positions to try when target is in collision. |

### Tuning Philosophy

The tested values above reflect a philosophy where:

- **Safety checker limits are set high** (`max_joint_velocity=20`, `max_accel=50`) so they don't interfere — the feedback filter and PVT streamer are the actual motion limiters.
- **Feedback gain is aggressive** (`25.0`) for tight tracking with high control rate (`100 Hz`).
- **Feedback max_vel is uncapped** (`50.0`) so it doesn't bottleneck the P-controller.
- **PVT velocity_scale is conservative** (`0.15`) — this is the real speed limit on the robot.
- **Catch-up velocity is low** (`0.1`) — if the robot falls behind, it catches up slowly and safely.
- **Butterworth filter at 2 Hz** removes jitter without adding noticeable lag.

The key insight: let the feedback loop run fast and responsive, and use `velocity_scale` on the PVT streamer as the single speed knob for the actual robot.

## Launch Files

| Launch File | Mode | Use Case |
|---|---|---|
| `vam_tm12s_headless.launch.py` | rviz | Inference + ghost robot, no real arm. Used by Mode 1 (rosbag) and Mode 4 (live ZED). Supports `active_model:=N` and `/vam/switch_model`. |
| `vam_tm12s_robot.launch.py` | robot | VAM inference + ghost robot + static transforms (no MoveIt Servo) |
| `tm12s_moveit_hw.launch.py` | robot | MoveIt + TM driver + RViz for real hardware |
| `zed2i_camera.launch.py` | rviz/robot | ZED2i with body tracking + RAPP override config |

**Standalone node** (run separately in rapp_hw):

| Node | Purpose |
|---|---|
| `vam_pvt_streamer` | Receives `/vam/joint_targets` -> collision-checks via MoveIt -> streams PVT points to TM12S. |

## Safety Architecture

Layered safety — each layer is independent:

**1. VAM Inference Node (100 Hz):**
- Joint limit clamping (TM12S URDF limits, joint_3 narrower at +/-162deg)
- Per-joint velocity and acceleration limiting (SafetyChecker)
- Feedback P-controller bounds output to actual robot position

**2. PVT Streamer (15 Hz):**
- Butterworth low-pass filtering of target signals
- Velocity clamping at fraction of hardware limits (`velocity_scale`)
- Acceleration ramping (~1 second to reach full speed from standstill)
- Position seeding (first PVT point matches current robot position with zero velocity)
- Watchdog (holds position if no target for 500ms)
- Self-collision checking via MoveIt `/check_state_validity`
- Catch-up via MoveIt trajectory planning for large gaps

**3. TM12S Firmware (final layer):**
- Rejects commands violating hardware limits (CPERR errors)
- Cubic spline interpolation between PVT points at 1kHz+ servo rate

## Key Differences from UR10

| Property | UR10 | TM12S |
|---|---|---|
| Joint names | `shoulder_pan_joint` .. `wrist_3_joint` | `joint_1` .. `joint_6` |
| MoveIt group | `ur_manipulator` | `tmr_arm` |
| Planning frame | `base_link` | `base` |
| EE frame | `tool0` | `flange` |
| Robot IP | 10.0.0.89 | 192.168.10.2 |
| Driver | `ur_robot_driver` | `tm_driver` (tm2_ros2) |

## Sign Conventions

The model outputs joint angles in TM12S space directly (native training data uses identity mapping). If a joint bends the wrong way, flip its sign multiplier:

```bash
ros2 launch vam_inference vam_tm12s_robot.launch.py \
    joint_sign_multipliers:="[1.0, -1.0, 1.0, 1.0, 1.0, 1.0]"
```

| Index | Joint | Function |
|---|---|---|
| 0 | joint_1 | Base rotation |
| 1 | joint_2 | Shoulder |
| 2 | joint_3 | Elbow (+/-162deg, narrower than UR10) |
| 3 | joint_4 | Wrist 1 |
| 4 | joint_5 | Wrist 2 |
| 5 | joint_6 | Wrist 3 |

## Configuration Files

| File | Purpose |
|---|---|
| `ros2_ws/src/vam_inference/config/vam_models.yaml` | Model registry (model dirs + norm stats paths) |
| `ros2_ws/src/vam_inference/vam_inference/vam_pvt_streamer.py` | PVT streamer node |
| `ros2_ws/src/vam_inference/vam_inference/vam_tm12s_node.py` | VAM inference node |
| `ros2_ws/src/vam_interfaces/srv/SwitchModel.srv` | Model switch service definition |
| `vam/vam_utils/data/robot_configs.py` | TM12S joint names, limits, velocities |
| `docker/docker-compose.hw.yml` | Hardware container compose file |
| `docker/docker-compose.yml` | VAM container compose file |

## Container Management

```bash
# Start
cd docker
docker compose -f docker-compose.hw.yml up -d   # hardware
docker compose up -d                              # VAM

# Shell access
docker exec -it rapp_hw bash
docker exec -it rapp_vam bash

# Stop
docker compose -f docker-compose.hw.yml down
docker compose down
```

## Troubleshooting

### TF lookup fails for 'base' frame
The TM12S uses `base` as its root frame (not `base_link` like UR10). The launch file publishes the required static transforms automatically. Verify with:
```bash
ros2 run tf2_ros tf2_echo base link_0
```

### forward_position_controller won't load
The servo and controller configs are auto-mounted by docker-compose.hw.yml into the tm2_ws install tree. If something is wrong:
```bash
docker exec rapp_hw cat /tm2_ws/install/tm12s_moveit_config/share/tm12s_moveit_config/config/ros2_controllers.yaml
```
Then restart MoveIt.

### Joint moves in wrong direction
Flip that joint's sign multiplier. See [Sign Conventions](#sign-conventions).

### ROS2 topics not visible between containers
Both containers must use the same ROS_DOMAIN_ID and RMW:
```bash
echo $ROS_DOMAIN_ID        # should be 0
echo $RMW_IMPLEMENTATION   # should be rmw_cyclonedds_cpp
```

### Elbow range clamping warnings
TM12S joint_3 has narrower range (+/-162deg) than UR10 elbow (+/-180deg). The safety checker clamps automatically. Occasional warnings are expected and harmless.

### CPERR 241
Position jump error from the TM12S firmware. Usually caused by:
- Safety checker velocity limit fighting the feedback filter (fix: raise `max_joint_velocity_rad_s`)
- PVT streamer velocity too high for the current gap (fix: lower `velocity_scale`)
- Model switch without proper pipeline reset (fix: use the `/vam/switch_model` service)

---

**Status:** Pipeline operational. Direct PVT streaming with model hot-swap verified on real TM12S hardware.
