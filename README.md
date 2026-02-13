# RAPP Lab 04: Vision-Action Model for Human-Robot Ensemble Performance

Real-time improvised physical theatre through Action Chunking Transformers. A UR10 robot mirrors and responds to human performers using skeletal tracking and learned movement patterns.

**Project Lead:** Dr. Maleen Jayasuriya, Piumi Wijesundara
**Institution:** University of Canberra - Collaborative Robotics Lab

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with driver 535+
- Docker and Docker Compose
- NVIDIA Container Toolkit

```bash
# Verify prerequisites
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Setup

### 1. Create directories and build container

```bash
./scripts/setup_volumes.sh
./scripts/build_containers.sh    # ~15 min first time
```

This creates host directories:
- `/home/maleen/rosbags/rapplab04/` - Rosbag files (read-only in container)
- `/home/maleen/csvdata/rapplab04/` - Processed CSV outputs
- `/home/maleen/models/rapplab04/` - Model checkpoints and logs

Copy rosbag files to `/home/maleen/rosbags/rapplab04/` before proceeding.

### 2. Start container

```bash
./scripts/run_jupyter.sh

# Access Jupyter: http://localhost:8888
# Access TensorBoard: http://localhost:6006
```

### 3. Verify

```bash
./scripts/verify_gpu.sh
```

### 4. Enable X11 for GUI (RViz, matplotlib)

```bash
xhost +local:docker
```

## Project Structure

```
RAPP_LAB_04/
├── vam_utils/               # Python package (config, data, model, inference)
├── notebooks/               # Jupyter notebooks (processing, training, evaluation)
├── ros2_ws/                 # ROS2 workspace
│   └── src/vam_inference/   # Inference node + launch files
├── config/                  # YAML configuration
├── docker/                  # Dockerfile + docker-compose
└── scripts/                 # Setup and utility scripts
```

## Workflow

### Phase 1: Data Processing

```bash
# In Jupyter (http://localhost:8888), run notebooks in order:
00_setup_urdf.ipynb           # Extract robot URDF from rosbag
01_process_rosbags.ipynb      # Convert rosbags → synchronized CSV
```

Output: CSV files in `/data/processed/csv/` with metadata in `recordings_metadata.csv`

### Phase 2: Training

```bash
02_prepare_training_data.ipynb  # Create PyTorch tensors with normalization
03_train_vam.ipynb              # Train Action Chunking Transformer
04_inference_test.ipynb         # Offline evaluation and parameter sweep
```

Output: Model checkpoint in `/data/models/vam_YYYYMMDD_HHMM/best.pt`

### Phase 3: ROS2 Inference

Build the ROS2 workspace (inside container):

```bash
docker exec -it rapp_vam bash
cd /workspace/ros2_ws
pip install setuptools==58.2.0    # one-time fix for --symlink-install
colcon build --symlink-install
source install/setup.bash
```

## Running the Inference Node

### Mode 1: RViz Visualization (rosbag replay)

Visualize model predictions against recorded data — no robot needed.

```bash
# Terminal 1: Play rosbag with clock
ros2 bag play /data/rosbags/<name> --clock

# Terminal 2: Launch VAM with RViz
ros2 launch vam_inference vam_rviz.launch.py use_sim_time:=true
```

### Mode 2: Real Robot with Rosbag Skeleton Data

Control the real UR10 using skeleton data from a rosbag (no live camera needed).
Uses MoveIt Servo for smooth 250Hz interpolation and self-collision avoidance.

```bash
# Terminal 1: UR10 driver
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur10 robot_ip:=10.0.0.89 reverse_port:=5002 launch_rviz:=true

# Terminal 2: Switch to forward_position_controller (once, after UR driver ready)
ros2 service call /controller_manager/switch_controller \
    controller_manager_msgs/srv/SwitchController \
    "{activate_controllers: ['forward_position_controller'], \
      deactivate_controllers: ['scaled_joint_trajectory_controller'], \
      strictness: 2}"

# Terminal 3: MoveIt Servo (from ZED container)
ros2 launch ur_moveit_config ur_moveit.launch.py \
    ur_type:=ur10 launch_servo:=true launch_rviz:=false

     ros2 service call /servo_node/start_servo std_srvs/srv/Trigger {}



# Terminal 4: Static transform — map → world (camera-to-robot, adjust for lab)
ros2 run tf2_ros static_transform_publisher \
    4.4 0.1 -0.25 1.5708 0 0 map world

# Terminal 5: Play rosbag — skeleton topic ONLY (no --clock!)
ros2 bag play /data/rosbags/<name> \
    --topics /zed/zed_node/body_trk/skeletons --loop

# Terminal 6: VAM inference (inside VAM container)
ros2 launch vam_inference vam_robot.launch.py
```

The `--topics` filter is critical — it prevents the rosbag from publishing
`/joint_states`, `/tf`, `/tf_static` which would conflict with the real robot.
No `--clock` because the real robot operates in wall-clock time.

**First-time safety:** Start with low P-gain for slow tracking:

```bash
ros2 launch vam_inference vam_robot.launch.py servo_proportional_gain:=2.0
```

### Mode 3: Live Performance (camera + robot)

```bash
# Terminal 1: UR10 driver
ros2 launch ur_robot_driver ur_control.launch.py \
    ur_type:=ur10 robot_ip:=10.0.0.89 reverse_port:=5002

# Terminal 2: Switch controller + MoveIt Servo (same as Mode 2, steps 2-3)

# Terminal 3: Static transform
ros2 run tf2_ros static_transform_publisher \
    4.4 0.1 -0.25 1.5708 0 0 map world

# Terminal 4: ZED camera (from ZED container)
# (ZED container publishes /zed/zed_node/body_trk/skeletons)

# Terminal 5: VAM inference
ros2 launch vam_inference vam_robot.launch.py
```

## Launch Files

| Launch File | Mode | Use Case |
|---|---|---|
| `vam_rviz.launch.py` | rviz | Visualization with rosbag replay (use_sim_time:=true) |
| `vam_robot.launch.py` | robot | Real robot control via MoveIt Servo |
| `vam_inference.launch.py` | configurable | Node only, no RViz or robot_state_publisher |

### Key Parameters

| Parameter | Default (rviz) | Default (robot) | Description |
|---|---|---|---|
| `servo_proportional_gain` | — | 5.0 | P-controller gain for MoveIt Servo (lower = slower) |
| `max_joint_velocity_rad_s` | 1.0 | 0.5 | SafetyChecker pre-filter velocity limit (rad/s) |
| `max_joint_acceleration_rad_s2` | 5.0 | 1.5 | SafetyChecker pre-filter acceleration limit (rad/s²) |
| `prediction_stride_K` | 1 | 1 | Re-predict every K frames |
| `ensemble_decay_weight` | 0.5 | 0.5 | Temporal smoothing (lower = smoother) |
| `tracking_timeout_sec` | 0.5 | 0.5 | Skeleton loss detection threshold |
| `target_skeleton_id` | -1 | -1 | Skeleton to track (-1 = auto) |

## Safety Architecture

Robot mode uses **MoveIt Servo** as the primary safety layer, with the VAM node's SafetyChecker as defense-in-depth:

**MoveIt Servo (primary — 250Hz):**

1. **Self-collision avoidance** — Full URDF collision model, scales/stops before collision
2. **Joint limit enforcement** — 0.1 rad margin from limits
3. **Singularity detection** — Scales velocity near singular configurations
4. **Butterworth smoothing** — Low-pass filter eliminates high-frequency jitter
5. **Auto-halt on timeout** — Stops robot if commands stop for >0.1s

**SafetyChecker (pre-filter — 15Hz):**

1. **Joint limit clamping** — Hard URDF limits (e.g., elbow: ±180°)
2. **Velocity limiting** — Caps max joint speed per step
3. **Acceleration limiting** — Prevents sudden speed changes

Additional protections:

- **SafetyChecker seeded** with robot's actual position on startup (prevents first-frame jump)
- **Hold-position on tracking loss** — Zero-velocity JointJog on skeleton timeout
- **Graceful shutdown** — Ctrl+C sends hold-position before exiting
- **UR10 built-in safety** — The robot controller rejects commands violating its internal limits

## Data Locations

| Location (Host) | Location (Container) | Contents |
|---|---|---|
| `/home/maleen/rosbags/rapplab04/` | `/data/rosbags/` | Raw rosbag recordings |
| `/home/maleen/csvdata/rapplab04/` | `/data/processed/` | CSV, tensors, URDF, metadata |
| `/home/maleen/models/rapplab04/` | `/data/models/` | Model checkpoints |

## Container Management

```bash
# Start
cd docker && docker-compose up -d

# Stop
cd docker && docker-compose down

# Shell access
docker exec -it rapp_vam bash

# View logs
docker logs rapp_vam

# Rebuild (after Dockerfile changes)
cd docker && docker-compose build --no-cache
```

## Troubleshooting

### GPU not detected

```bash
./scripts/verify_gpu.sh

# If Docker GPU fails, reinstall NVIDIA Container Toolkit:
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### ZED messages not found

The Dockerfile installs `ros-humble-zed-msgs` automatically. If you need to install manually in a running container:

```bash
docker exec -it rapp_vam bash
apt-get update && apt-get install -y ros-humble-zed-msgs
```

### ROS2 topics not visible between containers

```bash
# Both containers must use the same ROS_DOMAIN_ID (default: 0)
docker exec rapp_vam bash -c 'echo $ROS_DOMAIN_ID'
docker exec rapp_zed bash -c 'echo $ROS_DOMAIN_ID'
```

### Permission errors on mounted volumes

```bash
sudo chown -R $USER:$USER /home/maleen/csvdata/rapplab04
sudo chown -R $USER:$USER /home/maleen/models/rapplab04
```

### Jupyter not accessible

```bash
docker logs rapp_vam | grep -i jupyter
# Access at: http://localhost:8888 (no password)
```

### Out of memory during training

Reduce batch size in the training notebook or via environment:
```bash
export VAM_TRAINING_BATCH_SIZE=16
```

## Development

Code on the host is mounted into the container — edits take effect immediately (restart Jupyter kernel for Python changes).

```bash
# Build ROS2 workspace after changes
docker exec -it rapp_vam bash
cd /workspace/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

## Citation

```bibtex
@inproceedings{jayasuriya2025rapp,
  title={Vision-Action Models for Human-Robot Ensemble Performance},
  author={Jayasuriya, Maleen and Wijesundara, Piumi and Herath, Damith},
  booktitle={International Conference on Social Robotics},
  year={2025}
}
```

---

**Status:** Training complete. Real-time inference and robot deployment in progress.
