# RAPP Lab 04: Vision-Action Model for Human-Robot Ensemble Performance

Real-time improvised physical theatre through Action Chunking Transformers. Teaching a UR10 robot to mirror and respond to human performers using skeletal tracking and learned movement patterns.

**Project Lead:** Dr. Maleen Jayasuriya
**Institution:** University of Canberra - Collaborative Robotics Lab
**Research Context:** Building on RAPP Lab 03's biomechanics work, introducing AI-enabled improvisation for genuine ensemble partnership.

## Quick Start

### 1. Setup Directories

```bash
# Create host directories for data storage
./scripts/setup_volumes.sh
```

### 2. Build Docker Container

```bash
# Build the VAM container (10-15 minutes first time)
./scripts/build_containers.sh
```

### 3. Start Container

```bash
# Start Jupyter Lab and services
./scripts/run_jupyter.sh

# Access Jupyter at: http://localhost:8888
# Access TensorBoard at: http://localhost:6006
```

### 4. Verify GPU

```bash
# Check GPU accessibility in container
./scripts/verify_gpu.sh
```

## System Overview

### Container Architecture

```
┌─────────────────────────────────────────────┐
│         RAPP Lab 04 Infrastructure          │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │ ZED2 Docker  │      │  VAM Docker  │   │
│  │ (Existing)   │◄────►│   (New)      │   │
│  │              │ ROS2 │              │   │
│  │ - ZED SDK    │DDS  │ - ROS2 Desktop│   │
│  │ - UR Driver  │      │ - PyTorch    │   │
│  │ - Tracking   │      │ - Jupyter    │   │
│  └──────────────┘      └──────────────┘   │
│                                             │
│  Host Volumes:                              │
│  ├─ /home/maleen/rosbags/rapplab04         │
│  ├─ /home/maleen/csvdata/rapplab04         │
│  └─ /home/maleen/models/rapplab04          │
└─────────────────────────────────────────────┘
```

### VAM Container Specifications

- **OS:** Ubuntu 22.04
- **CUDA:** 12.1 with cuDNN 8 (RTX 5070 Ti compatible)
- **ROS2:** Humble Desktop (full installation)
- **PyTorch:** 2.1.2 with CUDA 12.1
- **Python:** 3.10
- **JupyterLab:** Latest with full ML stack

## Project Structure

```
RAPP_LAB_04/
├── vam_utils/           # Installable Python package (modular utilities)
├── notebooks/           # Jupyter notebooks (development)
├── config/              # YAML configuration files
├── docker/              # Container definitions
├── scripts/             # Setup and utility scripts
├── ros2_ws/             # ROS2 workspace (inference node)
└── RAPP_Lab_04_Design_Brief.md
```

## Workflow

### Phase 1: Data Processing

```bash
# 1. Start container and access Jupyter
./scripts/run_jupyter.sh

# 2. Run notebooks in order:
#    - 01_extract_urdf.ipynb        → Extract robot URDF
#    - 02_process_rosbags.ipynb     → Convert rosbags to CSV
#    - 03_data_exploration.ipynb    → Explore and validate data
```

**Output:** Synchronized CSV files in `/home/maleen/csvdata/rapplab04/`

### Phase 2: Training

```bash
# 3. Continue with training notebooks:
#    - 04_prepare_training_data.ipynb  → Create PyTorch datasets
#    - 05_train_vam.ipynb              → Train Action Chunking Transformer
#    - 06_evaluate_model.ipynb         → Evaluate and visualize results
```

**Output:** Trained models in `/home/maleen/models/rapplab04/`

### Phase 3: Deployment

```bash
# 4. Test inference offline:
#    - 07_test_inference.ipynb         → Test with rosbag playback

# 5. Deploy to robot:
docker exec -it rapp_vam bash
cd /workspace/ros2_ws
colcon build --packages-select vam_inference
source install/setup.bash
ros2 launch vam_inference vam_inference.launch.py
```

## Container Management

### Start Container

```bash
cd docker
docker-compose up -d
```

### Stop Container

```bash
cd docker
docker-compose down
```

### Access Container Shell

```bash
docker exec -it rapp_vam bash
```

### View Logs

```bash
docker logs rapp_vam
```

### Restart Container

```bash
cd docker
docker-compose restart
```

## Data Locations

### Host Machine (persistent storage)

- **Rosbags:** `/home/maleen/rosbags/rapplab04/` (read-only in container)
- **Processed CSV:** `/home/maleen/csvdata/rapplab04/` (read-write)
- **Models:** `/home/maleen/models/rapplab04/` (read-write)
- **Logs:** `/home/maleen/models/rapplab04/logs/` (read-write)

### Inside Container

- **Rosbags:** `/data/rosbags/` (mounted read-only)
- **Processed CSV:** `/data/processed/` (mounted read-write)
- **Models:** `/data/models/` (mounted read-write)
- **Config:** `/config/` (mounted read-only)
- **Code:** `/workspace/vam_utils/`, `/workspace/notebooks/`

## Code Reuse Architecture

The `vam_utils` package provides modular, reusable components:

```python
# Example: Using shared visualization across notebooks
from vam_utils.visualization import plot_skeleton_and_robot
from vam_utils.kinematics import load_urdf, compute_fk

# Same code works in data processing, training, and inference
fig = plot_skeleton_and_robot(skeleton_data, joint_angles, urdf_path)
```

**Key utilities used everywhere:**
- Skeleton visualization (notebooks 02, 06, 07)
- Forward kinematics (notebooks 01, 02, 03, 06, 07, inference)
- PyTorch Dataset (notebooks 04, 05, 06)
- Configuration loading (all notebooks + inference)

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver on host
nvidia-smi

# Verify GPU in container
./scripts/verify_gpu.sh

# If issues persist, check Docker NVIDIA runtime:
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### ROS2 Communication Issues

```bash
# Ensure both containers use same ROS_DOMAIN_ID
docker exec rapp_vam bash -c 'echo $ROS_DOMAIN_ID'
docker exec rapp_zed bash -c 'echo $ROS_DOMAIN_ID'

# Both should show: 0

# Test ROS2 topic visibility
docker exec rapp_vam bash -c 'source /opt/ros/humble/setup.bash && ros2 topic list'
```

### Jupyter Not Accessible

```bash
# Check if container is running
docker ps | grep rapp_vam

# Check Jupyter logs
docker logs rapp_vam | grep -i jupyter

# Access on: http://localhost:8888 (no password required)
```

### Permission Issues

```bash
# If you get permission errors on mounted volumes:
sudo chown -R $USER:$USER /home/maleen/csvdata/rapplab04
sudo chown -R $USER:$USER /home/maleen/models/rapplab04
```

## Configuration

Configuration files use YAML with environment variable overrides:

```yaml
# config/training.yaml
model:
  T_in: 10   # Input history frames
  T_out: 50  # Prediction horizon

training:
  batch_size: 32
  learning_rate: 1.0e-4
```

Override via environment:

```bash
export VAM_TRAINING_BATCH_SIZE=64
export VAM_TRAINING_LEARNING_RATE=5e-5
```

## Development

### Installing vam_utils in Editable Mode

```bash
# Inside container (automatic via entrypoint)
cd /workspace/vam_utils
pip install -e .

# Changes to vam_utils/*.py immediately available to notebooks
```

### Running Tests

```bash
docker exec -it rapp_vam bash
cd /workspace
pytest tests/ -v
```

### Building ROS2 Workspace

```bash
docker exec -it rapp_vam bash
cd /workspace/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Technical Specifications

### Model Architecture

- **Type:** Action Chunking Transformer (ACT)
- **Input:** 10 frames of skeleton (48D) + robot state (6D)
- **Output:** 50 frames of predicted robot joint angles (6D)
- **Skeleton Encoder:** 6-layer Transformer (8 heads, 256D)
- **Robot Encoder:** MLP (64→128)
- **Decoder:** 4-layer Transformer (8 heads)

### Hardware Requirements

- **GPU:** NVIDIA RTX 5070 Ti (or similar, CUDA 12.x compatible)
- **RAM:** 16GB minimum (32GB recommended)
- **Storage:** 100GB+ for rosbags, models, and data
- **OS:** Ubuntu 22.04 (host)

## Documentation

- **Design Brief:** [RAPP_Lab_04_Design_Brief.md](RAPP_Lab_04_Design_Brief.md) - Comprehensive technical overview
- **Implementation Plan:** `.claude/plans/peppy-jingling-wave.md` - Detailed implementation roadmap
- **VAM Utils:** [vam_utils/README.md](vam_utils/README.md) - Package documentation

## License

MIT License - See LICENSE file for details

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{jayasuriya2025rapp,
  title={Vision-Action Models for Human-Robot Ensemble Performance},
  author={Jayasuriya, Maleen and Wijesundara, Piumi and Herath, Damith},
  booktitle={International Conference on Social Robotics},
  year={2025}
}
```

## Contact

**Dr. Maleen Jayasuriya**
University of Canberra - Collaborative Robotics Lab
Email: maleen@example.com

---

**Status:** Phase 1 - Infrastructure Setup Complete ✓
