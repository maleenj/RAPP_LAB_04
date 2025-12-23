# RAPP Lab 04 - Quick Reference Card

## One-Time Setup

```bash
# 1. Create directories
./scripts/setup_volumes.sh

# 2. Build container
./scripts/build_containers.sh

# 3. Configure calibration (IMPORTANT!)
cp config/calibration.yaml.example config/calibration.yaml
nano config/calibration.yaml  # Add your calibration values
```

## Daily Usage

```bash
# Start container
./scripts/run_jupyter.sh

# Access Jupyter
# → http://localhost:8888

# Access container shell
docker exec -it rapp_vam bash

# Stop container
cd docker && docker-compose down
```

## Common Commands

### Inside Container

```bash
# Build ROS2 workspace
cd /workspace/ros2_ws && colcon build --symlink-install

# Source ROS2 workspace
source /workspace/ros2_ws/install/setup.bash

# List ROS2 topics
ros2 topic list

# Check GPU
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"

# Start TensorBoard
tensorboard --logdir=/data/logs/tensorboard --host=0.0.0.0
```

### From Host

```bash
# View container logs
docker logs rapp_vam

# Check GPU access
./scripts/verify_gpu.sh

# Restart container
cd docker && docker-compose restart

# Rebuild container (after Dockerfile changes)
cd docker && docker-compose build --no-cache
```

## File Locations

### Configuration
- **Calibration:** `config/calibration.yaml` (EDIT THIS!)
- **Training:** `config/training.yaml`
- **Data processing:** `config/data_processing.yaml`
- **Inference:** `config/inference.yaml`

### Data (Host)
- **Rosbags:** `/home/maleen/rosbags/rapplab04/`
- **CSV files:** `/home/maleen/csvdata/rapplab04/`
- **Models:** `/home/maleen/models/rapplab04/`
- **Logs:** `/home/maleen/models/rapplab04/logs/`

### Data (Container)
- **Rosbags:** `/data/rosbags/`
- **CSV files:** `/data/processed/`
- **Models:** `/data/models/`
- **Config:** `/config/`

### Code
- **Utilities:** `vam_utils/` (auto-installed in container)
- **Notebooks:** `notebooks/` (access via Jupyter)
- **ROS2:** `ros2_ws/`

## Jupyter Notebooks Workflow

```
00_setup_environment.ipynb      → Verify setup
01_extract_urdf.ipynb           → Extract robot URDF
02_process_rosbags.ipynb        → Rosbag → CSV
03_data_exploration.ipynb       → Explore data
04_prepare_training_data.ipynb  → Create datasets
05_train_vam.ipynb              → Train model
06_evaluate_model.ipynb         → Evaluate results
07_test_inference.ipynb         → Test inference
```

## Python Package Usage

```python
# Import utilities (available in all notebooks)
from vam_utils.config import load_config
from vam_utils.kinematics import load_urdf, compute_fk
from vam_utils.visualization import plot_skeleton_and_robot
from vam_utils.data import VAMDataset

# Load configuration
config = load_config('/config/training.yaml')

# Forward kinematics
urdf = load_urdf('/config/ur10.urdf')
pose = compute_fk(joint_angles, urdf)

# Visualization
fig = plot_skeleton_and_robot(skeleton, joints, urdf_path)
```

## Troubleshooting Quick Fixes

### GPU Not Working
```bash
./scripts/verify_gpu.sh
# If fails, check: nvidia-smi on host
```

### Jupyter Not Accessible
```bash
docker logs rapp_vam | grep -i jupyter
# Check: http://localhost:8888
```

### ROS2 Topics Not Visible
```bash
docker exec rapp_vam bash -c 'echo $ROS_DOMAIN_ID'
# Should be: 0
```

### Permission Errors
```bash
sudo chown -R $USER:$USER /home/maleen/csvdata/rapplab04
sudo chown -R $USER:$USER /home/maleen/models/rapplab04
```

### Out of Memory
```bash
# Edit config/training.yaml
# Reduce: training.batch_size from 32 to 16
```

## Environment Variables

```bash
# Override config values
export VAM_TRAINING_BATCH_SIZE=16
export VAM_TRAINING_LEARNING_RATE=5e-5
export VAM_MODEL_CHECKPOINT=/data/models/vam_best.pth
```

## Docker Compose Commands

```bash
cd docker

# Start (detached)
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# Rebuild
docker-compose build

# View logs
docker-compose logs -f

# Remove everything (clean slate)
docker-compose down -v
docker rmi rapp_vam
```

## ROS2 Launch Commands

```bash
# Inside container
cd /workspace/ros2_ws
source install/setup.bash

# Launch inference node
ros2 launch vam_inference vam_inference.launch.py

# Launch with RViz
ros2 launch vam_inference vam_with_rviz.launch.py

# Test with rosbag
ros2 bag play /data/rosbags/test_recording.db3
```

## Useful Aliases (Inside Container)

Already configured in container:

```bash
build          # Build ROS2 workspace
source_ros     # Source ROS2 Humble
source_ws      # Source ROS2 workspace
```

## Port Reference

- **8888** - Jupyter Lab
- **6006** - TensorBoard
- **ROS2 DDS** - Host network (shared with ZED container)

## File Sync

Changes to code on host are immediately available in container:
- Edit: `vam_utils/visualization/skeleton_viz.py` on host
- Use: Restart Jupyter kernel, changes are live

## Getting Help

1. Check logs: `docker logs rapp_vam`
2. Run diagnostics: `./scripts/verify_gpu.sh`
3. Review: [SETUP.md](SETUP.md) for detailed troubleshooting
4. Check: [README.md](README.md) for full documentation
