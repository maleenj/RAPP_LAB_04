# RAPP Lab 04 - Setup Instructions

Step-by-step guide to get your Vision-Action Model development environment running.

## Prerequisites

Ensure you have on your host machine:
- Ubuntu 22.04
- NVIDIA GPU driver installed (535+ recommended)
- Docker installed
- Docker Compose installed
- NVIDIA Container Toolkit installed

### Verify Prerequisites

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker
docker --version
docker compose version

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If any of these fail, install the missing components first.

## Step 1: Create Data Directories

```bash
cd /home/maleen/git/RAPP_LAB_04
./scripts/setup_volumes.sh
```

This creates:
- `/home/maleen/rosbags/rapplab04` - Store your rosbag files here
- `/home/maleen/csvdata/rapplab04` - Processed CSV outputs
- `/home/maleen/models/rapplab04` - Model checkpoints
- `/home/maleen/models/rapplab04/logs` - Training logs

**ACTION REQUIRED:** Copy your rosbag files to `/home/maleen/rosbags/rapplab04/`

## Step 2: Configure Calibration

```bash
# Copy example calibration file
cp config/calibration.yaml.example config/calibration.yaml

# Edit with your actual calibration values
nano config/calibration.yaml
```

Update the camera-to-robot transform with your calibration data:

```yaml
camera_to_robot_transform:
  translation:
    x: 1.234  # Replace with your values
    y: -0.567
    z: 0.890
  rotation:
    x: 0.0    # Quaternion [x, y, z, w]
    y: 0.0
    z: 0.0
    w: 1.0
```

## Step 3: Build Docker Container

```bash
./scripts/build_containers.sh
```

This will take 10-15 minutes on first build. It installs:
- Ubuntu 22.04 base
- CUDA 12.1 + cuDNN 8
- ROS2 Humble Desktop (full)
- PyTorch 2.1.2 with CUDA support
- JupyterLab + all ML dependencies
- Robotics libraries (yourdfpy, roboticstoolbox-python)

**Grab a coffee while it builds!**

## Step 4: Enable X11 for GUI Applications

If you want to use RViz, matplotlib windows, etc. from the container:

```bash
# Allow local Docker containers to access X server
xhost +local:docker
```

Add this to your `~/.bashrc` to make it permanent:

```bash
echo "xhost +local:docker > /dev/null 2>&1" >> ~/.bashrc
```

## Step 5: Start Container

```bash
./scripts/run_jupyter.sh
```

Wait 3-5 seconds for Jupyter to start, then access:
- **Jupyter Lab:** http://localhost:8888 (no password)
- **TensorBoard:** http://localhost:6006

## Step 6: Verify GPU Access

```bash
./scripts/verify_gpu.sh
```

You should see:
- ✓ GPU detected via nvidia-smi
- ✓ PyTorch CUDA available: True
- ✓ ROS2 Humble installed

## Step 7: Access Container Shell

```bash
docker exec -it rapp_vam bash

# Inside container, verify vam_utils installed
python3 -c "import vam_utils; print(vam_utils.__version__)"
```

## Step 8: Start Working with Notebooks

Open Jupyter Lab at http://localhost:8888 and navigate to `/workspace/notebooks/`

Start with:
1. `00_setup_environment.ipynb` - Verify everything works
2. `01_extract_urdf.ipynb` - Extract robot URDF from rosbag
3. `02_process_rosbags.ipynb` - Process your data

## Common Issues & Solutions

### Issue: GPU Not Detected

**Solution:**

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If this fails, reinstall NVIDIA Container Toolkit:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: Permission Denied on Volumes

**Solution:**

```bash
sudo chown -R $USER:$USER /home/maleen/rosbags/rapplab04
sudo chown -R $USER:$USER /home/maleen/csvdata/rapplab04
sudo chown -R $USER:$USER /home/maleen/models/rapplab04
```

### Issue: Port 8888 Already in Use

**Solution:**

```bash
# Check what's using the port
sudo lsof -i :8888

# Kill the process or change the port in docker-compose.yml
nano docker/docker-compose.yml
# Change "8888:8888" to "8889:8888"
# Then access at http://localhost:8889
```

### Issue: ROS2 Topics Not Visible from ZED Container

**Solution:**

```bash
# Verify both containers on same domain
docker exec rapp_vam bash -c 'echo $ROS_DOMAIN_ID'
docker exec rapp_zed bash -c 'echo $ROS_DOMAIN_ID'

# Both should be: 0

# If not, check docker/.env file:
cat docker/.env | grep ROS_DOMAIN_ID
```

### Issue: Out of Memory During Training

**Solution:**

```bash
# Reduce batch size in config/training.yaml
nano config/training.yaml
# Change batch_size from 32 to 16 or 8

# Or set via environment:
docker exec rapp_vam bash -c 'export VAM_TRAINING_BATCH_SIZE=16'
```

### Issue: Container Keeps Restarting

**Solution:**

```bash
# Check logs
docker logs rapp_vam

# Common causes:
# 1. GPU driver incompatibility - update NVIDIA driver
# 2. Insufficient resources - check with: docker stats
# 3. Entrypoint script error - check syntax in docker/vam/entrypoint.sh
```

## Development Workflow

### 1. Edit Code (on host)

```bash
# Edit files in your favorite editor
code /home/maleen/git/RAPP_LAB_04/vam_utils/visualization/skeleton_viz.py
```

### 2. Changes Available Immediately

```python
# In Jupyter notebook, just restart kernel and re-import
from vam_utils.visualization import skeleton_viz  # Gets latest code
```

### 3. Test Changes

```bash
# Run tests in container
docker exec -it rapp_vam bash
cd /workspace
pytest tests/ -v
```

## Next Steps

Once setup is complete:

1. **Familiarize yourself** with the [README.md](README.md)
2. **Review the design brief** [RAPP_Lab_04_Design_Brief.md](RAPP_Lab_04_Design_Brief.md)
3. **Start with Jupyter notebooks** at http://localhost:8888
4. **Process your first rosbag** using `02_process_rosbags.ipynb`

## Getting Help

- Check logs: `docker logs rapp_vam`
- Run diagnostics: `./scripts/verify_gpu.sh`
- Test ROS2: `docker exec rapp_vam ros2 topic list`
- Check GPU: `docker exec rapp_vam nvidia-smi`

## Clean Rebuild (if needed)

```bash
# Stop and remove container
cd docker
docker-compose down

# Remove image
docker rmi rapp_vam

# Rebuild from scratch
./scripts/build_containers.sh
```

---

**Setup Status Checklist:**

- [ ] Prerequisites verified (GPU, Docker, NVIDIA runtime)
- [ ] Data directories created (`./scripts/setup_volumes.sh`)
- [ ] Rosbag files copied to `/home/maleen/rosbags/rapplab04/`
- [ ] Calibration configured (`config/calibration.yaml`)
- [ ] Docker container built (`./scripts/build_containers.sh`)
- [ ] X11 access enabled (`xhost +local:docker`)
- [ ] Container running (`./scripts/run_jupyter.sh`)
- [ ] GPU verified (`./scripts/verify_gpu.sh`)
- [ ] Jupyter accessible (http://localhost:8888)
- [ ] `vam_utils` package installed (check in notebook)

**Once all boxes are checked, you're ready to start development!**
