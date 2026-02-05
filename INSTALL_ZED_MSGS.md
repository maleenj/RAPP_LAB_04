# Installing ZED Messages for Skeleton Processing

To process skeleton data from the ZED camera rosbags, you need to install the `zed_msgs` ROS2 package in your Docker container.

## Important: Docker Image Updates

**Good news!** The Dockerfile has been updated to automatically include `zed_msgs` in future builds.

- **If you rebuild your Docker image**, zed_msgs will be automatically included
- **For your current running container**, use the quick setup below

## Quick Setup (Current Container)

### Option 1: Install via apt (Recommended - Fastest)

1. **Enter the running VAM Docker container:**
   ```bash
   docker exec -it rapp_vam bash
   ```

2. **Install the package:**
   ```bash
   apt-get update && apt-get install -y ros-humble-zed-msgs
   ```

3. **Restart your Jupyter kernel** (in JupyterLab: Kernel → Restart Kernel)

4. **Verify installation** - Run this in a notebook cell:
   ```python
   from zed_msgs.msg import ObjectsStamped
   print("zed_msgs installed successfully!")
   ```

### Option 2: Build from Source (Alternative)

If you need the absolute latest version or want to modify the messages:

1. **Enter the Docker container:**
   ```bash
   docker exec -it rapp_vam bash
   ```

2. **Run the setup script:**
   ```bash
   /workspace/scripts/setup_zed_interfaces.sh
   ```

This will clone and build from source. Then restart your Jupyter kernel.

## What Gets Installed

The `zed-ros2-interfaces` package provides all ZED camera message definitions including:

- `zed_msgs/msg/ObjectsStamped` - Contains detected skeletons
- `zed_msgs/msg/Object` - Individual skeleton object
- `zed_msgs/msg/Skeleton` - Skeleton keypoint data
- And many other ZED-specific message types

## Troubleshooting

### "zed_msgs not found" after installation

- Make sure you restarted your Jupyter kernel
- Verify the workspace was built: `ls /workspace/ros2_ws/install/`
- Check if sourced: `echo $AMENT_PREFIX_PATH` should include `/workspace/ros2_ws/install`

### Build errors

Make sure you have internet access in the Docker container and sufficient disk space.

### Import still fails in notebook

The Docker entrypoint automatically sources the ROS2 workspace on container start. If you built it after the container started, you need to either:
- Restart the container: `docker restart rapp_vam`
- Or manually source in your notebook:
  ```python
  import subprocess
  subprocess.run(['bash', '-c', 'source /workspace/ros2_ws/install/setup.bash'])
  ```

## Rebuilding the Docker Image

If you want to start fresh with zed_msgs pre-installed:

```bash
# Stop and remove the current container
docker-compose -f docker/docker-compose.yml down

# Rebuild the image (this will include zed_msgs)
docker-compose -f docker/docker-compose.yml build

# Start the new container
docker-compose -f docker/docker-compose.yml up -d
```

After rebuilding, zed_msgs will be automatically available - no manual setup needed!

## What Changed in the Dockerfile

The Dockerfile now includes `ros-humble-zed-msgs` as a pre-built apt package (line 82).

This is much simpler and faster than building from source:
- No compilation needed during Docker build
- Smaller image size (no build artifacts)
- Faster build times
- Official ROS2 repository package

Future container instances will have zed_msgs ready to use immediately!
