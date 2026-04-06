#!/bin/bash
set -e

# RAPP Lab 04 - VAM Container Entrypoint Script

echo "=================================================="
echo "RAPP Lab 04 - Vision-Action Model Container"
echo "=================================================="

# Source ROS2 Humble
echo "Sourcing ROS2 Humble..."
source /opt/ros/humble/setup.bash

# Set ROS2 environment
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Check if vam_utils exists and install in editable mode
if [ -d "/workspace/vam_utils" ]; then
    echo "Installing vam_utils package in editable mode..."
    cd /workspace/vam_utils
    pip3 install -e . --no-deps 2>/dev/null || echo "vam_utils will be installed when setup.py is created"
    cd /workspace
else
    echo "vam_utils directory not found. Will be available after mounting."
fi

# Pin setuptools for ROS2 Humble colcon build compatibility
echo "Pinning setuptools for colcon build..."
pip3 install -q setuptools==58.2.0

# Build and source ROS2 workspace
if [ -d "/workspace/ros2_ws/src" ]; then
    echo "Building ROS2 workspace (vam_interfaces + vam_inference)..."
    cd /workspace/ros2_ws
    colcon build --symlink-install --packages-select vam_interfaces vam_inference 2>&1 | tail -5
    source /workspace/ros2_ws/install/setup.bash
    echo "ROS2 workspace built and sourced."
    cd /workspace
elif [ -d "/workspace/ros2_ws/install" ]; then
    echo "Sourcing ROS2 workspace..."
    source /workspace/ros2_ws/install/setup.bash
fi

# Check GPU availability
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "PyTorch not fully loaded yet"
else
    echo "WARNING: nvidia-smi not found. GPU may not be accessible."
fi

echo ""
echo "Container ready!"
echo "ROS2 Domain ID: $ROS_DOMAIN_ID"
echo "RMW Implementation: $RMW_IMPLEMENTATION"
echo ""
echo "Quick commands:"
echo "  - jupyter lab (already running if default CMD)"
echo "  - build: Build ROS2 workspace"
echo "  - source_ws: Source ROS2 workspace"
echo "=================================================="
echo ""

# Execute the CMD
exec "$@"
