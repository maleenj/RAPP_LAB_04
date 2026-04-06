#!/bin/bash
set -e

# RAPP Lab 04 - Hardware Container Entrypoint Script
# Wraps the base ZED ROS2 entrypoint with VAM workspace setup

echo "=================================================="
echo "RAPP Lab 04 - Hardware Container (ZED + TM12S)"
echo "=================================================="

# Source ROS2 Humble
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Source tm2_ws if it exists (ZED + TM12S drivers)
if [ -d "/tm2_ws/install" ]; then
    echo "Sourcing tm2_ws..."
    source /tm2_ws/install/setup.bash
fi

# Install dependencies needed inside hw container
echo "Installing setuptools + numpy..."
pip3 install -q setuptools==58.2.0 "numpy<2"

# Build VAM ROS2 workspace into a HW-specific directory
# (the shared ros2_ws mount has build artifacts from the VAM container — different
# base image, libraries, numpy — so each container needs its own build/install)
HW_BUILD_DIR=/workspace/ros2_ws_hw_build
if [ -d "/workspace/ros2_ws/src" ]; then
    echo "Building ROS2 workspace (vam_interfaces + vam_inference) into ${HW_BUILD_DIR}..."
    mkdir -p ${HW_BUILD_DIR}
    cd ${HW_BUILD_DIR}
    colcon build --symlink-install \
        --base-paths /workspace/ros2_ws \
        --build-base ${HW_BUILD_DIR}/build \
        --install-base ${HW_BUILD_DIR}/install \
        --packages-select vam_interfaces vam_inference 2>&1 | tail -5
    source ${HW_BUILD_DIR}/install/setup.bash
    echo "ROS2 workspace built and sourced."
    cd /workspace
elif [ -f "${HW_BUILD_DIR}/install/setup.bash" ]; then
    echo "Sourcing HW workspace build..."
    source ${HW_BUILD_DIR}/install/setup.bash
fi

# Ensure docker exec sessions auto-source everything
# Using /etc/bash.bashrc (runs before user .bashrc guard that blocks non-interactive shells)
grep -q "vam_workspace_setup" /etc/bash.bashrc 2>/dev/null || cat >> /etc/bash.bashrc << 'BASHRC'
# vam_workspace_setup
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
if [ -d /tm2_ws/install ]; then source /tm2_ws/install/setup.bash; fi
if [ -f /workspace/ros2_ws_hw_build/install/setup.bash ]; then source /workspace/ros2_ws_hw_build/install/setup.bash; fi
BASHRC

echo ""
echo "Container ready!"
echo "ROS2 Domain ID: $ROS_DOMAIN_ID"
echo "RMW Implementation: $RMW_IMPLEMENTATION"
echo "=================================================="
echo ""

exec "$@"
