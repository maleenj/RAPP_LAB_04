#!/bin/bash
# Quick setup for ZED ROS2 message interfaces
# Run this inside the VAM Docker container

set -e

echo "=================================================="
echo "Installing ZED ROS2 Interfaces"
echo "=================================================="

# Install via apt (simple and fast)
echo ""
echo "Installing ros-humble-zed-msgs via apt..."
apt-get update
apt-get install -y ros-humble-zed-msgs

echo ""
echo "=================================================="
echo "ZED Messages Installed Successfully!"
echo "=================================================="
echo ""
echo "Package installed: ros-humble-zed-msgs"
echo ""
echo "To use in your Python code:"
echo "  from zed_msgs.msg import ObjectsStamped"
echo ""
echo "Next step:"
echo "  Restart your Jupyter kernel to load the new messages"
echo "=================================================="
