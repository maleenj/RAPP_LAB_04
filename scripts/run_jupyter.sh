#!/bin/bash
# RAPP Lab 04 - Start Jupyter Lab in VAM Container

set -e

echo "=================================================="
echo "Starting Jupyter Lab in VAM Container"
echo "=================================================="

# Check if container is running
if [ ! "$(docker ps -q -f name=rapp_vam)" ]; then
    echo "ERROR: rapp_vam container is not running."
    echo "Start it first, then re-run this script."
    exit 1
fi

echo "Container is running. Starting Jupyter Lab..."
docker exec -d rapp_vam bash -c \
    "source /opt/ros/humble/setup.bash && \
     jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
     --notebook-dir=/workspace/notebooks 2>&1 &"

sleep 2

echo ""
echo "=================================================="
echo "Access points:"
echo "  Jupyter Lab:  http://localhost:8888"
echo ""
echo "Container shell:"
echo "  docker exec -it rapp_vam bash"
echo "=================================================="
