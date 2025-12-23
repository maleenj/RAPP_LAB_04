#!/bin/bash
# RAPP Lab 04 - Build Docker Containers

set -e

echo "=================================================="
echo "RAPP Lab 04 - Building Docker Containers"
echo "=================================================="

# Change to docker directory
cd "$(dirname "$0")/../docker"

echo ""
echo "Building VAM container (this may take 10-15 minutes)..."
echo "Container includes:"
echo "  - Ubuntu 22.04"
echo "  - CUDA 12.1 + cuDNN 8"
echo "  - ROS2 Humble Desktop (full)"
echo "  - PyTorch 2.1.2 with CUDA support"
echo "  - JupyterLab + all ML dependencies"
echo ""

# Build with progress output
docker compose build --progress=plain vam

echo ""
echo "=================================================="
echo "Build complete!"
echo ""
echo "To start the container:"
echo "  cd docker"
echo "  docker compose up -d"
echo ""
echo "To access Jupyter Lab:"
echo "  http://localhost:8888"
echo ""
echo "To access the container shell:"
echo "  docker exec -it rapp_vam bash"
echo "=================================================="
