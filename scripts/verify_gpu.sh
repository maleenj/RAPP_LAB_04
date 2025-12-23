#!/bin/bash
# RAPP Lab 04 - Verify GPU Accessibility in Container

set -e

echo "=================================================="
echo "RAPP Lab 04 - GPU Verification"
echo "=================================================="

# Check if container is running
if [ ! "$(docker ps -q -f name=rapp_vam)" ]; then
    echo "ERROR: Container 'rapp_vam' is not running!"
    echo "Start it with: cd docker && docker-compose up -d"
    exit 1
fi

echo ""
echo "1. Checking nvidia-smi in container..."
echo "=================================================="
docker exec rapp_vam nvidia-smi

echo ""
echo "2. Checking PyTorch CUDA availability..."
echo "=================================================="
docker exec rapp_vam python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'cuDNN version: {torch.backends.cudnn.version()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"

echo ""
echo "3. Checking ROS2 installation..."
echo "=================================================="
docker exec rapp_vam bash -c "source /opt/ros/humble/setup.bash && ros2 --version"

echo ""
echo "=================================================="
echo "Verification complete!"
echo ""
echo "If all checks passed, your setup is ready for:"
echo "  ✓ GPU-accelerated PyTorch training"
echo "  ✓ ROS2 Humble development"
echo "  ✓ Data processing and visualization"
echo "=================================================="
