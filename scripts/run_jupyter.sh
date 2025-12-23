#!/bin/bash
# RAPP Lab 04 - Start Jupyter Lab in VAM Container

set -e

cd "$(dirname "$0")/../docker"

echo "=================================================="
echo "Starting RAPP Lab 04 VAM Container"
echo "=================================================="

# Check if container is already running
if [ "$(docker ps -q -f name=rapp_vam)" ]; then
    echo "Container is already running!"
    echo ""
    echo "Jupyter Lab: http://localhost:8888"
    echo "TensorBoard: http://localhost:6006"
    echo ""
    echo "To access shell: docker exec -it rapp_vam bash"
else
    echo "Starting container..."
    docker-compose up -d

    echo ""
    echo "Waiting for Jupyter Lab to start..."
    sleep 3

    echo ""
    echo "=================================================="
    echo "Container started successfully!"
    echo ""
    echo "Access points:"
    echo "  Jupyter Lab:  http://localhost:8888"
    echo "  TensorBoard:  http://localhost:6006"
    echo ""
    echo "Container shell:"
    echo "  docker exec -it rapp_vam bash"
    echo ""
    echo "View logs:"
    echo "  docker logs rapp_vam"
    echo ""
    echo "Stop container:"
    echo "  docker-compose down"
    echo "=================================================="
fi
