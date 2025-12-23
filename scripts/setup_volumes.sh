#!/bin/bash
# RAPP Lab 04 - Setup Volume Directories
# Creates all necessary host directories for Docker volume mounts

set -e

echo "=================================================="
echo "RAPP Lab 04 - Setting up volume directories"
echo "=================================================="

# Define directories
ROSBAGS_DIR="/home/maleen/rosbags/rapplab04"
CSV_DIR="/home/maleen/csvdata/rapplab04"
MODELS_DIR="/home/maleen/models/rapplab04"
LOGS_DIR="/home/maleen/models/rapplab04/logs"
TENSORBOARD_DIR="/home/maleen/models/rapplab04/logs/tensorboard"

# Create directories if they don't exist
echo ""
echo "Creating data directories..."

if [ ! -d "$ROSBAGS_DIR" ]; then
    mkdir -p "$ROSBAGS_DIR"
    echo "✓ Created: $ROSBAGS_DIR"
else
    echo "✓ Already exists: $ROSBAGS_DIR"
fi

if [ ! -d "$CSV_DIR" ]; then
    mkdir -p "$CSV_DIR"
    echo "✓ Created: $CSV_DIR"
else
    echo "✓ Already exists: $CSV_DIR"
fi

if [ ! -d "$MODELS_DIR" ]; then
    mkdir -p "$MODELS_DIR"
    echo "✓ Created: $MODELS_DIR"
else
    echo "✓ Already exists: $MODELS_DIR"
fi

if [ ! -d "$LOGS_DIR" ]; then
    mkdir -p "$LOGS_DIR"
    echo "✓ Created: $LOGS_DIR"
else
    echo "✓ Already exists: $LOGS_DIR"
fi

if [ ! -d "$TENSORBOARD_DIR" ]; then
    mkdir -p "$TENSORBOARD_DIR"
    echo "✓ Created: $TENSORBOARD_DIR"
else
    echo "✓ Already exists: $TENSORBOARD_DIR"
fi

echo ""
echo "Volume directories setup complete!"
echo ""
echo "Directory structure:"
echo "  Rosbags:     $ROSBAGS_DIR"
echo "  CSV Data:    $CSV_DIR"
echo "  Models:      $MODELS_DIR"
echo "  Logs:        $LOGS_DIR"
echo "  TensorBoard: $TENSORBOARD_DIR"
echo ""
echo "Next steps:"
echo "  1. Copy your rosbag files to: $ROSBAGS_DIR"
echo "  2. Run: cd docker && docker-compose build"
echo "=================================================="
