#!/bin/bash
# ENACT - Setup Volume Directories
# Creates the host data directories that the Docker containers mount.
# The root is ENACT_DATA (from docker/.env), default ~/enact_local.

set -e

# Read ENACT_DATA from docker/.env if present
ENV_FILE="$(dirname "$0")/../docker/.env"
if [ -f "$ENV_FILE" ]; then
    # shellcheck disable=SC1090
    set -a; source "$ENV_FILE"; set +a
fi
ENACT_DATA="${ENACT_DATA:-$HOME/enact_local}"
# expand a literal leading ~
ENACT_DATA="${ENACT_DATA/#\~/$HOME}"

echo "=================================================="
echo "ENACT - Setting up data directories"
echo "Root: $ENACT_DATA"
echo "=================================================="

for sub in rosbags csvdata models logs logs/tensorboard; do
    dir="$ENACT_DATA/$sub"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  created: $dir"
    else
        echo "  exists:  $dir"
    fi
done

echo ""
echo "Data directories ready. Container mapping:"
echo "  $ENACT_DATA/rosbags  -> /data/rosbags"
echo "  $ENACT_DATA/csvdata  -> /data/processed"
echo "  $ENACT_DATA/models   -> /data/models"
echo "  $ENACT_DATA/logs     -> /data/logs"
echo ""
echo "Next steps:"
echo "  1. Put rosbags / the starter data pack under $ENACT_DATA"
echo "  2. cd docker && docker compose up -d"
echo "=================================================="
