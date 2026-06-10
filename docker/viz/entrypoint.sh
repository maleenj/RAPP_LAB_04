#!/bin/bash
set -e

echo "=================================================="
echo "RAPP Lab 04 - VAM Visualization Bridge"
echo "=================================================="

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}

# Build only the bridge package into an isolated install dir so we never touch
# the inference container's build artifacts (they share the mounted ros2_ws).
if [ -d "/ws/ros2_ws/src/vam_viz_bridge" ]; then
    echo "Building vam_viz_bridge..."
    cd /ws/ros2_ws
    colcon build --symlink-install \
        --packages-select vam_viz_bridge \
        --build-base build_viz --install-base install_viz 2>&1 | tail -5
    source /ws/ros2_ws/install_viz/setup.bash
    echo "vam_viz_bridge built and sourced."
    cd /ws
else
    echo "ERROR: /ws/ros2_ws/src/vam_viz_bridge not found (check the volume mount)."
    exit 1
fi

# Show the host IPs clients should connect to.
echo ""
echo "Connect clients to ws://<HOST-IP>:8765  — host IP(s):"
hostname -I 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sed 's/^/    /' || true
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  RMW=$RMW_IMPLEMENTATION"
echo "=================================================="
echo ""

exec "$@"
