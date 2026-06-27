#!/bin/bash
set -e

echo "=================================================="
echo "RAPP Lab 04 - VAM Visualization Bridge"
echo "=================================================="

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}

# Optional overlay: the inference (rapp_vam) container builds vam_interfaces +
# vam_inference into the SHARED ros2_ws/install. Sourcing it here (read-only use)
# gives the vote_robot_bridge node the SwitchModel service type without building
# any rosidl interfaces in this lightweight image. The streaming bridge does not
# need it; this is purely additive and skipped if the overlay isn't present.
INFER_OVERLAY=/ws/ros2_ws/install/setup.bash
if [ -f "$INFER_OVERLAY" ]; then
    source "$INFER_OVERLAY"
    grep -q "ros2_ws/install/setup.bash" /root/.bashrc 2>/dev/null || \
        echo "source $INFER_OVERLAY 2>/dev/null" >> /root/.bashrc
    echo "Sourced inference overlay (vam_interfaces available for vote_robot_bridge)."
else
    echo "NOTE: $INFER_OVERLAY not found — vote_robot_bridge needs the rapp_vam"
    echo "      container built at least once (it produces vam_interfaces). The"
    echo "      streaming bridge works regardless."
fi

# Build only the bridge package into an isolated install dir so we never touch
# the inference container's build artifacts (they share the mounted ros2_ws).
if [ -d "/ws/ros2_ws/src/vam_viz_bridge" ]; then
    echo "Building vam_viz_bridge..."
    cd /ws/ros2_ws
    colcon build --symlink-install \
        --packages-select vam_viz_bridge \
        --build-base build_viz --install-base install_viz 2>&1 | tail -5
    source /ws/ros2_ws/install_viz/setup.bash
    # Make interactive `docker exec ... bash` sessions auto-source the overlay too.
    grep -q "install_viz/setup.bash" /root/.bashrc 2>/dev/null || \
        echo "source /ws/ros2_ws/install_viz/setup.bash 2>/dev/null" >> /root/.bashrc
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
