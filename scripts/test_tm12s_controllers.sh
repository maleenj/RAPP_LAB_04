#!/usr/bin/env bash
###############################################
# Test TM12S controller availability
#
# Run this inside the TM12S container AFTER the tm_driver is running.
# It checks what controllers and hardware interfaces are available,
# and recommends which MoveIt Servo config to use.
#
# Usage (inside TM12S container):
#   bash /workspace/scripts/test_tm12s_controllers.sh
###############################################

set -e

echo "============================================"
echo "TM12S Controller Discovery"
echo "============================================"
echo ""

# Check if controller_manager is running
echo "--- Checking controller_manager ---"
if ros2 service list 2>/dev/null | grep -q controller_manager; then
    echo "[OK] controller_manager is running"
else
    echo "[WARN] controller_manager not found. Is the TM12S driver running?"
    echo "  Start it with: ros2 run tm_driver tm_driver robot_ip:=192.168.10.2"
    exit 1
fi

echo ""
echo "--- Active Controllers ---"
ros2 control list_controllers 2>/dev/null || echo "[WARN] Could not list controllers"

echo ""
echo "--- Hardware Interfaces ---"
ros2 control list_hardware_interfaces 2>/dev/null || echo "[WARN] Could not list hardware interfaces"

echo ""
echo "--- Checking for position command interfaces ---"
POSITION_IFACES=$(ros2 control list_hardware_interfaces 2>/dev/null | grep "position" | grep "command" || true)
if [ -n "$POSITION_IFACES" ]; then
    echo "[OK] Position command interfaces found:"
    echo "$POSITION_IFACES"
    echo ""
    echo ">>> RECOMMENDATION: Use tm12s_servo_vam.yaml (ForwardCommandController)"
    echo "    You can load a forward_position_controller with:"
    echo "    ros2 control load_controller forward_position_controller"
    echo "    ros2 control set_controller_state forward_position_controller active"
else
    echo "[INFO] No position command interfaces found"
    echo ""
    echo ">>> RECOMMENDATION: Use tm12s_servo_vam_traj.yaml (JointTrajectory fallback)"
    echo "    This publishes trajectory commands to tmr_arm_controller"
fi

echo ""
echo "--- TF Frames ---"
echo "Checking TM12S base frame name..."
ros2 run tf2_ros tf2_echo base world 2>/dev/null &
TF_PID=$!
sleep 2
kill $TF_PID 2>/dev/null || true

echo ""
echo "Available frames (first 20):"
ros2 run tf2_tools view_frames --no-wait 2>/dev/null || \
    ros2 topic echo /tf --once 2>/dev/null | head -20 || \
    echo "[INFO] Could not enumerate frames. Check /tf topic manually."

echo ""
echo "============================================"
echo "Done. Check recommendations above."
echo "============================================"
