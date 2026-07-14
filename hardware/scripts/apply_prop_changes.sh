#!/usr/bin/env bash
# Apply xacro / SRDF edits to the running system WITHOUT restarting the container.
#
# Works because:
#   - The bind-mounted active files live at docker/hw/urdf_override/{tm12s.urdf.xacro,tm12s.srdf}
#   - When you overwrite them with `cp` (truncate-in-place), the inode is preserved,
#     so the container sees the new content immediately -- no restart needed.
#   - `set_profile.sh` already uses cp internally, so this holds for the normal workflow.
#
# What this script does:
#   1. Regenerate the flat URDF at /data/processed/tm12s.urdf for VAM-prediction RViz
#      visualization. MoveIt doesn't need this; it reads the xacro directly.
#
# You MUST still Ctrl+C and re-run the three ROS launches after running this --
# move_group caches the URDF in memory at launch time.
#
# WARNING: If you edit docker/hw/urdf_override/tm12s.urdf.xacro or tm12s.srdf
# DIRECTLY with an editor that does atomic-save-by-rename (VSCode, most IDEs),
# the bind mount will lose its link to the new inode and the container will
# show stale content until `docker compose restart`. Prefer editing profile
# files and using set_profile.sh, which uses cp internally.

set -euo pipefail

CONTAINER="rapp_hw"
XACRO_IN_CONTAINER="/tm2_ws/install/tm12s_moveit_config/share/tm12s_moveit_config/config/tm12s.urdf.xacro"
FLAT_URDF_IN_CONTAINER="/data/processed/tm12s.urdf"

echo "Regenerating flat URDF (for RViz VAM-prediction visualization)..."
docker exec "$CONTAINER" bash -c "
  source /opt/ros/humble/setup.bash &&
  source /tm2_ws/install/setup.bash &&
  xacro '$XACRO_IN_CONTAINER' > '$FLAT_URDF_IN_CONTAINER'
"

echo ""
echo "Done. Now Ctrl+C and re-run your three ROS launches:"
echo "  ros2 launch vam_inference tm12s_moveit_hw.launch.py robot_ip:=192.168.10.2"
echo "  ros2 run   vam_inference vam_pvt_streamer --ros-args ..."
echo "  ros2 launch vam_inference vam_tm12s_robot.launch.py ..."
