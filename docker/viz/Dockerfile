###############################################
# VAM Visualization Bridge Container
#
# Lightweight ROS2 node that streams ROS topics to Unity/Unreal/browser clients
# over WebSocket. No CUDA, no PyTorch, no GPU — it never runs the model, it only
# taps the DDS topic graph (host networking + shared ROS_DOMAIN_ID).
###############################################
FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# CycloneDDS to match the rest of the stack, plus python deps for the bridge.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rmw-cyclonedds-cpp \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Pin setuptools for ROS2 Humble colcon/ament_python build compatibility.
RUN pip3 install --no-cache-dir \
        "setuptools==58.2.0" \
        "websockets>=12.0" \
        "pyyaml"

WORKDIR /ws
COPY entrypoint.sh /entrypoint_viz.sh
RUN chmod +x /entrypoint_viz.sh

ENTRYPOINT ["/entrypoint_viz.sh"]
CMD ["ros2", "launch", "vam_viz_bridge", "viz_bridge.launch.py"]
