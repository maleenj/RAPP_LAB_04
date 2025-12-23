# RAPP Lab 04 - Vision-Action Model Docker Container
# Ubuntu 22.04 + ROS2 Humble Desktop + PyTorch + JupyterLab
# Compatible with RTX 5070 Ti (CUDA 12.1)

ARG CUDA_VERSION=12.1.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS2_DIST=humble

# Set environment variables for NVIDIA
ENV NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}compute,video,utility,graphics
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DOMAIN_ID=0
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    wget curl git nano vim \
    build-essential cmake \
    software-properties-common \
    lsb-release \
    gnupg2 \
    locales \
    python3 python3-pip python3-dev \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Setup ROS2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble Desktop (full installation with rviz2, rqt, etc.)
RUN apt-get update && apt-get install -y \
    ros-${ROS2_DIST}-desktop \
    ros-${ROS2_DIST}-ros-base \
    python3-flake8-docstrings \
    python3-pytest-cov \
    python3-rosdep \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 tools and packages
RUN apt-get update && apt-get install -y \
    # Rosbag and data recording tools
    ros-${ROS2_DIST}-rosbag2 \
    ros-${ROS2_DIST}-rosbag2-storage-default-plugins \
    ros-${ROS2_DIST}-rosbag2-storage-mcap \
    ros-${ROS2_DIST}-rosbag2-compression \
    ros-${ROS2_DIST}-rosbag2-compression-zstd \
    # RViz and visualization
    ros-${ROS2_DIST}-rviz2 \
    ros-${ROS2_DIST}-rviz-common \
    ros-${ROS2_DIST}-rviz-default-plugins \
    ros-${ROS2_DIST}-rviz-visual-tools \
    # RQT tools
    ros-${ROS2_DIST}-rqt \
    ros-${ROS2_DIST}-rqt-common-plugins \
    ros-${ROS2_DIST}-rqt-graph \
    ros-${ROS2_DIST}-rqt-tf-tree \
    # TF and state publishers
    ros-${ROS2_DIST}-tf2-tools \
    ros-${ROS2_DIST}-tf2-ros \
    ros-${ROS2_DIST}-tf2-geometry-msgs \
    ros-${ROS2_DIST}-robot-state-publisher \
    ros-${ROS2_DIST}-joint-state-publisher \
    ros-${ROS2_DIST}-joint-state-publisher-gui \
    # ZED camera interface messages (for compatibility with ZED container)
    ros-${ROS2_DIST}-sensor-msgs \
    ros-${ROS2_DIST}-geometry-msgs \
    ros-${ROS2_DIST}-std-msgs \
    ros-${ROS2_DIST}-trajectory-msgs \
    ros-${ROS2_DIST}-visualization-msgs \
    # Development tools
    ros-dev-tools \
    python3-argcomplete \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init || true && rosdep update

# Install Python dependencies for ML and robotics
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    # Force reinstall of packages that conflict with system-installed versions
    pip3 install --no-cache-dir --ignore-installed \
        blinker \
        pyyaml \
        numpy \
        packaging \
        pygments && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install PyTorch with CUDA 12.1 support (compatible with RTX 5070 Ti)
RUN pip3 install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Setup Jupyter Lab configuration
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py

# Create workspace directory
WORKDIR /workspace

# Copy and set up entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Setup bashrc for convenient ROS2 usage
RUN echo "source /opt/ros/${ROS2_DIST}/setup.bash" >> /root/.bashrc && \
    echo "export ROS_DOMAIN_ID=0" >> /root/.bashrc && \
    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /root/.bashrc && \
    echo "alias build='cd /workspace/ros2_ws && colcon build --symlink-install'" >> /root/.bashrc && \
    echo "alias source_ros='source /opt/ros/${ROS2_DIST}/setup.bash'" >> /root/.bashrc && \
    echo "alias source_ws='source /workspace/ros2_ws/install/setup.bash'" >> /root/.bashrc && \
    echo "if [ -d /workspace/ros2_ws/install ]; then source /workspace/ros2_ws/install/setup.bash; fi" >> /root/.bashrc

# Expose Jupyter Lab and TensorBoard ports
EXPOSE 8888 6006

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command: launch Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
