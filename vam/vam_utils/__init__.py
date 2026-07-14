"""
RAPP Lab 04 - Vision-Action Model Utilities

A modular Python package for processing skeletal tracking data, training
Action Chunking Transformers, and deploying real-time robot control for
human-robot ensemble performance.

Modules:
    - config: Configuration management (YAML + environment variables)
    - kinematics: Robot kinematics and URDF utilities
    - data: Rosbag processing, synchronization, and PyTorch datasets
    - model: Action Chunking Transformer architecture
    - training: Training loops, evaluation, and checkpointing
    - inference: Real-time inference and safety checking
    - visualization: 3D visualization for skeletons, robots, and trajectories
"""

__version__ = "0.1.0"
__author__ = "Dr. Maleen Jayasuriya"
__email__ = "maleen@example.com"

# Package-level imports for convenience
from . import config
from . import kinematics
from . import data
from . import model
from . import training
from . import inference
from . import visualization

__all__ = [
    "config",
    "kinematics",
    "data",
    "model",
    "training",
    "inference",
    "visualization",
]
