"""Robot configuration data for supported manipulators.

Defines joint names, limits, velocity limits, and sign conventions
for each robot. Used by robot-specific inference nodes and safety checkers.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from vam_utils.data.episode import UR10_JOINT_LIMITS


@dataclass
class RobotConfig:
    """Configuration for a specific robot arm."""
    name: str
    joint_names: List[str]
    joint_limits: List[Tuple[float, float]]  # (lower, upper) radians
    max_joint_velocities: List[float]         # rad/s per joint
    move_group_name: str
    planning_frame: str
    ee_frame: str
    sign_multipliers: List[float] = field(default_factory=lambda: [1.0] * 6)
    # Per-joint angular offsets for mapping from UR10 joint space (radians).
    # Applied as: target_angle = sign * ur10_angle + offset
    joint_offsets: List[float] = field(default_factory=lambda: [0.0] * 6)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def joint_limits_array(self) -> np.ndarray:
        """Joint limits as [N, 2] numpy array."""
        return np.array(self.joint_limits, dtype=np.float32)

    @property
    def sign_multipliers_array(self) -> np.ndarray:
        return np.array(self.sign_multipliers, dtype=np.float32)

    @property
    def joint_offsets_array(self) -> np.ndarray:
        return np.array(self.joint_offsets, dtype=np.float32)


# ---------------------------------------------------------------------------
# UR10
# ---------------------------------------------------------------------------
UR10_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR10_CONFIG = RobotConfig(
    name="ur10",
    joint_names=UR10_JOINT_NAMES,
    joint_limits=UR10_JOINT_LIMITS,
    max_joint_velocities=[2.09] * 6,  # UR10 hardware max ~2.09 rad/s
    move_group_name="ur_manipulator",
    planning_frame="base_link",
    ee_frame="tool0",
)

# ---------------------------------------------------------------------------
# TM12S (TechMan S-Series, from tm2_ros2 URDF)
# ---------------------------------------------------------------------------
TM12S_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

TM12S_JOINT_LIMITS = [
    (-6.283, 6.283),   # joint_1 (base rotation)
    (-6.283, 6.283),   # joint_2 (shoulder)
    (-2.827, 2.827),    # joint_3 (elbow) — narrower than UR10
    (-6.283, 6.283),   # joint_4 (wrist 1)
    (-6.283, 6.283),   # joint_5 (wrist 2)
    (-6.283, 6.283),   # joint_6 (wrist 3)
]

TM12S_MAX_JOINT_VELOCITIES = [
    2.27,   # joint_1
    2.27,   # joint_2
    3.67,   # joint_3
    3.93,   # joint_4
    3.93,   # joint_5
    7.85,   # joint_6
]

TM12S_CONFIG = RobotConfig(
    name="tm12s",
    joint_names=TM12S_JOINT_NAMES,
    joint_limits=TM12S_JOINT_LIMITS,
    max_joint_velocities=TM12S_MAX_JOINT_VELOCITIES,
    move_group_name="tmr_arm",
    planning_frame="base",
    ee_frame="flange",
    # --- No mapping needed: model trained on native TM12S joint data ---
    sign_multipliers=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    joint_offsets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    # --- OLD UR10→TM12S mapping (kept for reference) ---
    # sign_multipliers=[1.0, -1.0, -1.0, 1.0, -1.0, 1.0],
    # joint_offsets=[-np.pi, 3*(np.pi / 2), 0.0 , np.pi / 2, np.pi, np.pi],
    # joint_offsets=[0.0, np.pi / 2, 0.0, np.pi / 2, np.pi, 0.0],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_CONFIGS = {
    "ur10": UR10_CONFIG,
    "tm12s": TM12S_CONFIG,
}


def get_robot_config(name: str) -> RobotConfig:
    """Look up a robot configuration by name."""
    key = name.lower()
    if key not in _CONFIGS:
        raise ValueError(f"Unknown robot '{name}'. Available: {list(_CONFIGS.keys())}")
    return _CONFIGS[key]
