"""Normalization statistics computation and application.

No hip-centering — skeleton data stays in robot_base_link frame to preserve
spatial position information (radius, arc position) for the model.
Only standardization (zero mean, unit variance) is applied.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from vam_utils.data.episode import UR10_JOINT_LIMITS, get_joint_limits_tensor


@dataclass
class NormalizationStats:
    """Stores normalization parameters. Serialized to disk for inference."""
    skeleton_mean: np.ndarray   # shape [48]
    skeleton_std: np.ndarray    # shape [48]
    joint_mean: np.ndarray      # shape [6]
    joint_std: np.ndarray       # shape [6]
    ee_mean: np.ndarray         # shape [6]
    ee_std: np.ndarray          # shape [6]

    def save(self, path: Path) -> None:
        """Save as a torch file (dict of tensors)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "skeleton_mean": torch.from_numpy(self.skeleton_mean),
            "skeleton_std": torch.from_numpy(self.skeleton_std),
            "joint_mean": torch.from_numpy(self.joint_mean),
            "joint_std": torch.from_numpy(self.joint_std),
            "ee_mean": torch.from_numpy(self.ee_mean),
            "ee_std": torch.from_numpy(self.ee_std),
        }, path)

    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        """Load from torch file."""
        data = torch.load(path, weights_only=True)
        return cls(
            skeleton_mean=data["skeleton_mean"].numpy(),
            skeleton_std=data["skeleton_std"].numpy(),
            joint_mean=data["joint_mean"].numpy(),
            joint_std=data["joint_std"].numpy(),
            ee_mean=data["ee_mean"].numpy(),
            ee_std=data["ee_std"].numpy(),
        )


def compute_normalization_stats(
    train_episodes: List[Dict[str, np.ndarray]],
) -> NormalizationStats:
    """
    Compute normalization statistics from TRAINING episodes only.

    Each episode dict has keys: "skeleton" [N, 48], "joints" [N, 6], "ee_pose" [N, 6].

    Uses two-pass computation for numerical stability:
    pass 1 computes means, pass 2 computes stds.
    """
    # Concatenate all training frames
    all_skeleton = np.concatenate([ep["skeleton"] for ep in train_episodes], axis=0)
    all_joints = np.concatenate([ep["joints"] for ep in train_episodes], axis=0)
    all_ee = np.concatenate([ep["ee_pose"] for ep in train_episodes], axis=0)

    stats = NormalizationStats(
        skeleton_mean=all_skeleton.mean(axis=0).astype(np.float32),
        skeleton_std=all_skeleton.std(axis=0).astype(np.float32),
        joint_mean=all_joints.mean(axis=0).astype(np.float32),
        joint_std=all_joints.std(axis=0).astype(np.float32),
        ee_mean=all_ee.mean(axis=0).astype(np.float32),
        ee_std=all_ee.std(axis=0).astype(np.float32),
    )

    # Guard against zero std (constant features)
    for attr in ["skeleton_std", "joint_std", "ee_std"]:
        arr = getattr(stats, attr)
        arr[arr < 1e-8] = 1.0

    return stats


def apply_normalization(
    skeleton: np.ndarray,
    joints: np.ndarray,
    stats: NormalizationStats,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize skeleton and joints: (x - mean) / std.

    Args:
        skeleton: shape [T, 48] or [N, T, 48]
        joints: shape [T, 6] or [N, T, 6]
        stats: pre-computed normalization stats

    Returns:
        (skeleton_normed, joints_normed) with same shapes
    """
    skeleton_normed = (skeleton - stats.skeleton_mean) / stats.skeleton_std
    joints_normed = (joints - stats.joint_mean) / stats.joint_std
    return skeleton_normed, joints_normed


def normalize_joint_limits(stats: NormalizationStats) -> torch.Tensor:
    """Convert UR10 physical joint limits to normalized space.

    Returns:
        Tensor of shape [6, 2] where [:, 0] = lower, [:, 1] = upper (normalized).
    """
    limits = get_joint_limits_tensor()  # [6, 2]
    mean = torch.from_numpy(stats.joint_mean).float()  # [6]
    std = torch.from_numpy(stats.joint_std).float()     # [6]
    limits_normed = (limits - mean.unsqueeze(1)) / std.unsqueeze(1)
    return limits_normed


def inverse_normalize_joints(
    joints_normed: np.ndarray,
    stats: NormalizationStats,
    clamp_to_limits: bool = False,
) -> np.ndarray:
    """Reverse normalization on predicted joint angles (for inference).

    Args:
        joints_normed: normalized joint angles, any shape with last dim = 6
        stats: normalization statistics
        clamp_to_limits: if True, clamp output to UR10 physical joint limits
    """
    joints = joints_normed * stats.joint_std + stats.joint_mean
    if clamp_to_limits:
        limits = np.array(UR10_JOINT_LIMITS, dtype=np.float32)  # [6, 2]
        joints = np.clip(joints, limits[:, 0], limits[:, 1])
    return joints
