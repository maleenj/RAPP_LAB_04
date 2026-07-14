"""Episode loading and validation for the data preparation pipeline."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from vam_utils.config.data_config import DataPipelineConfig

logger = logging.getLogger(__name__)

# UR10 joint limits (radians) from URDF
UR10_JOINT_LIMITS = [
    (-6.283, 6.283),   # shoulder_pan
    (-6.283, 6.283),   # shoulder_lift
    (-3.142, 3.142),   # elbow
    (-6.283, 6.283),   # wrist_1
    (-6.283, 6.283),   # wrist_2
    (-6.283, 6.283),   # wrist_3
]


def get_joint_limits_tensor() -> torch.Tensor:
    """Return UR10 joint limits as a tensor of shape [6, 2] (lower, upper)."""
    return torch.tensor(UR10_JOINT_LIMITS, dtype=torch.float32)


@dataclass
class EpisodeMetadata:
    """Metadata for a single recording episode."""
    episode_id: str
    csv_path: Path
    num_frames: int
    duration_sec: float
    avg_frequency_hz: float
    quality_nan_count: int = 0
    quality_jumps: int = 0
    quality_joint_violations: int = 0
    split: Optional[str] = None


class EpisodeLoader:
    """Loads and validates CSV episodes."""

    def __init__(self, config: DataPipelineConfig, joint_limits=None):
        self.config = config
        self.joint_limits = joint_limits if joint_limits is not None else UR10_JOINT_LIMITS

    def discover_episodes(self) -> List[EpisodeMetadata]:
        """Read recordings_metadata.csv and build list of EpisodeMetadata."""
        meta_df = pd.read_csv(self.config.metadata_path)
        episodes = []

        for _, row in meta_df.iterrows():
            csv_path = self.config.csv_dir / row["csv_filename"]
            if not csv_path.exists():
                logger.warning(f"CSV not found: {csv_path}, skipping")
                continue

            episodes.append(EpisodeMetadata(
                episode_id=row["rosbag_name"],
                csv_path=csv_path,
                num_frames=int(row["num_samples"]),
                duration_sec=float(row["duration_sec"]),
                avg_frequency_hz=float(row["avg_frequency_hz"]),
                quality_nan_count=int(row["quality_nan_count"]),
                quality_jumps=int(row["quality_jumps"]),
                quality_joint_violations=int(row["quality_joint_violations"]),
            ))

        episodes.sort(key=lambda e: e.episode_id)
        logger.info(f"Discovered {len(episodes)} episodes")
        return episodes

    def load_episode_arrays(self, meta: EpisodeMetadata) -> Dict[str, np.ndarray]:
        """
        Load a single CSV into separate numpy arrays.

        Returns:
            {
                "timestamps": shape [N],
                "skeleton": shape [N, 48],
                "joints": shape [N, 6],
                "ee_pose": shape [N, 6],
            }
        """
        df = pd.read_csv(meta.csv_path)

        expected_cols = ["timestamp"] + self.config.skeleton_cols + self.config.joint_cols + self.config.ee_cols
        missing = set(expected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Episode {meta.episode_id}: missing columns {missing}")

        return {
            "timestamps": df["timestamp"].values.astype(np.float64),
            "skeleton": df[self.config.skeleton_cols].values.astype(np.float32),
            "joints": df[self.config.joint_cols].values.astype(np.float32),
            "ee_pose": df[self.config.ee_cols].values.astype(np.float32),
        }

    def validate_episode(
        self, meta: EpisodeMetadata, arrays: Dict[str, np.ndarray]
    ) -> List[str]:
        """
        Run quality checks on loaded episode data.

        Returns list of warning strings (empty if all clean).
        """
        warnings = []
        timestamps = arrays["timestamps"]
        skeleton = arrays["skeleton"]
        joints = arrays["joints"]

        # Check for NaN
        for name, arr in [("skeleton", skeleton), ("joints", joints)]:
            nan_count = np.isnan(arr).sum()
            if nan_count > 0:
                warnings.append(f"{name} has {nan_count} NaN values")

        # Monotonically increasing timestamps
        dt = np.diff(timestamps)
        if np.any(dt <= 0):
            n_bad = (dt <= 0).sum()
            warnings.append(f"{n_bad} non-monotonic timestamp(s)")

        # Large time gaps (> 0.5s at ~15Hz means ~7 missed frames)
        large_gaps = dt > 0.5
        if np.any(large_gaps):
            n_gaps = large_gaps.sum()
            max_gap = dt.max()
            warnings.append(f"{n_gaps} time gap(s) > 0.5s (max: {max_gap:.3f}s)")

        # Joint limits
        for j_idx, (lo, hi) in enumerate(self.joint_limits):
            col = joints[:, j_idx]
            violations = ((col < lo) | (col > hi)).sum()
            if violations > 0:
                warnings.append(f"j{j_idx}: {violations} joint limit violations")

        return warnings
