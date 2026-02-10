"""PyTorch Dataset and DataLoader factory for the VAM training pipeline."""

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vam_utils.config.data_config import DataPipelineConfig
from vam_utils.data.augmentation import apply_skeleton_noise, apply_temporal_jitter
from vam_utils.data.normalization import NormalizationStats
from vam_utils.data.windowing import WindowIndex


class VAMDataset(Dataset):
    """
    PyTorch Dataset for Vision-Action Model training.

    Stores episode arrays in memory and slices windows on the fly.
    Augmentation is applied dynamically during __getitem__ for training.
    """

    def __init__(
        self,
        window_indices: List[WindowIndex],
        episode_data: Dict[str, Dict[str, np.ndarray]],
        norm_stats: NormalizationStats,
        config: DataPipelineConfig,
        split: str = "train",
    ):
        """
        Args:
            window_indices: list of WindowIndex (episode_id + start_frame)
            episode_data: {episode_id: {"skeleton": [N,48], "joints": [N,6]}}
            norm_stats: pre-computed normalization statistics
            config: pipeline configuration
            split: "train", "val", or "test"
        """
        self.window_indices = window_indices
        self.episode_data = episode_data
        self.norm_stats = norm_stats
        self.config = config
        self.split = split
        self.is_training = (split == "train")
        self._rng = np.random.default_rng(config.split_seed)

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                "input": tensor [T_in, 54],   # skeleton(48) + joints(6)
                "target": tensor [T_out, 6],   # future joint angles
                "episode_id": str,
                "start_frame": int,
            }
        """
        wi = self.window_indices[idx]
        ep = self.episode_data[wi.episode_id]
        skeleton_all = ep["skeleton"]  # [N, 48]
        joints_all = ep["joints"]      # [N, 6]

        T_in = self.config.T_in
        T_out = self.config.T_out

        # Determine actual start frame (with jitter for training)
        if self.is_training and self.config.augmentation_enabled:
            actual_start = apply_temporal_jitter(
                wi.start_frame, self.config.temporal_jitter_max, self._rng
            )
        else:
            actual_start = wi.start_frame

        # Slice windows
        input_skeleton = skeleton_all[actual_start:actual_start + T_in].copy()
        input_joints = joints_all[actual_start:actual_start + T_in].copy()
        target_joints = joints_all[actual_start + T_in:actual_start + T_in + T_out].copy()

        # Apply skeleton noise (training only, in physical units before standardization)
        if self.is_training and self.config.augmentation_enabled:
            input_skeleton = apply_skeleton_noise(
                input_skeleton, self.config.skeleton_noise_std, self._rng
            )

        # Standardize
        input_skeleton = (
            (input_skeleton - self.norm_stats.skeleton_mean) / self.norm_stats.skeleton_std
        )
        input_joints_normed = (
            (input_joints - self.norm_stats.joint_mean) / self.norm_stats.joint_std
        )
        target_joints_normed = (
            (target_joints - self.norm_stats.joint_mean) / self.norm_stats.joint_std
        )

        # Concatenate input features: skeleton + joints
        input_tensor = np.concatenate(
            [input_skeleton, input_joints_normed], axis=-1
        )  # [T_in, 54]

        return {
            "input": torch.from_numpy(input_tensor).float(),
            "target": torch.from_numpy(target_joints_normed).float(),
            "episode_id": wi.episode_id,
            "start_frame": actual_start,
        }

    def worker_init_fn(self, worker_id: int):
        """Re-seed RNG per DataLoader worker for different augmentations."""
        self._rng = np.random.default_rng(self.config.split_seed + worker_id)


def create_dataloaders(
    config: DataPipelineConfig,
    train_dataset: VAMDataset,
    val_dataset: VAMDataset,
    test_dataset: VAMDataset,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train/val/test splits."""
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            drop_last=True,
            worker_init_fn=train_dataset.worker_init_fn,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        ),
    }
