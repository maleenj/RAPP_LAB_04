"""Train/val/test splitting by recording episode."""

import logging
import math
from typing import Dict, List

import numpy as np

from vam_utils.data.episode import EpisodeMetadata

logger = logging.getLogger(__name__)


def split_episodes_by_recording(
    episodes: List[EpisodeMetadata],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[EpisodeMetadata]]:
    """
    Assign entire episodes to train/val/test splits.

    Episodes are sorted by ID for determinism, then shuffled with a fixed seed.
    Split counts are determined by ratio (rounding ensures all episodes assigned).
    Each EpisodeMetadata.split is set in place.
    """
    n = len(episodes)
    if n == 0:
        raise ValueError("No episodes to split")

    sorted_episodes = sorted(episodes, key=lambda e: e.episode_id)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    n_train = n - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Not enough episodes ({n}) for the requested split ratios. "
            f"Need at least 3 episodes."
        )

    splits: Dict[str, List[EpisodeMetadata]] = {"train": [], "val": [], "test": []}
    for i, idx in enumerate(indices):
        ep = sorted_episodes[idx]
        if i < n_train:
            ep.split = "train"
            splits["train"].append(ep)
        elif i < n_train + n_val:
            ep.split = "val"
            splits["val"].append(ep)
        else:
            ep.split = "test"
            splits["test"].append(ep)

    # Log the split
    for split_name, eps in splits.items():
        total_frames = sum(e.num_frames for e in eps)
        ids = [e.episode_id for e in eps]
        logger.info(
            f"{split_name}: {len(eps)} episodes, {total_frames} frames — {ids}"
        )

    return splits
