"""On-the-fly data augmentation for training."""

import numpy as np


def apply_temporal_jitter(
    start_frame: int,
    jitter_max: int,
    rng: np.random.Generator,
) -> int:
    """
    Shift window start by a random offset in [-jitter_max, +jitter_max].

    The windowing code reserves jitter_margin = 2 * jitter_max extra frames,
    so the canonical center position is at start_frame + jitter_max.
    The jitter shifts around this center.

    Returns the jittered start frame.
    """
    offset = rng.integers(-jitter_max, jitter_max + 1)
    return start_frame + jitter_max + offset


def apply_skeleton_noise(
    skeleton: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add isotropic Gaussian noise to skeleton coordinates.

    noise_std is in meters (physical units), applied BEFORE standardization.

    Args:
        skeleton: shape [T, 48]
        noise_std: standard deviation of noise in meters
        rng: numpy random generator

    Returns:
        skeleton + noise, same shape
    """
    noise = rng.normal(0.0, noise_std, size=skeleton.shape).astype(skeleton.dtype)
    return skeleton + noise
