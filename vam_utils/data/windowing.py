"""Episode-aware sliding window index generation."""

import logging
from dataclasses import dataclass
from typing import List

from vam_utils.data.episode import EpisodeMetadata

logger = logging.getLogger(__name__)


@dataclass
class WindowIndex:
    """Lightweight index for one window's location within an episode."""
    episode_id: str
    start_frame: int
    # Input:  [start_frame, start_frame + T_in)
    # Output: [start_frame + T_in, start_frame + T_in + T_out)


def compute_window_indices(
    episodes: List[EpisodeMetadata],
    T_in: int,
    T_out: int,
    stride: int = 1,
    jitter_margin: int = 0,
) -> List[WindowIndex]:
    """
    Generate window indices for each episode independently.

    Windows NEVER cross episode boundaries because each episode
    is processed independently.

    Args:
        episodes: List of episode metadata (must have num_frames populated)
        T_in: Input window length (frames)
        T_out: Output prediction length (frames)
        stride: Step between consecutive window starts
        jitter_margin: Extra frames reserved for temporal jitter augmentation.
            Use 2 * jitter_max for training, 0 for val/test.

    Returns:
        Flat list of WindowIndex objects across all episodes.
    """
    indices = []
    required_span = T_in + T_out + jitter_margin

    for ep in episodes:
        if ep.num_frames < required_span:
            logger.warning(
                f"Episode {ep.episode_id}: {ep.num_frames} frames < "
                f"{required_span} required. Skipping (0 windows)."
            )
            continue

        max_start = ep.num_frames - required_span
        n_windows = 0
        for start in range(0, max_start + 1, stride):
            indices.append(WindowIndex(
                episode_id=ep.episode_id,
                start_frame=start,
            ))
            n_windows += 1

        logger.debug(
            f"Episode {ep.episode_id}: {n_windows} windows "
            f"(frames={ep.num_frames}, span={required_span}, stride={stride})"
        )

    logger.info(
        f"Generated {len(indices)} windows from {len(episodes)} episodes "
        f"(T_in={T_in}, T_out={T_out}, stride={stride}, jitter_margin={jitter_margin})"
    )
    return indices
