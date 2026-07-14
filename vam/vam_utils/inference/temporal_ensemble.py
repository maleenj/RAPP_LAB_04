"""Temporal ensemble for smooth trajectory blending.

Implements the temporal ensemble strategy from the ACT paper (Tony Zhao et al.,
"Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware").

At each timestep, multiple overlapping chunk predictions are blended using
exponential decay weighting. This eliminates chunk boundary discontinuities
without adding filtering lag.
"""

from collections import deque
from typing import Optional

import numpy as np


class TemporalEnsemble:
    """Manages overlapping predictions and computes blended joint targets.

    Usage:
        ensemble = TemporalEnsemble(T_out=10, K=2, decay_weight=0.01)

        for each_frame:
            if ensemble.should_predict():
                chunk = model.predict(input)   # [T_out, 6]
                ensemble.add_prediction(chunk)

            target = ensemble.query()          # [6] blended radians
            ensemble.step()
    """

    def __init__(
        self,
        T_out: int = 10,
        K: int = 2,
        decay_weight: float = 0.01,
        n_joints: int = 6,
    ):
        """
        Args:
            T_out: prediction horizon (frames per chunk).
            K: prediction stride (re-predict every K frames).
            decay_weight: exponential decay lambda. Smaller = smoother.
                0.01 (ACT default) ≈ uniform weighting, very smooth.
                0.1–0.5 biases toward newest prediction, more responsive.
            n_joints: number of output dimensions (6 for UR10).
        """
        self.T_out = T_out
        self.K = K
        self.decay_weight = decay_weight
        self.n_joints = n_joints
        self.max_history = max(T_out // K, 1)
        self._predictions: deque = deque(maxlen=self.max_history)
        self._global_step: int = 0

    def add_prediction(self, chunk: np.ndarray) -> None:
        """Add a new T_out-frame prediction at the current global step.

        Args:
            chunk: np.ndarray of shape [T_out, n_joints] in physical units.
        """
        assert chunk.shape == (self.T_out, self.n_joints), (
            f"Expected ({self.T_out}, {self.n_joints}), got {chunk.shape}"
        )
        self._predictions.append((self._global_step, chunk.copy()))

    def query(self, step: Optional[int] = None) -> np.ndarray:
        """Get blended joint target at the given (or current) global step.

        Args:
            step: global timestep to query. Defaults to current step.

        Returns:
            np.ndarray of shape [n_joints] — blended joint angles.

        Raises:
            RuntimeError: if no predictions are available.
        """
        if len(self._predictions) == 0:
            raise RuntimeError("No predictions available in ensemble")

        if step is None:
            step = self._global_step

        weighted_sum = np.zeros(self.n_joints, dtype=np.float64)
        weight_sum = 0.0

        for pred_step, chunk in self._predictions:
            frame_idx = step - pred_step
            if frame_idx < 0 or frame_idx >= self.T_out:
                continue

            w = np.exp(-self.decay_weight * frame_idx)
            weighted_sum += w * chunk[frame_idx]
            weight_sum += w

        if weight_sum > 0:
            return (weighted_sum / weight_sum).astype(np.float32)

        # Fallback: return first frame of most recent prediction
        return self._predictions[-1][1][0].copy()

    def query_trajectory(self, n_frames: int) -> np.ndarray:
        """Query blended targets for the next n_frames starting from current step.

        Useful for building a multi-point JointTrajectory message.

        Args:
            n_frames: how many future frames to query.

        Returns:
            np.ndarray of shape [n_frames, n_joints].
        """
        trajectory = np.zeros((n_frames, self.n_joints), dtype=np.float32)
        for i in range(n_frames):
            trajectory[i] = self.query(self._global_step + i)
        return trajectory

    def step(self) -> None:
        """Advance the global timestep counter by 1."""
        self._global_step += 1

    def should_predict(self) -> bool:
        """True if it's time to run a new model inference (every K steps)."""
        return self._global_step % self.K == 0

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def num_predictions(self) -> int:
        return len(self._predictions)

    @property
    def is_warmed_up(self) -> bool:
        """True when the ensemble has enough overlapping predictions."""
        return len(self._predictions) >= self.max_history

    def reset(self) -> None:
        """Clear all state."""
        self._predictions.clear()
        self._global_step = 0
