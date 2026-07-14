"""Input assembler: rolling buffer + normalization for inference.

Maintains ring buffers of skeleton and joint state frames, applies
normalization, and produces model-ready input tensors.
"""

from collections import deque

import numpy as np
import torch

from vam_utils.data.normalization import NormalizationStats, apply_normalization


class InputAssembler:
    """Assembles normalized model input from streaming skeleton and joint data.

    Usage:
        assembler = InputAssembler(norm_stats, T_in=10)

        for skeleton_frame, joint_frame in data_stream:
            assembler.add_frame(skeleton_frame, joint_frame)

            if assembler.is_ready():
                input_tensor = assembler.get_input_tensor()  # [1, T_in, 54]
                prediction = model.predict(input_tensor)

    For skeleton-only models:
        assembler = InputAssembler(norm_stats, T_in=10, skeleton_only=True)

        for skeleton_frame in data_stream:
            assembler.add_frame(skeleton_frame)

            if assembler.is_ready():
                input_tensor = assembler.get_input_tensor()  # [1, T_in, skeleton_dim]
    """

    def __init__(
        self,
        norm_stats: NormalizationStats,
        T_in: int = 10,
        skeleton_only: bool = False,
    ):
        """
        Args:
            norm_stats: pre-computed normalization statistics from training.
            T_in: number of input context frames (must match model).
            skeleton_only: if True, only buffer skeleton data (no joints in input).
        """
        self.norm_stats = norm_stats
        self.T_in = T_in
        self.skeleton_only = skeleton_only
        self._skeleton_buffer: deque = deque(maxlen=T_in)
        self._joints_buffer: deque = deque(maxlen=T_in)

    def add_frame(
        self, skeleton: np.ndarray, joints: np.ndarray | None = None
    ) -> None:
        """Add a single frame of observations.

        Args:
            skeleton: shape [skeleton_dim] — keypoints x 3 coordinates, in physical
                units (meters, robot base frame).
            joints: shape [6] — 6 joint angles in radians. Ignored when
                skeleton_only=True.
        """
        self._skeleton_buffer.append(skeleton.astype(np.float32))
        if not self.skeleton_only and joints is not None:
            self._joints_buffer.append(joints.astype(np.float32))

    def is_ready(self) -> bool:
        """True when the buffer has T_in frames accumulated."""
        return len(self._skeleton_buffer) == self.T_in

    @property
    def buffer_fill(self) -> int:
        """Number of frames currently in the buffer."""
        return len(self._skeleton_buffer)

    def get_input_tensor(self) -> torch.Tensor:
        """Build the normalized input tensor for the model.

        Returns:
            torch.Tensor of shape [1, T_in, 54] (default) or
            [1, T_in, skeleton_dim] (skeleton_only), normalized using training statistics.

        Raises:
            RuntimeError: if buffer is not full yet.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer not ready: {self.buffer_fill}/{self.T_in} frames"
            )

        skeleton = np.stack(list(self._skeleton_buffer))  # [T_in, skeleton_dim]

        if self.skeleton_only:
            skel_normed = (
                (skeleton - self.norm_stats.skeleton_mean)
                / self.norm_stats.skeleton_std
            )
            return torch.from_numpy(skel_normed).float().unsqueeze(0)  # [1, T_in, skeleton_dim]

        joints = np.stack(list(self._joints_buffer))  # [T_in, 6]
        skel_normed, joints_normed = apply_normalization(
            skeleton, joints, self.norm_stats
        )
        combined = np.concatenate(
            [skel_normed, joints_normed], axis=-1
        )  # [T_in, 54]
        return torch.from_numpy(combined).float().unsqueeze(0)  # [1, T_in, 54]

    def reset(self) -> None:
        """Clear the buffers."""
        self._skeleton_buffer.clear()
        self._joints_buffer.clear()
