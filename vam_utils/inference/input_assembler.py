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
    """

    def __init__(self, norm_stats: NormalizationStats, T_in: int = 10):
        """
        Args:
            norm_stats: pre-computed normalization statistics from training.
            T_in: number of input context frames (must match model).
        """
        self.norm_stats = norm_stats
        self.T_in = T_in
        self._skeleton_buffer: deque = deque(maxlen=T_in)
        self._joints_buffer: deque = deque(maxlen=T_in)

    def add_frame(self, skeleton: np.ndarray, joints: np.ndarray) -> None:
        """Add a single frame of observations.

        Args:
            skeleton: shape [48] — 16 keypoints x 3 coordinates, in physical
                units (meters, robot_base_link frame).
            joints: shape [6] — 6 joint angles in radians.
        """
        self._skeleton_buffer.append(skeleton.astype(np.float32))
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
            torch.Tensor of shape [1, T_in, 54] (skeleton_dim + joint_dim),
            normalized using training statistics.

        Raises:
            RuntimeError: if buffer is not full yet.
        """
        if not self.is_ready():
            raise RuntimeError(
                f"Buffer not ready: {self.buffer_fill}/{self.T_in} frames"
            )

        skeleton = np.stack(list(self._skeleton_buffer))  # [T_in, 48]
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
