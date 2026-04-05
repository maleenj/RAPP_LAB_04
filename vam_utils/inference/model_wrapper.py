"""Model wrapper for inference: load checkpoint, run predictions, denormalize."""

import json
import logging
from pathlib import Path

import numpy as np
import torch

from vam_utils.config.inference_config import InferenceConfig
from vam_utils.config.model_config import ModelConfig
from vam_utils.data.normalization import NormalizationStats, inverse_normalize_joints
from vam_utils.model.act import ActionChunkingTransformer, SkeletonOnlyACT

logger = logging.getLogger(__name__)


class VAMModelWrapper:
    """Thin wrapper around ActionChunkingTransformer for inference.

    Handles model loading, device placement, and denormalization so callers
    get physical joint angles (radians) directly.

    Auto-detects skeleton-only models via model_type.json in the checkpoint
    directory. Falls back to checking checkpoint keys if the file is missing.

    Usage:
        wrapper = VAMModelWrapper(config)
        # input_tensor: [1, T_in, 54] or [1, T_in, 48] (skeleton-only)
        joints_rad = wrapper.predict(input_tensor)  # [T_out, 6] radians
    """

    def __init__(self, config: InferenceConfig, joint_limits=None):
        self.device = torch.device(config.device)

        # Load model config and trained weights
        self.model_config = ModelConfig.load(config.model_config_path)
        checkpoint = torch.load(
            config.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Auto-detect model type
        self.skeleton_only = self._detect_skeleton_only(
            config.checkpoint_path, checkpoint
        )

        # Instantiate correct architecture
        if self.skeleton_only:
            self.model = SkeletonOnlyACT(self.model_config)
        else:
            self.model = ActionChunkingTransformer(self.model_config)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load normalization stats for denormalization
        self.norm_stats = NormalizationStats.load(config.norm_stats_path)
        self.clamp_to_limits = config.clamp_to_joint_limits
        self.joint_limits = joint_limits

        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("best_val_loss", "?")
        model_type_str = "SkeletonOnlyACT" if self.skeleton_only else "ActionChunkingTransformer"
        logger.info(
            f"Model loaded: {model_type_str}, {self.model.count_parameters():,} params, "
            f"epoch {epoch}, val_loss {val_loss}"
        )

    @staticmethod
    def _detect_skeleton_only(checkpoint_path: Path, checkpoint: dict) -> bool:
        """Detect if this is a skeleton-only model."""
        # Check for model_type.json in the checkpoint directory
        model_type_path = Path(checkpoint_path).parent / "model_type.json"
        if model_type_path.exists():
            with open(model_type_path) as f:
                info = json.load(f)
            return info.get("model_type") == "skeleton_only"

        # Fallback: check if checkpoint has robot_proj keys
        state_keys = checkpoint["model_state_dict"].keys()
        has_robot_proj = any(k.startswith("robot_proj") for k in state_keys)
        return not has_robot_proj

    @torch.inference_mode()
    def predict(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Run a single forward pass and return denormalized joint angles.

        Args:
            input_tensor: [1, T_in, 54] normalized (skeleton + joints).

        Returns:
            np.ndarray of shape [T_out, 6] — joint angles in radians,
            optionally clamped to UR10 physical limits.
        """
        pred_normed = self.model(input_tensor.to(self.device))  # [1, T_out, 6]
        pred_np = pred_normed.cpu().numpy()[0]  # [T_out, 6]
        return inverse_normalize_joints(
            pred_np, self.norm_stats,
            clamp_to_limits=self.clamp_to_limits,
            joint_limits=self.joint_limits,
        )

    @torch.inference_mode()
    def predict_normalized(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Run forward pass and return predictions in normalized space.

        Useful for comparison with normalized ground truth during evaluation.

        Args:
            input_tensor: [1, T_in, 54] normalized.

        Returns:
            np.ndarray of shape [T_out, 6] in normalized space.
        """
        pred = self.model(input_tensor.to(self.device))
        return pred.cpu().numpy()[0]
