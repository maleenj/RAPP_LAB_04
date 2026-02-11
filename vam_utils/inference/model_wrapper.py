"""Model wrapper for inference: load checkpoint, run predictions, denormalize."""

import logging

import numpy as np
import torch

from vam_utils.config.inference_config import InferenceConfig
from vam_utils.config.model_config import ModelConfig
from vam_utils.data.normalization import NormalizationStats, inverse_normalize_joints
from vam_utils.model.act import ActionChunkingTransformer

logger = logging.getLogger(__name__)


class VAMModelWrapper:
    """Thin wrapper around ActionChunkingTransformer for inference.

    Handles model loading, device placement, and denormalization so callers
    get physical joint angles (radians) directly.

    Usage:
        wrapper = VAMModelWrapper(config)
        # input_tensor: [1, T_in, 54] normalized
        joints_rad = wrapper.predict(input_tensor)  # [T_out, 6] radians
    """

    def __init__(self, config: InferenceConfig):
        self.device = torch.device(config.device)

        # Load model config and instantiate architecture
        self.model_config = ModelConfig.load(config.model_config_path)
        self.model = ActionChunkingTransformer(self.model_config)

        # Load trained weights
        checkpoint = torch.load(
            config.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Load normalization stats for denormalization
        self.norm_stats = NormalizationStats.load(config.norm_stats_path)
        self.clamp_to_limits = config.clamp_to_joint_limits

        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("best_val_loss", "?")
        logger.info(
            f"Model loaded: {self.model.count_parameters():,} params, "
            f"epoch {epoch}, val_loss {val_loss}"
        )

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
            pred_np, self.norm_stats, clamp_to_limits=self.clamp_to_limits
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
