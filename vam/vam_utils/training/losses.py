"""Loss functions for Vision-Action Model training.

Composite loss: prediction (MSE) + smoothness (velocity) + acceleration.
All operate on normalized joint space.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from vam_utils.config.model_config import TrainingConfig


def joint_limit_penalty(
    pred: torch.Tensor,
    limits_normed: torch.Tensor,
) -> torch.Tensor:
    """Compute squared penalty for joint limit violations in normalized space.

    Args:
        pred: [batch, T_out, joint_dim] predicted joint angles (normalized)
        limits_normed: [joint_dim, 2] lower/upper limits in normalized space

    Returns:
        Scalar penalty (mean squared violation).
    """
    lower = limits_normed[:, 0]  # [joint_dim]
    upper = limits_normed[:, 1]  # [joint_dim]
    violation = F.relu(lower - pred) + F.relu(pred - upper)  # [B, T, J]
    return (violation ** 2).mean()


def vam_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: TrainingConfig,
    joint_limits_normed: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute composite VAM training loss.

    Args:
        pred:   [batch, T_out, joint_dim] predicted joint angles (normalized)
        target: [batch, T_out, joint_dim] ground truth joint angles (normalized)
        config: training config with loss weights
        joint_limits_normed: [joint_dim, 2] normalized joint limits (optional)

    Returns:
        Dict with keys: "total", "prediction", "smoothness", "acceleration",
        and optionally "joint_limit"
    """
    # Position loss (MSE on joint angles)
    prediction = F.mse_loss(pred, target)

    # Velocity loss (temporal smoothness)
    pred_vel = pred[:, 1:] - pred[:, :-1]
    target_vel = target[:, 1:] - target[:, :-1]
    smoothness = F.mse_loss(pred_vel, target_vel)

    # Acceleration loss (smooth motion)
    pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
    target_acc = target_vel[:, 1:] - target_vel[:, :-1]
    acceleration = F.mse_loss(pred_acc, target_acc)

    total = (
        config.prediction_weight * prediction
        + config.smoothness_weight * smoothness
        + config.acceleration_weight * acceleration
    )

    result = {
        "total": total,
        "prediction": prediction,
        "smoothness": smoothness,
        "acceleration": acceleration,
    }

    # Joint limit penalty (optional)
    if joint_limits_normed is not None and config.joint_limit_weight > 0:
        jl_penalty = joint_limit_penalty(pred, joint_limits_normed)
        total = total + config.joint_limit_weight * jl_penalty
        result["joint_limit"] = jl_penalty
        result["total"] = total

    return result
