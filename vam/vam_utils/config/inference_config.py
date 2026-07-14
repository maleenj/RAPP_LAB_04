"""Configuration dataclass for the real-time inference pipeline."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class InferenceConfig:
    """All parameters for the inference pipeline.

    Usage:
        config = InferenceConfig()
        config = InferenceConfig(prediction_stride_K=3, ensemble_decay_weight=0.05)
        config = InferenceConfig.load("path/to/inference_config.json")
    """

    # --- Model paths (container paths) ---
    checkpoint_path: Path = Path("/data/models/vam_20260210_2342/best.pt")
    model_config_path: Path = Path("/data/models/vam_20260210_2342/model_config.json")
    norm_stats_path: Path = Path("/data/processed/tensors/2026_02_10_tin10_tout10/norm_stats.pt")
    device: str = "cuda"

    # --- Temporal ensemble ---
    prediction_stride_K: int = 1        # re-predict every K frames (K=1 empirically smoothest)
    ensemble_decay_weight: float = 0.5   # exponential decay lambda (sweep: best RMSE + smoothness)

    # --- Execution ---
    control_rate_hz: float = 15.0       # match training data rate

    # --- Safety ---
    max_joint_velocity_rad_s: float = 1.0
    max_joint_acceleration_rad_s2: float = 5.0
    clamp_to_joint_limits: bool = True

    # --- Derived ---
    max_ensemble_history: int = field(init=False)
    warmup_frames: int = field(init=False)
    frame_dt: float = field(init=False)

    def __post_init__(self):
        self.max_ensemble_history = 10 // self.prediction_stride_K  # T_out // K
        self.warmup_frames = 10 + self.max_ensemble_history * self.prediction_stride_K
        self.frame_dt = 1.0 / self.control_rate_hz

    def save(self, path: Path) -> Path:
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = asdict(self)
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "InferenceConfig":
        """Load config from JSON."""
        with open(path) as f:
            config_dict = json.load(f)
        for derived in ["max_ensemble_history", "warmup_frames", "frame_dt"]:
            config_dict.pop(derived, None)
        for key in ["checkpoint_path", "model_config_path", "norm_stats_path"]:
            if key in config_dict:
                config_dict[key] = Path(config_dict[key])
        return cls(**config_dict)
