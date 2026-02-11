"""Configuration dataclasses for model architecture and training."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Action Chunking Transformer architecture hyperparameters.

    Starting with a small model (~1.2M params) appropriate for ~7K training
    windows. Scale up only if underfitting is observed.

    Usage:
        config = ModelConfig()
        config = ModelConfig(d_model=256, n_heads=8)  # larger variant
        config = ModelConfig.load("path/to/model_config.json")
    """

    # --- Transformer dimensions ---
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512          # feedforward dim (typically 4 * d_model)
    dropout: float = 0.1

    # --- Encoder (processes fused skeleton + robot sequence) ---
    n_encoder_layers: int = 3

    # --- Decoder (generates action chunk via cross-attention) ---
    n_decoder_layers: int = 2

    # --- Robot state MLP ---
    d_robot_hidden: int = 64

    # --- Data dimensions (must match DataPipelineConfig) ---
    skeleton_dim: int = 48   # 16 keypoints * 3 coordinates
    joint_dim: int = 6       # 6 DOF UR10
    T_in: int = 10           # input context frames
    T_out: int = 10          # prediction horizon frames

    # --- Derived ---
    input_dim: int = field(init=False)

    def __post_init__(self):
        self.input_dim = self.skeleton_dim + self.joint_dim

    def save(self, path: Path = None) -> Path:
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = asdict(self)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        """Load config from JSON."""
        with open(path) as f:
            config_dict = json.load(f)
        config_dict.pop("input_dim", None)
        return cls(**config_dict)


@dataclass
class TrainingConfig:
    """Training loop hyperparameters.

    Usage:
        config = TrainingConfig()
        config = TrainingConfig(epochs=100, learning_rate=3e-4)
        config = TrainingConfig.load("path/to/training_config.json")
    """

    # --- Optimizer ---
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # --- Scheduler (linear warmup + cosine decay) ---
    warmup_epochs: int = 5

    # --- Loss weights ---
    prediction_weight: float = 1.0
    smoothness_weight: float = 0.1
    acceleration_weight: float = 0.05
    joint_limit_weight: float = 0.1

    # --- Training ---
    epochs: int = 200
    grad_clip_norm: float = 1.0

    # --- Checkpointing ---
    checkpoint_dir: Path = Path("/data/models")
    save_every_n_epochs: int = 10

    # --- Early stopping ---
    early_stopping_patience: int = 30

    # --- Logging ---
    log_dir: Path = Path("/data/logs")

    def save(self, path: Path = None) -> Path:
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
    def load(cls, path: str | Path) -> "TrainingConfig":
        """Load config from JSON."""
        with open(path) as f:
            config_dict = json.load(f)
        for key in ["checkpoint_dir", "log_dir"]:
            if key in config_dict:
                config_dict[key] = Path(config_dict[key])
        return cls(**config_dict)
