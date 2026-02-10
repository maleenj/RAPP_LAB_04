"""Configuration dataclass for the data preparation pipeline."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataPipelineConfig:
    """All hyperparameters for the data preparation pipeline.

    Derived fields (min_episode_length, jitter_margin, output paths) are
    auto-computed in __post_init__ so you only need to change T_in/T_out
    and everything stays consistent.

    Usage:
        # Defaults
        config = DataPipelineConfig()

        # Override for experiment
        config = DataPipelineConfig(T_in=10, T_out=15, experiment_name="tout15")

        # Reload from a previous run
        config = DataPipelineConfig.load("/data/processed/tensors/tout15/pipeline_config.json")
    """

    # --- Experiment naming ---
    experiment_name: str = "default"

    # --- Base paths (container paths) ---
    csv_dir: Path = Path("/data/processed/csv")
    metadata_path: Path = Path("/data/processed/recordings_metadata.csv")
    base_output_dir: Path = Path("/data/processed/tensors")

    # --- Window parameters (TUNE THESE) ---
    T_in: int = 10
    T_out: int = 10
    stride: int = 1

    # --- Feature dimensions ---
    skeleton_dim: int = 48  # 16 keypoints * 3 coordinates
    joint_dim: int = 6      # 6 DOF UR10
    ee_dim: int = 6         # end-effector pose (xyz + rpy)
    input_dim: int = 54     # skeleton_dim + joint_dim

    # --- Column names ---
    skeleton_cols: List[str] = field(default_factory=lambda: [
        f"sk_{i}_{axis}" for i in range(16) for axis in ["x", "y", "z"]
    ])
    joint_cols: List[str] = field(default_factory=lambda: [
        f"j{i}" for i in range(6)
    ])
    ee_cols: List[str] = field(default_factory=lambda: [
        "ee_x", "ee_y", "ee_z", "ee_roll", "ee_pitch", "ee_yaw"
    ])

    # --- Split ratios ---
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42

    # --- Augmentation (training only, TUNE THESE) ---
    augmentation_enabled: bool = True
    temporal_jitter_max: int = 2
    skeleton_noise_std: float = 0.01  # meters

    # --- DataLoader ---
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # --- Derived fields (auto-computed, do not set manually) ---
    min_episode_length: int = field(init=False)
    jitter_margin: int = field(init=False)
    output_dir: Path = field(init=False)
    normalization_stats_path: Path = field(init=False)

    def __post_init__(self):
        """Auto-compute derived fields from primary parameters."""
        self.min_episode_length = self.T_in + self.T_out
        self.jitter_margin = 2 * self.temporal_jitter_max
        self.output_dir = self.base_output_dir / self.experiment_name
        self.normalization_stats_path = self.output_dir / "norm_stats.pt"

    def save(self, path: Path = None) -> Path:
        """Save config to JSON. Defaults to output_dir/pipeline_config.json."""
        if path is None:
            path = self.output_dir / "pipeline_config.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self)
        # Convert Path objects to strings for JSON
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DataPipelineConfig":
        """Load config from a saved JSON file.

        Useful for inference: reload the exact config used during training.
        """
        with open(path) as f:
            config_dict = json.load(f)

        # Remove derived fields (they'll be recomputed by __post_init__)
        for derived in ["min_episode_length", "jitter_margin", "output_dir",
                        "normalization_stats_path"]:
            config_dict.pop(derived, None)

        # Convert string paths back to Path objects
        path_fields = ["csv_dir", "metadata_path", "base_output_dir"]
        for key in path_fields:
            if key in config_dict:
                config_dict[key] = Path(config_dict[key])

        return cls(**config_dict)
