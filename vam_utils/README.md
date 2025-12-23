# VAM Utils - Vision-Action Model Utilities

Modular Python package for RAPP Lab 04: Training and deploying Action Chunking Transformers for human-robot ensemble performance.

## Installation

```bash
# From within the Docker container (automatic via entrypoint)
cd /workspace/vam_utils
pip install -e .

# Or from host (if developing outside container)
cd vam_utils
pip install -e .
```

## Package Structure

```
vam_utils/
├── config/              # Configuration management
│   ├── config_loader.py
│   └── schema_validator.py
│
├── kinematics/          # Robot kinematics
│   ├── urdf_loader.py
│   ├── forward_kinematics.py
│   ├── joint_limits.py
│   └── workspace.py
│
├── data/                # Data processing
│   ├── rosbag_reader.py
│   ├── synchronizer.py
│   ├── transforms.py
│   ├── csv_handler.py
│   ├── dataset.py
│   ├── augmentation.py
│   └── quality_checks.py
│
├── model/               # Model architecture
│   ├── act_transformer.py
│   ├── skeleton_encoder.py
│   ├── robot_encoder.py
│   ├── decoder.py
│   ├── losses.py
│   └── model_wrapper.py
│
├── training/            # Training utilities
│   ├── trainer.py
│   ├── evaluator.py
│   ├── checkpointing.py
│   └── logger.py
│
├── inference/           # Real-time inference
│   ├── buffer_manager.py
│   ├── trajectory_smoother.py
│   ├── safety_checker.py
│   └── ros_bridge.py
│
└── visualization/       # Visualization tools
    ├── skeleton_viz.py
    ├── robot_viz.py
    ├── combined_viz.py
    ├── trajectory_viz.py
    └── attention_viz.py
```

## Usage Examples

### Configuration Loading

```python
from vam_utils.config import load_config

config = load_config('/config/training.yaml')
batch_size = config['training']['batch_size']
```

### Forward Kinematics

```python
from vam_utils.kinematics import load_urdf, compute_fk

urdf_model = load_urdf('/config/ur10.urdf')
joint_angles = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
end_effector_pose = compute_fk(joint_angles, urdf_model)
```

### Visualization

```python
from vam_utils.visualization import plot_skeleton_and_robot

fig = plot_skeleton_and_robot(
    skeleton_keypoints=skeleton_data,
    joint_angles=robot_joint_angles,
    urdf_path='/config/ur10.urdf'
)
fig.show()
```

### PyTorch Dataset

```python
from vam_utils.data import VAMDataset
from torch.utils.data import DataLoader

dataset = VAMDataset(
    csv_paths=['rec_001.csv', 'rec_002.csv'],
    T_in=10,
    T_out=50
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Development

### Running Tests

```python
pytest tests/
```

### Type Checking

```bash
mypy vam_utils/
```

### Code Formatting

```bash
black vam_utils/
```

## License

MIT License - See LICENSE file for details
