# VAM — the Vision-Action Model

The heart of ENACT: an Action Chunking Transformer (ACT, ~1.1M parameters)
that maps a live human skeleton stream (ZED BODY_18, 18 keypoints) to TM12S
robot joint targets. Current models are **skeleton-only**: the input is the
skeleton window alone, with no robot joint feedback in the model input (this
makes inference robust to the robot's starting pose).

The VAM lifecycle has **two halves**:

## 1. Training (offline — `rapp_vam` container + Jupyter)

Record human/robot demonstrations as rosbags, then walk the notebooks in
order to produce a trained checkpoint:

| Notebook | Does |
|---|---|
| `notebooks/00_setup_urdf.ipynb` | One-time: extract the TM12S URDF, validate forward kinematics |
| `notebooks/01_process_rosbags.ipynb` | Rosbags → synchronized CSVs (skeleton ↔ joint states), leader selection |
| `notebooks/02_prepare_training_data.ipynb` | Episode splits, normalization stats, PyTorch tensors |
| `notebooks/03_train_vam_skeleton_only.ipynb` | Train the skeleton-only ACT model |
| `notebooks/04_inference_test_skeleton_only.ipynb` | Offline evaluation / parameter sweeps |
| `notebooks/05_pvt_diagnostics.ipynb` | Diagnose robot-side streaming behavior from CSV logs |

Output: a model folder in `/data/models/<name>/` (`best.pt` +
`model_config.json`) and `norm_stats.pt` in `/data/processed/tensors/...`.

**→ Step-by-step guide: [TRAINING.md](TRAINING.md)**

## 2. Inference (live — `rapp_hw` + `rapp_vam` containers)

A trained checkpoint is registered in the model registry
[`ros2_ws/src/vam_inference/config/vam_models.yaml`](../ros2_ws/src/vam_inference/config/vam_models.yaml),
then driven by the ROS 2 pipeline:

```text
ZED skeleton ─▶ vam_tm12s_node_viz (ACT inference, safety clamps, feedback filter)
                        │ /vam/joint_targets
                        ▼
             vam_pvt_streamer (Butterworth filter, collision check, velocity scaling)
                        │ PVT points
                        ▼
                TM12S firmware (1 kHz spline interpolation)
```

- Inference code lives in the [`vam_inference`](../ros2_ws/src/vam_inference/)
  ROS package and reuses this folder's [`vam_utils/`](vam_utils/) library
  (`inference/` — model wrapper, input assembler, temporal ensemble, safety
  checker; `model/act.py` — the network).
- Models **hot-swap at runtime** (e.g. mirror ↔ contrast during a show):
  `ros2 service call /vam/switch_model vam_interfaces/srv/SwitchModel "{model_id: 2}"`
- **→ Run procedures, tuning tables, and safety architecture:
  [hardware/README.md](../hardware/README.md)**

## What's in this folder

| Path | Purpose |
|---|---|
| `notebooks/` | The training pipeline (00→05 above) |
| `vam_utils/` | Python library shared by notebooks AND the ROS inference nodes (config, data, model, training, inference submodules). Installed editable inside the containers. |
| `TRAINING.md` | Detailed training walkthrough |

Background reading: [docs/design_brief.md](../docs/design_brief.md) (project
concept) and [docs/vam_technical_brief.md](../docs/vam_technical_brief.md)
(architecture, latency, training details).
