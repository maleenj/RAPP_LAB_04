# Training a VAM model, end to end

This guide takes you from raw demonstration recordings to a trained model
registered for live inference. Everything runs inside the `rapp_vam`
container via Jupyter.

## 0. Environment

```bash
cd docker && docker compose up -d          # starts rapp_vam (Jupyter on :8888)
../scripts/run_jupyter.sh                  # if Jupyter isn't already serving
```

Open <http://localhost:8888> → the notebooks are at
`/workspace/notebooks_tm` (this repo's `vam/notebooks/`).

Host ↔ container data map (host root = `ENACT_DATA` in `docker/.env`,
default `~/enact_local`):

| Host | Container | Used for |
|---|---|---|
| `enact_local/rosbags/` | `/data/rosbags` | input recordings |
| `enact_local/csvdata/` | `/data/processed` | CSVs, tensors, norm stats, URDF |
| `enact_local/models/` | `/data/models` | trained checkpoints |
| `enact_local/logs/` | `/data/logs` | TensorBoard logs |

## 1. Record demonstrations

Record rosbags containing the ZED skeleton topic and the robot joint states
while an actor leads and the robot (tele-operated or teach-pendant driven)
follows — see `ros2_ws/src/tm_teach_playback/` for waypoint playback during
recording sessions. Copy the bags into `~/enact_local/rosbags/`.

## 2. Run the notebooks in order

| # | Notebook | Key outputs | Watch out for |
|---|---|---|---|
| 00 | `00_setup_urdf.ipynb` | `/data/processed/tm12s.urdf`, static transforms | One-time per robot. Validates forward kinematics against the URDF. |
| 01 | `01_process_rosbags.ipynb` | Synchronized per-recording CSVs + metadata | **Skeleton ID selection**: ZED assigns arbitrary IDs (e.g. 5 and 22, not 0/1). Use the notebook's 3D visualization to pick the leader; verify detected IDs before setting `SELECTED_SKELETON_ID`. |
| 02 | `02_prepare_training_data.ipynb` | Train/val/test tensors + `norm_stats.pt` under `/data/processed/tensors/<dataset>/` | Splits are per-episode (70/15/15) to avoid leakage. Normalization uses training stats only. |
| 03 | `03_train_vam_skeleton_only.ipynb` | Model folder `/data/models/<name>/` with `best.pt` + `model_config.json` | Skeleton-only input `[B, T_in, 54]` (18 keypoints × 3). Monitor with TensorBoard (`/data/logs`). Rough expectation: minutes-to-an-hour on a modern GPU, not days — the model is ~1.1M params. |
| 04 | `04_inference_test_skeleton_only.ipynb` | Offline predictions, temporal-ensemble parameter sweeps | Use this to sanity-check smoothness BEFORE going near the robot. |
| 05 | `05_pvt_diagnostics.ipynb` | Plots from streamer/inference diagnostic CSVs | Only needed when tuning robot-side motion (vibration, CPERR errors). |

Data format notes:

- Skeleton: ZED **BODY_18** — all 18 keypoints, 3 coordinates each (54
  features/frame). The legacy UR10 pipeline used 16 keypoints; TM notebooks
  use all 18.
- The camera→robot transform is currently an identity placeholder; a
  calibrated transform belongs in `config/` when available.

## 3. Register the model for live use

Add an entry to
[`ros2_ws/src/vam_inference/config/vam_models.yaml`](../ros2_ws/src/vam_inference/config/vam_models.yaml):

```yaml
models:
  3:
    name: "my_new_model"
    description: "What/when/who this was trained on"
    model_dir: "/data/models/vam_skelonly_tm12_<timestamp>"
    norm_stats_path: "/data/processed/tensors/<dataset>/norm_stats.pt"
```

Rebuild the package inside the container (`colcon build --packages-select
vam_inference`), then select it at launch (`active_model:=3`) or hot-swap at
runtime (`/vam/switch_model`). Run procedure and tuning:
[hardware/README.md](../hardware/README.md).

## 4. Share it (optional)

Upload the model folder + its `norm_stats.pt` to the versioned ENACT data
folder on Google Drive so others can drop it into their own
`~/enact_local/` — see `docs/PUBLISHING.md`.
