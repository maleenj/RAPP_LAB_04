# ENACT — Embodied Neural Action Translation

ENACT is a Vision-Action Model (VAM) system that lets an industrial robot arm
perform live with human actors. A ZED stereo camera tracks a performer's
skeleton (18 keypoints), a small Action Chunking Transformer (~1.1M params)
translates the pose stream into robot joint targets in real time, and a
Techman TM12S arm mirrors — or deliberately contrasts — the performer's
movement on stage. A WebSocket bridge streams the robot's "thoughts" (joints,
attention, activations) to Unity/browser visualizations, and an audience
voting web app can hot-swap the active model mid-performance.

Developed at the UC Collaborative Robotics Lab (RAPP Lab). Workshop-proven on
real hardware.

> **⚠️ Safety:** this system moves a full-size industrial robot arm based on
> live human motion. Always follow the layered safety setup described in
> [hardware/README.md](hardware/README.md) (velocity scaling, collision
> checking, watchdogs), keep the e-stop within reach, and never share the
> robot's workspace during operation.

## Architecture

Three Docker containers, all on `network_mode: host`, sharing one ROS 2
Humble DDS graph (CycloneDDS, `ROS_DOMAIN_ID=0`):

```text
                 ┌────────────────────────────────────────────────┐
                 │                    host machine                │
  ZED 2i ──USB──▶│ ┌────────────┐   ┌────────────┐  ┌───────────┐ │
                 │ │  rapp_hw   │   │  rapp_vam  │  │ rapp_viz  │ │
  TM12S ◀──LAN──▶│ │ ZED SDK    │   │ PyTorch    │  │ WebSocket │ │──▶ Unity /
                 │ │ TM driver  │◀─▶│ VAM model  │◀▶│ bridge    │ │    browser /
                 │ │ MoveIt     │DDS│ inference  │DD│ :8765     │ │    audience
                 │ └────────────┘   └────────────┘  └───────────┘ │    voting
                 │        shared: ros2_ws/ (colcon workspace)     │
                 └────────────────────────────────────────────────┘
```

| Container | Purpose | GPU | Details |
|---|---|---|---|
| `rapp_hw` | ZED camera, TM12S driver, MoveIt, PVT streamer | yes | [hardware/README.md](hardware/README.md) |
| `rapp_vam` | Model training (Jupyter) + live inference | yes | [vam/README.md](vam/README.md) |
| `rapp_viz` | ROS→WebSocket JSON bridge for visualizations | no | [viz/README.md](viz/README.md) |

## Repository layout

| Folder | What's in it |
|---|---|
| [hardware/](hardware/README.md) | TM12S operation runbook, robot/prop scripts |
| [vam/](vam/README.md) | Training notebooks, `vam_utils` Python library, training guide |
| [viz/](viz/README.md) | Visualization guide, student pack (Unity/browser clients), audience voting app |
| [ros2_ws/](ros2_ws/) | Shared ROS 2 workspace: `vam_inference`, `vam_interfaces`, `vam_viz_bridge`, `tm_teach_playback` |
| [docker/](docker/) | Dockerfiles, compose files, `.env` configuration |
| [docs/](docs/) | Design brief, technical brief, guides; `docs/archive/` holds R&D notes |
| [legacy/](legacy/README.md) | Retired UR10 pipeline and superseded experiments (kept runnable/readable) |

## Quickstart

Prerequisites: Ubuntu 22.04, an NVIDIA GPU (driver 535+), Docker,
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Sanity check: `docker run --rm --runtime=nvidia nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi`.

```bash
git clone <this-repo> && cd <this-repo>

# 1. Configure paths (defaults are fine for most setups)
cp docker/.env.example docker/.env

# 2. Create the data folder tree (~/enact_local by default)
./scripts/setup_volumes.sh

# 3. Get the starter data pack (trained models, norm stats, URDF)
#    — download link in the ENACT data folder (see Data section below),
#    unzip into ~/enact_local

# 4. Pull and start the containers ("pull for use, build only if modifying")
cd docker
docker compose up -d                              # vam (inference + Jupyter)
docker compose -f docker-compose.viz.yml up -d    # viz bridge
docker compose -f docker-compose.hw.yml up -d     # hardware (needs GPU + devices)
```

Then follow the run modes in [hardware/README.md](hardware/README.md) —
including a **no-hardware mode** (rosbag replay + RViz ghost robot) you can
try without a robot or camera.

## Data

Everything lives under one host folder (`ENACT_DATA` in `docker/.env`,
default `~/enact_local`), mounted into the containers as `/data/...`:

| Host | Container | Contents |
|---|---|---|
| `enact_local/rosbags/` | `/data/rosbags` | recorded demonstrations |
| `enact_local/csvdata/` | `/data/processed` | CSVs, tensors, norm stats, URDFs |
| `enact_local/models/` | `/data/models` | trained checkpoints (see `vam_models.yaml`) |
| `enact_local/logs/` | `/data/logs` | training / TensorBoard logs |

Curated models and sample data are hosted in the **ENACT data folder on
Google Drive** — _link TBD_ (versioned subfolders match release tags).

## Development workflow

- `main` — stable; what workshops run; matches the published Docker image tags.
- `dev` — active development (may be messy).
- Releases: merge `dev`→`main`, tag `vX.Y.Z`, push matching image tags.

See [CONTRIBUTING.md](CONTRIBUTING.md). To cite this work, see
[CITATION.cff](CITATION.cff). Licensed under [Apache-2.0](LICENSE).
