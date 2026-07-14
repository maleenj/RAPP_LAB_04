# enact-hw image provenance

`ghcr.io/collaborativeroboticslab/enact-hw` (local tag:
`zed_ros2_desktop_u22.04_sdk_5.1.0_cuda_12.1.1`) is a **hand-built** image —
it is not produced by any Dockerfile in this repository. This file records
what is inside so it can be reproduced if ever needed.

## Contents

| Layer | Version |
|---|---|
| Base | Ubuntu 22.04, CUDA 12.1.1 (devel) |
| ZED SDK | 5.1.0 (Stereolabs), with ZED ROS 2 wrapper |
| ROS 2 | Humble desktop |
| MoveIt 2 | humble binaries incl. `moveit-servo` |
| UR driver stack | `ros-humble-ur*` (UR10 fallback support) |
| Techman driver | `tm2_ros2` (humble branch), built in `/tm2_ws` — includes `tm_driver`, `tm12s_moveit_config`, `tm_description` |
| ros2_control | humble binaries + controllers |

Entrypoint expectations: the compose file bind-mounts
`docker/hw/entrypoint.sh` over `/entrypoint_hw.sh`, servo/controller YAMLs
and URDF/SRDF prop overrides into
`/tm2_ws/install/tm12s_moveit_config/share/tm12s_moveit_config/config/`.

## Distribution / licensing

The image embeds the ZED SDK. Stereolabs distributes SDK-containing Docker
images publicly themselves, but until their EULA terms are explicitly
confirmed for third-party redistribution, keep the GHCR package **private**
and grant access on request.

## Reconstruction path (future work)

Base a new Dockerfile on Stereolabs' official ZED + ROS 2 images, then add
the TM2 driver clone/build and the apt packages above.
`legacy/docker_zed/Dockerfile.desktop-humble` is an older reference (SDK
4.2.x) — a starting point, not buildable as-is.

## Backups

A compressed export of the working image exists at
`~/enact_backups/rapp_hw_image_2026-07-14.tar.gz` (restore with
`docker load < file.tar.gz`). Keep a copy off-machine.
