# legacy/docker_zed — stale reference only, NOT buildable

`Dockerfile.desktop-humble` is an old reference copy (ZED SDK 4.2.x /
CUDA 12.6.3) and references files (`tmp_sources/`, `ros_entrypoint.sh`)
that are not in this repository — it will **not** build as-is.

The hardware container actually runs a pre-built image
(`zed_ros2_desktop_u22.04_sdk_5.1.0_cuda_12.1.1`, published as
`enact-hw`) that was hand-built outside this repo. See
`docker/hw/PROVENANCE.md` for what that image contains, and
`docker/docker-compose.hw.yml` for how it is run.

Kept only as a starting point if the hw image ever needs to be
reconstructed from scratch (a buildable Dockerfile based on Stereolabs'
official ZED images is the eventual goal).
