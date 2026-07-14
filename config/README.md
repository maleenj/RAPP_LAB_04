# config/

Mounted read-only into all containers at `/config`.

Intended for machine/site-specific calibration files that are **not**
committed (see `.gitignore`), e.g.:

- `calibration.yaml` — sensor calibration
- `camera_to_robot_transform.yaml` — calibrated camera→robot extrinsics
  (currently the pipeline uses an identity-transform placeholder; put the
  real calibration here when available)

This README exists so the folder is present in fresh clones — if it were
missing, `docker compose up` would auto-create it owned by root.
