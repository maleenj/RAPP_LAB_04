# Contributing

- `main` is stable: it is what workshops run, and it always matches the
  published Docker image tags (`ghcr.io/collaborativeroboticslab/enact-*`).
- Day-to-day work happens on `dev` (or feature branches off it). Messy is
  fine there.
- When `dev` is workshop-proven: merge to `main`, tag `vX.Y.Z`, and push
  Docker images with the same tag (see `docs/PUBLISHING.md`).
- Don't commit data: rosbags, CSVs, checkpoints, and `.env` are gitignored
  by design. Models and datasets are shared via the ENACT data folder
  (Google Drive), not git.
- Before pushing changes that touch the ROS packages or compose files, run
  the smoke checks in `docs/PUBLISHING.md` (containers start, package
  builds, launch files dry-run with `-s`, viz bridge serves on :8765).
- Working code is sacred: prefer adding a new module over rewriting a
  workshop-proven one mid-season; superseded iterations get archived to
  `legacy/experiments/` at release time.
