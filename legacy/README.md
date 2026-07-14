# Legacy — retired but intact

Nothing here is part of the live TM12S pipeline. It is kept because it is
either a usable fallback (UR10) or a documented dead end that explains how
the current design was reached. Do not start here — start at the
[root README](../README.md).

| Folder / file | What it is |
|---|---|
| `notebooks_ur/` | The original UR10 training pipeline (16-keypoint models, includes non-skeleton-only variants). Superseded by `vam/notebooks/` (TM12S, 18 keypoints, skeleton-only). |
| `ur_trajectory_publisher/` | UR10 trajectory publishing + MoveIt waypoints + UR10 URDF/SRDF. The UR10 fallback robot stack. |
| `README_UR10.md` | The old root README — full UR10 setup and operation guide. |
| `experiments/` | Superseded `vam_inference` node/streamer/bridge iterations, with a [README](experiments/README.md) table explaining each. |
| `docker_zed/` | Stale ZED Dockerfile — **not buildable**, see its [README](docker_zed/README.md). |
| `misc/Lab06/` | Unrelated scan-matching/ICP lab content that once lived in the notebooks folder. |

## Running the UR10 fallback

The UR10 path predates the reorg: follow [`README_UR10.md`](README_UR10.md),
which still references the old folder layout (`notebooks/` →
`legacy/notebooks_ur/`, `vam_utils/` → `vam/vam_utils/`). The UR10-era
inference node and launch files are in [`experiments/`](experiments/README.md)
and would need their `setup.py` entries restored to run — see the note at the
bottom of that README.
