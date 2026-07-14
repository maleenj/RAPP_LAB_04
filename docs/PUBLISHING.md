# Publishing guide (maintainer notes)

How to publish this repo, its Docker images, and its data. Written for the
maintainer; external users never need this file.

## 1. One-time: hand off to the public `enact` repo

Create the public repo (GitHub → collaborativeroboticslab → New →
`enact`), then copy the reorganized tree WITHOUT git history or local junk:

```bash
rsync -a \
  --exclude '.git' \
  --exclude 'ros2_ws/build*' --exclude 'ros2_ws/install*' --exclude 'ros2_ws/log*' \
  --exclude 'docker/.env' \
  --exclude '.ipynb_checkpoints' --exclude '__pycache__' --exclude '*.egg-info' \
  --exclude '.claude' \
  ~/git/RAPP_LAB_04/ ~/git/enact/

cd ~/git/enact
git init && git add -A
git commit -m "ENACT v1.0.0 — first public release"
git branch -M main
git remote add origin git@github.com:collaborativeroboticslab/enact.git
git push -u origin main
git checkout -b dev && git push -u origin dev
git tag v1.0.0 && git push --tags
```

After this, **develop in `enact`** (dev branch); the old private
RAPP_LAB_04 repo is a frozen archive.

## 2. Push Docker images to GHCR

Needs a GitHub login with `write:packages` on the org (once per machine):

```bash
gh auth token | docker login ghcr.io -u <your-github-username> --password-stdin
```

Tag + push (repeat with a version tag for releases):

```bash
for img in enact-vam enact-viz enact-hw; do
  docker tag ghcr.io/collaborativeroboticslab/$img:latest \
             ghcr.io/collaborativeroboticslab/$img:v1.0.0
done
docker push ghcr.io/collaborativeroboticslab/enact-vam:latest
docker push ghcr.io/collaborativeroboticslab/enact-vam:v1.0.0
docker push ghcr.io/collaborativeroboticslab/enact-viz:latest
docker push ghcr.io/collaborativeroboticslab/enact-viz:v1.0.0
docker push ghcr.io/collaborativeroboticslab/enact-hw:latest    # ~23GB, slow
docker push ghcr.io/collaborativeroboticslab/enact-hw:v1.0.0
```

Then on github.com → org → Packages:

- `enact-vam`, `enact-viz` → **public**
- `enact-hw` → **private** until ZED SDK redistribution is confirmed with
  Stereolabs (see `docker/hw/PROVENANCE.md`); grant access on request.

Optional later (v2): a GitHub Actions workflow that builds and pushes
enact-vam/enact-viz on every `v*` tag (`docker/build-push-action`,
`permissions: packages: write`). enact-hw stays manual — it has no
Dockerfile.

## 3. Data: the ENACT Google Drive folder

Layout (versioned subfolders so code releases match their data):

```text
ENACT_data/
├── v1.0/
│   └── enact_local_starter.zip
├── recordings/
│   └── enact_viz_01.jsonl.txt      (89MB viz replay, from ~/rapplab04_release_assets/)
└── training_data/                  (optional: rosbags/tensors, or "on request")
```

Build the starter pack (minimum needed to RUN without training —
checkpoints referenced by `vam_models.yaml`, norm stats, URDF):

```bash
cd ~
mkdir -p pack/enact_local/{rosbags,logs,models,csvdata/tensors}
cp -r enact_local/models/vam_skelonly_tm12_mirror_pom20260405_0314 \
      enact_local/models/vam_skelonly_tm12_contrast_pom20260413_1323 \
      pack/enact_local/models/
cp -r enact_local/csvdata/tensors/2026_04_05_tm12_mirror_pom \
      pack/enact_local/csvdata/tensors/   # norm_stats.pt (+ contrast set)
cp enact_local/csvdata/tm12s.urdf enact_local/csvdata/vam_tm12s.rviz \
      pack/enact_local/csvdata/ 2>/dev/null
cd pack && zip -r enact_local_starter.zip enact_local && cd ..
```

(Adjust the model/tensor names to whatever `vam_models.yaml` currently
lists.) Upload to `ENACT_data/v1.0/`, set link-sharing to "anyone with the
link", and paste the folder link into the root `README.md` and
`viz/student_pack/recordings/README.md` where marked _TBD_.

Never overwrite an old `vX.Y/` subfolder — add a new one, so old code
versions keep matching their data. When a paper is published, snapshot the
starter pack to Zenodo for a citable DOI.

## 4. Release checklist

1. `dev` is workshop-proven → merge to `main`.
2. Smoke checks (no robot needed):
   - `cd docker && docker compose config` (×3 compose files) renders cleanly
   - all three containers start; `rapp_vam` builds the packages
     (`ros2 pkg executables vam_inference` → 6 entries) and sees CUDA
   - `ros2 launch vam_inference vam_tm12s_headless_viz.launch.py -s` loads
   - viz bridge answers on :8765
3. `git tag vX.Y.Z && git push --tags`
4. Re-tag + push images with the same version (section 2).
5. If models changed: new `ENACT_data/vX.Y/enact_local_starter.zip`
   (section 3).
6. GitHub → Releases → draft release from the tag; mention the matching
   image tags and data folder version in the notes.
