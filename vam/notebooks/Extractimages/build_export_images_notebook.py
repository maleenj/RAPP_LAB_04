"""Generate the export_images notebook. Run once; produces export_images.ipynb."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "export_images.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

cells.append(md("""# Export Images from Rosbags

**RAPP Lab 04 — Time-lapse frame extraction**

Reads the ZED RGB image topic (`/zed/zed_node/rgb/color/rect/image`) from up to 8 rosbags
and writes JPEGs into one folder per bag. Each output folder lives under
`OUTPUT_BASE / <rosbag_name>/` and contains stride-sampled frames named by index and
timestamp (e.g. `frame_000123_1775647003456789012.jpg`).

Edit the **CONFIG** cell, then **Run All**.
"""))

cells.append(md("## CONFIG — edit these"))

cells.append(code('''# =============================================================================
# CONFIG — edit for each run
# =============================================================================

# Eight rosbag folder names under /data/rosbags/. Fill these in.
ROSBAG_NAMES = [
    "",  # 1
    "",  # 2
    "",  # 3
    "",  # 4
    "",  # 5
    "",  # 6
    "",  # 7
    "",  # 8
]

# Image topic to extract.
IMAGE_TOPIC = "/zed/zed_node/rgb/color/rect/image"

# Sample every Nth message. 1 = every frame. Source is ~15 Hz, so stride=5 → ~3 fps.
STRIDE = 5

# JPEG quality (1-100). 90 is a good time-lapse default.
JPEG_QUALITY = 90

# Resize longest edge to this many pixels. None = original resolution.
RESIZE_LONGEST_EDGE = None

# Skip bags whose output folder already has images (idempotent re-runs).
SKIP_IF_OUTPUT_EXISTS = True

# =============================================================================
'''))

cells.append(md("## 1. Imports and paths"))

cells.append(code('''import os
import sys
import sqlite3
from pathlib import Path

import numpy as np
import cv2

from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

ROSBAG_DIR = Path("/data/rosbags")
OUTPUT_BASE = Path("/data/processed/timelapse_frames")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

print(f"Rosbag source : {ROSBAG_DIR}")
print(f"Output base   : {OUTPUT_BASE}")
print(f"Topic         : {IMAGE_TOPIC}")
print(f"Stride        : every {STRIDE} frame(s)")
'''))

cells.append(md("## 2. Helpers"))

cells.append(code('''def find_db(rosbag_name: str) -> Path:
    """Locate the .db3 file inside a rosbag folder."""
    bag_dir = ROSBAG_DIR / rosbag_name
    if not bag_dir.exists():
        raise FileNotFoundError(f"Rosbag folder not found: {bag_dir}")
    dbs = sorted(bag_dir.glob("*.db3"))
    if not dbs:
        raise FileNotFoundError(f"No .db3 file inside {bag_dir}")
    return dbs[0]


def image_msg_to_bgr(msg):
    """Convert sensor_msgs/Image to a BGR numpy array without cv_bridge."""
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, w, 3)
        return img[..., ::-1].copy() if enc == "rgb8" else img.copy()
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(h, w, 4)
        bgra = img[..., [2, 1, 0, 3]] if enc == "rgba8" else img
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    if enc in ("16uc1", "mono16"):
        img16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
        norm = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def maybe_resize(img, longest_edge):
    if longest_edge is None:
        return img
    h, w = img.shape[:2]
    scale = longest_edge / max(h, w)
    if scale >= 1.0:
        return img
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_topic_id(conn, topic_name):
    row = conn.execute("SELECT id FROM topics WHERE name = ?", (topic_name,)).fetchone()
    if row is None:
        raise KeyError(f"Topic {topic_name} not in bag")
    return row[0]
'''))

cells.append(md("## 3. Extract a single bag"))

cells.append(code('''def export_bag(rosbag_name):
    """Extract images from one rosbag into its own subfolder. Returns a summary dict."""
    summary = {"bag": rosbag_name, "written": 0, "total_msgs": 0, "skipped": False, "error": None}

    try:
        db_path = find_db(rosbag_name)
    except FileNotFoundError as e:
        summary["error"] = str(e)
        return summary

    out_dir = OUTPUT_BASE / rosbag_name
    if SKIP_IF_OUTPUT_EXISTS and out_dir.exists() and any(out_dir.glob("*.jpg")):
        summary["skipped"] = True
        summary["written"] = sum(1 for _ in out_dir.glob("*.jpg"))
        return summary
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        topic_id = get_topic_id(conn, IMAGE_TOPIC)
        cur = conn.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC",
            (topic_id,),
        )

        idx = 0
        written = 0
        for ts_ns, blob in cur:
            summary["total_msgs"] += 1
            if idx % STRIDE == 0:
                msg = deserialize_message(bytes(blob), Image)
                bgr = image_msg_to_bgr(msg)
                bgr = maybe_resize(bgr, RESIZE_LONGEST_EDGE)
                fname = f"frame_{written:06d}_{ts_ns}.jpg"
                cv2.imwrite(
                    str(out_dir / fname),
                    bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
                )
                written += 1
            idx += 1

        summary["written"] = written
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
    finally:
        conn.close()

    return summary
'''))

cells.append(md("## 4. Run on all configured bags"))

cells.append(code('''bags = [b.strip() for b in ROSBAG_NAMES if b and b.strip()]
print(f"Processing {len(bags)} bag(s)\\n")

results = []
for i, name in enumerate(bags, 1):
    print(f"[{i}/{len(bags)}] {name} ...", flush=True)
    summary = export_bag(name)
    results.append(summary)
    if summary["error"]:
        print(f"    ERROR: {summary['error']}")
    elif summary["skipped"]:
        print(f"    skipped (output exists, {summary['written']} jpg already present)")
    else:
        print(f"    wrote {summary['written']} / {summary['total_msgs']} frames "
              f"-> {OUTPUT_BASE / name}")

print("\\nDone.")
'''))

cells.append(md("## 5. Summary"))

cells.append(code('''import pandas as pd

df = pd.DataFrame(results)
df["output_dir"] = df["bag"].apply(lambda n: str(OUTPUT_BASE / n))
df
'''))

cells.append(md("""## Notes

- The bridge from `sensor_msgs/Image` to a BGR numpy array is done by hand, so `cv_bridge`
  is not required.
- Filenames embed the original ROS timestamp (`frame_<index>_<nanoseconds>.jpg`), which
  lets you reconstruct real-time playback later via `ffmpeg -framerate ...` or pandas.
- To stitch a folder into a video once frames are exported, from the host:
  ```bash
  cd /home/maleen/csvdata/rapplab04/timelapse_frames/<bag_name>
  ffmpeg -framerate 30 -pattern_type glob -i 'frame_*.jpg' -c:v libx264 -pix_fmt yuv420p timelapse.mp4
  ```
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1))
print(f"Wrote {OUT}")
