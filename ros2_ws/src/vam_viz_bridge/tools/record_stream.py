"""Record a running bridge's WebSocket stream into a portable .jsonl dataset.

This is an OPERATOR tool (run on the host, no ROS needed beyond a live bridge).
It connects to the bridge as an ordinary WebSocket client and writes each frame
as one JSON line, tagged with a wall-clock receive time `t_recv`. The resulting
file is tiny (joints @15Hz ≈ a few MB/minute), cross-platform, and replayable by
viz_student_pack/player.py — students never need ROS or the robot.

The recording captures EXACTLY what clients receive, so it is forward-compatible:
when Phase 2 adds the `activations` channel, recordings include it automatically.

Usage:
    # with a bridge running on :8765 (and a rosbag playing through it)
    python3 record_stream.py ws://localhost:8765 --out rec.jsonl --duration 90
    python3 record_stream.py ws://localhost:8765 --out rec.jsonl   # until Ctrl-C
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install websockets") from exc


async def _record(url: str, out_path: str, duration: float | None) -> None:
    print(f"[record] connecting to {url} ...")
    n = 0
    channels: dict = {}
    start = None
    with open(out_path, "w") as f:
        async with websockets.connect(url, max_size=None) as ws:
            print(f"[record] writing to {out_path}. Ctrl-C to stop.")
            deadline = None
            async for message in ws:
                now = time.time()
                if start is None:
                    start = now
                    deadline = (start + duration) if duration else None
                try:
                    frame = json.loads(message)
                except json.JSONDecodeError:
                    continue
                channel = frame.get("channel", "?")
                # Skip the connection greeting — it isn't replayable data.
                if channel == "__status__":
                    continue
                frame["t_recv"] = now
                f.write(json.dumps(frame, separators=(",", ":")) + "\n")
                n += 1
                channels[channel] = channels.get(channel, 0) + 1
                if n % 200 == 0:
                    print(f"[record] {n} frames "
                          + ", ".join(f"{k}={v}" for k, v in channels.items()))
                if deadline and now >= deadline:
                    break
    dur = (time.time() - start) if start else 0.0
    print(f"\n[record] done: {n} frames over {dur:.1f}s -> {out_path}")
    print("[record] per channel: " + ", ".join(f"{k}={v}" for k, v in channels.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description="Record a viz-bridge WebSocket stream to .jsonl")
    ap.add_argument("url", nargs="?", default="ws://localhost:8765",
                    help="bridge WebSocket URL (default ws://localhost:8765)")
    ap.add_argument("--out", required=True, help="output .jsonl path")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to record (default: until Ctrl-C)")
    args = ap.parse_args()
    try:
        asyncio.run(_record(args.url, args.out, args.duration))
    except KeyboardInterrupt:
        print("\n[record] stopped")


if __name__ == "__main__":
    main()
