"""CLI test client for the viz bridge — verify the stream without Unity/Unreal.

Connects to the WebSocket server, prints each channel's shape and the first few
values, and reports a per-channel frame rate. This proves the end-to-end path
(ROS topic -> bridge -> WebSocket) from the host before any game engine is involved.

Usage:
    ros2 run vam_viz_bridge test_client
    ros2 run vam_viz_bridge test_client --ros-args -p url:=ws://192.168.10.1:8765
    # or standalone (no ROS):
    python3 -m vam_viz_bridge.test_client ws://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict

try:
    import websockets
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install websockets") from exc


async def _run(url: str) -> None:
    print(f"[test_client] connecting to {url} ...")
    async with websockets.connect(url) as ws:
        print("[test_client] connected. Ctrl-C to quit.\n")
        counts: dict = defaultdict(int)
        last_report = time.time()
        last_counts: dict = defaultdict(int)
        async for message in ws:
            frame = json.loads(message)
            channel = frame.get("channel", "?")
            counts[channel] += 1

            if "tensors" in frame:
                desc = ", ".join(
                    f"{name}{t['shape']}" for name, t in frame["tensors"].items()
                )
                preview = f"tensors: {desc}"
            else:
                shape = frame.get("shape", [])
                data = frame.get("data", [])
                head = ", ".join(f"{v:+.3f}" for v in data[:6])
                preview = f"shape={shape} data=[{head}{' ...' if len(data) > 6 else ''}]"

            print(f"  {channel:<20} {preview}")

            now = time.time()
            if now - last_report >= 2.0:
                rates = {
                    ch: (counts[ch] - last_counts[ch]) / (now - last_report)
                    for ch in counts
                }
                rate_str = ", ".join(f"{ch}={hz:.1f}Hz" for ch, hz in rates.items())
                print(f"  --- rates: {rate_str} ---")
                last_report = now
                last_counts = dict(counts)


def main(args=None) -> None:
    # Allow either a bare positional URL (standalone) or a ROS param.
    url = "ws://localhost:8765"
    argv = sys.argv[1:]
    positional = [a for a in argv if not a.startswith("-") and a != "--ros-args"]
    if positional:
        url = positional[0]
    else:
        try:
            import rclpy

            rclpy.init(args=args)
            node = rclpy.create_node("viz_test_client")
            node.declare_parameter("url", url)
            url = node.get_parameter("url").get_parameter_value().string_value
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    try:
        asyncio.run(_run(url))
    except KeyboardInterrupt:
        print("\n[test_client] bye")


if __name__ == "__main__":
    main()
