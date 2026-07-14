#!/usr/bin/env python3
"""Standalone replay server for recorded VAM datasets — no ROS, no robot.

Reads a recorded .jsonl dataset and re-broadcasts each frame over WebSocket on
the SAME protocol the live bridge uses, paced to the original timing. Your Unity
/ Unreal / browser client connects to ws://localhost:8765 and behaves exactly as
if the live robot were running — only the URL changes.

Cross-platform (Windows / macOS / Linux). The ONLY dependency is `websockets`:

    pip install websockets
    python player.py recordings/r1g1.jsonl --loop
    # then point your client at ws://localhost:8765

Options:
    --host 0.0.0.0     interface to bind (default: all interfaces)
    --port 8765        port (default: 8765)
    --loop             repeat the recording forever
    --speed 1.0        playback speed multiplier (2.0 = twice as fast)

This file is fully self-contained on purpose: a student can copy just player.py
plus a .jsonl recording and run it anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys

try:
    import websockets
except ImportError:
    sys.exit("This tool needs the 'websockets' package. Install it with:\n"
             "    pip install websockets")


# --------------------------------------------------------------------------- #
# Minimal broadcast server (latest-frame-wins per channel), mirrors the bridge.
# --------------------------------------------------------------------------- #
class _Client:
    def __init__(self) -> None:
        self.mailbox: dict = {}
        self.event = asyncio.Event()

    def put(self, channel: str, payload: str) -> None:
        self.mailbox[channel] = payload
        self.event.set()

    def drain(self) -> list:
        items = list(self.mailbox.values())
        self.mailbox.clear()
        self.event.clear()
        return items


class BroadcastServer:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.clients: "set[_Client]" = set()

    async def handler(self, connection, path=None) -> None:
        client = _Client()
        self.clients.add(client)
        peer = getattr(connection, "remote_address", ("?", 0))
        print(f"[player] client connected: {peer} (total={len(self.clients)})")
        try:
            hello = {"channel": "__status__", "msg": "connected", "server": "vam_player"}
            await connection.send(json.dumps(hello, separators=(",", ":")))
        except Exception:
            pass
        try:
            while True:
                await client.event.wait()
                for payload in client.drain():
                    await connection.send(payload)
        except Exception:
            pass
        finally:
            self.clients.discard(client)
            print(f"[player] client disconnected: {peer} (total={len(self.clients)})")

    def broadcast(self, channel: str, frame: dict) -> None:
        if not self.clients:
            return
        payload = json.dumps(frame, separators=(",", ":"))
        for client in self.clients:
            client.put(channel, payload)


# --------------------------------------------------------------------------- #
# Recording load + paced playback
# --------------------------------------------------------------------------- #
def load_frames(path: str) -> list:
    frames = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not frames:
        sys.exit(f"[player] no frames found in {path}")
    return frames


def _host_ips() -> list:
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        ips.extend(socket.gethostbyname_ex(socket.gethostname())[2])
    except Exception:
        pass
    # de-dup, drop loopback noise but keep at least something
    out = []
    for ip in ips:
        if ip not in out and not ip.startswith("127."):
            out.append(ip)
    return out or ["127.0.0.1"]


async def play(server: BroadcastServer, frames: list, loop: bool, speed: float) -> None:
    # Pre-compute inter-frame delays from t_recv (fall back to a flat 15Hz).
    times = [fr.get("t_recv") for fr in frames]
    have_times = all(t is not None for t in times)
    base = times[0] if have_times else None

    while True:
        prev = base
        for fr in frames:
            if have_times:
                delay = (fr["t_recv"] - prev) / max(speed, 1e-6)
                prev = fr["t_recv"]
                if delay > 0:
                    await asyncio.sleep(min(delay, 5.0))  # cap pathological gaps
            else:
                await asyncio.sleep((1.0 / 15.0) / max(speed, 1e-6))
            out = {k: v for k, v in fr.items() if k != "t_recv"}
            server.broadcast(out.get("channel", "?"), out)
        if not loop:
            break
        await asyncio.sleep(0.2)
    print("[player] playback finished")


async def _main(args) -> None:
    frames = load_frames(args.file)
    channels = sorted({fr.get("channel", "?") for fr in frames})
    server = BroadcastServer(args.host, args.port)

    await websockets.serve(server.handler, args.host, args.port, ping_interval=20)

    print("=" * 60)
    print(f"[player] serving '{args.file}'  ({len(frames)} frames, channels: {channels})")
    print(f"[player] loop={args.loop}  speed={args.speed}x")
    print("[player] connect your client to one of:")
    for ip in _host_ips():
        print(f"             ws://{ip}:{args.port}")
    print(f"         (on this machine: ws://localhost:{args.port})")
    print("=" * 60)

    await play(server, frames, args.loop, args.speed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a recorded VAM dataset over WebSocket.")
    ap.add_argument("file", help="recorded .jsonl dataset")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--loop", action="store_true", help="repeat forever")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = ap.parse_args()
    try:
        asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\n[player] bye")


if __name__ == "__main__":
    main()
