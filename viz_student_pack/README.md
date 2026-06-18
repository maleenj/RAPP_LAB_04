# VAM Visualization — Student Pack

Everything you need to receive the robot's live data stream and visualize it in
**Unity** (or a browser), **without ROS2 and without the robot**. Works on
**Windows, macOS, and Linux**.

The stream is the same whether it comes from the live robot, the instructor's
replay server, or a recorded file on your own laptop — so build your
visualization once and it works in all three.

```
What's in here
├── README.md            ← you are here
├── player.py            ← replay recorded data on your own machine (offline)
├── recordings/          ← recorded datasets (*.jsonl) to play back
├── web/index.html       ← browser viewer + connection tester (no install)
└── unity/               ← Unity scripts + step-by-step guide
```

---

## The data

Every message is one JSON object, same shape for every channel:

```json
{ "channel": "robot_joint_states", "shape": [6],
  "data": [0.01, -1.57, 1.2, 0.0, 0.3, -0.1],
  "labels": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"] }
```

`data` is **joint angles in radians**. Channels you'll see: `robot_joint_states`
(real robot), `joint_states` (the model's predicted "ghost" joints),
`joint_targets`. (Later in the workshop an `activations` channel with neural-net
internals appears.)

---

## Three ways to use it

### 1. Browser connection test (do this first — any OS, no install)
1. Open `web/index.html` (double-click).
2. Type the address the instructor gives you, e.g. `ws://192.168.1.50:8765`.
3. Click **Connect**. Green banner = you're on the network and receiving data.

This is the fastest way to confirm your laptop can reach the stream **before**
opening Unity. If it works here but not in Unity, the problem is in Unity, not
the network.

### 2. Unity — live (connect to the robot / the instructor's server)
Follow `unity/README.md`. The Unity pack does the data plumbing for you: add a
**VamInspector** to see every channel instantly, then add a **VamData** component
(pick a channel) + copy **VamVisualizerTemplate** and write only your visual.
No JSON package needed. In short: new 3D project → install NativeWebSocket →
add the three scripts → set `VamClient.Url` to the instructor's address → Play.
Six cubes rotate with the live motion.

### 3. Unity / browser — fully offline (your own recorded playback)
Run a recording on your own machine, then connect to yourself:

**Windows** (PowerShell):
```powershell
py -m pip install websockets
py player.py recordings\r1g1.jsonl --loop
```
**macOS / Linux**:
```bash
python3 -m pip install websockets
python3 player.py recordings/r1g1.jsonl --loop
```
It prints `ws://localhost:8765`. Point the browser page or Unity (`Url =
ws://localhost:8765`) at it — same as live.

> **No Python?** Unity can read a recording directly with **zero install** — see
> "Run fully offline" in `unity/README.md` (set Source = FilePlayback).

---

## player.py options
```
python player.py <recording.jsonl> [--loop] [--speed 2.0] [--port 8765] [--host 0.0.0.0]
```
- `--loop` repeat forever
- `--speed` playback speed (2.0 = twice as fast)
- `--port` / `--host` change where it serves

---

## Troubleshooting
| Problem | Fix |
|---|---|
| Browser won't connect to the instructor's address | Wrong IP, or the WiFi blocks laptop-to-laptop traffic ("client isolation"), or a firewall blocks port 8765. Confirm the address; try another network. |
| `player.py`: "needs the 'websockets' package" | Run the `pip install websockets` line above. |
| `py`/`python3` not found | Install Python 3 from python.org (Windows: tick "Add to PATH"), or use the Unity file-playback option instead. |
| Unity connects but nothing moves | Set the visualizer's `channel` to `robot_joint_states`. |
