# Unity quick-start — receive & visualize the VAM stream

These three scripts get a live (or recorded) joint-angle stream into Unity and
moving on screen in a few minutes. The **same scripts** work against the live
robot, the host's `player.py`, or a recorded file with no network at all.

| Script | Role |
|---|---|
| `VamClient.cs` | Connects (WebSocket) or plays a recorded `.jsonl`; parses frames |
| `JointVisualizer.cs` | Rotates 6 cubes from the joint angles |
| `ConnectionStatusUI.cs` | On-screen connected / rate / values overlay |

---

## 1. Create the project
- Unity Hub → **New Project** → **3D (Built-in/URP, either)** → create.

## 2. Install NativeWebSocket (for live / player.py)
- **Window → Package Manager → `+` → Add package from git URL…**
- Paste: `https://github.com/endel/NativeWebSocket.git#upm` → **Add**.

> Doing **offline file playback only**? You can skip this and comment out the
> `#define USE_NATIVE_WEBSOCKET` line at the top of `VamClient.cs`.

## 3. Add the scripts
- Drag `VamClient.cs`, `JointVisualizer.cs`, `ConnectionStatusUI.cs` into
  `Assets/` (e.g. an `Assets/Scripts/` folder).

## 4. Wire up the scene
- Create an empty GameObject, name it **VAM**.
- **Add Component → VamClient**, **JointVisualizer**, **ConnectionStatusUI**.
- On JointVisualizer and ConnectionStatusUI, drag the **VAM** object into the
  `client` field (or leave empty — they auto-find it).
- Point the camera at the origin (the cubes spawn along +X from the object).

## 5a. Run against live data or the host player
- In **VamClient**, set **Source = WebSocket**.
- Set **Url** to what the instructor gives you, e.g. `ws://192.168.1.50:8765`
  (use `ws://localhost:8765` if `player.py` runs on your own machine).
- Press **Play** → status overlay turns green, 6 cubes rotate with the motion.

## 5b. Run fully offline (no network)
- Put a recording in the project: copy `recordings/r1g1.jsonl` into
  `Assets/recordings/` and **rename it to end in `.txt`** (e.g.
  `r1g1.jsonl.txt`) so Unity imports it as a `TextAsset`.
- In **VamClient**: set **Source = FilePlayback**, drag the text asset into
  **Recording**, tick **Loop**.
- Press **Play** → cubes move with no server, no WiFi.

---

## Which channel?
On **JointVisualizer** / **ConnectionStatusUI**, set `channel` to one of:
- `robot_joint_states` — the real robot's measured joints (best for the demo data)
- `joint_states` — the VAM's predicted "ghost" joints (when inference is running)
- `joint_targets` — normalized targets sent to the robot

## The data format (for your own visualizations)
Every frame is the same shape:
```json
{ "channel": "robot_joint_states", "shape": [6],
  "data": [0.01, -1.57, 1.2, 0.0, 0.3, -0.1],
  "labels": ["joint_1", ...] }
```
`data` is joint angles in **radians**. Subscribe to `VamClient.OnFrame` or read
`VamClient.LatestByChannel[...]` and drive whatever rig you build.

> Later in the workshop a `activations` channel appears with nested `tensors`
> (NN internals). `JsonUtility` can't parse nested objects — add
> **com.unity.nuget.newtonsoft-json** via Package Manager and use `JObject` for
> that channel. The joints test here needs nothing extra.

## Troubleshooting
- **Status stays red / never connects:** wrong IP, or the WiFi blocks
  device-to-device traffic (AP "client isolation"), or a firewall blocks port
  8765. Test first with the browser page (`web/index.html`) — it's the quickest
  way to tell a network problem from a Unity problem.
- **Connected but cubes don't move:** check the `channel` name matches what the
  server streams (try `robot_joint_states`).
