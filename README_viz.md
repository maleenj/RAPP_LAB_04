# README_viz — Live Visualization Streaming (VAM → Unity/Unreal)

Everything for streaming the robot's live data (joint angles now; neural-net
internals later) over WiFi to **Unity, Unreal, or a browser**, plus **recording**
datasets and letting **students replay them offline without ROS or the robot**.

This is the master guide. Component-level docs:
- `ros2_ws/src/vam_viz_bridge/README.md` — the bridge package in depth
- `viz_student_pack/README.md` — the hand-to-students bundle
- `viz_student_pack/unity/README.md` — Unity step-by-step

---

## Contents
1. [How it works](#1-how-it-works)
2. [Prerequisites](#2-prerequisites)
3. [Part A — Docker: build & run the bridge](#part-a--docker-build--run-the-bridge)
4. [Part B — Test the stream (no game engine)](#part-b--test-the-stream-no-game-engine)
5. [Part C — Connect Unity / Unreal](#part-c--connect-unity--unreal)
6. [Part D — Record data streams](#part-d--record-data-streams)
7. [Part E — Play recordings back](#part-e--play-recordings-back)
8. [Part F — Student setup & Workshop Day-1](#part-f--student-setup--workshop-day-1)
9. [Part G — Add a new data source](#part-g--add-a-new-data-source)
10. [Part H — Neural-net activations: the "brain scan" (Phase 2)](#part-h--neural-net-activations-the-brain-scan-phase-2)
11. [Troubleshooting](#troubleshooting)
12. [File map](#file-map)

---

## 1. How it works

```
 existing ROS2 topics ─┐
  /vam/joint_states     │   ┌──────────────────┐   ws://<host-ip>:8765   ┌── Unity laptop
  /vam/joint_targets    ├──▶│  vam_viz_bridge  │──── JSON frames ────────┼── Unreal laptop
  /joint_states         │   │ (rapp_viz docker)│   (broadcast)           ├── browser viewer
  /vam/activations*     ┘   └──────────────────┘                         └── big screen
  (* Phase 2, opt-in)
```

- The **bridge** is a small ROS2 node in its own Docker container (`rapp_viz`). It
  subscribes to a config-driven list of topics and re-broadcasts each message as
  a **uniform JSON frame** over a WebSocket server on port **8765**.
- Because every container runs with **host networking + the same `ROS_DOMAIN_ID`**,
  the bridge sees all existing topics with zero setup, and `ws://<host-ip>:8765`
  is reachable directly from any laptop on the LAN.
- **Recordings** capture the exact frames clients receive into a tiny `.jsonl`.
  The **player** replays them over the same protocol — so student clients behave
  identically whether the source is the live robot, the bridge, or a file.

Every frame looks the same, so each client needs only one parser:
```json
{ "channel": "robot_joint_states", "shape": [6],
  "data": [0.01, -1.57, 1.2, 0.0, 0.3, -0.1],
  "labels": ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"] }
```
`data` is joint angles in **radians**.

---

## 2. Prerequisites

**On the host (the lab machine):**
- Docker + docker compose (already used by this project).
- The existing stack works (`rapp_vam` / `rapp_hw`). The viz container is fully
  independent and never modifies them.
- For running the recorder/player **on the host** (optional, simplest): Python 3
  with `websockets`:
  ```bash
  pip install websockets        # one-time
  ```

**On student laptops (Windows/Mac/Linux):** nothing required for the browser
viewer; Unity + the free NativeWebSocket package for the Unity path; Python 3 +
`websockets` only if they run `player.py` (there is also a no-Python option).

---

## Part A — Docker: build & run the bridge

The bridge has its own compose file: `docker/docker-compose.viz.yml` (container
`rapp_viz`, host networking, **no GPU/PyTorch**). It does **not** touch
`docker-compose.yml` or `docker-compose.hw.yml`.

```bash
cd docker
docker compose -f docker-compose.viz.yml up -d --build
```

Watch the log — it builds the package in-container and prints the address to
hand to clients:
```bash
docker logs -f rapp_viz
```
You'll see something like:
```
vam_viz_bridge ready: 3 channel(s) on ws://0.0.0.0:8765
Connect clients to ws://<HOST-IP>:8765  — host IP(s):
    192.168.1.50
```

**Stop / restart:**
```bash
docker compose -f docker-compose.viz.yml down
docker compose -f docker-compose.viz.yml restart      # after editing config
```

> The bridge only relays topics that exist. To actually see data you need a
> source publishing (live robot, headless inference, **or** a rosbag — see Part D).
> With no source, clients still connect and get the `__status__` "connected"
> greeting, just no data frames yet.

Which channels are streamed is set in
`ros2_ws/src/vam_viz_bridge/config/bridge_channels.yaml` (default:
`joint_states`, `joint_targets`, `robot_joint_states`). After editing it,
restart the container.

---

## Part B — Test the stream (no game engine)

Do these on the host before involving Unity — they isolate "is data flowing?"
from "is my Unity scene wired right?".

**1. Is the upstream topic even publishing? (pure ROS)**
```bash
docker exec rapp_vam bash -lc 'source /opt/ros/humble/setup.bash && ros2 topic hz /joint_states'
```

**2. CLI test client (prints channels + rate):**
```bash
docker exec -it rapp_viz bash
ros2 run vam_viz_bridge test_client          # inside the container
# or from any machine with python:
python3 -m vam_viz_bridge.test_client ws://<host-ip>:8765
```
Expected:
```
  robot_joint_states   shape=[6] data=[-1.480, -0.030, +0.079, ...]
  --- rates: robot_joint_states=65.0Hz ---
```

**3. Browser viewer (any OS, zero install):**
- Open `viz_student_pack/web/index.html` (double-click).
- Type `ws://<host-ip>:8765`, click **Connect**.
- Banner turns **green** and joint bars move.

**4. Multi-client / WiFi check:** repeat step 3 from a second laptop on the
workshop WiFi — both receive identical data simultaneously.

---

## Part C — Connect Unity / Unreal

Full Unity walkthrough: `viz_student_pack/unity/README.md`. The Unity pack is
designed so students **focus on visuals, not plumbing** — no JSON package needed
(a tiny parser is bundled).

**Unity — fastest test (the inspector):**
1. Install **NativeWebSocket** (Package Manager → Add from git URL:
   `https://github.com/endel/NativeWebSocket.git#upm`).
2. Drag the `viz_student_pack/unity/` folder into `Assets/`.
3. Empty GameObject → **Add Component → VamClient** (set `Url = ws://<host-ip>:8765`)
   → **Add Component → VamInspector** → press **Play**.
4. An overlay lists **every** channel: joints as bars, attention/activation
   channels as colored heatmap grids. Instant "is it working + what's in here".

**Unity — students build their own visual:** add a **`VamData`** component, pick a
`channel`, then duplicate **`VamVisualizerTemplate.cs`** and fill in the marked
block. Reading data is 1–2 lines:
```csharp
VamData data = GetComponent<VamData>();
float[] joints = data.Values;                 // flat channels
VamTensor attn = data.Tensor("decoder_xattn");// activation matrices
```
Two ready examples to remix: `ExampleAttentionHeatmap` and `ExampleJointBars`.

**Unreal** — built-in WebSockets module (add `"WebSockets"` + `"Json"` to your
`.Build.cs`); connect to `ws://<host-ip>:8765` and parse the JSON `data` array /
nested `tensors`. Snippet in `ros2_ws/src/vam_viz_bridge/README.md`.

---

## Part D — Record data streams

A recording is a small `.jsonl` capturing exactly what clients receive — perfect
for giving students offline data. The recorder is a plain WebSocket client
(`ros2_ws/src/vam_viz_bridge/tools/record_stream.py`).

### Option 1 — Record from a rosbag (no robot, no model needed)
This is how the bundled sample datasets were made. Three pieces: bridge up, a
rosbag playing its `/joint_states`, and the recorder.

```bash
# 0) bridge running (Part A)

# 1) play a rosbag's joints (in the VAM container, which has /data/rosbags)
docker exec -it rapp_vam bash -lc '
  source /opt/ros/humble/setup.bash
  ros2 bag play /data/rosbags/26_04_08_RAPP_M_R1G1_01 --clock --topics /joint_states'

# 2) in another terminal, record for 60s straight to the student pack (host):
python3 ros2_ws/src/vam_viz_bridge/tools/record_stream.py \
    ws://localhost:8765 --out viz_student_pack/recordings/r1g1.jsonl --duration 60
```
*(`python3 ros2_ws/.../record_stream.py` runs on the host — needs
`pip install websockets`. Alternatively run it inside `rapp_viz` with
`docker exec`, then `docker cp` the file out.)*

### Option 2 — Record the live robot (or the VAM's predicted "ghost" joints)
Same recorder; just have the real pipeline running so `/vam/joint_states` and
`/joint_states` are live, then:
```bash
python3 ros2_ws/src/vam_viz_bridge/tools/record_stream.py \
    ws://localhost:8765 --out viz_student_pack/recordings/live_demo.jsonl --duration 90
```
Whatever channels the bridge streams at that moment get recorded.

**Recorder flags:** `--out FILE` (required), `--duration SECONDS` (omit = until
Ctrl-C). Files are ~1 MB/minute for joints — safe to commit / put on a USB.

---

## Part E — Play recordings back

`viz_student_pack/player.py` is a **standalone, cross-platform** replay server.
Only dependency: `websockets`. No ROS, no robot. It serves the recording on the
same WebSocket protocol, paced to the original timing.

```bash
cd viz_student_pack
python3 player.py recordings/r1g1.jsonl --loop
```
It prints the addresses to connect to:
```
[player] serving 'recordings/r1g1.jsonl' (2922 frames, channels: ['robot_joint_states'])
[player] connect your client to one of:
             ws://192.168.1.50:8765
         (on this machine: ws://localhost:8765)
```
Point any client (browser viewer, Unity, `test_client`) at that URL — exactly as
if it were live.

**Flags:** `--loop` (repeat forever), `--speed 2.0` (2× faster), `--port 8765`,
`--host 0.0.0.0`.

> Note: the bridge and the player both listen on **8765** by default. Don't run
> both on the same machine on the same port — stop the bridge, or pass the
> player a different `--port`.

---

## Part F — Student setup & Workshop Day-1

Everything students need is the folder **`viz_student_pack/`**. Hand it over via
USB / shared drive / zip. It contains:
```
viz_student_pack/
├── README.md            ← student-facing instructions (Windows/Mac)
├── player.py            ← offline replay server
├── recordings/*.jsonl   ← sample datasets
├── web/index.html       ← browser viewer + connection tester
└── unity/               ← Unity scripts + step-by-step guide
```

### Day-1 runbook (host — no robot required)
Stream a looping recording so many students can connect to a stable, repeatable
source while they set up:
```bash
cd viz_student_pack
python3 player.py recordings/r1g1.jsonl --loop
```
Then tell students: **open `web/index.html`, enter `ws://<host-ip>:8765`, click
Connect.** Green banner = they're on the network and receiving data. This clears
WiFi/firewall issues before anyone opens Unity.

### What students do
1. **Connection test (any OS, no install):** browser viewer → `ws://<host-ip>:8765`.
2. **Unity, live:** connect to the host's `ws://<host-ip>:8765` (see `unity/README.md`).
3. **Unity/browser, fully offline:** run their own `player.py recordings/r1g1.jsonl --loop`
   and connect to `ws://localhost:8765`.
4. **No Python at all:** Unity reads a recording file directly — set
   `VamClient.Source = FilePlayback` and assign the `.jsonl` (renamed `.jsonl.txt`).
   Zero install, fully offline.

Because the protocol is identical in all cases, **a visualization students build
offline works unchanged against the live robot at the workshop.**

---

## Part G — Add a new data source

Anything on a ROS topic becomes a stream with **one config line**. Edit
`ros2_ws/src/vam_viz_bridge/config/bridge_channels.yaml`:
```yaml
  - topic: "/my/new_topic"
    type: "std_msgs/msg/Float64MultiArray"
    channel: "my_data"
```
…then `docker compose -f docker-compose.viz.yml restart`.

A **Python value that isn't on ROS yet** — publish it to a topic first (the topic
graph is the universal bus), then add the config line:
```python
from std_msgs.msg import Float32MultiArray
pub = node.create_publisher(Float32MultiArray, "/vam/my_state", 10)
pub.publish(Float32MultiArray(data=my_vector.tolist()))
```
Supported message types out of the box: `JointState`, `Float32/64MultiArray`,
ZED `ObjectsStamped`. A genuinely new type needs one serializer function in
`ros2_ws/src/vam_viz_bridge/vam_viz_bridge/serializers.py`.

---

## Part H — Neural-net activations: the "brain scan" (Phase 2)

Stream the model's internal "thinking" for live visualization and explainability.
This runs a **separate** inference node (`vam_tm12s_node_viz`) — your original
`vam_tm12s_node` / `vam_tm12s_robot.launch.py` are **never modified**.

### Run it — real robot
Same params as your normal command, just the `_viz` launch:
```bash
ros2 launch vam_inference vam_tm12s_robot_viz.launch.py \
    feedback_gain:=25.0 feedback_max_vel:=50.0 \
    max_joint_velocity_rad_s:=400.0 max_joint_acceleration_rad_s2:=400.0 \
    control_rate_hz:=100. \
    publish_saliency:=true        # optional perceptual scan (off by default)
```

### Run it — headless / RViz replay (no robot, Mode 1)
Test the activation stream from a rosbag on a ghost robot — no hardware needed:
```bash
# Terminal 1 (rapp_vam): play rosbag with clock
ros2 bag play /data/rosbags/<name> --clock

# Terminal 2 (rapp_vam): inference (headless) WITH activations
ros2 launch vam_inference vam_tm12s_headless_viz.launch.py use_sim_time:=true
#   add publish_saliency:=true for the perceptual scan

# Terminal 3 (rapp_hw): RViz with the TM12S config (ghost robot)
rviz2 -d /data/processed/vam_tm12s.rviz
```
This is the drop-in viz equivalent of `vam_tm12s_headless.launch.py` (which stays
unchanged). It runs the model on the rosbag's skeleton, so `/vam/activations`
carries **real attention** — ideal for testing the Unity attention visuals
without the robot. (Needs a model from the registry; for joints-only replay with
no model, use the recorded-`.jsonl` player instead.)

In both cases the `activations` channel and the `derived:` block are already
enabled in `bridge_channels.yaml`, so once the viz node runs the bridge streams
everything. If you rebuilt the bridge earlier, it already has `numpy` for the
derived metrics.

### What you get (channels)
**Raw (published by the viz node):**
| Channel sub-tensor | Shape | Meaning |
|---|---|---|
| `decoder_xattn` | `[2,4,10,10]` | which input frames drove each predicted action (explainability anchor) |
| `encoder_selfattn` | `[3,4,10,10]` | how the model integrates the recent motion across time |
| `encoder_out` | `[10,128]` | latent "situation" state (pulsing heatmap) |
| `decoder_out` | `[10,128]` | motor-plan latent |
| `input_saliency` | `[10, K]` | *(saliency on)* which human keypoints, when, drove the action |

**Derived (computed in the `rapp_viz` docker, not the robot):**
| Channel | Meaning |
|---|---|
| `attn_entropy` | per-layer attention entropy — focused vs diffuse |
| `activation_energy` | per-timestep latent L2 norm — "activity meter" |

Choose which raw signals with `--ros-args -p activation_set:='[decoder_xattn, encoder_out]'`.

### Notes
- **Robot behavior is unchanged:** capturing attention shifts predictions by ~1e-6 rad (≈6e-5°), verified negligible. Saliency runs as a separate forward off the control path and is **off by default**.
- **Recordings include all of it** automatically (the recorder captures the bridge's output — raw *and* derived channels). Raw tensors are recorded too, so new explainability metrics can be recomputed offline later.
- **Add a new derived metric** = one function in `ros2_ws/src/vam_viz_bridge/vam_viz_bridge/transforms.py` + one `derived:` entry in `bridge_channels.yaml`, then restart only `rapp_viz`. The robot pipeline is never touched.

Details in `ros2_ws/src/vam_viz_bridge/README.md`.

---

## Troubleshooting

| Symptom | Check |
|---|---|
| Client connects but **no data** | Is a source publishing? `ros2 topic hz /joint_states`. With no robot, start a rosbag (Part D) or `player.py` (Part E). |
| Client **can't reach** `ws://<host-ip>:8765` | Wrong IP; or WiFi **client isolation** (AP blocks laptop↔laptop) — use a router/switch that allows it; or host firewall blocking port 8765. Test with the browser viewer first. |
| Bridge log says a channel was **skipped** | Its message type couldn't be imported (e.g. `zed_msgs` not in the lightweight image). The bridge logs a warning and continues with the other channels. |
| Recorder error `__aenter__` / connection refused | The bridge/player isn't running on that URL/port yet. Start it first (Part A/E). |
| `rapp_viz` won't start: `exec: "/entrypoint_viz.sh": permission denied` | The entrypoint is bind-mounted, so the **host** file needs the execute bit: `chmod +x docker/viz/entrypoint.sh`, then `docker compose -f docker-compose.viz.yml up -d`. |
| `player.py`: "needs the 'websockets' package" | `pip install websockets`. |
| Port **8765 already in use** | The bridge and player both default to 8765 — stop one, or give the player `--port 8770`. |
| Stream looks **laggy/jumpy** on one laptop | Expected: the server is latest-frame-wins, so a slow client just sees newer frames — it never backs up the robot or other clients. |

Quick diagnostic ladder: `ros2 topic hz` (is the data there?) → `test_client`
(does the bridge serve it?) → browser viewer (can a client reach it?) → Unity.

---

## File map

**Bridge (ships in the `rapp_viz` container):**
```
ros2_ws/src/vam_viz_bridge/
├── config/bridge_channels.yaml          # topic → channel map (add sources here)
├── launch/viz_bridge.launch.py
├── vam_viz_bridge/
│   ├── viz_bridge_node.py               # config-driven subs + WS fan-out
│   ├── serializers.py                   # ROS msg → uniform JSON (add new types here)
│   ├── websocket_server.py              # asyncio broadcast server
│   └── test_client.py                   # CLI verification client
├── tools/record_stream.py               # record the stream to .jsonl
├── web/index.html                       # browser viewer
└── README.md                            # bridge deep-dive
docker/
├── docker-compose.viz.yml               # the rapp_viz stack (independent)
└── viz/{Dockerfile,entrypoint.sh}
```

**Student pack (hand to students):**
```
viz_student_pack/
├── README.md                            # student instructions
├── player.py                            # offline replay server (no ROS)
├── recordings/{r1g1,r2g1}.jsonl         # sample datasets
├── web/index.html                       # browser viewer + connection test
└── unity/                                # VamClient + VamData + VamInspector
    ├── VamClient.cs  VamData.cs           #   connection + per-object data input
    ├── VamJson.cs    VamFrame.cs          #   bundled parser + data model
    ├── VamInspector.cs                    #   default overlay (every channel)
    ├── VamVisualizerTemplate.cs           #   copy-me starter for new visuals
    ├── ExampleAttentionHeatmap.cs  ExampleJointBars.cs  JointVisualizer.cs
    ├── ConnectionStatusUI.cs
    └── README.md
```

**Edited later for Phase 2 only (gated, default-off):**
`ros2_ws/src/vam_inference/vam_inference/vam_tm12s_node.py` and the two
`vam_tm12s_*.launch.py` files.
