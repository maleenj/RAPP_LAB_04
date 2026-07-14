# vam_viz_bridge — live visualization streaming

Stream live data from ENACT (joint angles, and later neural-network internal
activations) over WiFi to **Unity, Unreal, or a browser** running on workshop
attendees' laptops. Plug-and-play: if you're on the network, you can receive the
stream.

This is a **standalone module**. It does not modify the inference pipeline — it
just taps the ROS2 topic graph (all containers run host networking + a shared
`ROS_DOMAIN_ID`, so the bridge sees every topic with zero setup) and fans each
message out as a uniform JSON frame.

```
 existing ROS topics ─┐
  /joint_states        │   ┌──────────────────┐   ws://<host-ip>:8765   ┌── Unity laptop
  /vam/skeleton        ├──▶│  vam_viz_bridge  │──── JSON frames ────────┼── Unreal laptop
  /joint_states        │   │ (config-driven)  │   (broadcast)           ├── browser test page
  /vam/activations*    ┘   └──────────────────┘                         └── big screen
  (* Phase 2, opt-in)
```

---

## Quick start

```bash
cd docker
docker compose -f docker-compose.viz.yml up -d --build
docker logs -f rapp_viz        # prints the host IP(s) to hand to clients
```

The log prints something like:

```
Connect clients to ws://<HOST-IP>:8765  — host IP(s):
    192.168.10.1
```

That `ws://192.168.10.1:8765` is what every client connects to.

> The bridge works against the **live robot**, **headless inference**, or a
> **rosbag replay** — it only needs the topics to exist on the network.

---

## Verify the stream (no game engine needed)

**CLI test client** (inside the container):

```bash
docker exec -it rapp_viz bash
ros2 run vam_viz_bridge test_client
# or from any machine with python:  python3 -m vam_viz_bridge.test_client ws://<host-ip>:8765
```

You should see per-channel frames and a measured rate (~15 Hz for joints).

**Browser test page** — open `web/index.html` (double-click, or
`python3 -m http.server` in the `web/` folder), type `ws://<host-ip>:8765`,
click **Connect**. Joint angles render as bars; 2-D tensors (attention maps)
render as a heatmap. This page is also the reference client implementation.

**Pure-ROS sanity check** (is the upstream data even there?):

```bash
ros2 topic hz /joint_states
ros2 topic echo /joint_states --once
```

---

## The JSON frame (one parser for every channel)

Each WebSocket message is one JSON object:

```json
{
  "channel": "robot_joint_states",
  "stamp": 1749567890.123,
  "shape": [6],
  "data": [0.01, -1.57, 1.2, 0.0, 0.3, -0.1],
  "labels": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
}
```

`shape` + flat `data` lets a client reshape without knowing anything about ROS.
Messages that carry several named tensors (e.g. NN activations) use `tensors`
instead of `shape`/`data`:

```json
{
  "channel": "activations",
  "stamp": 1749567890.130,
  "tensors": {
    "encoder_out":     { "shape": [10, 128], "data": [ ... ] },
    "encoder_selfattn":{ "shape": [3, 4, 10, 10], "data": [ ... ] }
  }
}
```

---

## Channels (default config)

| Channel | Source topic | Shape | Notes |
|---|---|---|---|
| `robot_joint_states` | `/joint_states` | `[6]` | the robot's **actual** measured joints (radians), has `labels` |
| `skeleton` | `/vam/skeleton` | `[18,3]` | human keypoints (ZED BODY_18), via `skeleton_relay` |
| `activations` | `/vam/activations` | tensors | **Phase 2** "brain scan", when the viz node runs (see below) |

> Only the robot's **physical** joint state is streamed. ENACT's predicted
> `/vam/joint_states` and commanded `/vam/joint_targets` are intentionally not
> bridged — visuals should track what the arm is actually doing. (Those ROS
> topics still exist for the robot pipeline.)

Edit channels in `config/bridge_channels.yaml`.

---

## Connecting from Unity

1. Install **NativeWebSocket** (Package Manager → add from git URL
   `https://github.com/endel/NativeWebSocket.git#upm`).
2. Minimal receiver:

```csharp
using NativeWebSocket;
using UnityEngine;

public class VamStream : MonoBehaviour {
    WebSocket ws;

    async void Start() {
        ws = new WebSocket("ws://192.168.10.1:8765");   // <- host IP from the log
        ws.OnMessage += (bytes) => {
            var json = System.Text.Encoding.UTF8.GetString(bytes);
            var frame = JsonUtility.FromJson<Frame>(json);
            if (frame.channel == "robot_joint_states") {
                // frame.data is float[]; drive your rig here
            }
        };
        await ws.Connect();
    }

    void Update() {
        #if !UNITY_WEBGL || UNITY_EDITOR
        ws?.DispatchMessageQueue();
        #endif
    }

    async void OnApplicationQuit() { if (ws != null) await ws.Close(); }

    [System.Serializable]
    public class Frame { public string channel; public double stamp; public int[] shape; public float[] data; public string[] labels; }
}
```

> `JsonUtility` does not handle the nested `tensors` object — for the activations
> channel use a JSON lib like Newtonsoft (com.unity.nuget.newtonsoft-json) and
> parse `tensors` as a dictionary.

## Connecting from Unreal

Unreal ships a built-in **WebSockets** module — no marketplace plugin needed.

1. In your `.Build.cs`: add `"WebSockets"` and `"Json"` to `PublicDependencyModuleNames`.
2. Minimal connect:

```cpp
#include "WebSocketsModule.h"
#include "IWebSocket.h"

TSharedPtr<IWebSocket> Socket =
    FWebSocketsModule::Get().CreateWebSocket(TEXT("ws://192.168.10.1:8765"));

Socket->OnMessage().AddLambda([](const FString& Msg) {
    TSharedPtr<FJsonObject> Obj;
    auto Reader = TJsonReaderFactory<>::Create(Msg);
    if (FJsonSerializer::Deserialize(Reader, Obj)) {
        const FString Channel = Obj->GetStringField(TEXT("channel"));
        const TArray<TSharedPtr<FJsonValue>>* Data;
        if (Channel == TEXT("robot_joint_states") && Obj->TryGetArrayField(TEXT("data"), Data)) {
            // (*Data)[i]->AsNumber()  -> drive your rig
        }
    }
});
Socket->Connect();
```

---

## Adding a new data source (the scalable bit)

**A new ROS topic** of an already-supported type — add ONE entry to
`config/bridge_channels.yaml` and restart the bridge:

```yaml
  - topic: "/my/new_topic"
    type: "std_msgs/msg/Float64MultiArray"
    channel: "my_data"
```

**An arbitrary Python state vector** that isn't on ROS yet — publish it to a
topic first (the topic graph is the universal bus), then add the config line:

```python
from std_msgs.msg import Float32MultiArray
pub = node.create_publisher(Float32MultiArray, "/vam/my_state", 10)
pub.publish(Float32MultiArray(data=my_vector.tolist()))
```

**A genuinely new message type** — add one function to `serializers.py` and
register it in `SERIALIZERS`, then reference it via `serializer:` in config.

Supported serializers out of the box: `joint_state`, `multiarray`,
`multiarray_named`, `zed_skeleton`, `generic_numeric` (auto-fallback).

---

## Recording datasets for offline use (students / no robot)

Capture the live stream into a small, portable `.jsonl` that anyone can replay
without ROS or the robot — ideal for handing students offline data to build
against:

```bash
# with a bridge running (e.g. while a rosbag plays through it)
python3 tools/record_stream.py ws://localhost:8765 --out rec.jsonl --duration 60
```

Replay it anywhere with the standalone, cross-platform player (only needs
`pip install websockets`, no ROS) — clients connect exactly as if it were live:

```bash
python3 ../../../viz/student_pack/player.py rec.jsonl --loop
# -> ws://localhost:8765
```

The full student-facing bundle (player, sample recordings, browser viewer, Unity
scripts + guide) lives in [`viz/student_pack/`](../../../viz/student_pack/). See
its `README.md` for the Windows/Mac day-1 runbook.

## Phase 2 — neural-network activations (the "brain scan")

Streamed by a **separate** inference node, `vam_tm12s_node_viz` — the original
`vam_tm12s_node` is never modified. Run the viz launch (same params as the
original robot launch):

```bash
ros2 launch vam_inference vam_tm12s_robot_viz.launch.py \
    feedback_gain:=25.0 feedback_max_vel:=50.0 \
    max_joint_velocity_rad_s:=400.0 max_joint_acceleration_rad_s2:=400.0 \
    control_rate_hz:=100. \
    publish_saliency:=true        # optional; off by default
```

The `activations` channel and the `derived:` block are already enabled in
`config/bridge_channels.yaml`. Channels produced:

- **Raw (viz node):** `activations` carrying `decoder_xattn [2,4,10,10]`,
  `encoder_selfattn [3,4,10,10]`, `encoder_out [10,128]`, `decoder_out [10,128]`,
  and (saliency on) `input_saliency [10,K]`.
- **Derived (computed here in the bridge):** `attn_entropy` (per-layer attention
  entropy) and `activation_energy` (per-timestep latent L2 norm).

Choose raw signals with `--ros-args -p activation_set:='[decoder_xattn, encoder_out]'`.
Capturing attention shifts predictions by ~1e-6 rad (negligible, verified);
saliency runs off the control path and is off by default. **Add a derived metric**
= one function in `transforms.py` + one `derived:` entry, then restart only the
bridge — the robot is never touched. The viz image needs `numpy` (already in
`docker/viz/Dockerfile`).

---

## Troubleshooting

| Symptom | Check |
|---|---|
| Client connects but no frames | Is the upstream topic publishing? `ros2 topic hz /joint_states` |
| Client can't reach `ws://...` | WiFi **client isolation** (AP setting) blocks laptop↔host; use a switch/router that allows it. Also check the host firewall on port 8765. |
| Bridge logs no channels | A message type couldn't be imported (e.g. `zed_msgs`) — it logs a warning and skips that channel. |
| Frames lag / stutter | Expected on a slow client — the bridge is latest-frame-wins, so a slow laptop just sees newer data, it never backs up the robot. |

---

## Files

```
vam_viz_bridge/
├── config/bridge_channels.yaml     # topic -> channel map (add sources here)
├── launch/viz_bridge.launch.py
├── vam_viz_bridge/
│   ├── viz_bridge_node.py          # config-driven subs + WebSocket fan-out
│   ├── serializers.py              # ROS msg -> uniform dict (add new types here)
│   ├── transforms.py               # derived metrics (entropy, norms) — add here
│   ├── websocket_server.py         # asyncio broadcast server (latest-frame-wins)
│   └── test_client.py              # CLI verification client
└── web/index.html                  # browser test page + reference client
```

Docker: `docker/docker-compose.viz.yml`, `docker/viz/{Dockerfile,entrypoint.sh}`.
