# Unity quick-start — receive & visualize the ENACT stream

Get the robot's live data into Unity and build your own visuals fast. The data
plumbing is done for you — you pick a channel and read raw numbers/matrices. The
**same scripts** work against the live robot, the host's `player.py`, or a
recorded file with no network at all.

**No JSON package needed.** The only thing to install is NativeWebSocket (for
live mode), below. A tiny JSON parser is bundled (`EnactJson.cs`).

## The scripts

| Script | Role |
|---|---|
| `EnactClient.cs` | The connection (WebSocket or recorded file). Add **one** to the scene. |
| `EnactData.cs` | **Your data input.** Put on your object, pick a `channel`, read the data. |
| `EnactVisualizerTemplate.cs` | **Copy me** to start a new visualization. |
| `EnactInspector.cs` | Default overlay showing every channel (test + inspiration). |
| `EnactJson.cs` / `EnactFrame.cs` | Bundled parser + data model (you don't edit these). |
| `ExampleAttentionHeatmap.cs` | Example: a 3D grid colored by attention. |
| `ExampleJointBars.cs` / `JointVisualizer.cs` | Examples: joints → bars / cube rotations. |
| `SkeletonVisualizer.cs` | Example: the human skeleton in 3D (joints + bones). |
| `FlyCamera.cs` | Free-fly camera (WASD + arrows + mouse) to move around your scene. |
| `ConnectionStatusUI.cs` | Small connected/rate overlay. |

---

## 1. Create the project
Unity Hub → **New Project** → **3D** → create.

## 2. Install NativeWebSocket (for live / player.py)
**Window → Package Manager → `+` → Add package from git URL…** →
`https://github.com/endel/NativeWebSocket.git#upm` → **Add**.

> Offline file playback only? You can skip this and comment out the
> `#define USE_NATIVE_WEBSOCKET` line at the top of `EnactClient.cs`.

## 3. Add the scripts
Drag the whole `unity/` folder into `Assets/` (e.g. `Assets/ENACT/`).

## 4. Fastest test — the inspector
- Create an empty GameObject named **ENACT**. **Add Component → EnactClient**.
- Set **Url** to the instructor's address, e.g. `ws://192.168.1.50:8765`
  (or `ws://localhost:8765` if you run `player.py` yourself).
- **Add Component → EnactInspector**. Press **Play**.
- An on-screen panel lists **every** live channel: joints as bars, attention/
  activation channels as colored heatmap grids. This confirms data is flowing and
  shows you what each channel looks like.

---

## 5. Build your own visualization (the point)

1. Create a GameObject for your visual. **Add Component → EnactData**.
2. On EnactData, set **Channel** (see the table below).
3. Duplicate **`EnactVisualizerTemplate.cs`**, rename it, add it to the same object,
   and fill in the marked block. Read the data any of these ways:

```csharp
EnactData data = GetComponent<EnactData>();

if (data.HasData) {
    // flat channels (joints):
    float[] v = data.Values;            // e.g. 6 joint angles (radians)
    float j2  = data.Get(1);

    // matrices / activation channels:
    EnactTensor attn = data.Tensor("decoder_xattn");  // [2,4,10,10]
    float[] map = attn.MeanPlane();      // collapse to one 10x10 heatmap
    float cell  = attn.Get(0, 3);        // [row, col] of the last plane
}
```
…or react only on new data: `data.OnData += frame => { ... };`

Ready examples to read/remix: **`ExampleAttentionHeatmap`** (attention → a 3D
grid of colored quads), **`ExampleJointBars`** (joints → 3D bars), and
**`SkeletonVisualizer`** (the human in 3D).

> **Special / advanced examples** live in `SpecialExamples/`, each with its own
> setup README. Start with **`SpecialExamples/01_ParticleFlowField/`** — a GPU
> particle cloud sculpted by the body (the skeleton points become attractors;
> you see only the turbulence they create). Stunning, GPU-cheap, fully tunable.

### See the human skeleton in 3D
1. On the host, run the relay so the skeleton becomes a stream:
   `ros2 run vam_inference skeleton_relay` (needs a skeleton source — live ZED or
   a rosbag playing `/zed/zed_node/body_trk/skeletons`).
2. In Unity: GameObject → **EnactData** (Channel = `skeleton`) + **SkeletonVisualizer**.
3. Add **FlyCamera** to your Main Camera and press Play. Move with **WASD**,
   up/down with the **arrow keys**, and **hold right-mouse to look around**.
4. If the figure looks rotated or mirrored, toggle `rosToUnity` (or tweak
   `scale`/`offset`) on SkeletonVisualizer — 3D coordinate conventions differ by
   capture setup.

> FlyCamera uses Unity's legacy Input. If you get an input error, set
> **Project Settings → Player → Active Input Handling** to "Input Manager (Old)"
> or "Both".

---

## Channels (set `channel` on EnactData)

| Channel | Type | Contents |
|---|---|---|
| `robot_joint_states` | vector | the robot's actual joints `[6]` (radians, has labels) |
| `skeleton` | matrix | human keypoints `[18,3]` (x,y,z), ZED BODY_18 |
| `activations` | tensors | `decoder_xattn [2,4,10,10]`, `encoder_selfattn [3,4,10,10]`, `encoder_out [10,128]`, `decoder_out [10,128]`, `input_saliency [10,K]` (if enabled) |
| `attn_entropy` | tensors | `decoder_xattn_entropy [2]`, `encoder_selfattn_entropy [3]` |
| `activation_energy` | tensors | `encoder_out_norm [10]`, `decoder_out_norm [10]` |

Use `data.Values` for vector channels and `data.Tensor("name")` for tensor channels.

> **Joint channels** stream from any source (live, player, recording). The
> **activation / attention** channels appear only when the instructor runs the
> visualization node on the robot. An offline recording with activations will be
> provided later so you can build those visuals without the robot.

---

## Run fully offline (no network)
- Copy a recording (e.g. `recordings/r1g1.jsonl`) into `Assets/recordings/` and
  **rename it to end in `.txt`** so Unity imports it as a `TextAsset`.
- On **EnactClient**: set **Source = FilePlayback**, drag the text asset into
  **Recording**, tick **Loop**, press **Play**. Everything else is identical.

## Troubleshooting
- **Inspector says "no EnactClient found":** add a EnactClient component to a scene object.
- **Status never connects:** wrong IP, WiFi **client isolation** (AP blocks
  laptop-to-laptop), or a firewall on port 8765. Check with the browser page
  (`../web/index.html`) first to tell a network problem from a Unity one.
- **Connected but my channel is empty:** confirm the `channel` name matches the
  table; attention channels need the robot's viz node running.
