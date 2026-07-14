# ENACT Visualization — Student Handout

You're going to build live visuals from a robot that **watches a person and
decides how to move**. This stream lets you tap into three things at once:

- what the robot's **body** is doing,
- what the robot **sees** (the person), and
- what's happening inside the robot's **"mind"** (the **ENACT** neural network).

Your job is the *visuals*. Getting the data is already done for you — you pick a
**channel**, you get clean numbers, you make something beautiful.

> No machine-learning background needed. Each data source below is explained at
> three levels — **🟢 the simple idea**, **🔵 a bit more technical**, and
> **🟣 the math** (totally optional). Read as deep as you like.

---

## Meet ENACT — the robot's brain

The neural network driving the robot is called **ENACT** — *Embodied Neural Action
Chunking Transformer*. It's a **vision–action model (VAM)**: it takes in what the
camera sees of a person and decides how the robot should move. In one breath:
ENACT watches the **last 10 snapshots** of the person's skeleton, runs them
through a **transformer encoder** (which relates those moments to one another),
and a **transformer decoder** then plans the **next 10 moves all at once** —
that's the "action chunking" part — by *attending back* to what it just saw. The
data sources below let you look inside ENACT at each stage: where it's **looking**
(attention), its inner **"thoughts"** (hidden state), and **what about you**
mattered (saliency). Throughout this guide, "the ENACT model / system / network"
all mean this same brain.

---

## Data sources at a glance

| Channel (set on `EnactData`) | What it is | Shape | Think of it as… |
|---|---|---|---|
| `robot_joint_states` | the robot's actual joint angles | `[6]` | the robot's **body pose** |
| `skeleton` | the tracked person's keypoints | `[18,3]` | the robot's **eyes** (what it sees of you) |
| `activations` → `decoder_xattn` | cross-attention | `[2,4,10,10]` | **what the robot focuses on** to decide each move |
| `activations` → `encoder_selfattn` | self-attention | `[3,4,10,10]` | how it **connects moments in time** |
| `activations` → `encoder_out` / `decoder_out` | hidden state | `[10,128]` | the robot's **inner thoughts** |
| `activations` → `input_saliency` | input attribution | `[10,K]` | **which parts of you** drive the action |
| `attn_entropy` | attention focus | small | a **"focused vs scattered" dial** |
| `activation_energy` | activation strength | `[10]` | a **"how active" dial** |

All angles are in **radians**. `activations`, `attn_entropy`, and
`activation_energy` only stream when the instructor runs the robot's
visualization node.

---

## 1. `robot_joint_states` — the robot's body

**🟢 The simple idea.** A robot arm is made of segments connected by joints, like
your shoulder, elbow, and wrist. This tells you the **angle of each of the
robot's 6 joints right now** — exactly how the arm is posed at this instant. It's
the real arm's actual position, so anything you build with it stays perfectly in
sync with the physical robot.

**🔵 A bit more technical.** Six numbers, one per joint, in **radians**, in joint
order (base → tip). These are the robot's *measured* encoder values (feedback
from the hardware), not a command — so it's ground truth for where the arm is.

**🟣 The math.** Each joint angle is `θᵢ ∈ ℝ` (radians). Stacking them gives the
arm's *configuration* `q = [θ₁…θ₆]`. "Forward kinematics" turns `q` into the 3D
position/orientation of the hand by chaining each joint's rotation. Handy
conversion: `degrees = radians × 180/π`.

**Shape:** `[6]` — one angle per joint.
```
Example frame:
  data = [ 0.00, -1.57,  1.20,  0.00,  1.05, -0.20 ]   (radians)
           j1     j2     j3     j4     j5     j6
  j2 = -1.57 rad ≈ -90°  (that joint is bent a quarter-turn)
```

**What it represents:** the robot's *body* — its physical pose.

---

## 2. `skeleton` — what the robot sees of you

**🟢 The simple idea.** This is the robot's **eyes**: a 3D stick-figure of the
person in front of it — where the head, shoulders, elbows, hands, hips, knees and
feet are in space. The camera finds these points on the body many times a second.
This is the robot's whole picture of *you*.

**🔵 A bit more technical.** 18 body keypoints (the "BODY_18" layout), each with
an `(x, y, z)` position in **metres**, from the ZED camera's body tracking — an
`[18, 3]` matrix. Drawing lines between the right pairs of points (neck→shoulder,
shoulder→elbow, …) gives a skeleton.

**🟣 The math.** Points `pₖ ∈ ℝ³`. "Bones" are a fixed list of index pairs
`(a, b)`; draw a segment from `pₐ` to `p_b`. The coordinates are in the camera's
reference frame (a robotics convention: x-forward, y-left, z-up), so to place them
in Unity you remap axes — the example does this for you (`rosToUnity`).

**Shape:** `[18,3]` — 18 keypoints × (x, y, z), in metres.
```
Example frame (first few keypoints):
  k0  nose          [ 0.05,  0.10, 1.85 ]
  k1  neck          [ 0.04,  0.00, 1.60 ]
  k2  right shoulder[ 0.22, -0.02, 1.58 ]
  k4  right wrist   [ 0.33, -0.08, 1.21 ]
  ...                (18 rows total)
  A "bone" pair (1,2) = a line from neck k1 to shoulder k2.
```

**What it represents:** the robot's *perception* — its understanding of where the
person is and how they're posed.

---

## 3. Attention — what the robot is paying attention to

*(channel `activations`, tensors `decoder_xattn` and `encoder_selfattn`)*

**🟢 The simple idea.** When you reach to catch a ball, your attention locks onto
the ball and the way it just moved — you ignore the background. The robot does
the same: out of everything it just saw, it **focuses on the moments that matter**
for its next move. Attention is a glowing grid where **brighter = "I'm paying
more attention here."** It's the closest thing to watching *what the robot cares
about* in real time.

**🔵 A bit more technical.** ENACT looks at the **last 10 snapshots** of the
person. Attention is a set of weight grids (it's a transformer):
- **`encoder_selfattn`** — how each of those 10 moments relates to the others
  (how it stitches your motion over time into a story).
- **`decoder_xattn`** ("cross-attention") — for each of the **10 future moves**
  it's planning, *which of the 10 input snapshots it leans on*. This is the
  clearest "**why** did it do that" signal: it links an action back to what it saw.

Each grid is 10×10 numbers between 0 and 1; each row adds up to 1 (it's dividing
100% of its attention across the moments). There are several grids (one per
"head" and per layer) — the examples average them into one map.

**🟣 The math.** Attention is `softmax(QKᵀ / √d) · V`. The part we visualize is
the weight matrix `A = softmax(QKᵀ / √d)`, shape `[rows × 10]`, where each row is
a probability distribution over the 10 input timesteps (`Σ = 1`). We expose it
per layer and head: `[layers, heads, rows, 10]`.

**Shape:** `decoder_xattn [2,4,10,10]`, `encoder_selfattn [3,4,10,10]` — layers × heads × rows × 10 input timesteps.
```
Example: ONE attention row (how one moment splits its focus over the 10 inputs)
  [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.12, 0.20, 0.25, 0.15]   (sums to 1.00)
   t0    t1    t2    t3    t4    t5    t6    t7    t8    t9
  → most attention on t7–t9 (the most recent frames). Brighter cells there.
  The full tensor is [2, 4, 10, 10] = 2 layers × 4 heads × 10 rows × 10 inputs;
  the examples average all of that into one 10×10 heatmap.
```

**What it represents:** the robot's *focus* — which moments of what it saw are
driving the decision it's making now.

> Honest caveat for the curious: attention shows where information *flows*, which
> is a strong hint about focus but not a literal "gaze." Great for intuition and
> art; don't treat it as a perfect explanation.

---

## 4. `encoder_out` / `decoder_out` — the robot's inner thoughts

*(channel `activations`, tensors `encoder_out` and `decoder_out`)*

**🟢 The simple idea.** After looking at you, the robot forms an **inner
impression** — not a picture or a word, just a pattern of internal signals that
shifts as the situation changes. It's like a **brain scan**: you can't read a
single brain cell, but you can watch the whole pattern light up, pulse, and
settle. `encoder_out` is "what I understood about the person," and `decoder_out`
is "the plan I'm forming."

**🔵 A bit more technical.** These are ENACT's **hidden activations** — for each
of 10 timesteps, a vector of **128 numbers**, so a `[10, 128]` grid. `encoder_out`
is ENACT's compressed understanding of the input; `decoder_out` is the
representation of its planned actions, just before it turns them into joint
commands.

**🟣 The math.** Vectors `hₜ ∈ ℝ¹²⁸`. Individual numbers aren't meaningful on
their own, but the **pattern** is: useful views are the per-timestep magnitude
`‖hₜ‖` (see `activation_energy`), or squashing the 128 dimensions down to 2–3 with
PCA/t-SNE to plot the "mental state" as a moving point.

**Shape:** `[10,128]` — 10 timesteps × 128 features (`encoder_out` and `decoder_out` each).
```
Example (one 128-vector per timestep; first 8 of 128 shown):
  h0 = [ 0.13, -0.42,  0.05,  0.88, -0.11,  0.30, -0.67,  0.24, … ]
  h1 = [ 0.10, -0.39,  0.02,  0.91, -0.08,  0.35, -0.70,  0.20, … ]
  ...                                              (10 rows total)
  Don't read one number — watch the whole row of 128 shift as the person moves.
```

**What it represents:** the robot's *internal state* — its evolving "thoughts"
about the situation.

---

## 5. `input_saliency` — which parts of you matter

*(channel `activations`, tensor `input_saliency`; only when saliency is enabled)*

**🟢 The simple idea.** This answers: **what about the person caused the robot to
act?** If the robot starts moving because your right hand came up, your right hand
**lights up**. It paints the importance of each body part back onto the skeleton —
the robot's way of saying "I moved *because of this part of you*."

**🔵 A bit more technical.** It's an **attribution**: how sensitive ENACT's
chosen action is to each input keypoint. A large value means "nudge this body
point a little and the action changes a lot." It comes as a value per keypoint
over the 10 input timesteps, shape `[10, K]`.

**🟣 The math.** Gradient-based saliency: `sₖ = |∂(action)/∂(inputₖ)|`, summed over
that keypoint's x/y/z. Bigger gradient magnitude = stronger influence on the
output. (It's a first-order, local sensitivity — a good visual cue, not a precise
causal proof.)

**Shape:** `[10,K]` — 10 timesteps × K keypoints (K=18).
```
Example: per-keypoint saliency for one timestep (one row of the grid)
  right wrist    0.91   ←★ matters most
  right elbow    0.40
  right shoulder 0.18
  neck           0.07
  left ankle     0.03   ← barely matters right now
  ...
  → light up the right arm; leave the rest dim.
```

**What it represents:** the robot's *attention onto the body itself* — which parts
of the human are driving its behaviour.

---

## 6. `attn_entropy` & `activation_energy` — the summary dials

These are small, friendly numbers computed from the big tensors — perfect for
simple meters, bars, or audio.

**`attn_entropy` — the focus dial.**
- 🟢 Is the robot **laser-focused on one moment, or scanning everywhere?** Low =
  focused; high = spread out.
- 🔵 The "spread" of the attention distribution, one number per layer.
- 🟣 Entropy `H = −Σ p·log p` over the attention weights (low H = peaky, high H =
  uniform). **Shape:** one value per layer (`decoder_xattn_entropy [2]`,
  `encoder_selfattn_entropy [3]`).
  ```
  focused  p = [0,0,0,0,0,0,0,0,0.9,0.1]  → H ≈ 0.33   (one number, small)
  scattered p = [0.1,0.1,…,0.1] (×10)      → H ≈ 2.30   (= ln 10, the max)
  ```

**`activation_energy` — the activity dial.**
- 🟢 **How "worked up" the robot's thoughts are** at each moment — a heartbeat for
  its mental state.
- 🔵 The strength (magnitude) of the hidden-state vector at each of the 10 timesteps.
- 🟣 The L2 norm `‖hₜ‖₂` per timestep. **Shape:** `[10]` — one value per
  timestep (`encoder_out_norm`, `decoder_out_norm`).
  ```
  per-timestep energy (shape [10]):
  [3.1, 3.3, 3.6, 4.8, 6.2, 6.5, 5.0, 4.1, 3.7, 3.4]
   t0                    ↑ peak: most "worked up" mid-motion        t9
  ```

**What they represent:** at-a-glance dials for *focus* and *mental activity* —
easy to map to size, colour, or sound.

---
---

# Quickstart — get something on screen

You need: Unity (any recent version, 3D project) and the address of the stream
(`ws://<host-ip>:8765`) **or** a recording file. The instructor gives you the
address; `localhost` works if you run a recording yourself (see the last step).

### 1. New project + the scripts
1. Unity Hub → **New Project → 3D**.
2. **Window → Package Manager → `+` → Add package from git URL** →
   `https://github.com/endel/NativeWebSocket.git#upm` → **Add**.
   *(This is the only package you need. No JSON package — it's bundled.)*
3. Drag the whole `unity/` folder into your project's `Assets/`.

### 2. See the data immediately (the inspector)
1. Create an empty GameObject, name it **ENACT**. **Add Component → EnactClient**.
2. On EnactClient set **Url** = `ws://<host-ip>:8765`.
3. **Add Component → EnactInspector**. Press **▶ Play**.

You should see an on-screen panel listing every channel — joints as bars,
attention/activation channels as little colour heatmaps. If it's there, you're
connected and ready to build. (If not, see Troubleshooting at the bottom.)

### 3. Run pre-recorded data (no robot needed)
Two options:

- **Server replay (matches "live" exactly):** on your machine run
  ```bash
  python3 player.py recordings/r1g1.jsonl --loop      # needs: pip install websockets
  ```
  then set `EnactClient.Url = ws://localhost:8765`.
- **No Python:** copy a recording into `Assets/recordings/` and **rename it to end
  in `.txt`** (e.g. `r1g1.jsonl.txt`). On **EnactClient** set **Source =
  FilePlayback**, drag the file into **Recording**, tick **Loop**. Press Play.

Either way the rest of your scene is identical — only the data source changes.

---

# The template — your starting point for any visual

Open **`EnactVisualizerTemplate.cs`**. This is the file you **copy** to start a new
visualization. It already does all the plumbing; you only write the visual.

**How to use it**
1. Duplicate `EnactVisualizerTemplate.cs`, rename the file *and* the class (e.g.
   `MyCoolViz`).
2. Put it on a GameObject. A **EnactData** component is added automatically.
3. On that **EnactData**, set **Channel** to the data you want (see the table at the
   top, e.g. `skeleton` or `activations`).
4. Fill in the marked block with your visual.

**Getting the data (three easy ways):**
```csharp
EnactData data = GetComponent<EnactData>();

if (data.HasData) {
    float[] v = data.Values;                 // a flat channel (e.g. joints)
    float   m = data.Get(2, 3);              // a matrix element [row, col]
    EnactTensor attn = data.Tensor("decoder_xattn");  // a named activation tensor
    float[] map = attn.MeanPlane();          // collapse it to one 2-D heatmap
}
```
…or react only when fresh data arrives, instead of every frame:
```csharp
data.OnData += frame => { /* runs once per new frame (~15×/sec) */ };
```
That's the whole API. `data.HasData`, `data.Values`, `data.Get(...)`,
`data.Tensor(...)`, `data.OnData`. Everything else is your creativity.

---

# The examples — read these, then remix them

Each is a small, commented script. Attach the listed components and press Play.

### `EnactInspector` — "show me everything"
A zero-setup overlay of every channel (numbers, bars, heatmaps). Your debugging
and exploration tool. **Attach:** one GameObject with `EnactClient` + `EnactInspector`.

### `JointVisualizer` — joints → spinning cubes
Rotates 6 cubes by the joint angles. The simplest "it's alive" demo.
**Attach:** `EnactData` (channel `robot_joint_states`) + `JointVisualizer`.

### `ExampleJointBars` — joints → 3D bars
A row of bars that rise and fall with each joint angle.
**Attach:** `EnactData` (channel `robot_joint_states`) + `ExampleJointBars`.

### `SkeletonVisualizer` — the person in 3D
Draws the tracked human as joints + bones. If it looks rotated or mirrored,
toggle `rosToUnity` (coordinate conventions differ). Needs the skeleton stream
running. **Attach:** `EnactData` (channel `skeleton`) + `SkeletonVisualizer`.

### `ExampleAttentionHeatmap` — the "brain scan" wall
A grid of tiles that glow with the robot's attention (`decoder_xattn`, averaged
to one 10×10 map). Point a camera at it. Needs the activation stream running.
**Attach:** `EnactData` (channel `activations`) + `ExampleAttentionHeatmap`.

### `FlyCamera` — move around your scene
Put it on your **Main Camera**. **W A S D** to move, **arrow up/down** for
height, **hold right-mouse** to look around, **Shift** to go faster.
*(Uses Unity's legacy Input — if you get an input error, set Project Settings →
Player → Active Input Handling to "Both".)*

### `ConnectionStatusUI` — a tiny status readout
Shows connected / rate / latest values in a corner. Attach anywhere.

---

# Troubleshooting

- **Inspector says "no EnactClient found"** → add a `EnactClient` component to a
  GameObject.
- **Status never turns green / can't connect** → wrong IP, the WiFi blocks
  laptop-to-laptop traffic, or a firewall is blocking port 8765. Test first with
  the browser page (`web/index.html`) — if that fails too, it's the network, not
  Unity.
- **Connected, but my channel is empty** → check the channel name matches the
  table exactly. `skeleton`, `activations`, etc. only stream when the instructor
  runs the matching node on the robot; `robot_joint_states` is always there.
- **Skeleton looks sideways/mirrored** → toggle `rosToUnity` on
  `SkeletonVisualizer`, or adjust `scale`/`offset`.

Have fun — the data is yours; make something that makes people *feel* what the
robot is thinking. 🤖
