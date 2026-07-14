# Special Example 03 — Saliency Field (full screen)

The model's **input-saliency** map — "which parts of you are driving the robot" —
blown up to **fill the screen** as a living heat-field. It reads the
`input_saliency` tensor (`[10 timesteps × K keypoints]`, K=18) and draws one quad
per cell. Each cell is **coloured** by a modern heat ramp and, driven by the *same*
value, **tilts** and **lifts toward the camera** — so the flat heatmap becomes a
rippling 3D surface. A built-in **camera auto-fit** frames the whole grid for you.

The depth lift is deliberately gentle so the grid stays one cohesive sheet (you
asked for the 3D to be *felt*, not flown) — `Depth Scale` controls exactly how much.

**Files in this folder**
- `SaliencyField.cs` — the driver (reads saliency, builds the grid, animates it, fits the camera)
- `SaliencyField.shader` — flat unlit HDR emissive (`Enact/SaliencyField`)

---

## Requirements
- **Built-in Render Pipeline** (default 3D template). On URP/HDRP the shader shows
  magenta — use a Built-in RP project, same as the other special examples.
- A **dark background** + **HDR/Bloom** makes the heat glow (Camera → Clear Flags =
  Solid Color, black; enable HDR + a Bloom post-process).
- The stream must contain saliency — i.e. **saliency enabled** on the model/bridge
  and the `activations` channel flowing.

## Setup
1. Make a **new scene**. Add an empty **GameObject** → **Add Component → SaliencyField**.
   (An **EnactData** is added automatically.)
2. On the **EnactData** component, set **Channel = `activations`**.
3. On **SaliencyField**, assign **Cell Shader** → the `Enact/SaliencyField` shader.
4. Ensure an **EnactClient** exists and is receiving data. Leave **Fit Camera** ON
   and press **▶ Play** — the camera frames the grid edge-to-edge automatically.

> **Camera:** with *Fit Camera* on, the script drives `Camera.main` (or the camera
> you assign) every frame to frame the grid for the current screen aspect. Turn it
> off to position the camera yourself.

---

## The dials (Inspector)

**Data**
| Dial | What it does |
|---|---|
| `Tensor Name` | which tensor in `activations` to draw (`input_saliency`) |
| `Collapse Timesteps` | average the 10 timesteps into a single row (one cell per keypoint) for a cleaner bar-like read |

**Grid layout**
| Dial | What it does |
|---|---|
| `Cell Size` | centre-to-centre spacing (the auto-fit uses this to frame the grid) |
| `Cell Gap` | gap between cells as a fraction of cell size (0 = touching) |

**Colour — the heatmap**
| Dial | What it does |
|---|---|
| `Colors` | the heat gradient (**Magma** by default — tweak freely) |
| `Color Gamma` | contrast shaping: >1 darkens mids (only the strongest cells pop), <1 lifts them |
| `Intensity` | emissive brightness — push >1 with Bloom for the glow |

**Normalisation**
| Dial | What it does |
|---|---|
| `Auto Normalize` | ON: scale by the brightest cell each frame. OFF: divide by `Manual Max` |
| `Manual Max` | fixed scale when auto is off (use if you want absolute, comparable values) |
| `Max Smoothing` | smooths the running max so the field doesn't flicker frame-to-frame |

**Motion — the same value drives all of these**
| Dial | What it does |
|---|---|
| `Value Smoothing` | per-cell easing toward the latest value (0 = snappy, 0.9 = silky) |
| `Rotation Max` | degrees a cell tilts at value = 1 |
| `Rotation Axis` | axis each cell tilts about (X = flip toward you, Y = swing sideways, mix for a diagonal ripple) |
| `Spin Max` | extra in-plane spin (about Z) at value = 1; 0 = off |
| `Depth Scale` | **metres a cell lifts toward the camera at value = 1.** Keep small (≈0.1–0.4) so the grid stays a connected sheet; raise for a more dramatic relief |

**Camera auto-fit**
| Dial | What it does |
|---|---|
| `Fit Camera` | drive a camera to frame the grid full-screen each frame |
| `Target Camera` | which camera (empty = `Camera.main`) |
| `Fill Fraction` | how much of the screen the grid fills (0.9 leaves a small margin) |
| `View Tilt` | tilt the view a few degrees so the depth/rotation reads as 3D. **0 = dead-on, flattest full-screen look** (depth becomes nearly invisible head-on, so a small tilt is what makes the lift visible) |

---

## Tuning the feel you described
- **Full-screen, flat heatmap first:** `View Tilt` 0, `Rotation Max` 0, `Depth Scale` 0.
  Confirm it fills the screen and the colours look right.
- **Add gentle 3D:** raise `View Tilt` to ~8–12°, `Depth Scale` to ~0.2, `Rotation Max`
  to ~30°. The cells lift and tilt but stay knit into the surface.
- **More pronounced relief:** bump `Depth Scale` and `View Tilt` together — the tilt
  is what makes the depth visible, so they go hand in hand.
- **Punchy, only-the-hotspots-move:** raise `Color Gamma` (≈2) and `Value Smoothing`
  (≈0.8).

## Troubleshooting
- **Nothing appears / empty:** saliency may be disabled upstream — confirm the
  `activations` channel carries `input_saliency` (check the inspector overlay).
- **Grid not centred / not full:** make sure `Fit Camera` is on and a camera is
  tagged MainCamera (or assign `Target Camera`). `Fill Fraction` controls the margin.
- **Magenta cells:** you're on URP/HDRP — use a Built-in RP project.
- **Depth looks invisible:** that's expected dead-on — raise `View Tilt`.
