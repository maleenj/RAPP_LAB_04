# Special Example 02 — Radial Joint Mandala

A **kinetic geometric sculpture** driven by the robot's 6 joint angles (radians).
Six concentric **ring (torus)** meshes are staggered in depth to form a **tunnel you
look into**; each ring tumbles about its **own axis**, rotated *directly* by its
joint angle. Crisp single-colour **emissive** materials with an **edge-glow** rim —
a clean, hypnotic counterpoint to the organic particle mode. When an angle crosses a
threshold, its ring snaps a bright **accent flash**.

This is **Option A (nested rings)** from the brief. See *Option B* at the bottom for
the radial-bars variant.

**Files in this folder**
- `RadialMandala.cs` — the driver (reads joints, builds the rings, rotates them)
- `RadialMandala.shader` — emissive + fresnel edge-glow (`Enact/RadialMandala`)

---

## Requirements
- **Built-in Render Pipeline** (the default 3D template). On URP/HDRP the shader
  shows magenta — use a Built-in RP project, same as Special Example 01.
- A **dark background** + **HDR/Bloom** makes the edge-glow and flashes pop
  (Camera → Clear Flags = Solid Color, black; enable HDR + a Bloom post-process).

## Setup (4 steps)
1. Create an empty **GameObject** → **Add Component → RadialMandala**.
   (An **EnactData** is added automatically.)
2. On the **EnactData** component, set **Channel = `robot_joint_states`**.
3. On **RadialMandala**, assign **Render Shader** → the `Enact/RadialMandala`
   shader (the `.shader` file).
4. Make sure an **EnactClient** exists in the scene and is receiving data.
   Press **▶ Play**. Add a **FlyCamera** to your Main Camera, or assign one to
   **Orbit Camera** below to let the sculpture present itself.

---

## The dials (Inspector)

**Geometry — the tunnel of rings**
| Dial | What it does |
|---|---|
| `Base Radius` | size of the innermost ring (metres) |
| `Radius Step` | how much larger each ring is → the concentric spread |
| `Tube Radius` | tube thickness; small = thin, crisp rings |
| `Depth Step` | spacing along Z → the depth of the tunnel |
| `Ring Segments` / `Tube Segments` | mesh smoothness (higher = rounder, costlier) |

**Motion — angle → rotation**
| Dial | What it does |
|---|---|
| `Degrees Per Radian` | tumble gain; default `Rad2Deg` = 1:1 angle mapping |
| `Smoothing` | 0 = snap to the latest angle, 0.95 = very smooth/laggy |
| `Ambient Spin` | optional idle spin (deg/s) so it breathes when the robot is still; 0 = pure data |

**Look — colour & glow**
| Dial | What it does |
|---|---|
| `Colors` | gradient across the rings (inner → outer) |
| `Intensity` | base emissive brightness (push >1 for Bloom) |
| `Edge Glow` | strength of the fresnel rim on each ring |

**Accent flashes**
| Dial | What it does |
|---|---|
| `Flash Every` | fire a flash each time a joint angle crosses a multiple of this many radians; 0 = off |
| `Flash Strength` | how bright the flash spikes |
| `Flash Decay` | how fast it fades back (per second) |

**Camera — optional slow orbit**
| Dial | What it does |
|---|---|
| `Orbit Camera` | assign a camera **Transform** to gently orbit the tunnel mouth (leave empty to fly yourself) |
| `Orbit Speed` | orbit rate, deg/s |
| `Orbit Distance` | how far in front of the tunnel the camera sits |
| `Orbit Tilt` | vertical tilt of the orbit (look slightly into the tunnel) |

> **Why each ring tumbles about a different axis:** a torus spun about its own
> symmetry axis looks motionless. The driver gives each ring a *distinct* tilt axis
> so the joint angle produces visible, interlocking motion. The six axes are
> hard-coded (`AXES` in the script) — reorder or replace them to taste.

---

## Looks to try
- **Hypnotic tunnel:** large `Depth Step` (0.8), small `Tube Radius` (0.03), high
  `Edge Glow` (4), assign **Orbit Camera** with low `Orbit Speed` (4).
- **Instrument cluster:** `Depth Step` 0, `Radius Step` 0.5 (flat concentric),
  `Smoothing` 0.2 for snappy reads, `Flash Every` 0.25 for frequent ticks.
- **Solid sculpture:** fat `Tube Radius` (0.12), low `Edge Glow` (1), high
  `Intensity` (3), warm gradient.

## Troubleshooting
- **Magenta rings:** you're on URP/HDRP — use a Built-in RP project, or port the
  shader.
- **Rings look static:** the robot may be idle. Add a little `Ambient Spin`, or
  confirm `robot_joint_states` is streaming (EnactData `Channel`, EnactClient
  connected).
- **No glow / flat:** enable HDR + Bloom, raise `Intensity`, use a dark background.
- **Geometry changes don't apply:** tweaking `Base Radius`/`Tube Radius`/segments
  rebuilds the meshes live; if not, toggle the component off/on.

## Option B — radial bars
Same data + flash logic, different mesh: instead of `BuildTorus()`, build a flat
bar/wedge and arrange six of them around a circle (`localPosition` on a ring,
`localRotation` facing outward). Encode each angle as the bar's rotation or length
(`localScale.y`). Keep the emissive shader for the crisp instrument-cluster read.
