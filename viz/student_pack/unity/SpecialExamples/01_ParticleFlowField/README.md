# Special Example 01 — Particle Flow Field

A living cloud of GPU particles sculpted by the body. The 18 skeleton points act
as **attractors**: particles are pulled toward the moving points, pushed away when
too close, and swirled by turbulence that grows where the body moves fast. You
never see the skeleton — only the current it creates.

Runs as a **self-contained GPU compute system** (no VFX Graph to author). An
RTX 2080 handles 150k–500k particles comfortably.

![data: skeleton channel]

**Files in this folder**
- `ParticleFlowField.cs` — the driver (reads skeleton, runs the sim, draws)
- `ParticleFlowField.compute` — the GPU simulation
- `ParticleFlowField.shader` — additive point-sprite rendering (`Enact/ParticleFlowField`)

---

## Requirements
- **Built-in Render Pipeline** (the default 3D template). URP/HDRP need a small
  shader tweak — see *Notes* at the bottom.
- A GPU that supports compute + DX11 (your RTX 2080 is way more than enough).

## Setup (5 steps)
1. Copy this `01_ParticleFlowField/` folder into your project's `Assets/`
   (anywhere under Assets is fine; it can sit next to the other Enact scripts).
2. Create an empty **GameObject** → **Add Component → ParticleFlowField**.
   (An **EnactData** is added automatically.)
3. On the **EnactData** component, set **Channel = `skeleton`**.
4. On **ParticleFlowField**, assign the two assets:
   - **Sim Shader** → `ParticleFlowField.compute`
   - **Render Shader** → the `Enact/ParticleFlowField` shader (the `.shader` file)
5. Make sure an **EnactClient** exists in the scene (the connection) and is
   receiving data. Press **▶ Play**.

Add a **FlyCamera** to your Main Camera to fly around the cloud. A **dark
background** makes the additive glow pop (Camera → Clear Flags = Solid Color,
black).

> **For the full glowing look:** turn on **HDR** and add a **Bloom** post-process.
> The `Intensity` dial pushes colours above 1.0 specifically so Bloom makes them
> bloom. Without Bloom it still works, just less dreamy.

---

## The dials (Inspector)

**Particles**
| Dial | What it does |
|---|---|
| `Particle Count` | how many particles (150k default; push to 300k–500k on a 2080) |
| `Particle Life` | seconds before a particle recycles near the body |
| `Particle Size` | sprite size in metres |

**Forces — pull / push**
| Dial | What it does |
|---|---|
| `Attract Strength` | how hard the body points pull particles in |
| `Attract Falloff` | higher = pull only acts up close → tighter cloud |
| `Repel Radius` / `Repel Strength` | push particles out when they get too close (keeps the cloud from collapsing onto a point) |

**Forces — turbulence**
| Dial | What it does |
|---|---|
| `Turbulence` | base swirl strength |
| `Turbulence Scale` | size of the eddies (higher = finer, wispier) |
| `Turbulence Speed` | how fast the noise field evolves |
| `Velocity Turbulence` | extra swirl added **where the body moves fast** (the signature effect) |
| `Drag` | damping; 0.85 = heavy/short trails, 0.98 = floaty/long trails |

**Emission**
| Dial | What it does |
|---|---|
| `Emit From Body` | **OFF (default & recommended):** particles fill the whole volume and the body just *sculpts* them into a flowing current — no clump. **ON:** particles are born at the body and flow outward (denser near the body). |
| `Spawn Radius` | only used when *Emit From Body* is ON — how tightly particles are born around a body point |
| `Spawn Speed` | initial outward speed of new particles so they flow instead of piling up |

> Spawning is now **spherical** (no boxy edges). If you previously saw a "cube" of
> particles, that was the old cube-shaped spawn with *Emit From Body* on — turn it
> **off** for a pure flowing cloud.

**Look — colour & glow**
| Dial | What it does |
|---|---|
| `Colors` | the **gradient** mapped by particle speed (slow → fast). This is your main colour control. |
| `Speed For Max Color` | the speed that maps to the END of the gradient |
| `Intensity` | HDR brightness (use with Bloom) |
| `Edge Softness` | how soft each particle dot is |

**Coordinate mapping**
| Dial | What it does |
|---|---|
| `Ros To Unity` | axis remap (same as SkeletonVisualizer). Toggle if the cloud sits in the wrong place/orientation. |
| `Scale` / `Offset` | reposition / resize the whole field |
| `Point Response` | how snappily attractors chase the latest skeleton (higher = tighter to the body) |
| `Bounds Radius` | cloud volume radius (and recycle distance) |

**Move the source with the actor** — slide the cloud along a line as the actor crosses the stage
| Dial | What it does |
|---|---|
| `Track Person` | **ON:** the whole cloud slides along a line based on where the actor is across the stage. **OFF:** cloud sits at the body in 3D. |
| `Interaction Width` | physical stage width in metres (e.g. **7**) — the actor moving this far moves the cloud the full `Track Width`. |
| `Stage Horizontal Axis` | which world axis is the actor's left-right movement (leave **X** with default `Ros To Unity`; flip if left/right is wrong). |
| `Horizontal Center` | the stage coordinate that maps to the **centre** of the track (set to where the actor stands when centred). |
| `Track Center` | the world position of the cloud when the actor is centred — **this sets the cloud's depth & height** (put it where your camera is looking). |
| `Track Axis` | the world direction the cloud slides along (default **+X** = screen-horizontal). |
| `Track Width` | how far (world metres) the cloud travels across the full stage. = `Interaction Width` for 1:1; larger = exaggerated travel; smaller = subtle. |
| `Stable Tracking` | **ON (recommended):** slide using the **torso joints** (neck + hips) only — barely affected by arm/leg swings, so the cloud glides smoothly. **OFF:** use the average of every visible joint (jumpier). |
| `Skip Zero Points` *(Coordinate mapping)* | **ON (recommended):** ignore dropped `(0,0,0)` keypoints. Leave this on — otherwise a momentarily-lost joint sits at the camera origin and **yanks the tracking toward (0,0,0)**, which is the usual cause of jittery / drifting tracking. |

**How to set it up**
1. Aim your camera at where the cloud should appear. Set **Track Center** to that spot (this fixes the cloud's depth & height); it only slides left/right from there.
2. Tick **Track Person**, set **Interaction Width = 7**.
3. Stand the actor centre-stage; if the cloud isn't centred, set **Horizontal Center** to their current stage coordinate (or nudge until centred).
4. Walk to the edges and set **Track Width** so the cloud travels as far as you want on screen. Wrong direction → flip **Track Axis** (e.g. to `-1,0,0`) or **Stage Horizontal Axis**.

The body's own motion still sculpts the turbulence inside the cloud — only the cloud's *position* slides. Depth/height stay fixed at `Track Center`, so it reads as a clean horizontal move across the screen.

---

## Looks to try
- **Aurora ribbon:** high `Drag` (0.96), low `Turbulence` (0.5), high
  `Velocity Turbulence` (8), thin `Particle Size` (0.02), blue→magenta gradient.
- **Fireflies:** low `Attract Strength` (3), big `Spawn Radius` (0.6), short
  `Particle Life` (2), warm gradient, small size, high `Intensity`.
- **Smoke vortex:** high `Turbulence Scale` (1.2), moderate `Turbulence` (2.5),
  `Drag` 0.9, grey→white gradient.

---

## Performance (RTX 2080)
- 150k is the default and very light. 300k–500k still runs at high FPS; 1M is
  possible but watch your frame time.
- Cost scales with `Particle Count × Attractor Count (18)`. It's all on the GPU —
  no CPU readback — so it stays cheap.
- Lower `Particle Size` and `Particle Count` first if you ever need more headroom.

## Troubleshooting
- **Nothing appears:** check (a) `Sim Shader` and `Render Shader` are assigned,
  (b) EnactData `Channel = skeleton`, (c) the EnactClient is connected. The
  particles also need the skeleton stream — confirm with the inspector overlay.
- **Cloud is off to one side / sideways:** toggle `Ros To Unity`, or set
  `Offset`/`Scale`. The camera auto-target is the body centroid, so fly toward it.
- **Looks flat, no glow:** enable HDR + Bloom, raise `Intensity`, use a dark
  background.
- **Pink particles / shader error:** you're likely on URP/HDRP — see Notes.

## Notes
- **URP / HDRP:** the renderer uses a Built-in-RP additive shader. On URP/HDRP it
  may show as magenta. Easiest fix: use a **Built-in RP** project for this visual.
  (Porting the `.shader` to URP is possible but out of scope for the workshop.)
- The simulation is fully on the GPU; the only CPU work per frame is uploading 18
  attractor positions — trivial.
