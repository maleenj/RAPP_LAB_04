# End-Effector Props & Static Obstacles (Swords, Masks, Poles, Ground)

This guide covers how to add, swap, and remove props on the robot end-effector
(and static obstacles like the mounting pole / ground plane) so that the VAM
PVT streamer's collision checking accounts for them.

## How It Works

The PVT streamer calls MoveIt's `/check_state_validity` service every cycle to
decide if a proposed joint target is safe. MoveIt uses the URDF collision
meshes + SRDF collision rules that were loaded at `move_group` startup.

Any link added to the URDF is automatically included in every collision check.
**No streamer code changes needed -- just URDF + SRDF edits.**

```
VAM joint targets -> PVT Streamer -> MoveIt /check_state_validity -> Robot
                                            ^
                                            | loads URDF + SRDF at startup
                                            |
                             tm12s.urdf.xacro + tm12s.srdf
                             (bind-mounted from host into container)
```

## Current Setup (Option A -- Bind Mounts)

Two files are bind-mounted from host into the hardware container. With the
profile workflow (see next section) you never need to restart the container
to pick up changes -- edits propagate through the bind mount immediately:

| Host path (edit here) | Container path (auto-mounted) |
|---|---|
| `/home/maleen/git/RAPP_LAB_04/docker/hw/urdf_override/tm12s.urdf.xacro` | `/tm2_ws/install/tm12s_moveit_config/share/tm12s_moveit_config/config/tm12s.urdf.xacro` |
| `/home/maleen/git/RAPP_LAB_04/docker/hw/urdf_override/tm12s.srdf` | `/tm2_ws/install/tm12s_moveit_config/share/tm12s_moveit_config/config/tm12s.srdf` |

The bind mounts are configured in [docker/docker-compose.hw.yml](../docker/docker-compose.hw.yml)
lines 46-48.

There is also a flat URDF used by RViz for VAM-prediction visualization:
`/home/maleen/csvdata/rapplab04/tm12s.urdf`. This one is auto-regenerated from
the xacro and doesn't need manual editing (see Step 4 below).

---

## Performance Profiles (Recommended for Production)

For performances where you need to swap between pre-built prop configurations
(e.g., one per team), use **profiles**. Each profile is a self-contained pair
of `tm12s.urdf.xacro` + `tm12s.srdf` files stored in
`docker/hw/urdf_override/profiles/<name>/`.

### Available profiles

```text
docker/hw/urdf_override/profiles/
├── bare/      # no prop, just pole + ground
├── team1/     # team 1's prop config
├── team2/     # team 2's prop config
└── team3/     # team 3's prop config
```

Seeded initial state: all four profiles currently contain the bare setup
(pole + ground, no end-effector prop). Fill in `team1/`, `team2/`, `team3/`
as each team finalizes their prop design.

### Swap between profiles

One command, **no docker restart needed**:

```bash
# List profiles (and show which is active)
./scripts/set_profile.sh

# Activate a profile (copies files in-place + regenerates flat URDF)
./scripts/set_profile.sh team1
```

How it stays container-restart-free: `set_profile.sh` uses `cp` to overwrite
the bind-mounted active files. On Linux, `cp` truncates the target in place
and writes new content, so the file's inode is preserved -- the running
container sees the new content immediately.

After `set_profile.sh` completes, Ctrl+C and re-run your three ROS launches.
The launch restart is still needed because `move_group` caches the URDF in
memory at launch time.

### Edit a profile

1. Edit `docker/hw/urdf_override/profiles/<name>/tm12s.urdf.xacro` and
   `tm12s.srdf` directly (using the geometry recipes in the
   "Step-by-Step: Add a New Prop" section below).
2. Run `./scripts/set_profile.sh <name>` to apply.

If the profile you edited is already active, re-running `set_profile.sh` still
works — it re-copies and re-applies.

### Create a new profile

```bash
cp -r docker/hw/urdf_override/profiles/bare docker/hw/urdf_override/profiles/my_new_profile
# then edit docker/hw/urdf_override/profiles/my_new_profile/*.xacro / *.srdf
./scripts/set_profile.sh my_new_profile
```

---

## Quick Reference: Modify the Currently Active Config (Without Profiles)

If you're iterating on a single config and don't need profile swap machinery:

### To SWAP the prop (e.g., sword → mask)

1. Edit `docker/hw/urdf_override/tm12s.urdf.xacro`: replace the geometry block inside `<link name="prop_link">`
2. Run `./scripts/apply_prop_changes.sh`
3. Ctrl+C and restart your three ROS launches

### To REMOVE the prop (bare robot)

1. In `tm12s.urdf.xacro`: delete or comment out the `prop_joint` and `prop_link` blocks
2. In `tm12s.srdf`: delete or comment out lines containing `prop_link`
3. Run `./scripts/apply_prop_changes.sh`
4. Restart ROS launches

### To ADD a second prop

1. In `tm12s.urdf.xacro`: add another `prop2_joint` + `prop2_link` block
2. In `tm12s.srdf`: add `<disable_collisions link1="link_6" link2="prop2_link" reason="Adjacent"/>` and `<disable_collisions link1="link_5" link2="prop2_link" reason="Never"/>`
3. Run `./scripts/apply_prop_changes.sh`
4. Restart ROS launches

**Note:** editing active files directly means your changes only live in the
active location, not in any profile. For performance work, prefer editing
inside a profile directory and using `set_profile.sh`.

---

## Step-by-Step: Add a New Prop

### Step 1 -- Measure your prop

| Prop type | Collision shape | Parameters needed |
|---|---|---|
| Sword, stick, pointer | `<cylinder>` | length, radius |
| Mask, paddle, flag | `<box>` | x (width) × y (thickness) × z (height) |
| Ball, head | `<sphere>` | radius |
| Complex mesh | `<mesh>` | STL file path |

**Oversize the collision by ~2cm** to give a safety margin. For the visual you
can use the true dimensions.

### Step 2 -- Figure out the mount offset

The prop attaches to `link_6` (the flange). In link_6's frame:

- **Z axis points OUTWARD** from the flange (along the tool direction)
- The joint `<origin xyz="X Y Z">` sets where the **center of the geometry**
  sits, not where it starts
- So for a 40cm cylinder with origin `z=0.30`, the cylinder extends from
  `z=0.10` to `z=0.50` in link_6 frame

**Rule of thumb:** leave 5-10cm between the flange and the start of your prop
so the collision geometry clears link_5's housing.

### Step 3 -- Edit the xacro

Open `/home/maleen/git/RAPP_LAB_04/docker/hw/urdf_override/tm12s.urdf.xacro`.

Find the existing `PROP` section and replace or modify. Template:

```xml
<!-- ==================== PROP: <NAME> ==================== -->
<!-- <describe dimensions and orientation> -->
<joint name="prop_joint" type="fixed">
    <parent link="link_6"/>
    <child link="prop_link"/>
    <origin xyz="0 0 0.30" rpy="0 0 0"/>   <!-- adjust offset here -->
</joint>

<link name="prop_link">
    <visual>
        <geometry>
            <!-- pick ONE: cylinder | box | sphere | mesh -->
            <cylinder length="0.40" radius="0.015"/>
        </geometry>
        <material name="prop_color">
            <color rgba="1.0 0.0 0.0 0.7"/>  <!-- R G B alpha -->
        </material>
    </visual>
    <collision>
        <geometry>
            <!-- SAME shape type, but optionally slightly larger -->
            <cylinder length="0.40" radius="0.02"/>
        </geometry>
    </collision>
    <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0"
                 iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
</link>
```

### Geometry recipes

**Cylinder (sword, stick):**
```xml
<cylinder length="0.40" radius="0.015"/>
```

**Box (mask, paddle):**
```xml
<box size="0.25 0.03 0.30"/>   <!-- x y z in meters -->
```

**Sphere (ball, head):**
```xml
<sphere radius="0.10"/>
```

**Mesh (complex shape, optional):**
```xml
<mesh filename="file:///data/processed/my_prop.stl" scale="1 1 1"/>
```

### Orientation tricks

If your prop isn't along link_6's Z axis, rotate the joint:

```xml
<!-- Mask flat, facing forward: flip so box Z -> link_6 Y -->
<origin xyz="0 0 0.20" rpy="1.5708 0 0"/>

<!-- Sword pointing sideways: rotate 90° around Y -->
<origin xyz="0.20 0 0.10" rpy="0 1.5708 0"/>
```

### Step 4 -- Edit the SRDF

Open `/home/maleen/git/RAPP_LAB_04/docker/hw/urdf_override/tm12s.srdf`.

You **must** disable the collision check between the prop and the links it
touches/is near. The standard two entries for a single prop:

```xml
<disable_collisions link1="link_6" link2="prop_link" reason="Adjacent"/>
<disable_collisions link1="link_5" link2="prop_link" reason="Never"/>
```

**Never disable** collisions between prop_link and `link_0` through `link_4`
or the pole/ground -- those are exactly the collisions you want detected.

### Step 5 -- Apply the changes

**No docker restart needed.** The profile swap script (and the underlying
apply script) work entirely through `cp`, which preserves the bind-mounted
inode -- the container sees new content immediately.

If you edited a **profile** (under `profiles/<name>/`), use:

```bash
./scripts/set_profile.sh <name>
```

If you edited the **active** files directly (no profile), use:

```bash
./scripts/apply_prop_changes.sh
```

Then Ctrl+C and re-run your three ROS launches:

```bash
ros2 launch vam_inference tm12s_moveit_hw.launch.py robot_ip:=192.168.10.2
ros2 run   vam_inference vam_pvt_streamer --ros-args -p velocity_scale:=0.15 ...
ros2 launch vam_inference vam_tm12s_robot.launch.py ...
```

> The ROS launch restart is unavoidable -- `move_group` caches the URDF in memory
> at launch time, so new xacro/srdf content only takes effect on fresh launches.

#### What the scripts do

**`set_profile.sh <name>`**

1. `cp profiles/<name>/*.{xacro,srdf}` into the active location. `cp` on GNU
   Linux truncates-in-place, preserving the inode, so the container's bind
   mount sees the new content immediately -- no restart needed.
2. Calls `apply_prop_changes.sh` to regenerate the flat URDF.
3. Writes `.active_profile` marker.

**`apply_prop_changes.sh`**

1. Runs `docker exec rapp_hw xacro ... > /data/processed/tm12s.urdf` to
   regenerate the flat URDF that `vam_tm12s_robot.launch.py` uses for
   RViz VAM-prediction visualization.

#### If you prefer the manual form

```bash
# Regenerate flat URDF only (one command inside the container)
docker exec rapp_hw bash -c "
  source /opt/ros/humble/setup.bash &&
  source /tm2_ws/install/setup.bash &&
  xacro /tm2_ws/install/tm12s_moveit_config/share/tm12s_moveit_config/config/tm12s.urdf.xacro \
    > /data/processed/tm12s.urdf
"
```

#### When you need what

| What you changed | Need container restart? | Need flat URDF regen? | Need ROS launch restart? |
| --- | --- | --- | --- |
| Edited profile files, ran `set_profile.sh` | **No** | Done by script | Yes |
| Edited active files with `cp` / `install` | **No** | Yes (run apply script) | Yes |
| Edited active files with VSCode/editor | **Yes** (inode changed) | Yes | Yes |
| Nothing URDF-related (launch params only) | No | No | Maybe |

### Step 6 -- Verify

In RViz:

1. The new prop appears on the flange with the right colour
2. The robot initially doesn't freeze (if it does → prop is in permanent collision, see Troubleshooting)
3. Command a pose where the prop SHOULD hit the body → streamer logs collision warnings
4. Command a safe pose → robot moves normally

---

## Inode Gotcha (only matters if you edit active files directly)

This only applies if you edit `docker/hw/urdf_override/tm12s.urdf.xacro` or
`tm12s.srdf` **directly** with an editor that does atomic-save-by-rename
(VSCode, most IDEs, the Claude `Edit` tool). In that case the host file gets a
new inode and the container's single-file bind mount keeps pointing at the
old inode, so the container sees stale content.

### How to avoid it

**Use the profile workflow.** Edit files under `profiles/<name>/` (inode
doesn't matter there -- nothing is bind-mounted from the profile dir). Then
run `./scripts/set_profile.sh <name>`, which uses `cp` to overwrite the active
files in-place -- this preserves the inode, so the container sees new content
with zero restart.

### If you did edit an active file directly and got stuck

```bash
docker compose -f docker/docker-compose.hw.yml restart
```

Takes ~5 seconds. The bind mount re-attaches to the current inode.

### Quick sanity check that your edits propagated

```bash
docker exec rapp_hw grep prop_link \
  /tm2_ws/install/tm12s_moveit_config/share/tm12s_moveit_config/config/tm12s.urdf.xacro
```

If this shows your new prop, you're good.

---

## Visual vs Collision: what RViz shows you

- **Visual mesh** = the pretty green robot housing in RViz
- **Collision mesh** = the simplified geometry MoveIt actually checks

These often differ (collision is usually smaller/simpler for speed). You can
see a prop visually clip through the robot in RViz **without** MoveIt flagging
it -- that means the prop passed through visual-only space, not a collision
volume.

**Is this safe?** It depends:

- Small cosmetic overlap in RViz, real-world prop doesn't actually touch the
  robot → fine, nothing to fix
- Prop visually passes *through* the robot body in RViz → real prop probably
  will strike real robot → enlarge the prop's collision geometry (not visual)
  and/or enlarge the relevant link collision geometry

For safety-critical cases: oversize the prop's **collision** shape by 2-5cm
relative to the visual shape.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Robot won't move at all after adding prop | Prop is in permanent collision with a non-disabled link | Check SRDF -- likely need to disable vs `link_5` too. Or prop geometry intersects another link's collision mesh -- increase Z offset |
| Edits in host file don't show in container | Inode mismatch from atomic save | `docker compose restart` |
| URDF parse error on launch | XML syntax issue | Look for unclosed tags; every `<link>` and `<joint>` needs matching close |
| Prop missing in RViz | Flat URDF wasn't regenerated | Re-run Step 5 part 2 |
| Prop shows in MoveIt but wrong position | Joint origin math | Remember the cylinder center is at the joint origin, not the near end |
| Robot moves through prop in RViz but real prop would hit | Collision mesh smaller than visual | Oversize the `<collision>` geometry relative to `<visual>` |
| Collision check causes lots of "holding position" warnings during normal motion | Prop collision geometry too conservative | Shrink the `<collision>` size |

---

## Current Config Snapshot (as of 2026-04-16)

### Props in place

**Sword (`prop_link`)**

- 40cm long, 3cm diameter cylinder
- Mounted on link_6 Z-axis, joint origin `z=0.30` (extends z=0.10 to z=0.50)
- SRDF: disabled vs link_5 and link_6

### Static obstacles in place

**Pedestal pole (`pole_link`)**

- 58cm tall, 15cm diameter cylinder
- Mounted on `base`, joint origin `z=-0.29`
- SRDF: disabled vs base, link_0, ground_link

**Ground plane (`ground_link`)**

- 20mm thick, 122cm diameter disc
- Mounted on `base`, joint origin `z=-0.59`
- SRDF: disabled vs base, pole_link, link_0

---

## Quick Rollback (Back to Bare Robot)

If something breaks and you just want the baseline tm12s back:

```bash
# Remove the two bind-mount lines from docker-compose.hw.yml
# (lines 46-48 are the URDF/SRDF override block)
cd /home/maleen/git/RAPP_LAB_04/docker
docker compose -f docker-compose.hw.yml down
docker compose -f docker-compose.hw.yml up -d
```

The container falls back to the pristine xacro/srdf baked into the image. Re-add the
mount lines when you want to resume using your prop setup. Host-side files are left intact.

---

## Example: Adding a Second Prop (e.g., sword + mask)

Say you want a sword in one hand (link_6) and a mask attached slightly
differently. URDF additions:

```xml
<!-- Sword (existing) -->
<joint name="prop_joint" type="fixed">
    <parent link="link_6"/>
    <child link="prop_link"/>
    <origin xyz="0 0 0.30" rpy="0 0 0"/>
</joint>
<link name="prop_link"> ... </link>

<!-- Mask (new) -->
<joint name="prop2_joint" type="fixed">
    <parent link="link_6"/>
    <child link="prop2_link"/>
    <origin xyz="0 -0.15 0.10" rpy="1.5708 0 0"/>
</joint>
<link name="prop2_link">
    <visual>
        <geometry><box size="0.20 0.03 0.25"/></geometry>
        <material name="mask_tan"><color rgba="0.8 0.6 0.4 0.9"/></material>
    </visual>
    <collision>
        <geometry><box size="0.22 0.05 0.27"/></geometry>
    </collision>
    <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
    </inertial>
</link>
```

SRDF additions:

```xml
<disable_collisions link1="link_6" link2="prop2_link" reason="Adjacent"/>
<disable_collisions link1="link_5" link2="prop2_link" reason="Never"/>
<disable_collisions link1="prop_link" link2="prop2_link" reason="Never"/>
```

Then: restart container, regenerate flat URDF, restart ROS launches.
