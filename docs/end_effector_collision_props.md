# Adding End-Effector Props (Masks, Swords, etc.) with Collision Checking

## Problem

The VAM PVT streamer checks self-collision via MoveIt's `GetStateValidity` service, which uses the TM12S URDF collision meshes. If you attach a prop (mask, sword, etc.) to the end effector, the system doesn't know it exists -- the robot can swing the prop into its own body.

## Solution: Add a Collision Link to the URDF

The easiest, most reliable method is to **add a fixed link to the URDF** representing the prop. MoveIt will automatically include it in all collision checks -- no code changes needed.

### How It Works

Your existing pipeline:
```
VAM joint targets --> PVT Streamer --> MoveIt GetStateValidity --> Robot
```
MoveIt loads the URDF at startup. If the URDF has an extra link attached to `link_6` (the end effector), every `GetStateValidity` call already checks collisions against it. **Zero code changes.**

---

## Step-by-Step Instructions

### Step 1: Understand the Two URDFs

There are **two URDF files** involved during live operation. They serve different purposes:

| File (host) | Container path | Launch file | Purpose |
|---|---|---|---|
| `tm2_ros2/tm12s_moveit_config/config/tm12s.urdf.xacro` | Built into ROS package | `tm12s_moveit_hw.launch.py` | **Collision checking** — MoveIt `move_group` loads this and serves `/check_state_validity` |
| `/home/maleen/csvdata/rapplab04/tm12s.urdf` | `/data/processed/tm12s.urdf` | `vam_tm12s_robot.launch.py` | **RViz visualization** — `robot_state_publisher` uses this for TF frames |

The PVT streamer queries MoveIt's `/check_state_validity` service, so **the xacro is what
controls collision checking**. The CSV folder URDF is used live too, but only for visualization.

**You must edit both** so that collision checking AND visualization include the prop:

1. **Edit the xacro** (collision checking):
   `/home/maleen/git/tm2_ros2/tm12s_moveit_config/config/tm12s.urdf.xacro`
   Add the prop link right before the closing `</robot>` tag.

2. **Regenerate the flat URDF** (visualization):
   ```bash
   # Inside the ROS container
   xacro $(ros2 pkg prefix tm12s_moveit_config)/share/tm12s_moveit_config/config/tm12s.urdf.xacro \
       > /data/processed/tm12s.urdf
   ```

3. **SRDF** (disable adjacent-link collision for prop):
   `/home/maleen/git/tm2_ros2/tm12s_moveit_config/config/tm12s.srdf`

### Step 2: Measure Your Prop

You only need rough dimensions. Use simple shapes:

| Prop Type | Best Shape | What to Measure |
|-----------|-----------|-----------------|
| Sword/stick | Cylinder | Length + diameter |
| Mask/paddle | Box | Width x height x thickness |
| Ball/round | Sphere | Radius |

**Oversize by ~2cm** for safety margin.

### Step 3: Add the Prop Link to the URDF

Open the URDF and find `link_6` (the end effector). **After** the `link_6` definition and its joint, add:

#### For a Sword / Stick (Cylinder)

```xml
<!-- ========== PROP: Sword ========== -->
<joint name="prop_joint" type="fixed">
  <parent link="link_6"/>
  <child link="prop_link"/>
  <!-- Offset from end-effector center. Z = forward along tool axis -->
  <!-- Adjust xyz so the cylinder center aligns with the prop center -->
  <origin xyz="0 0 0.30" rpy="0 0 0"/>
</joint>

<link name="prop_link">
  <collision>
    <geometry>
      <!-- length = prop length, radius = prop radius + margin -->
      <cylinder length="0.50" radius="0.03"/>
    </geometry>
  </collision>
  <!-- Minimal inertia so URDF parser doesn't complain -->
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>
```

#### For a Mask / Flat Object (Box)

```xml
<!-- ========== PROP: Mask ========== -->
<joint name="prop_joint" type="fixed">
  <parent link="link_6"/>
  <child link="prop_link"/>
  <origin xyz="0 0 0.15" rpy="0 0 0"/>
</joint>

<link name="prop_link">
  <collision>
    <geometry>
      <!-- width x thickness x height -->
      <box size="0.25 0.05 0.30"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>
```

### Step 4: Get the Offset Right

The `<origin xyz="0 0 Z">` in the joint sets where the prop center sits relative to `link_6`.

- **Z axis** = along the tool (pointing outward from flange)
- If your sword is 50cm long and starts at the flange: `Z = 0.25` (center at half length)
- If your mask is 15cm from the flange: `Z = 0.15`

**Quick way to verify:** Temporarily add a `<visual>` block with the same geometry so you can see it in RViz:
```xml
<link name="prop_link">
  <visual>
    <geometry>
      <cylinder length="0.50" radius="0.03"/>
    </geometry>
    <material name="red">
      <color rgba="1 0 0 0.5"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.50" radius="0.03"/>
    </geometry>
  </collision>
  ...
</link>
```

### Step 5: Update the SRDF (Important!)

MoveIt uses the SRDF to disable collision checks between adjacent links (they always touch). You need to tell it **not** to check prop_link vs link_6 (they're welded together) but **do** check it vs everything else.

The SRDF file is at:

```
/home/maleen/git/tm2_ros2/tm12s_moveit_config/config/tm12s.srdf
```

Add this line inside the `<robot>` tag:
```xml
<disable_collisions link1="link_6" link2="prop_link" reason="Adjacent"/>
```

**Do NOT add any other disable_collisions entries for prop_link** -- you want it checked against link_0 through link_5.

### Step 6: Restart Everything

The URDF is loaded at launch time. You must restart:

```bash
# Stop all containers
docker compose -f docker/docker-compose.hw.yml down

# Restart
docker compose -f docker/docker-compose.hw.yml up
```

### Step 7: Verify in RViz

1. Open RViz and load the robot model
2. Confirm the prop geometry appears attached to link_6
3. Manually jog the robot (or replay a rosbag) and watch for collision highlights
4. Test a pose where the sword/mask would hit the robot body -- MoveIt should flag it

---

## Swapping Props Quickly

For fast prop changes between performances, you can parameterize the MoveIt launch file:

1. **Create one xacro per prop** in `/home/maleen/git/tm2_ros2/tm12s_moveit_config/config/`:

   ```
   tm12s.urdf.xacro          (original, no prop)
   tm12s_sword.urdf.xacro    (copy of original + sword link)
   tm12s_mask.urdf.xacro     (copy of original + mask link)
   ```

2. **Add a launch argument** to `tm12s_moveit_hw.launch.py`:

   ```python
   # Add a prop argument
   DeclareLaunchArgument('prop', default_value='', description='Prop URDF suffix'),

   # Change the MoveIt config builder to use it
   urdf_file = 'config/tm12s.urdf.xacro'  # or tm12s_sword.urdf.xacro etc.
   ```

3. **Launch with the prop URDF:**

   ```bash
   # This is the MoveIt launch -- it's where the URDF is loaded
   ros2 launch vam_inference tm12s_moveit_hw.launch.py robot_ip:=192.168.10.2
   ```

   The PVT streamer and VAM node launch commands stay exactly the same -- they don't need to know about the prop.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Robot won't move at all after adding prop | Prop is too large or offset is wrong -- it's always in collision. Reduce size or fix origin. |
| URDF parse error on launch | Check XML syntax. Every `<joint>` needs matching `<link>`. Inertial block is required. |
| Prop not showing in RViz | Add a `<visual>` block (see Step 4). Collision geometry is invisible by default. |
| Collisions not being detected | Check SRDF -- make sure you only disabled `link_6 <-> prop_link`, not others. |
| Robot moves but ignores prop | MoveIt might not have reloaded. Fully restart all nodes (not just the streamer). |
| PVT streamer holds position too often | Prop geometry is too conservative (oversized). Shrink it slightly. |

## Key Numbers to Remember

- Current self-collision threshold: **10mm** (`self_collision_proximity_threshold: 0.01`)
- Current scene collision threshold: **20mm**
- Collision check timeout in PVT streamer: **50ms**
- Perturbation search radius: **0.08 rad (~4.6 deg)**
- Adding a prop link adds negligible compute cost (one extra primitive in the collision check)
