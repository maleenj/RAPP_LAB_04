# tm_teach_playback

Record a fixed sequence of joint waypoints for the TM12S arm and play them back —
a **scripted movement** for demos, warm-ups, or as a deterministic fallback when the
VAM is not driving the robot.

It provides two nodes:

| Node | Command | Purpose |
|------|---------|---------|
| `record_waypoints` | `ros2 run tm_teach_playback record_waypoints` | Interactively capture the current arm pose into a YAML sequence |
| `play_waypoints` | `ros2 run tm_teach_playback play_waypoints ...` | Replay a saved YAML sequence through MoveIt |

Playback sends goals to the MoveIt `move_action` server using planning group
`tmr_arm` and the six joints `joint_1` … `joint_6`.

---

## Prerequisites

1. **Build & source the workspace**
   ```bash
   cd ~/git/RAPP_LAB_04/ros2_ws
   colcon build --packages-select tm_teach_playback
   source install/setup.bash
   ```

2. **Bring up the robot + MoveIt** (playback needs the `move_action` server, and
   recording needs `/joint_states`). Use the hardware MoveIt launch — **not**
   `demo.launch.py` / `tm12s_moveit.launch.py`, which start a fake ros2_control
   stack that publishes competing zero joint states (see [hardware/README](../../../hardware/README.md)):
   ```bash
   ros2 launch <tm bringup> tm12s_moveit_hw.launch.py
   ```

> For recording you only need `/joint_states` (you can hand-guide the arm), but
> playback requires the full MoveIt stack to be running.

---

## 1. Record a sequence

```bash
ros2 run tm_teach_playback record_waypoints
```

Move/jog the arm to each pose you want, then use the interactive menu:

```
[c]apture  [l]ist  [d]elete last  [s]ave  [q]uit
```

- **c** — capture the current joint angles. Prompts for:
  - `move_time` (s, default **3.0**) — how long the move *to* this waypoint should take during playback.
  - `wait_time` (s, default **0.5**) — how long to pause *at* this waypoint before moving on.
- **l** — list captured waypoints.
- **d** — delete the last captured waypoint.
- **s** — save to YAML. Defaults to `/data/rosbags/waypoints_<timestamp>.yaml`
  (host: `~/rosbags/rapplab04/`). Press Enter to accept or type your own path.
- **q** — quit.

---

## 2. Play the scripted movement

```bash
ros2 run tm_teach_playback play_waypoints \
  --ros-args \
  -p waypoints_file:=/data/rosbags/waypoints_20260622_101500.yaml \
  -p loop:=false
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `waypoints_file` | string | `''` (required) | Absolute path to a saved waypoints YAML. The node exits if unset. |
| `loop` | bool | `false` | If `true`, repeat the whole sequence indefinitely. Stop with `Ctrl-C`. |

The node moves through each waypoint in order, scaling MoveIt velocity/acceleration
by `min(1.0, 1.0 / move_time)`, then sleeping for that waypoint's `wait_time`.
If any motion is rejected or fails to plan, playback **stops** at that waypoint and
logs the MoveIt error code.

---

## Waypoints file format

```yaml
joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
waypoints:
  - joints: [0.0000, -0.5000, 1.2000, 0.0000, 1.5708, 0.0000]
    move_time: 3.0      # seconds to reach this pose
    wait_time: 0.5      # seconds to pause once reached
  - joints: [0.3000, -0.3000, 1.0000, 0.0000, 1.4000, 0.2000]
    move_time: 2.0
    wait_time: 1.0
```

- `joints` — six joint positions in **radians**, ordered to match `joint_names`.
- `move_time` — optional, defaults to `3.0` if omitted.
- `wait_time` — optional, defaults to `0.0` at playback time if omitted.

You can hand-author a file in this format instead of recording one.

---

## Notes

- No sample waypoint files ship with the repo — record one (step 1) or hand-write
  a YAML before running playback.
- Inside the container, `/data/rosbags` maps to the host `~/rosbags/rapplab04`.
- Joint tolerances are tight (±0.0001 rad), so MoveIt plans to the exact recorded
  pose. Large gaps between consecutive waypoints may fail to plan — keep steps small.
