# Legacy experiments (vam_inference variants)

Superseded development iterations of the `vam_inference` package, kept for
reference. **None of these are built or installed** — the live package is
`ros2_ws/src/vam_inference/`, and the current streamer is
`vam_pvt_streamer_new2.py` (run as `ros2 run vam_inference vam_pvt_streamer`).

| File | What it was | Why superseded |
|---|---|---|
| `vam_pvt_streamer.py` | First PVT streamer (direct joint streaming to the TM12S) | Replaced by later iterations with CPERR/safety fixes |
| `vam_pvt_streamer_new.py` | Second iteration (CPERR 241 debugging) | Intermediate experiment |
| `vam_pvt_streamer_new_fix.py` | Patch experiment on `_new` | Intermediate experiment |
| `vam_pvt_streamer_new3.py` | Later experiment branched from `_new2` | `_new2` (with the STO/safety-limit fixes) is the workshop-proven version |
| `vam_pvt_streamer_diag.py` | Diagnostics streamer (CSV logging of PVT behavior) | Debugging complete; pairs with `vam_tm12s_robot_diag.launch.py` |
| `vam_tm12s_node_diag.py` | Diagnostics inference node | Debugging complete |
| `diag_csv_logger.py` | CSV logger used only by the `_diag` variants | Moves with them |
| `vam_node.py` | Original UR10 inference node | UR10 pipeline retired (see `legacy/` root) |
| `servo_to_tm_ptp_bridge.py` | Point-to-point motion strategy (Plan C) | PVT streaming (Plan B) shipped; see `docs/archive/` |
| `servo_to_tm_vjog_bridge.py` | Velocity-jog motion strategy (Plan A) | PVT streaming (Plan B) shipped; see `docs/archive/` |
| `launch/vam_robot.launch.py` | UR10 robot bringup | UR10 retired |
| `launch/vam_rviz.launch.py` | UR10 RViz visualization | UR10 retired |
| `launch/vam_inference.launch.py` | UR10-era inference launch | UR10 retired |
| `launch/vam_tm12s_robot_diag.launch.py` | Diagnostics bringup (uses the `_diag` nodes) | Debugging complete |

To resurrect one: copy it back into
`ros2_ws/src/vam_inference/vam_inference/` (or `launch/`), re-add its
`console_scripts` entry in `setup.py`, and rebuild the container.
