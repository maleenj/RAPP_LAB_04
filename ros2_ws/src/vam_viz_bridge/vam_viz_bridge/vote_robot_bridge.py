#!/usr/bin/env python3
"""vote_robot_bridge — drive the robot's mirror/contrast mode from the audience vote.

Polls the Fringe voting backend (Hostinger PHP, `state.php`) over plain HTTPS and
calls the EXISTING `/vam/switch_model` service so a sustained majority switches the
robot between the mirror model (id 1) and the contrast model (id 2).

This node only ADDS behaviour: it never modifies the inference node — it is just
another client of a service the robot already exposes. If it dies or is frozen, you
switch by hand exactly as before (`ros2 service call /vam/switch_model ...`).

Safety / anti-thrash:
  * deadband — switch to contrast only when ratio > enter_contrast, back to mirror
    only when ratio < enter_mirror (neutral band in between holds current mode);
  * hold — the new majority must persist for `hold_seconds` before committing;
  * cooldown — at most one switch per `min_switch_interval` (a model swap briefly
    resets the inference pipeline, so we don't flip-flop);
  * manual override — set param `auto_switch:=false` OR publish False on
    `/vote/auto_enable` to freeze auto-switching and take manual control instantly.

ratio convention (from state.php): 0 = all mirror, 1 = all contrast, 0.5 = no votes.

Run:
  ros2 launch vam_viz_bridge vote_bridge.launch.py \
       state_url:=https://yourdomain.com/fringe/api/tally.php
"""

import json
import urllib.request
import urllib.parse

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

from vam_interfaces.srv import SwitchModel


class VoteRobotBridge(Node):
    def __init__(self):
        super().__init__("vote_robot_bridge")

        # --- parameters (all tunable at launch / via `ros2 param set`) ---
        # NOTE: backend file is tally.php (NOT state.php — Hostinger blocks that name).
        self.declare_parameter("state_url", "http://localhost/fringe/api/tally.php")
        self.declare_parameter("state_token", "")        # if state.php is token-protected
        self.declare_parameter("poll_interval", 1.0)     # seconds between polls
        self.declare_parameter("http_timeout", 2.0)      # seconds per request
        self.declare_parameter("active_window", 25)      # forwarded to state.php ?window=
        self.declare_parameter("enter_contrast", 0.55)   # ratio above -> want contrast
        self.declare_parameter("enter_mirror", 0.45)     # ratio below -> want mirror
        self.declare_parameter("hold_seconds", 3.0)      # majority must persist this long
        self.declare_parameter("min_switch_interval", 12.0)  # cooldown between switches
        self.declare_parameter("mirror_model_id", 1)
        self.declare_parameter("contrast_model_id", 2)
        self.declare_parameter("auto_switch", True)      # master enable (manual override)

        self._state_url = self.get_parameter("state_url").value
        self._token = self.get_parameter("state_token").value
        self._timeout = float(self.get_parameter("http_timeout").value)
        self._window = int(self.get_parameter("active_window").value)
        self._enter_contrast = float(self.get_parameter("enter_contrast").value)
        self._enter_mirror = float(self.get_parameter("enter_mirror").value)
        self._hold = float(self.get_parameter("hold_seconds").value)
        self._cooldown = float(self.get_parameter("min_switch_interval").value)
        self._mirror_id = int(self.get_parameter("mirror_model_id").value)
        self._contrast_id = int(self.get_parameter("contrast_model_id").value)

        # --- runtime state ---
        self._committed = None          # 'mirror' | 'contrast' | None (unknown at start)
        self._candidate = None          # mode the current ratio favours
        self._candidate_since = 0.0
        self._last_switch = -1e9
        self._switch_inflight = False
        self._topic_enable = True       # /vote/auto_enable gate (independent of param)
        self._frozen_target = None      # last target announced while frozen (de-spam)

        # --- ROS wiring ---
        self._cli = self.create_client(SwitchModel, "/vam/switch_model")
        self.create_subscription(Bool, "/vote/auto_enable", self._on_enable, 10)
        period = float(self.get_parameter("poll_interval").value)
        self.create_timer(period, self._poll)

        self.get_logger().info(
            f"vote_robot_bridge polling {self._state_url} every {period:.1f}s | "
            f"deadband [{self._enter_mirror},{self._enter_contrast}] hold {self._hold}s "
            f"cooldown {self._cooldown}s | mirror=id{self._mirror_id} contrast=id{self._contrast_id}"
        )
        if not self._cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                "/vam/switch_model not available yet — will keep retrying on each switch."
            )

    # ---- helpers ----------------------------------------------------------
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _enabled(self) -> bool:
        return bool(self.get_parameter("auto_switch").value) and self._topic_enable

    def _on_enable(self, msg: Bool):
        if msg.data != self._topic_enable:
            self.get_logger().info(
                f"/vote/auto_enable -> {'ENABLED' if msg.data else 'FROZEN (manual control)'}"
            )
        self._topic_enable = bool(msg.data)

    def _fetch_ratio(self):
        url = self._state_url
        params = {"window": self._window}
        if self._token:
            params["token"] = self._token
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data

    # ---- main loop --------------------------------------------------------
    def _poll(self):
        try:
            state = self._fetch_ratio()
            ratio = float(state.get("ratio", 0.5))
        except Exception as e:  # noqa: BLE001 — network hiccups are expected, keep polling
            self.get_logger().warn(f"state poll failed: {e}", throttle_duration_sec=10.0)
            return

        # ratio -> desired mode with deadband
        if ratio > self._enter_contrast:
            want = "contrast"
        elif ratio < self._enter_mirror:
            want = "mirror"
        else:
            want = None  # neutral / no clear majority -> hold

        now = self._now()
        self.get_logger().info(
            f"ratio={ratio:.2f} m={state.get('mirror')} c={state.get('contrast')} "
            f"active={state.get('active')} want={want} committed={self._committed} "
            f"{'' if self._enabled() else '[FROZEN]'}",
            throttle_duration_sec=2.0,
        )

        if want is None or want == self._committed:
            self._candidate = None
            return

        # a new majority is forming — require it to hold, then respect the cooldown
        if self._candidate != want:
            self._candidate = want
            self._candidate_since = now
            return
        if (now - self._candidate_since) < self._hold:
            return
        if (now - self._last_switch) < self._cooldown:
            return

        self._do_switch(want)

    def _do_switch(self, mode: str):
        model_id = self._contrast_id if mode == "contrast" else self._mirror_id
        if not self._enabled():
            # Frozen: announce once per distinct target, but DON'T touch committed state —
            # so when unfrozen we still re-assert the crowd's choice over manual control.
            if self._frozen_target != mode:
                self.get_logger().info(
                    f"[FROZEN] majority wants {mode} (model_id {model_id}); auto_switch off — not switching"
                )
                self._frozen_target = mode
            return
        if self._switch_inflight:
            return
        if not self._cli.service_is_ready():
            self.get_logger().warn("/vam/switch_model not ready; deferring switch")
            return

        self._switch_inflight = True
        self._last_switch = self._now()
        req = SwitchModel.Request()
        req.model_id = model_id
        self.get_logger().info(f"SWITCH -> {mode} (model_id {model_id})")
        future = self._cli.call_async(req)
        future.add_done_callback(lambda f, m=mode: self._on_switch_done(f, m))

    def _on_switch_done(self, future, mode: str):
        self._switch_inflight = False
        try:
            res = future.result()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"switch_model call failed: {e}")
            return
        if res.success:
            self._committed = mode
            self._candidate = None
            self._frozen_target = None
            self.get_logger().info(f"now in '{res.active_model_name}' ({mode})")
        else:
            self.get_logger().warn(f"switch_model refused: {res.message}")


def main(args=None):
    rclpy.init(args=args)
    node = VoteRobotBridge()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
