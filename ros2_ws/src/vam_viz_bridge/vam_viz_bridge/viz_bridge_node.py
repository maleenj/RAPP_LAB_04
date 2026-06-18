"""vam_viz_bridge — config-driven ROS2 -> WebSocket bridge for live workshop viz.

Subscribes to a list of ROS2 topics defined in a YAML config and re-publishes
each message as a uniform JSON frame over a WebSocket server. Any laptop on the
network (Unity, Unreal, or a browser) connects to ws://<host-ip>:<port> and
receives every channel through one parser.

This node never imports or touches the inference code. Because all containers run
with host networking and a shared ROS_DOMAIN_ID, it simply taps the existing DDS
topic graph. To add a new data source: add one entry to bridge_channels.yaml.

Usage:
    ros2 launch vam_viz_bridge viz_bridge.launch.py
    ros2 run vam_viz_bridge viz_bridge_node --ros-args -p config:=/path/to/bridge_channels.yaml
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from vam_viz_bridge import serializers
from vam_viz_bridge.websocket_server import WebSocketBroadcastServer


def _import_message_class(type_str: str):
    """Import a ROS2 message class from its 'pkg/msg/Type' string."""
    parts = type_str.split("/")
    if len(parts) != 3:
        raise ValueError(f"Bad message type '{type_str}', expected 'pkg/msg/Type'.")
    pkg, sub, name = parts
    module = importlib.import_module(f"{pkg}.{sub}")
    return getattr(module, name)


def _qos_for(name: str) -> QoSProfile:
    """Map a friendly QoS name to a profile. Sensor data uses BEST_EFFORT."""
    if name == "sensor":
        return QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
    # "default" — reliable, depth 10
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=10,
    )


def _stamp_seconds(msg) -> Optional[float]:
    """Extract header stamp in float seconds if the message has a header."""
    header = getattr(msg, "header", None)
    if header is None:
        return None
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class VizBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("vam_viz_bridge")

        self.declare_parameter("config", "")
        config_path = self.get_parameter("config").get_parameter_value().string_value
        if not config_path:
            raise RuntimeError(
                "No config provided. Pass -p config:=/path/to/bridge_channels.yaml"
            )

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        server_cfg = (cfg.get("server") or {}).get("websocket") or {}
        host = server_cfg.get("host", "0.0.0.0")
        port = int(server_cfg.get("port", 8765))

        self._server = WebSocketBroadcastServer(host=host, port=port, logger=self.get_logger())
        self._server.start()

        self._subs = []
        self._frame_counts: Dict[str, int] = {}
        # derived[source_channel] -> list of (out_channel, transform_fn)
        self._derived: Dict[str, list] = {}
        self._setup_channels(cfg.get("channels") or [])
        self._setup_derived(cfg.get("derived") or [])

        # Periodic heartbeat so operators see life + client count in the log.
        self.create_timer(5.0, self._heartbeat)
        self.get_logger().info(
            f"vam_viz_bridge ready: {len(self._subs)} channel(s) on ws://{host}:{port}"
        )

    def _setup_channels(self, channels: list) -> None:
        for entry in channels:
            topic = entry["topic"]
            type_str = entry["type"]
            channel = entry.get("channel", topic.strip("/").replace("/", "_"))
            serializer_name = entry.get("serializer")
            qos_name = entry.get("qos", "default")

            try:
                msg_class = _import_message_class(type_str)
            except Exception as exc:
                # e.g. zed_msgs not installed in the lightweight image — skip gracefully.
                self.get_logger().warn(
                    f"Skipping channel '{channel}' ({topic}): cannot import "
                    f"message type '{type_str}': {exc}"
                )
                continue

            try:
                serializer = serializers.resolve_serializer(type_str, serializer_name)
            except KeyError as exc:
                self.get_logger().error(f"Channel '{channel}': {exc}")
                continue

            self._frame_counts[channel] = 0
            cb = self._make_callback(channel, serializer)
            sub = self.create_subscription(msg_class, topic, cb, _qos_for(qos_name))
            self._subs.append(sub)
            self.get_logger().info(
                f"bridging {topic} [{type_str}] -> channel '{channel}'"
                f" (serializer={serializer.__name__})"
            )

    def _setup_derived(self, derived: list) -> None:
        """Wire config-driven derived channels: source channel -> transform -> new channel.

        Transforms run in THIS (viz) container, keeping all explainability math out
        of the inference node. Imported lazily so the bridge still runs without numpy
        when no derived channels are configured.
        """
        if not derived:
            return
        from vam_viz_bridge import transforms  # lazy: needs numpy

        for entry in derived:
            src = entry["from"]
            out_channel = entry["channel"]
            try:
                fn = transforms.resolve_transform(entry["transform"])
            except KeyError as exc:
                self.get_logger().error(f"Derived '{out_channel}': {exc}")
                continue
            self._derived.setdefault(src, []).append((out_channel, fn))
            self._frame_counts[out_channel] = 0
            self.get_logger().info(
                f"derived '{out_channel}' = {entry['transform']}({src})"
            )

    def _make_callback(self, channel: str, serializer):
        def _cb(msg) -> None:
            try:
                frame: Dict[str, Any] = serializer(msg)
            except Exception as exc:
                self.get_logger().warn(f"serialize failed on '{channel}': {exc}")
                return
            frame["channel"] = channel
            stamp = _stamp_seconds(msg)
            if stamp is not None:
                frame["stamp"] = stamp
            self._frame_counts[channel] += 1
            self._server.broadcast(channel, frame)
            self._emit_derived(channel, frame)

        return _cb

    def _emit_derived(self, source_channel: str, frame: Dict[str, Any]) -> None:
        """Run any transforms registered on this channel and broadcast results."""
        configs = self._derived.get(source_channel)
        if not configs or "tensors" not in frame:
            return
        for out_channel, fn in configs:
            try:
                dframe = fn(frame["tensors"])
            except Exception as exc:
                self.get_logger().warn(f"transform for '{out_channel}' failed: {exc}")
                continue
            dframe["channel"] = out_channel
            if "stamp" in frame:
                dframe["stamp"] = frame["stamp"]
            self._frame_counts[out_channel] = self._frame_counts.get(out_channel, 0) + 1
            self._server.broadcast(out_channel, dframe)

    def _heartbeat(self) -> None:
        n = self._server.client_count
        if n == 0:
            return
        counts = ", ".join(f"{k}={v}" for k, v in self._frame_counts.items())
        self.get_logger().info(f"clients={n} | frames sent: {counts}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VizBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
