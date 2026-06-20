"""skeleton_relay — republish the ZED body skeleton as a generic numeric topic.

Why: the lightweight viz bridge container does not have `zed_msgs`, so it can't
subscribe to /zed/.../skeletons directly. This tiny node (run in a container that
HAS zed_msgs, e.g. rapp_vam/rapp_hw) converts the skeleton into a plain
std_msgs/Float32MultiArray on /vam/skeleton — shape [K,3] keypoints — which the
bridge streams with one config line (no zed_msgs needed downstream).

This is the project's standard "publish anything to a ROS topic, then bridge it"
pattern. It does not touch any working node.

Usage:
    ros2 run vam_inference skeleton_relay
    ros2 run vam_inference skeleton_relay --ros-args \
        -p target_skeleton_id:=5 -p skeleton_topic:=/zed/zed_node/body_trk/skeletons
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout
from zed_msgs.msg import ObjectsStamped


class SkeletonRelay(Node):
    def __init__(self):
        super().__init__("skeleton_relay")
        self.declare_parameter("skeleton_topic", "/zed/zed_node/body_trk/skeletons")
        self.declare_parameter("output_topic", "/vam/skeleton")
        self.declare_parameter("target_skeleton_id", -1)  # -1 = first detected body
        self.declare_parameter("max_keypoints", 18)        # ZED BODY_18

        in_topic = self.get_parameter("skeleton_topic").value
        out_topic = self.get_parameter("output_topic").value
        self._target_id = int(self.get_parameter("target_skeleton_id").value)
        self._max_kp = int(self.get_parameter("max_keypoints").value)

        # Best-effort: compatible with both reliable and best-effort publishers.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._pub = self.create_publisher(Float32MultiArray, out_topic, 10)
        self.create_subscription(ObjectsStamped, in_topic, self._cb, qos)
        self.get_logger().info(
            f"skeleton_relay: {in_topic} -> {out_topic} "
            f"(target_id={self._target_id}, max_kp={self._max_kp})"
        )

    def _select(self, objects):
        if self._target_id >= 0:
            for obj in objects:
                if obj.label_id == self._target_id:
                    return obj
            return None
        return objects[0]

    def _cb(self, msg: ObjectsStamped) -> None:
        if len(msg.objects) == 0:
            return
        obj = self._select(msg.objects)
        if obj is None:
            return

        keypoints = obj.skeleton_3d.keypoints[: self._max_kp]
        pts = np.array(
            [[kp.kp[0], kp.kp[1], kp.kp[2]] for kp in keypoints],
            dtype=np.float32,
        )
        # Zero any non-finite values so the array stays a clean [K,3].
        pts = np.where(np.isfinite(pts), pts, 0.0).astype(np.float32)
        k = pts.shape[0]
        flat = pts.flatten().tolist()

        out = Float32MultiArray()
        layout = MultiArrayLayout()
        layout.dim.append(MultiArrayDimension(label="keypoints", size=k, stride=k * 3))
        layout.dim.append(MultiArrayDimension(label="xyz", size=3, stride=3))
        out.layout = layout
        out.data = flat
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SkeletonRelay()
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
