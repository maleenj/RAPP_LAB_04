"""VAM Inference Node — real-time skeleton-to-joint prediction for UR10.

Subscribes to ZED skeleton tracking and UR10 joint states, runs the trained
Action Chunking Transformer through the temporal ensemble pipeline, and
publishes smooth joint commands.

Modes:
    rviz  — publish JointState on /vam/joint_states for RViz visualization
    robot — stream joint velocity commands to MoveIt Servo via JointJog

Robot mode requires:
    1. forward_position_controller active (MoveIt Servo publishes to it)
    2. MoveIt Servo running (from ur_moveit_config)
"""

import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from rclpy.time import Time
from control_msgs.msg import JointJog
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from zed_msgs.msg import ObjectsStamped

import tf2_ros

# Ensure vam_utils is importable (mounted at /workspace in container)
_workspace = Path("/workspace")
if str(_workspace) not in sys.path:
    sys.path.insert(0, str(_workspace))

from vam_utils.config import InferenceConfig
from vam_utils.inference import (
    InputAssembler,
    VAMModelWrapper,
    TemporalEnsemble,
    SafetyChecker,
)

# UR10 joint names in URDF order (matches j0–j5 in training data)
UR10_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class VAMInferenceNode(Node):
    def __init__(self):
        super().__init__("vam_inference_node")

        # --- Declare parameters ---
        self.declare_parameter("mode", "rviz")
        self.declare_parameter(
            "checkpoint_path", "/data/models/vam_20260210_2342/best.pt"
        )
        self.declare_parameter(
            "model_config_path", "/data/models/vam_20260210_2342/model_config.json"
        )
        self.declare_parameter(
            "norm_stats_path",
            "/data/processed/tensors/2026_02_10_tin10_tout10/norm_stats.pt",
        )
        self.declare_parameter("device", "cuda")
        self.declare_parameter("prediction_stride_K", 1)
        self.declare_parameter("ensemble_decay_weight", 0.5)
        self.declare_parameter("max_joint_velocity_rad_s", 1.0)
        self.declare_parameter("max_joint_acceleration_rad_s2", 5.0)
        self.declare_parameter("target_skeleton_id", -1)
        self.declare_parameter("tracking_timeout_sec", 0.5)
        self.declare_parameter("trajectory_lookahead_frames", 5)
        self.declare_parameter("servo_proportional_gain", 5.0)

        # --- Read parameters ---
        self._mode = self.get_parameter("mode").value
        self._target_skeleton_id = self.get_parameter("target_skeleton_id").value
        self._tracking_timeout = self.get_parameter("tracking_timeout_sec").value
        self._lookahead = self.get_parameter("trajectory_lookahead_frames").value
        self._servo_gain = self.get_parameter("servo_proportional_gain").value

        # --- Build inference config from parameters ---
        config = InferenceConfig(
            checkpoint_path=Path(self.get_parameter("checkpoint_path").value),
            model_config_path=Path(self.get_parameter("model_config_path").value),
            norm_stats_path=Path(self.get_parameter("norm_stats_path").value),
            device=self.get_parameter("device").value,
            prediction_stride_K=self.get_parameter("prediction_stride_K").value,
            ensemble_decay_weight=self.get_parameter("ensemble_decay_weight").value,
            max_joint_velocity_rad_s=self.get_parameter(
                "max_joint_velocity_rad_s"
            ).value,
            max_joint_acceleration_rad_s2=self.get_parameter(
                "max_joint_acceleration_rad_s2"
            ).value,
        )

        # --- Load model and create pipeline components ---
        self.get_logger().info("Loading VAM model...")
        self._model = VAMModelWrapper(config)
        self._assembler = InputAssembler(
            norm_stats=self._model.norm_stats, T_in=10
        )
        self._ensemble = TemporalEnsemble(
            T_out=10,
            K=config.prediction_stride_K,
            decay_weight=config.ensemble_decay_weight,
        )
        # In robot mode, MoveIt Servo handles velocity/acceleration/collision
        # safety — SafetyChecker only does joint limit clamping.
        self._safety = SafetyChecker(
            max_joint_velocity_rad_s=config.max_joint_velocity_rad_s,
            max_joint_acceleration_rad_s2=config.max_joint_acceleration_rad_s2,
            dt=config.frame_dt,
            joint_limits_only=(self._mode == "robot"),
        )
        self.get_logger().info(
            f"Model loaded. Mode={self._mode}, K={config.prediction_stride_K}, "
            f"lambda={config.ensemble_decay_weight}"
        )

        # --- State ---
        self._latest_skeleton: np.ndarray | None = None  # [48]
        self._latest_joints: np.ndarray | None = None  # [6]
        self._skeleton_stamp: Time | None = None
        self._joints_stamp: Time | None = None
        self._pipeline_active = False
        self._safety_seeded = False
        self._last_target: np.ndarray | None = None  # for feedforward

        # --- TF2 for coordinate transforms ---
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # --- Subscribers ---
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._skeleton_sub = self.create_subscription(
            ObjectsStamped,
            "/zed/zed_node/body_trk/skeletons",
            self._skeleton_cb,
            sensor_qos,
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_states_cb,
            10,
        )

        # --- Publisher ---
        self._joint_state_pub = self.create_publisher(
            JointState, "/vam/joint_states", 10
        )

        # --- MoveIt Servo JointJog publisher (robot mode) ---
        # Publishes joint velocity commands to MoveIt Servo, which handles
        # interpolation (250Hz), collision checking, and streaming to
        # forward_position_controller.
        self._jog_pub = None
        if self._mode == "robot":
            self._jog_pub = self.create_publisher(
                JointJog, "/servo_node/delta_joint_cmds", 10
            )
            self._servo_status_sub = self.create_subscription(
                Int8, "/servo_node/status", self._servo_status_cb, 10
            )
            self.get_logger().info(
                f"MoveIt Servo mode: publishing JointJog to /servo_node/delta_joint_cmds "
                f"(Kp={self._servo_gain}). Ensure MoveIt Servo is running."
            )

        # --- 15 Hz timer ---
        self._timer = self.create_timer(1.0 / 15.0, self._timer_cb)
        self.get_logger().info("VAM inference node started at 15 Hz")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _skeleton_cb(self, msg: ObjectsStamped) -> None:
        """Extract skeleton keypoints from ZED body tracking message."""
        if len(msg.objects) == 0:
            return

        # Select skeleton
        obj = self._select_skeleton(msg.objects)
        if obj is None:
            return

        # Extract 16 keypoints → [48] array
        keypoints = obj.skeleton_3d.keypoints[:16]
        skeleton_48 = np.array(
            [[kp.kp[0], kp.kp[1], kp.kp[2]] for kp in keypoints],
            dtype=np.float32,
        ).flatten()

        # Check for NaN/inf (bad tracking)
        if not np.all(np.isfinite(skeleton_48)):
            return

        # Transform to base_link frame via tf2
        skeleton_transformed = self._transform_skeleton(
            skeleton_48, msg.header.frame_id, msg.header.stamp
        )
        if skeleton_transformed is None:
            return

        self._latest_skeleton = skeleton_transformed
        self._skeleton_stamp = self.get_clock().now()

    def _joint_states_cb(self, msg: JointState) -> None:
        """Store latest joint positions in j0–j5 order."""
        if len(msg.name) == 0:
            return

        # Map from JointState name order to our j0–j5 order
        joints = np.zeros(6, dtype=np.float32)
        for i, target_name in enumerate(UR10_JOINT_NAMES):
            for j, msg_name in enumerate(msg.name):
                if msg_name == target_name:
                    joints[i] = msg.position[j]
                    break

        self._latest_joints = joints
        self._joints_stamp = self.get_clock().now()

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _timer_cb(self) -> None:
        """15 Hz control loop: assemble → predict → ensemble → safety → publish."""
        now = self.get_clock().now()

        # Check data freshness
        if self._skeleton_stamp is None or self._latest_skeleton is None:
            return
        if self._joints_stamp is None or self._latest_joints is None:
            return

        skeleton_age = (now - self._skeleton_stamp).nanoseconds / 1e9
        if skeleton_age > self._tracking_timeout:
            if self._pipeline_active:
                self.get_logger().warn(
                    f"Skeleton tracking lost ({skeleton_age:.1f}s), holding position"
                )
                if self._mode == "robot":
                    self._hold_position()
                self._pipeline_active = False
            return

        self._pipeline_active = True

        # Seed safety checker with actual robot position on first active frame
        if not self._safety_seeded and self._latest_joints is not None:
            self._safety.seed(self._latest_joints)
            self._safety_seeded = True
            self.get_logger().info("SafetyChecker seeded with current robot position")

        # Feed frame to assembler
        self._assembler.add_frame(self._latest_skeleton, self._latest_joints)

        if not self._assembler.is_ready():
            self._ensemble.step()
            return

        # Run model if it's time
        if self._ensemble.should_predict():
            input_tensor = self._assembler.get_input_tensor()
            chunk = self._model.predict(input_tensor)  # [T_out, 6] radians
            self._ensemble.add_prediction(chunk)

        # Query ensemble and apply safety
        if self._ensemble.num_predictions > 0:
            target = self._ensemble.query()
            report = self._safety.check(target)

            if report.warnings:
                for w in report.warnings[:1]:  # log at most 1 per frame
                    self.get_logger().warn(w)

            # Publish predicted joint state (always, for RViz)
            self._publish_joint_state(report.target)

            # Stream to robot controller (every frame for smooth motion)
            if self._mode == "robot":
                self._stream_to_robot(report.target)

        self._ensemble.step()

    # ------------------------------------------------------------------
    # Skeleton helpers
    # ------------------------------------------------------------------

    def _select_skeleton(self, objects) -> object | None:
        """Select which skeleton to track from detected objects."""
        target_id = self._target_skeleton_id

        if target_id >= 0:
            # Track a specific label_id
            for obj in objects:
                if obj.label_id == target_id:
                    return obj
            return None

        # Auto-select: pick the first object (simplest heuristic)
        # In practice with a single performer this is sufficient
        return objects[0]

    def _transform_skeleton(
        self, skeleton_48: np.ndarray, source_frame: str, stamp
    ) -> np.ndarray | None:
        """Transform skeleton keypoints from source_frame to base_link."""
        try:
            transform = self._tf_buffer.lookup_transform(
                "base_link", source_frame, Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            return None

        # Extract rotation matrix and translation from transform
        t = transform.transform.translation
        q = transform.transform.rotation
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)
        rotation = self._quat_to_rotation_matrix(q.x, q.y, q.z, q.w)

        # Reshape to [16, 3], transform, flatten back to [48]
        points = skeleton_48.reshape(16, 3).astype(np.float64)
        transformed = (rotation @ points.T).T + translation
        return transformed.flatten().astype(np.float32)

    @staticmethod
    def _quat_to_rotation_matrix(x, y, z, w) -> np.ndarray:
        """Convert quaternion (xyzw) to 3x3 rotation matrix."""
        n = np.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / n, y / n, z / n, w / n
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_joint_state(self, joint_angles: np.ndarray) -> None:
        """Publish predicted joint positions for RViz."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(UR10_JOINT_NAMES)
        msg.position = joint_angles.tolist()
        self._joint_state_pub.publish(msg)

    def _hold_position(self) -> None:
        """Command the robot to hold its current position.

        Publishes zero-velocity JointJog so MoveIt Servo stops motion.
        Servo also auto-halts after its incoming_command_timeout (0.1s).
        """
        if self._jog_pub is None:
            return
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.joint_names = list(UR10_JOINT_NAMES)
        msg.velocities = [0.0] * 6
        self._jog_pub.publish(msg)

    def _stream_to_robot(self, target: np.ndarray) -> None:
        """Stream joint velocity commands to MoveIt Servo.

        Computes velocity via P-controller + feedforward:
            v = Kp * (target - current) + (target - last_target) / dt

        The feedforward term anticipates target motion so the robot doesn't
        lag behind. MoveIt Servo handles interpolation (250Hz), collision
        checking, and streaming to forward_position_controller.
        """
        if self._jog_pub is None or self._latest_joints is None:
            return

        # P-controller: corrects tracking error
        error = target - self._latest_joints
        velocities = self._servo_gain * error

        # Feedforward: anticipates target motion (eliminates steady-state lag)
        if self._last_target is not None:
            dt = 1.0 / 15.0
            feedforward = (target - self._last_target) / dt
            velocities = velocities + feedforward

        self._last_target = target.copy()

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.joint_names = list(UR10_JOINT_NAMES)
        msg.velocities = velocities.tolist()
        self._jog_pub.publish(msg)

    def _servo_status_cb(self, msg: Int8) -> None:
        """Log MoveIt Servo status warnings."""
        if msg.data != 0:
            self.get_logger().warn(
                f"MoveIt Servo status: {msg.data}", throttle_duration_sec=2.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = VAMInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested, holding position...")
        if node._mode == "robot":
            node._hold_position()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
