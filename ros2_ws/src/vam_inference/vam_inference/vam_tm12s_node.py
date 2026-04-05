"""VAM Inference Node — real-time skeleton-to-joint prediction for TM12S.

Subscribes to ZED skeleton tracking and TM12S joint states, runs the trained
Action Chunking Transformer through the temporal ensemble pipeline, applies
optional sign convention mapping, and publishes smooth joint commands.

Model output is mapped to TM12S joint space via per-joint sign multipliers
and angular offsets (configurable via parameters or TM12S_CONFIG):
    tm12s_angle = sign * model_angle + offset

When trained on native TM12S data, use identity mapping (signs=1, offsets=0).
When trained on UR10 data, use the UR10→TM12S mapping from robot_configs.py.

After mapping, angles are normalized to [-π, π] via arctan2(sin, cos) so
they match the range reported by /joint_states (TM12S encoder values).

Modes:
    rviz  — publish JointState on /vam/joint_states for RViz visualization
    robot — publish Float64MultiArray on /vam/joint_targets for PVT streamer

Robot mode requires:
    vam_pvt_streamer node running (handles PVT streaming + collision checking)
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
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from zed_msgs.msg import ObjectsStamped

import tf2_ros

# Ensure vam_utils is importable (mounted at /workspace in container)
_workspace = Path("/workspace")
if str(_workspace) not in sys.path:
    sys.path.insert(0, str(_workspace))

from vam_utils.config import InferenceConfig
from vam_utils.data.robot_configs import TM12S_JOINT_NAMES, TM12S_JOINT_LIMITS, TM12S_CONFIG
from vam_utils.inference import (
    InputAssembler,
    VAMModelWrapper,
    TemporalEnsemble,
)
from vam_utils.inference.tm12s_safety_checker import TM12SSafetyChecker
from vam_utils.inference.one_euro_filter import OneEuroFilter

# Frame for skeleton TF transform — the model was trained with skeletons in
# the UR10 "base_link" frame. On TM12S the equivalent frame is "base" (there
# is no "base_link" in the TM12S URDF). Both frames sit at the robot's
# mounting surface with Z-up, so the skeleton orientation seen by the model
# is identical.
SKELETON_TRANSFORM_FRAME = "base"


class VAMTM12SInferenceNode(Node):
    def __init__(self):
        super().__init__("vam_tm12s_inference_node")

        # --- Declare parameters ---
        self.declare_parameter("mode", "rviz")
        self.declare_parameter(
            "checkpoint_path", "/data/models/vam_skelonly_tm12_20260404_0631/best.pt"
        )
        self.declare_parameter(
            "model_config_path", "/data/models/vam_skelonly_tm12_20260404_0631/model_config.json"
        )
        self.declare_parameter(
            "norm_stats_path",
            "/data/processed/tensors/2026_04_04_tm12/norm_stats.pt",
        )
        self.declare_parameter("device", "cuda")
        self.declare_parameter("prediction_stride_K", 1)
        self.declare_parameter("ensemble_decay_weight", 0.5)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("max_joint_velocity_rad_s", 2.0)
        self.declare_parameter("max_joint_acceleration_rad_s2", 5.0)
        self.declare_parameter("smoothing_alpha", 0.3)
        self.declare_parameter("filter_type", "feedback")  # "feedback", "one_euro", or "ema"
        self.declare_parameter("feedback_gain", 3.0)
        self.declare_parameter("feedback_max_vel", 1.4)
        self.declare_parameter("one_euro_min_cutoff", 0.4)
        self.declare_parameter("one_euro_beta", 0.1)
        self.declare_parameter("one_euro_d_cutoff", 1.0)
        self.declare_parameter("deadzone_rad", 0.12)
        self.declare_parameter("still_frames_threshold", 2)
        self.declare_parameter("target_skeleton_id", -1)
        self.declare_parameter("tracking_timeout_sec", 0.5)
        self.declare_parameter("trajectory_lookahead_frames", 5)
        # Per-joint sign multipliers and offsets for UR10 → TM12S mapping.
        # Defaults from TM12S_CONFIG (derived from URDF comparison).
        self.declare_parameter(
            "joint_sign_multipliers", TM12S_CONFIG.sign_multipliers
        )
        self.declare_parameter(
            "joint_offsets", TM12S_CONFIG.joint_offsets
        )

        # --- Read parameters ---
        self._mode = self.get_parameter("mode").value
        self._target_skeleton_id = self.get_parameter("target_skeleton_id").value
        self._tracking_timeout = self.get_parameter("tracking_timeout_sec").value
        self._lookahead = self.get_parameter("trajectory_lookahead_frames").value
        self._sign_multipliers = np.array(
            self.get_parameter("joint_sign_multipliers").value, dtype=np.float32
        )
        self._joint_offsets = np.array(
            self.get_parameter("joint_offsets").value, dtype=np.float32
        )
        # When using identity mapping (native TM12S training data), arctan2
        # wrapping must be skipped — the training data has joints outside
        # [-pi, pi] (e.g. j4/wrist_2 ranges up to 6.28 rad).
        self._identity_mapping = (
            np.allclose(self._sign_multipliers, 1.0)
            and np.allclose(self._joint_offsets, 0.0)
        )

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
            control_rate_hz=self.get_parameter("control_rate_hz").value,
        )

        # --- Load model and create pipeline components ---
        self.get_logger().info("Loading VAM model for TM12S...")
        self._model = VAMModelWrapper(config, joint_limits=TM12S_JOINT_LIMITS)
        self._skeleton_only = self._model.skeleton_only
        self._assembler = InputAssembler(
            norm_stats=self._model.norm_stats,
            T_in=10,
            skeleton_only=self._skeleton_only,
        )
        self._ensemble = TemporalEnsemble(
            T_out=10,
            K=config.prediction_stride_K,
            decay_weight=config.ensemble_decay_weight,
        )
        # In feedback mode: joint_limits_only=True because the feedback
        # smoother already rate-limits targets (max_vel parameter). Adding
        # safety checker velocity/accel limiting on top causes stop-start
        # jerkiness from the two limiters fighting each other.
        # In other modes: full safety checking needed to prevent CPERR 241.
        self._filter_type = self.get_parameter("filter_type").value
        use_joint_limits_only = False
        self._safety = TM12SSafetyChecker(
            max_joint_velocity_rad_s=config.max_joint_velocity_rad_s,
            max_joint_acceleration_rad_s2=config.max_joint_acceleration_rad_s2,
            dt=config.frame_dt,
            joint_limits_only=use_joint_limits_only,
        )
        model_type_str = "skeleton_only" if self._skeleton_only else "skeleton+joints"
        self.get_logger().info(
            f"Model loaded ({model_type_str}). Mode={self._mode}, "
            f"K={config.prediction_stride_K}, lambda={config.ensemble_decay_weight}, "
            f"sign_multipliers={self._sign_multipliers.tolist()}, "
            f"joint_offsets={[round(o, 3) for o in self._joint_offsets.tolist()]}"
        )

        # --- Smoothing filter ---
        self._smoothing_alpha = self.get_parameter("smoothing_alpha").value
        self._ema_target: np.ndarray | None = None
        self._feedback_gain = self.get_parameter("feedback_gain").value
        self._feedback_max_vel = self.get_parameter("feedback_max_vel").value
        self._control_rate = self.get_parameter("control_rate_hz").value
        self._feedback_dt = 1.0 / self._control_rate
        self._one_euro = OneEuroFilter(
            n_signals=6,
            rate=15.0,
            min_cutoff=self.get_parameter("one_euro_min_cutoff").value,
            beta=self.get_parameter("one_euro_beta").value,
            d_cutoff=self.get_parameter("one_euro_d_cutoff").value,
        )
        if self._filter_type == "feedback":
            self.get_logger().info(
                f"Filter: feedback (Kp={self._feedback_gain}, "
                f"max_vel={self._feedback_max_vel} rad/s)"
            )
        elif self._filter_type == "one_euro":
            self.get_logger().info(
                f"Filter: one_euro (min_cutoff={self._one_euro.min_cutoff}, "
                f"beta={self._one_euro.beta})"
            )
        else:
            self.get_logger().info(
                f"Filter: ema (alpha={self._smoothing_alpha})"
            )

        # --- Stillness / deadzone detection ---
        self._deadzone_rad = self.get_parameter("deadzone_rad").value
        self._still_frames_thresh = self.get_parameter("still_frames_threshold").value
        self._stable_target: np.ndarray | None = None
        self._still_count = 0

        # --- State ---
        self._latest_skeleton: np.ndarray | None = None  # [54]
        self._latest_joints: np.ndarray | None = None  # [6]
        self._skeleton_stamp: Time | None = None
        self._joints_stamp: Time | None = None
        self._pipeline_active = False
        self._safety_seeded = False
        self._frame_count = 0

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

        # --- Publishers ---
        self._joint_state_pub = self.create_publisher(
            JointState, "/vam/joint_states", 10
        )
        # Robot mode: publish normalized targets for PVT streamer
        self._joint_target_pub = None
        if self._mode == "robot":
            self._joint_target_pub = self.create_publisher(
                Float64MultiArray, "/vam/joint_targets", 10
            )
            self.get_logger().info(
                "Robot mode: publishing targets to /vam/joint_targets "
                "for vam_pvt_streamer."
            )

        # --- 15 Hz timer ---
        self._timer = self.create_timer(1.0 / self._control_rate, self._timer_cb)
        self.get_logger().info(f"VAM TM12S inference node started at {self._control_rate:.0f} Hz")

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _skeleton_cb(self, msg: ObjectsStamped) -> None:
        """Extract skeleton keypoints from ZED body tracking message."""
        if len(msg.objects) == 0:
            self.get_logger().debug(
                "Skeleton msg received but objects list is empty",
                throttle_duration_sec=2.0,
            )
            return

        ids = [obj.label_id for obj in msg.objects]
        self.get_logger().info(
            f"Skeleton msg: frame_id='{msg.header.frame_id}', "
            f"{len(msg.objects)} bodies, label_ids={ids}",
            throttle_duration_sec=5.0,
        )

        obj = self._select_skeleton(msg.objects)
        if obj is None:
            self.get_logger().warn(
                f"target_skeleton_id={self._target_skeleton_id} not found "
                f"in detected ids={ids}",
                throttle_duration_sec=2.0,
            )
            return

        # Extract 18 keypoints (ZED BODY_18 format) → [54] array
        keypoints = obj.skeleton_3d.keypoints[:18]
        skeleton_54 = np.array(
            [[kp.kp[0], kp.kp[1], kp.kp[2]] for kp in keypoints],
            dtype=np.float32,
        ).flatten()

        if not np.all(np.isfinite(skeleton_54)):
            nan_count = np.sum(~np.isfinite(skeleton_54))
            self.get_logger().warn(
                f"Skeleton has {nan_count}/54 non-finite values, dropping frame",
                throttle_duration_sec=2.0,
            )
            return

        # Transform to TM12S planning frame via tf2
        skeleton_transformed = self._transform_skeleton(
            skeleton_54, msg.header.frame_id, msg.header.stamp
        )
        if skeleton_transformed is None:
            return

        self._latest_skeleton = skeleton_transformed
        self._skeleton_stamp = self.get_clock().now()

    def _joint_states_cb(self, msg: JointState) -> None:
        """Store latest joint positions in joint_1–joint_6 order."""
        if len(msg.name) == 0:
            return

        joints = np.zeros(6, dtype=np.float32)
        for i, target_name in enumerate(TM12S_JOINT_NAMES):
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
        """15 Hz control loop: assemble → predict → sign map → safety → publish."""
        now = self.get_clock().now()

        # Check data freshness
        if self._skeleton_stamp is None or self._latest_skeleton is None:
            self.get_logger().warn(
                "Waiting for skeleton data — no valid skeleton received yet",
                throttle_duration_sec=5.0,
            )
            return
        if not self._skeleton_only or self._mode == "robot":
            if self._joints_stamp is None or self._latest_joints is None:
                self.get_logger().warn(
                    "Waiting for joint state data",
                    throttle_duration_sec=5.0,
                )
                return

        skeleton_age = (now - self._skeleton_stamp).nanoseconds / 1e9
        if skeleton_age > self._tracking_timeout:
            if self._pipeline_active:
                self.get_logger().warn(
                    f"Skeleton tracking lost ({skeleton_age:.1f}s)"
                )
                self._pipeline_active = False
            return

        self._pipeline_active = True

        # Seed safety checker with actual robot position on first active frame
        # (robot mode only — rviz mode passes raw model output through)
        if self._mode == "robot":
            if not self._safety_seeded and self._latest_joints is not None:
                self._safety.seed(self._latest_joints)
                self._safety_seeded = True
                self.get_logger().info("SafetyChecker seeded with current TM12S position")

        # Feed frame to assembler
        if self._skeleton_only:
            self._assembler.add_frame(self._latest_skeleton)
        else:
            self._assembler.add_frame(self._latest_skeleton, self._latest_joints)

        if not self._assembler.is_ready():
            self._ensemble.step()
            return

        # Run model if it's time
        if self._ensemble.should_predict():
            input_tensor = self._assembler.get_input_tensor()
            chunk = self._model.predict(input_tensor)  # [T_out, 6] radians (UR10 space)
            self._ensemble.add_prediction(chunk)

        # Query ensemble and apply sign mapping
        if self._ensemble.num_predictions > 0:
            target = self._ensemble.query()

            # Map UR10 joint angles → TM12S: sign flip + angular offset
            target = self._sign_multipliers * target + self._joint_offsets

            # Normalize to [-π, π] only when sign/offset mapping is active.
            # With identity mapping (native TM12S data), wrapping is
            # destructive: j4/wrist_2 trains at 0–6.28 rad and wrapping
            # snaps predictions above π to negative values (-360°).
            if not self._identity_mapping:
                target = np.arctan2(np.sin(target), np.cos(target))

            raw_target = target.copy()  # before any filtering

            # Apply safety limits (joint limits + velocity/accel clamping)
            report = self._safety.check(target)
            if report.warnings:
                for w in report.warnings[:1]:
                    self.get_logger().warn(w)
            target = report.target

            post_safety = target.copy()

            # --- Smoothing filter ---
            if self._filter_type == "feedback" and self._latest_joints is not None:
                # Feedback-based smoother: P-controller from actual joints.
                # Output is always within max_vel*dt of actual — bounded, no drift.
                # The PVT Butterworth filter smooths any stepping.
                error = target - self._latest_joints
                velocity = self._feedback_gain * error
                velocity = np.clip(velocity, -self._feedback_max_vel,
                                   self._feedback_max_vel)
                target = self._latest_joints + velocity * self._feedback_dt
            elif self._filter_type == "one_euro":
                target = self._one_euro(target)
            else:
                if self._ema_target is None:
                    self._ema_target = target.copy()
                else:
                    self._ema_target = (self._smoothing_alpha * target
                                        + (1.0 - self._smoothing_alpha)
                                        * self._ema_target)
                target = self._ema_target

            post_filter = target.copy()

            # Deadzone (skipped for feedback filter — feedback loop handles it)
            if self._filter_type != "feedback":
                if self._stable_target is None:
                    self._stable_target = target.copy()
                    self._still_count = 0
                delta = np.max(np.abs(target - self._stable_target))
                if delta < self._deadzone_rad:
                    self._still_count += 1
                    if self._still_count >= self._still_frames_thresh:
                        target = self._stable_target
                else:
                    self._stable_target = target.copy()
                    self._still_count = 0

            # --- Diagnostic logging ---
            self._frame_count += 1
            if self._latest_joints is not None and (
                self._frame_count <= 5 or self._frame_count % 30 == 0
            ):
                actual = self._latest_joints
                # Max error at each stage (degrees)
                raw_err = np.degrees(np.max(np.abs(raw_target - actual)))
                safety_err = np.degrees(np.max(np.abs(post_safety - raw_target)))
                filter_err = np.degrees(np.max(np.abs(post_filter - post_safety)))
                final_err = np.degrees(np.max(np.abs(target - actual)))
                # Per-joint final error
                per_joint = np.degrees(target - actual)
                jstr = " ".join(f"{e:+.1f}" for e in per_joint)
                self.get_logger().info(
                    f"F#{self._frame_count}: "
                    f"raw→act={raw_err:.1f}° "
                    f"safety_clip={safety_err:.1f}° "
                    f"filter_lag={filter_err:.1f}° "
                    f"final→act={final_err:.1f}° "
                    f"[{jstr}]"
                )

            # Publish predicted joint state (always, for RViz)
            self._publish_joint_state(target)

            # Publish target for PVT streamer (robot mode)
            if self._mode == "robot":
                self._publish_joint_target(target)

        self._ensemble.step()

    # ------------------------------------------------------------------
    # Skeleton helpers
    # ------------------------------------------------------------------

    def _select_skeleton(self, objects) -> object | None:
        """Select which skeleton to track from detected objects."""
        target_id = self._target_skeleton_id

        if target_id >= 0:
            for obj in objects:
                if obj.label_id == target_id:
                    return obj
            return None

        return objects[0]

    def _transform_skeleton(
        self, skeleton_54: np.ndarray, source_frame: str, stamp
    ) -> np.ndarray | None:
        """Transform skeleton keypoints from source_frame to model training frame."""
        try:
            transform = self._tf_buffer.lookup_transform(
                SKELETON_TRANSFORM_FRAME, source_frame, Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warn(
                f"TF lookup failed: '{SKELETON_TRANSFORM_FRAME}' ← '{source_frame}': {e}",
                throttle_duration_sec=2.0,
            )
            return None

        t = transform.transform.translation
        q = transform.transform.rotation
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)
        rotation = self._quat_to_rotation_matrix(q.x, q.y, q.z, q.w)

        points = skeleton_54.reshape(18, 3).astype(np.float64)
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
        msg.name = list(TM12S_JOINT_NAMES)
        msg.position = joint_angles.tolist()
        self._joint_state_pub.publish(msg)

    def _publish_joint_target(self, joint_angles: np.ndarray) -> None:
        """Publish normalized target positions for PVT streamer."""
        if self._joint_target_pub is None:
            return
        msg = Float64MultiArray()
        msg.data = joint_angles.tolist()
        self._joint_target_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VAMTM12SInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
