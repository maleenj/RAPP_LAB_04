"""VAM Inference Node — DIAGNOSTIC CLONE with CSV logging.

Identical to vam_tm12s_node.py but logs every pipeline stage to CSV:
  raw (post-mapping) → safety → filter → final → actual

Usage:
    ros2 run vam_inference vam_tm12s_node_diag --ros-args \
        -p diag_label:=vel02_acc01

CSV output: ~/csvdata/rapplab04/diagnostics/inference_diag_<label>_<timestamp>.csv
"""

import sys
import time as _time
from pathlib import Path

import numpy as np
import yaml
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
from vam_interfaces.srv import SwitchModel

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

from vam_inference.diag_csv_logger import DiagCSVLogger

SKELETON_TRANSFORM_FRAME = "base"

# CSV column definitions
_JOINT_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5"]

DIAG_COLUMNS = (
    ["timestamp", "wall_time", "tick", "dt_actual"]
    + [f"raw_{j}" for j in _JOINT_NAMES]
    + [f"safety_{j}" for j in _JOINT_NAMES]
    + [f"filtered_{j}" for j in _JOINT_NAMES]
    + [f"final_{j}" for j in _JOINT_NAMES]
    + [f"actual_{j}" for j in _JOINT_NAMES]
    + ["safety_was_constrained", "filter_type",
       "feedback_gain", "feedback_max_vel", "control_rate_hz",
       "max_joint_velocity_rad_s", "max_joint_acceleration_rad_s2"]
)


class VAMTM12SInferenceNodeDiag(Node):
    def __init__(self):
        super().__init__("vam_tm12s_inference_node")

        # --- Declare parameters (identical to original) ---
        self.declare_parameter("mode", "rviz")
        self.declare_parameter("models_config", "")
        self.declare_parameter("active_model", 1)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("prediction_stride_K", 1)
        self.declare_parameter("ensemble_decay_weight", 0.5)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("max_joint_velocity_rad_s", 2.0)
        self.declare_parameter("max_joint_acceleration_rad_s2", 5.0)
        self.declare_parameter("smoothing_alpha", 0.3)
        self.declare_parameter("filter_type", "feedback")
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
        self.declare_parameter(
            "joint_sign_multipliers", TM12S_CONFIG.sign_multipliers
        )
        self.declare_parameter(
            "joint_offsets", TM12S_CONFIG.joint_offsets
        )

        # --- Diagnostic parameters ---
        self.declare_parameter("diag_output_dir", "/data/processed/diagnostics")
        self.declare_parameter("diag_label", "")

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
        self._identity_mapping = (
            np.allclose(self._sign_multipliers, 1.0)
            and np.allclose(self._joint_offsets, 0.0)
        )

        # --- Load model registry from YAML ---
        self._models_registry = {}
        models_config_path = self.get_parameter("models_config").value
        if models_config_path:
            cfg_path = Path(models_config_path)
            if cfg_path.exists():
                with open(cfg_path) as f:
                    registry = yaml.safe_load(f)
                self._models_registry = {
                    int(k): v for k, v in registry.get("models", {}).items()
                }
                self.get_logger().info(
                    f"Loaded model registry: {len(self._models_registry)} models "
                    f"from {cfg_path}"
                )
            else:
                self.get_logger().warn(
                    f"Models config not found: {cfg_path}"
                )

        self._active_model_id = int(self.get_parameter("active_model").value)
        self._switching = False

        checkpoint_path, model_config_path, norm_stats_path = (
            self._resolve_model_paths(self._active_model_id)
        )

        config = self._build_inference_config(
            checkpoint_path, model_config_path, norm_stats_path
        )

        self._load_pipeline(config)

        self._filter_type = self.get_parameter("filter_type").value
        use_joint_limits_only = False
        self._safety = TM12SSafetyChecker(
            max_joint_velocity_rad_s=config.max_joint_velocity_rad_s,
            max_joint_acceleration_rad_s2=config.max_joint_acceleration_rad_s2,
            dt=config.frame_dt,
            joint_limits_only=use_joint_limits_only,
        )

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

        self._deadzone_rad = self.get_parameter("deadzone_rad").value
        self._still_frames_thresh = self.get_parameter("still_frames_threshold").value
        self._stable_target: np.ndarray | None = None
        self._still_count = 0

        self._latest_skeleton: np.ndarray | None = None
        self._latest_joints: np.ndarray | None = None
        self._skeleton_stamp: Time | None = None
        self._joints_stamp: Time | None = None
        self._pipeline_active = False
        self._safety_seeded = False
        self._frame_count = 0

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

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

        self._joint_state_pub = self.create_publisher(
            JointState, "/vam/joint_states", 10
        )
        self._joint_target_pub = None
        if self._mode == "robot":
            self._joint_target_pub = self.create_publisher(
                Float64MultiArray, "/vam/joint_targets", 10
            )
            self.get_logger().info(
                "Robot mode: publishing targets to /vam/joint_targets "
                "for vam_pvt_streamer."
            )

        self._switch_srv = self.create_service(
            SwitchModel, "/vam/switch_model", self._switch_model_cb
        )

        # --- Diagnostic CSV logger ---
        self._diag_logger = DiagCSVLogger(
            prefix="inference_diag",
            columns=DIAG_COLUMNS,
            output_dir=self.get_parameter("diag_output_dir").value,
            label=self.get_parameter("diag_label").value,
        )
        self._diag_last_tick_time = _time.monotonic()
        self.get_logger().info(
            f"[DIAG] CSV logging to: {self._diag_logger.path}"
        )

        self._timer = self.create_timer(1.0 / self._control_rate, self._timer_cb)
        self.get_logger().info(f"VAM TM12S inference node (DIAG) started at {self._control_rate:.0f} Hz")

    # ------------------------------------------------------------------
    # Model registry helpers (identical to original)
    # ------------------------------------------------------------------

    def _resolve_model_paths(self, model_id: int) -> tuple[str, str, str]:
        if model_id in self._models_registry:
            entry = self._models_registry[model_id]
            model_dir = entry["model_dir"]
            checkpoint_path = f"{model_dir}/best.pt"
            model_config_path = f"{model_dir}/model_config.json"
            norm_stats_path = entry["norm_stats_path"]
            name = entry.get("name", "unnamed")
            desc = entry.get("description", "")
            self.get_logger().info(
                f'=== Active model: #{model_id} "{name}" - {desc} ==='
            )
            self.get_logger().info(f"  Model dir: {model_dir}")
            self.get_logger().info(f"  Norm stats: {norm_stats_path}")
            return checkpoint_path, model_config_path, norm_stats_path

        self.get_logger().warn(
            f"Model #{model_id} not found in registry "
            f"(available: {list(self._models_registry.keys())}). "
            f"Using first available model."
        )
        if self._models_registry:
            first_id = min(self._models_registry.keys())
            return self._resolve_model_paths(first_id)

        self.get_logger().warn("No model registry loaded, using hardcoded defaults.")
        return (
            "/data/models/vam_skelonly_tm12_20260404_0631/best.pt",
            "/data/models/vam_skelonly_tm12_20260404_0631/model_config.json",
            "/data/processed/tensors/2026_04_04_tm12/norm_stats.pt",
        )

    def _build_inference_config(
        self, checkpoint_path: str, model_config_path: str, norm_stats_path: str
    ) -> InferenceConfig:
        return InferenceConfig(
            checkpoint_path=Path(checkpoint_path),
            model_config_path=Path(model_config_path),
            norm_stats_path=Path(norm_stats_path),
            device=self.get_parameter("device").value,
            prediction_stride_K=self.get_parameter("prediction_stride_K").value,
            ensemble_decay_weight=self.get_parameter(
                "ensemble_decay_weight"
            ).value,
            max_joint_velocity_rad_s=self.get_parameter(
                "max_joint_velocity_rad_s"
            ).value,
            max_joint_acceleration_rad_s2=self.get_parameter(
                "max_joint_acceleration_rad_s2"
            ).value,
            control_rate_hz=self.get_parameter("control_rate_hz").value,
        )

    def _load_pipeline(self, config: InferenceConfig) -> None:
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
        model_type_str = "skeleton_only" if self._skeleton_only else "skeleton+joints"
        self.get_logger().info(
            f"Model loaded ({model_type_str}). Mode={self._mode}, "
            f"K={config.prediction_stride_K}, "
            f"lambda={config.ensemble_decay_weight}"
        )

    # ------------------------------------------------------------------
    # Model switch service (identical to original)
    # ------------------------------------------------------------------

    def _switch_model_cb(self, request, response):
        model_id = request.model_id

        if model_id not in self._models_registry:
            available = list(self._models_registry.keys())
            response.success = False
            response.message = (
                f"Model #{model_id} not found. Available: {available}"
            )
            response.active_model_name = self._models_registry.get(
                self._active_model_id, {}
            ).get("name", "unknown")
            self.get_logger().warn(response.message)
            return response

        if model_id == self._active_model_id:
            entry = self._models_registry[model_id]
            response.success = True
            response.message = (
                f'Model #{model_id} "{entry["name"]}" is already active.'
            )
            response.active_model_name = entry["name"]
            self.get_logger().info(response.message)
            return response

        old_entry = self._models_registry.get(self._active_model_id, {})
        new_entry = self._models_registry[model_id]
        self.get_logger().info(
            f'Model switch requested: #{self._active_model_id} '
            f'"{old_entry.get("name", "?")}" -> #{model_id} '
            f'"{new_entry["name"]}"'
        )
        self.get_logger().info("Pausing inference for model swap...")

        self._switching = True

        try:
            checkpoint_path, model_config_path, norm_stats_path = (
                self._resolve_model_paths(model_id)
            )
            config = self._build_inference_config(
                checkpoint_path, model_config_path, norm_stats_path
            )
            self._load_pipeline(config)

            self._ema_target = None
            self._one_euro = OneEuroFilter(
                n_signals=6,
                rate=15.0,
                min_cutoff=self.get_parameter("one_euro_min_cutoff").value,
                beta=self.get_parameter("one_euro_beta").value,
                d_cutoff=self.get_parameter("one_euro_d_cutoff").value,
            )
            self._stable_target = None
            self._still_count = 0

            self._safety_seeded = False
            if self._latest_joints is not None:
                self._safety.seed(self._latest_joints)
                self._safety_seeded = True

            self._active_model_id = model_id
            self._frame_count = 0

            response.success = True
            response.message = (
                f'Switched to model #{model_id} "{new_entry["name"]}".'
            )
            response.active_model_name = new_entry["name"]
            self.get_logger().info(
                f"Model swap complete, resuming inference."
            )
        except Exception as e:
            response.success = False
            response.message = f"Model swap failed: {e}"
            response.active_model_name = old_entry.get("name", "unknown")
            self.get_logger().error(f"Model swap failed: {e}")
        finally:
            self._switching = False

        return response

    # ------------------------------------------------------------------
    # Subscriber callbacks (identical to original)
    # ------------------------------------------------------------------

    def _skeleton_cb(self, msg: ObjectsStamped) -> None:
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

        skeleton_transformed = self._transform_skeleton(
            skeleton_54, msg.header.frame_id, msg.header.stamp
        )
        if skeleton_transformed is None:
            return

        self._latest_skeleton = skeleton_transformed
        self._skeleton_stamp = self.get_clock().now()

    def _joint_states_cb(self, msg: JointState) -> None:
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
    # Main control loop (with CSV logging)
    # ------------------------------------------------------------------

    def _timer_cb(self) -> None:
        if self._switching:
            return

        now = self.get_clock().now()
        wall_now = _time.monotonic()
        dt_actual = wall_now - self._diag_last_tick_time
        self._diag_last_tick_time = wall_now

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

        if self._mode == "robot":
            if not self._safety_seeded and self._latest_joints is not None:
                self._safety.seed(self._latest_joints)
                self._safety_seeded = True
                self.get_logger().info("SafetyChecker seeded with current TM12S position")

        if self._skeleton_only:
            self._assembler.add_frame(self._latest_skeleton)
        else:
            self._assembler.add_frame(self._latest_skeleton, self._latest_joints)

        if not self._assembler.is_ready():
            self._ensemble.step()
            return

        if self._ensemble.should_predict():
            input_tensor = self._assembler.get_input_tensor()
            chunk = self._model.predict(input_tensor)
            self._ensemble.add_prediction(chunk)

        if self._ensemble.num_predictions > 0:
            target = self._ensemble.query()

            # Map UR10 joint angles -> TM12S
            target = self._sign_multipliers * target + self._joint_offsets
            if not self._identity_mapping:
                target = np.arctan2(np.sin(target), np.cos(target))

            raw_target = target.copy()  # [DIAG] post-mapping, pre-safety

            # Safety limits
            report = self._safety.check(target)
            safety_constrained = bool(report.warnings)
            if report.warnings:
                for w in report.warnings[:1]:
                    self.get_logger().warn(w)
            target = report.target

            post_safety = target.copy()  # [DIAG]

            # Smoothing filter
            if self._filter_type == "feedback" and self._latest_joints is not None:
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

            post_filter = target.copy()  # [DIAG]

            # Deadzone
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

            final_target = target.copy()  # [DIAG]

            # --- Diagnostic logging (every tick) ---
            self._frame_count += 1
            ros_ts = now.nanoseconds / 1e9
            actual = self._latest_joints if self._latest_joints is not None else np.zeros(6)

            self._diag_logger.log(
                [ros_ts, wall_now, self._frame_count, f"{dt_actual:.6f}"]
                + [f"{v:.6f}" for v in raw_target]
                + [f"{v:.6f}" for v in post_safety]
                + [f"{v:.6f}" for v in post_filter]
                + [f"{v:.6f}" for v in final_target]
                + [f"{v:.6f}" for v in actual]
                + [int(safety_constrained), self._filter_type,
                   self._feedback_gain, self._feedback_max_vel,
                   self._control_rate,
                   self.get_parameter("max_joint_velocity_rad_s").value,
                   self.get_parameter("max_joint_acceleration_rad_s2").value]
            )

            # Text logging (sampled, same as original)
            if self._latest_joints is not None and (
                self._frame_count <= 5 or self._frame_count % 30 == 0
            ):
                raw_err = np.degrees(np.max(np.abs(raw_target - actual)))
                safety_err = np.degrees(np.max(np.abs(post_safety - raw_target)))
                filter_err = np.degrees(np.max(np.abs(post_filter - post_safety)))
                final_err = np.degrees(np.max(np.abs(target - actual)))
                per_joint = np.degrees(target - actual)
                jstr = " ".join(f"{e:+.1f}" for e in per_joint)
                self.get_logger().info(
                    f"F#{self._frame_count}: "
                    f"raw->act={raw_err:.1f} deg "
                    f"safety_clip={safety_err:.1f} deg "
                    f"filter_lag={filter_err:.1f} deg "
                    f"final->act={final_err:.1f} deg "
                    f"[{jstr}]"
                )

            self._publish_joint_state(target)

            if self._mode == "robot":
                self._publish_joint_target(target)

        self._ensemble.step()

    # ------------------------------------------------------------------
    # Skeleton helpers (identical to original)
    # ------------------------------------------------------------------

    def _select_skeleton(self, objects) -> object | None:
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
                f"TF lookup failed: '{SKELETON_TRANSFORM_FRAME}' <- '{source_frame}': {e}",
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
    # Publishing (identical to original)
    # ------------------------------------------------------------------

    def _publish_joint_state(self, joint_angles: np.ndarray) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(TM12S_JOINT_NAMES)
        msg.position = joint_angles.tolist()
        self._joint_state_pub.publish(msg)

    def _publish_joint_target(self, joint_angles: np.ndarray) -> None:
        if self._joint_target_pub is None:
            return
        msg = Float64MultiArray()
        msg.data = joint_angles.tolist()
        self._joint_target_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VAMTM12SInferenceNodeDiag()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested")
    finally:
        node._diag_logger.close()
        node.get_logger().info(
            f"[DIAG] CSV saved: {node._diag_logger.path} "
            f"({node._frame_count} frames)"
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
