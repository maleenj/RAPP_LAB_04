"""Safety checker for TM12S: validates and constrains joint targets.

Standalone implementation using TM12S joint limits and per-joint velocity
limits. Same interface as SafetyChecker (check/seed/reset) so it can be
used as a drop-in replacement in the TM12S inference node.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from vam_utils.data.robot_configs import TM12S_JOINT_LIMITS, TM12S_MAX_JOINT_VELOCITIES


@dataclass
class SafetyReport:
    """Result of a safety check on a single joint target."""
    target: np.ndarray
    warnings: List[str] = field(default_factory=list)
    was_constrained: bool = False


class TM12SSafetyChecker:
    """Validates and constrains joint targets for TM12S robot.

    Applies:
    1. Joint limit clamping (from TM12S URDF)
    2. Per-joint velocity limiting (TM12S has different max vel per joint)
    3. Acceleration limiting

    In joint_limits_only mode (when MoveIt Servo handles dynamic safety),
    only joint limit clamping is applied.
    """

    def __init__(
        self,
        max_joint_velocity_rad_s: float = 1.0,
        max_joint_acceleration_rad_s2: float = 5.0,
        dt: float = 1.0 / 15.0,
        joint_limits_only: bool = False,
    ):
        self.joint_limits = np.array(TM12S_JOINT_LIMITS, dtype=np.float32)  # [6, 2]
        hw_limits = np.array(TM12S_MAX_JOINT_VELOCITIES, dtype=np.float32)  # [6]
        # Use the more restrictive of operational limit and hardware limit per joint
        self.max_velocities = np.minimum(hw_limits, max_joint_velocity_rad_s)
        self.max_acc = max_joint_acceleration_rad_s2
        self.dt = dt
        self.joint_limits_only = joint_limits_only
        self._last_position: Optional[np.ndarray] = None
        self._last_velocity: Optional[np.ndarray] = None

    def check(self, target: np.ndarray) -> SafetyReport:
        """Validate and constrain a joint target."""
        target = target.copy().astype(np.float32)
        warnings: List[str] = []
        was_constrained = False

        # 1. Joint limits
        clamped = np.clip(target, self.joint_limits[:, 0], self.joint_limits[:, 1])
        if not np.allclose(target, clamped):
            violations = np.where(target != clamped)[0]
            warnings.append(f"Joint limit clamp on joints {violations.tolist()}")
            target = clamped
            was_constrained = True

        if self.joint_limits_only:
            self._last_position = target.copy()
            return SafetyReport(
                target=target, warnings=warnings, was_constrained=was_constrained
            )

        # 2. Per-joint velocity limiting
        if self._last_position is not None:
            delta = target - self._last_position
            velocity = delta / self.dt
            abs_vel = np.abs(velocity)

            over_limit = abs_vel > self.max_velocities
            if np.any(over_limit):
                # Scale each violating joint individually
                scale = np.where(
                    over_limit,
                    self.max_velocities / np.maximum(abs_vel, 1e-8),
                    1.0,
                )
                # Use the most restrictive scale to maintain trajectory shape
                min_scale = np.min(scale)
                target = self._last_position + delta * min_scale
                warnings.append(
                    f"Velocity limited: max {np.max(abs_vel):.2f} rad/s "
                    f"(limits {self.max_velocities.tolist()})"
                )
                was_constrained = True

            # 3. Acceleration limiting
            new_velocity = (target - self._last_position) / self.dt
            if self._last_velocity is not None:
                acceleration = (new_velocity - self._last_velocity) / self.dt
                max_a = np.max(np.abs(acceleration))

                if max_a > self.max_acc:
                    scale_a = self.max_acc / max_a
                    constrained_vel = (
                        self._last_velocity
                        + (new_velocity - self._last_velocity) * scale_a
                    )
                    target = self._last_position + constrained_vel * self.dt
                    new_velocity = constrained_vel
                    warnings.append(
                        f"Acceleration limited: {max_a:.2f} > {self.max_acc:.2f} rad/s^2"
                    )
                    was_constrained = True

            self._last_velocity = new_velocity

        self._last_position = target.copy()

        return SafetyReport(
            target=target, warnings=warnings, was_constrained=was_constrained
        )

    def seed(self, position: np.ndarray) -> None:
        """Initialize with the robot's current position."""
        self._last_position = position.copy().astype(np.float32)
        self._last_velocity = np.zeros(6, dtype=np.float32)

    def reset(self) -> None:
        """Clear state."""
        self._last_position = None
        self._last_velocity = None
