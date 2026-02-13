"""Safety checker: validates and constrains joint targets for safe execution.

Enforces joint limits, velocity bounds, and acceleration bounds. Returns
constrained targets along with any warnings generated.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from vam_utils.data.episode import UR10_JOINT_LIMITS


@dataclass
class SafetyReport:
    """Result of a safety check on a single joint target."""
    target: np.ndarray          # [6] constrained joint angles (radians)
    warnings: List[str] = field(default_factory=list)
    was_constrained: bool = False


class SafetyChecker:
    """Validates and constrains joint targets for safe robot execution.

    Applies three layers of protection:
    1. Joint limit clamping (hard limits from URDF)
    2. Velocity limiting (max rad/s per joint)
    3. Acceleration limiting (max rad/s^2 per joint)

    Usage:
        checker = SafetyChecker(max_vel=1.0, max_acc=5.0, dt=1/15)

        for target in predictions:
            report = checker.check(target)
            safe_target = report.target
            if report.warnings:
                print(report.warnings)
    """

    def __init__(
        self,
        max_joint_velocity_rad_s: float = 1.0,
        max_joint_acceleration_rad_s2: float = 5.0,
        dt: float = 1.0 / 15.0,
    ):
        """
        Args:
            max_joint_velocity_rad_s: per-joint velocity limit.
            max_joint_acceleration_rad_s2: per-joint acceleration limit.
            dt: time between control steps (1/control_rate_hz).
        """
        self.joint_limits = np.array(UR10_JOINT_LIMITS, dtype=np.float32)  # [6, 2]
        self.max_vel = max_joint_velocity_rad_s
        self.max_acc = max_joint_acceleration_rad_s2
        self.dt = dt
        self._last_position: Optional[np.ndarray] = None
        self._last_velocity: Optional[np.ndarray] = None

    def check(self, target: np.ndarray) -> SafetyReport:
        """Validate and constrain a joint target.

        Args:
            target: [6] joint angles in radians (from temporal ensemble).

        Returns:
            SafetyReport with constrained target and any warnings.
        """
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

        # 2. Velocity limiting
        if self._last_position is not None:
            delta = target - self._last_position
            velocity = delta / self.dt
            max_v = np.max(np.abs(velocity))

            if max_v > self.max_vel:
                scale = self.max_vel / max_v
                target = self._last_position + delta * scale
                warnings.append(
                    f"Velocity limited: {max_v:.2f} > {self.max_vel:.2f} rad/s"
                )
                was_constrained = True

            # 3. Acceleration limiting
            new_velocity = (target - self._last_position) / self.dt
            if self._last_velocity is not None:
                acceleration = (new_velocity - self._last_velocity) / self.dt
                max_a = np.max(np.abs(acceleration))

                if max_a > self.max_acc:
                    scale = self.max_acc / max_a
                    constrained_vel = (
                        self._last_velocity
                        + (new_velocity - self._last_velocity) * scale
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
        """Initialize safety state with the robot's current position.

        Call this BEFORE the first check() so that the very first prediction
        is velocity/acceleration-limited relative to where the robot actually
        is, preventing a sudden jump.

        Args:
            position: [6] current joint angles in radians (from /joint_states).
        """
        self._last_position = position.copy().astype(np.float32)
        self._last_velocity = np.zeros(6, dtype=np.float32)

    def reset(self) -> None:
        """Clear state (call when restarting inference)."""
        self._last_position = None
        self._last_velocity = None
