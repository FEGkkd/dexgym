"""Bounded six-axis wrench controller for a floating PhysX hand articulation.

Commands are world-frame translations and wrist-local rotation vectors.
Quaternions in this module and Isaac Lab initial/root pose buffers are xyzw.
The controller applies forces and torques; it never writes a pose per step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


Vector3 = tuple[float, float, float]
QuaternionXYZW = tuple[float, float, float, float]


def _norm(v) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _clip(v: Vector3, maximum: float) -> Vector3:
    length = _norm(v)
    factor = min(1.0, maximum / length) if length else 1.0
    return tuple(float(x) * factor for x in v)  # type: ignore[return-value]


def _quat_mul(a: QuaternionXYZW, b: QuaternionXYZW) -> QuaternionXYZW:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_normalize(q: QuaternionXYZW) -> QuaternionXYZW:
    length = _norm(q)
    if length < 1e-12:
        raise ValueError("zero quaternion")
    return tuple(float(x) / length for x in q)  # type: ignore[return-value]


def _orientation_error(target: QuaternionXYZW, current: QuaternionXYZW) -> Vector3:
    cx, cy, cz, cw = _quat_normalize(current)
    x, y, z, w = _quat_mul(_quat_normalize(target), (-cx, -cy, -cz, cw))
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    radius = _norm((x, y, z))
    if radius < 1e-10:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(radius, max(0.0, w))
    return (angle * x / radius, angle * y / radius, angle * z / radius)


@dataclass(frozen=True)
class WristLimits:
    max_command_translation: float = 0.01  # m per control update
    max_command_rotation: float = 0.08  # rad per control update
    max_target_radius: float = 0.60  # m from initial wrist position
    max_target_rotation: float = 1.2  # rad from initial wrist orientation
    min_height: float = 0.10  # world z, m
    max_force: float = 40.0  # N
    max_torque: float = 2.0  # N m
    max_linear_speed: float = 0.50  # m/s, soft resistance threshold
    max_angular_speed: float = 3.0  # rad/s, soft resistance threshold
    kp_position: float = 1200.0  # N/m
    kd_position: float = 80.0  # N s/m
    kp_rotation: float = 8.0  # N m/rad
    kd_rotation: float = 1.2  # N m s/rad

    def __post_init__(self):
        if any(value <= 0 for key, value in vars(self).items() if key != "min_height"):
            raise ValueError("Wrist limits and gains must be positive")


class Wrist6DoFController:
    """Pose-tracking wrench source for one PhysX floating articulation root."""

    def __init__(
        self, initial_pos: Vector3, initial_quat: QuaternionXYZW,
        limits: WristLimits | None = None,
    ):
        self.limits = limits or WristLimits()
        self.initial_pos = tuple(float(x) for x in initial_pos)
        self.target_pos = self.initial_pos
        self.initial_quat = _quat_normalize(initial_quat)
        self.target_quat = self.initial_quat

    @property
    def target_rot_angle(self) -> float:
        return _norm(_orientation_error(self.target_quat, self.initial_quat))

    def command(self, translation_world: Vector3, rotation_local: Vector3) -> None:
        delta_p = _clip(translation_world, self.limits.max_command_translation)
        next_pos = tuple(a + b for a, b in zip(self.target_pos, delta_p))
        displacement = _clip(
            tuple(a - b for a, b in zip(next_pos, self.initial_pos)),
            self.limits.max_target_radius,
        )
        self.target_pos = (
            self.initial_pos[0] + displacement[0],
            self.initial_pos[1] + displacement[1],
            max(self.limits.min_height, self.initial_pos[2] + displacement[2]),
        )
        delta_r = _clip(rotation_local, self.limits.max_command_rotation)
        angle = _norm(delta_r)
        if angle:
            sin_half_over_angle = math.sin(angle / 2.0) / angle
            delta_q = (
                delta_r[0] * sin_half_over_angle,
                delta_r[1] * sin_half_over_angle,
                delta_r[2] * sin_half_over_angle,
                math.cos(angle / 2.0),
            )
            candidate = _quat_normalize(_quat_mul(self.target_quat, delta_q))
            from_initial = _clip(
                _orientation_error(candidate, self.initial_quat),
                self.limits.max_target_rotation,
            )
            total_angle = _norm(from_initial)
            if total_angle:
                factor = math.sin(total_angle / 2.0) / total_angle
                total_q = (from_initial[0] * factor, from_initial[1] * factor,
                           from_initial[2] * factor, math.cos(total_angle / 2.0))
                self.target_quat = _quat_normalize(_quat_mul(total_q, self.initial_quat))
            else:
                self.target_quat = self.initial_quat

    def wrench(
        self, position_world: Vector3, quaternion_xyzw: QuaternionXYZW,
        linear_velocity_world: Vector3, angular_velocity_world: Vector3,
    ) -> tuple[Vector3, Vector3]:
        """Return bounded world-frame force and torque about the wrist root COM."""
        pos_error = tuple(a - b for a, b in zip(self.target_pos, position_world))
        rot_error = _orientation_error(self.target_quat, quaternion_xyzw)
        linear_overspeed = max(0.0, _norm(linear_velocity_world) - self.limits.max_linear_speed)
        angular_overspeed = max(0.0, _norm(angular_velocity_world) - self.limits.max_angular_speed)
        force = tuple(
            self.limits.kp_position * e
            - (self.limits.kd_position + 2.0 * linear_overspeed) * v
            for e, v in zip(pos_error, linear_velocity_world)
        )
        torque = tuple(
            self.limits.kp_rotation * e
            - (self.limits.kd_rotation + 0.2 * angular_overspeed) * v
            for e, v in zip(rot_error, angular_velocity_world)
        )
        return _clip(force, self.limits.max_force), _clip(torque, self.limits.max_torque)
