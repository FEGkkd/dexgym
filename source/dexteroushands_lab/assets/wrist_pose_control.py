"""Ideal, speed-limited SE(3) wrist command trajectory (xyzw quaternions).

This module computes command poses and velocities. The Isaac Lab adapter must
verify PhysX contact before using root-state writes as a production wrist.
The older bounded wrench PD remains in ``wrist_6dof.py`` for experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from dexteroushands_lab.assets.wrist_6dof import (
    _norm, _orientation_error, _quat_mul, _quat_normalize,
)


def _exp_rotation(vector):
    angle = _norm(vector)
    if angle < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    s = math.sin(angle / 2.0) / angle
    return (vector[0] * s, vector[1] * s, vector[2] * s,
            math.cos(angle / 2.0))


@dataclass(frozen=True)
class WristPoseLimits:
    max_linear_speed: float = 0.15
    max_angular_speed: float = 1.0
    max_target_radius: float = 0.6
    min_height: float = 0.1
    position_tolerance: float = 1e-5
    rotation_tolerance: float = 1e-4

    def __post_init__(self):
        if (self.max_linear_speed <= 0 or self.max_angular_speed <= 0
                or self.max_target_radius <= 0 or self.min_height < 0
                or self.position_tolerance <= 0 or self.rotation_tolerance <= 0):
            raise ValueError("Invalid ideal wrist trajectory limits")


class WristPoseController:
    """One environment's ideal wrist target and finite-speed command state."""

    def __init__(self, initial_pos, initial_quat, limits: WristPoseLimits | None = None):
        self.limits = limits or WristPoseLimits()
        self.reset_to(initial_pos, initial_quat)

    def reset_to(self, position_world, quaternion_xyzw, *, target_pos=None,
                 target_quat=None):
        """Re-anchor the episode and clear trajectory velocity and progress."""
        self.initial_pos = tuple(float(x) for x in position_world)
        self.initial_quat = _quat_normalize(tuple(quaternion_xyzw))
        self.current_pos = self.initial_pos
        self.current_quat = self.initial_quat
        self.current_velocity = (0.0,) * 6
        self.target_pos = self.initial_pos
        self.target_quat = self.initial_quat
        self.set_target(target_pos or self.initial_pos,
                        target_quat or self.initial_quat)

    @property
    def current_pose(self):
        return (*self.current_pos, *self.current_quat)

    @property
    def target_pose(self):
        return (*self.target_pos, *self.target_quat)

    def set_target(self, position_xyz, quaternion_xyzw):
        pos = tuple(float(x) for x in position_xyz)
        if (len(pos) != 3 or not all(math.isfinite(x) for x in pos)
                or pos[2] < self.limits.min_height
                or math.dist(pos, self.initial_pos) > self.limits.max_target_radius):
            raise ValueError("Ideal wrist target is outside the finite workspace")
        quat = _quat_normalize(tuple(quaternion_xyzw))
        if not all(math.isfinite(x) for x in quat):
            raise ValueError("Ideal wrist quaternion must be finite")
        self.target_pos = pos
        self.target_quat = quat

    def command_relative(self, translation_world, rotation_local):
        """Compose a world translation and a wrist-local rotation vector."""
        translation = tuple(float(x) for x in translation_world)
        rotation = tuple(float(x) for x in rotation_local)
        if (len(translation) != 3 or len(rotation) != 3
                or not all(math.isfinite(x) for x in (*translation, *rotation))):
            raise ValueError("Relative wrist command requires six finite values")
        self.set_target(tuple(a + b for a, b in zip(self.target_pos, translation)),
                        _quat_mul(self.target_quat, _exp_rotation(rotation)))

    def command(self, translation_world, rotation_local):
        """Compatibility with the existing 6D incremental command interface."""
        self.command_relative(translation_world, rotation_local)

    def at_target(self):
        return (math.dist(self.current_pos, self.target_pos)
                <= self.limits.position_tolerance and
                _norm(_orientation_error(self.target_quat, self.current_quat))
                <= self.limits.rotation_tolerance)

    def step(self, physics_dt):
        """Advance one command substep and return pose plus world-frame 6D velocity."""
        dt = float(physics_dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Physics dt must be finite and positive")
        dp = tuple(a - b for a, b in zip(self.target_pos, self.current_pos))
        distance = _norm(dp)
        amount = min(distance, self.limits.max_linear_speed * dt)
        move = tuple(x * amount / distance for x in dp) if distance else (0.0,) * 3
        dr = _orientation_error(self.target_quat, self.current_quat)
        angle = _norm(dr)
        turned = min(angle, self.limits.max_angular_speed * dt)
        rotation = tuple(x * turned / angle for x in dr) if angle else (0.0,) * 3
        self.current_pos = tuple(a + b for a, b in zip(self.current_pos, move))
        self.current_quat = _quat_normalize(
            _quat_mul(_exp_rotation(rotation), self.current_quat))
        self.current_velocity = tuple(x / dt for x in (*move, *rotation))
        return self.current_pose, self.current_velocity


def split_bimanual_action_52(values):
    """Parse [left Δxyz, Δrot_local, 20 finger radians, right same].

    Translation is world-frame metres; rotation is a wrist-local rotation
    vector in radians. Finger values are absolute canonical joint targets.
    The parser is independent of Isaac and does not define an RL policy.
    """
    action = tuple(float(value) for value in values)
    if len(action) != 52 or any(not math.isfinite(value) for value in action):
        raise ValueError("A bimanual wrist/finger command requires 52 finite values")
    return ((action[0:3], action[3:6], action[6:26]),
            (action[26:29], action[29:32], action[32:52]))
