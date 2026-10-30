"""Equal-and-opposite compliant screw coupling for two free PhysX rigid bodies.

No pose writes occur here. The scene applies these wrenches every physics step.
The finite travel currently keeps the cap engaged; detachment is not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import BottleConfig
from ..wrist_6dof import _clip, _norm, _quat_mul, _quat_normalize


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _scale(v, scalar):
    return tuple(scalar * x for x in v)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rotate(q, v):
    q = _quat_normalize(q)
    x, y, z, w = q
    qvec = (x, y, z)
    t = _scale(_cross(qvec, v), 2.0)
    return _add(v, _add(_scale(t, w), _cross(qvec, t)))


@dataclass(frozen=True)
class ThreadState:
    theta_unwrapped: float
    axial: float
    desired_axial: float
    constraint_error: float
    lateral_error: float
    axis_world: tuple[float, float, float]


@dataclass(frozen=True)
class ThreadWrenches:
    bottle_force: tuple[float, float, float]
    bottle_torque: tuple[float, float, float]
    cap_force: tuple[float, float, float]
    cap_torque: tuple[float, float, float]
    state: ThreadState
    resistance_torque: float
    breakaway_resistance_torque: float
    end_stop_resistance_torque: float
    applied_coupling_torque_nm: float


class ThreadCoupler:
    """Stateful angle unwrap and compliance controller for one bottle/cap pair."""

    def __init__(self, config: BottleConfig):
        self.config = config
        self._last_wrapped: float | None = None
        self._theta = 0.0
        self._initial_theta = 0.0
        self._initial_axial: float | None = None
        self._stick_reference = 0.0
        self._sliding = False

    def reset(self) -> None:
        self._last_wrapped = None
        self._theta = 0.0
        self._initial_theta = 0.0
        self._initial_axial = None
        self._stick_reference = 0.0
        self._sliding = False

    def observe(self, bottle_pos, bottle_quat, cap_pos, cap_quat) -> ThreadState:
        axis = _rotate(bottle_quat, (0.0, 0.0, 1.0))
        displacement = _sub(cap_pos, bottle_pos)
        axial = _dot(displacement, axis)
        lateral = _sub(displacement, _scale(axis, axial))
        bx, by, bz, bw = _quat_normalize(bottle_quat)
        relative = _quat_mul((-bx, -by, -bz, bw), _quat_normalize(cap_quat))
        wrapped = math.atan2(math.sin(2 * math.atan2(relative[2], relative[3])),
                             math.cos(2 * math.atan2(relative[2], relative[3])))
        if self._last_wrapped is None:
            self._last_wrapped = wrapped
            self._theta = wrapped
            self._initial_theta = wrapped
            self._initial_axial = axial
        else:
            delta = math.atan2(math.sin(wrapped - self._last_wrapped),
                               math.cos(wrapped - self._last_wrapped))
            self._theta += delta
            self._last_wrapped = wrapped
        travel_angle = max(-self.config.thread_turns * 2 * math.pi,
                           min(self.config.thread_turns * 2 * math.pi,
                               self._theta - self._initial_theta))
        desired = self._initial_axial + (
            self.config.thread_sign * self.config.thread_pitch * travel_angle / (2 * math.pi)
        )
        return ThreadState(
            self._theta - self._initial_theta, axial, desired, axial - desired, _norm(lateral), axis,
        )

    def wrenches(
        self, bottle_pos, bottle_quat, bottle_lin_vel, bottle_ang_vel,
        cap_pos, cap_quat, cap_lin_vel, cap_ang_vel,
    ) -> ThreadWrenches:
        cfg = self.config
        state = self.observe(bottle_pos, bottle_quat, cap_pos, cap_quat)
        axis = state.axis_world
        displacement = _sub(cap_pos, bottle_pos)
        lateral = _sub(displacement, _scale(axis, state.axial))
        # COM velocity difference includes frame transport when the complete
        # assembly rotates. Subtract ω_bottle × r before damping joint motion.
        relative_velocity = _sub(
            _sub(cap_lin_vel, bottle_lin_vel), _cross(bottle_ang_vel, displacement)
        )
        axial_velocity = _dot(relative_velocity, axis)
        lateral_velocity = _sub(relative_velocity, _scale(axis, axial_velocity))
        lateral_force = _add(
            _scale(lateral, -cfg.lateral_stiffness),
            _scale(lateral_velocity, -cfg.lateral_damping),
        )
        axial_force = -cfg.axial_stiffness * state.constraint_error - cfg.axial_damping * axial_velocity
        cap_force = _clip(_add(lateral_force, _scale(axis, axial_force)), cfg.max_constraint_force)
        bottle_force = _scale(cap_force, -1.0)

        cap_axis = _rotate(cap_quat, (0.0, 0.0, 1.0))
        relative_omega = _sub(cap_ang_vel, bottle_ang_vel)
        spin = _dot(relative_omega, axis)
        tilt_omega = _sub(relative_omega, _scale(axis, spin))
        tilt_torque = _add(
            _scale(_cross(cap_axis, axis), cfg.tilt_stiffness),
            _scale(tilt_omega, -cfg.tilt_damping),
        )
        lead = cfg.thread_sign * cfg.thread_pitch / (2 * math.pi)
        screw_torque = cfg.axial_stiffness * state.constraint_error * lead
        # A small elastic angular range holds the engaged cap. The extra
        # angle/speed condition avoids false slip on one explicit high-
        # acceleration substep. During slip only running resistance remains;
        # when spin settles, re-anchor the static branch at the current angle.
        if self._sliding and abs(spin) < cfg.restick_speed:
            self._sliding = False
            self._stick_reference = state.theta_unwrapped
        if not self._sliding and (
            abs(state.theta_unwrapped - self._stick_reference)
            >= cfg.slip_angle_ratio * cfg.breakaway_torque / cfg.breakaway_stiffness
            and abs(spin) >= cfg.slip_speed
        ):
            self._sliding = True
        trial = (-cfg.breakaway_stiffness * (state.theta_unwrapped - self._stick_reference)
                 - cfg.breakaway_damping * spin)
        breakaway = (0.0 if self._sliding else
                     max(-cfg.breakaway_torque, min(cfg.breakaway_torque, trial)))
        resistance = -cfg.running_resistance_torque * math.tanh(
            spin / cfg.friction_velocity_scale
        )
        limit = cfg.thread_turns * 2 * math.pi
        excess = state.theta_unwrapped - max(-limit, min(limit, state.theta_unwrapped))
        end_stop = 0.0
        if excess:
            end_stop = -cfg.end_stop_stiffness * excess
            if spin * excess > 0:
                end_stop -= cfg.end_stop_damping * spin
        coupling_torque = _clip(
            _add(tilt_torque, _scale(axis, screw_torque + resistance + breakaway + end_stop)),
            cfg.max_constraint_torque,
        )
        # Both force vectors act at the same point on the neck axis; the
        # moment arms preserve equal-and-opposite world wrench transfer.
        anchor = _add(bottle_pos, _scale(axis, cfg.bottle_height / 2 + cfg.neck_height / 2))
        cap_torque = _add(coupling_torque, _cross(_sub(anchor, cap_pos), cap_force))
        bottle_torque = _add(_scale(coupling_torque, -1.0),
                             _cross(_sub(anchor, bottle_pos), bottle_force))
        return ThreadWrenches(
            bottle_force, bottle_torque, cap_force, cap_torque,
            state, resistance, breakaway, end_stop, _norm(coupling_torque),
        )
