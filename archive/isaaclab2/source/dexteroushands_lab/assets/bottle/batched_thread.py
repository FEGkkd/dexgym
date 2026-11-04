"""Vectorized GPU screw compliance for independent free bottle/cap pairs.

Equivalent to :mod:`thread` for the engaged, non-detaching Phase 1 task.
No body pose is written by this module; returned equal/opposite wrenches are
applied through PhysX on every substep. All quaternions are xyzw.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .config import BottleConfig
from ..batched_wrist_pose import _mul, _normalize


def _rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = _normalize(q)
    t = 2.0 * torch.cross(q[:, :3], v, dim=-1)
    return v + q[:, 3:] * t + torch.cross(q[:, :3], t, dim=-1)


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1)


def _clip(v: torch.Tensor, maximum: float) -> torch.Tensor:
    length = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return v * (maximum / length.clamp_min(1e-12)).clamp(max=1)


@dataclass
class BatchedThreadWrenches:
    bottle_force: torch.Tensor
    bottle_torque: torch.Tensor
    cap_force: torch.Tensor
    cap_torque: torch.Tensor
    theta_unwrapped: torch.Tensor
    axial: torch.Tensor
    desired_axial: torch.Tensor
    constraint_error: torch.Tensor
    lateral_error: torch.Tensor
    axis_world: torch.Tensor
    engagement_fraction: torch.Tensor
    resistance_torque: torch.Tensor
    breakaway_resistance_torque: torch.Tensor
    breakaway_limit: torch.Tensor
    running_limit: torch.Tensor


class BatchedThreadCoupler:
    """No per-env Python work or GPU-to-CPU sync during physics substeps."""

    def __init__(self, num_envs: int, config: BottleConfig, *, device: str,
                 dtype=torch.float32):
        if config.allow_detachment:
            raise ValueError("Phase 1 batched thread currently requires allow_detachment=False")
        self.num_envs = num_envs
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        def zeros():
            return torch.zeros(num_envs, device=device, dtype=dtype)
        self.last_wrapped = zeros()
        self.theta = zeros()
        self.initial_theta = zeros()
        self.initial_axial = zeros()
        self.stick_reference = zeros()
        self.sliding = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)

    @property
    def theta_unwrapped(self) -> torch.Tensor:
        return self.theta - self.initial_theta

    def _geometry(self, bottle_pose: torch.Tensor, cap_pose: torch.Tensor):
        axis = _rotate(bottle_pose[:, 3:],
                       torch.tensor([0., 0., 1.], device=self.device, dtype=self.dtype).expand(self.num_envs, 3))
        displacement = cap_pose[:, :3] - bottle_pose[:, :3]
        axial = _dot(displacement, axis)
        lateral = displacement - axis * axial[:, None]
        bq = _normalize(bottle_pose[:, 3:])
        cq = _normalize(cap_pose[:, 3:])
        relative = _mul(torch.cat((-bq[:, :3], bq[:, 3:]), dim=-1), cq)
        wrapped = torch.atan2(torch.sin(2 * torch.atan2(relative[:, 2], relative[:, 3])),
                              torch.cos(2 * torch.atan2(relative[:, 2], relative[:, 3])))
        return axis, displacement, axial, lateral, wrapped

    def reset_from_pose(self, env_ids: torch.Tensor, bottle_pose: torch.Tensor,
                        cap_pose: torch.Tensor, *, initial_angle: torch.Tensor | None = None,
                        snapshot: dict[str, torch.Tensor] | None = None) -> None:
        ids = env_ids.to(device=self.device, dtype=torch.long)
        bp = bottle_pose.to(self.device, self.dtype)
        cp = cap_pose.to(self.device, self.dtype)
        if bp.shape != (len(ids), 7) or cp.shape != (len(ids), 7):
            raise ValueError("Thread reset poses must match selected environment count")
        # Compute local-axis geometry only for selected rows, without mutating peers.
        bq, cq = _normalize(bp[:, 3:]), _normalize(cp[:, 3:])
        axis = _rotate(bq, torch.tensor([0., 0., 1.], device=self.device, dtype=self.dtype).expand(len(ids), 3))
        axial = _dot(cp[:, :3] - bp[:, :3], axis)
        relative = _mul(torch.cat((-bq[:, :3], bq[:, 3:]), dim=-1), cq)
        wrapped = torch.atan2(torch.sin(2 * torch.atan2(relative[:, 2], relative[:, 3])),
                              torch.cos(2 * torch.atan2(relative[:, 2], relative[:, 3])))
        if snapshot is None:
            theta = wrapped if initial_angle is None else initial_angle.to(self.device, self.dtype)
            mismatch = torch.atan2(torch.sin(theta - wrapped), torch.cos(theta - wrapped))
            if torch.any(mismatch.abs() > 1e-4):
                raise ValueError("Thread angle and cap pose disagree")
            self.last_wrapped[ids] = wrapped
            self.theta[ids] = theta
            self.initial_theta[ids] = theta
            self.initial_axial[ids] = axial
            self.stick_reference[ids] = theta
            self.sliding[ids] = False
        else:
            for key, field in (("last_wrapped_rad", self.last_wrapped),
                               ("theta_unwrapped_rad", self.theta),
                               ("initial_theta_rad", self.initial_theta),
                               ("initial_axial_m", self.initial_axial),
                               ("stick_reference_rad", self.stick_reference),
                               ("sliding", self.sliding)):
                field[ids] = snapshot[key].to(device=self.device, dtype=field.dtype)
            if torch.any(snapshot["detached"]):
                raise ValueError("Detached cap cannot enter Phase 1 engaged reset")
        self.initialized[ids] = True

    def snapshot(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        ids = env_ids.to(device=self.device, dtype=torch.long)
        return {"last_wrapped_rad": self.last_wrapped[ids].clone(),
                "theta_unwrapped_rad": self.theta[ids].clone(),
                "initial_theta_rad": self.initial_theta[ids].clone(),
                "initial_axial_m": self.initial_axial[ids].clone(),
                "stick_reference_rad": self.stick_reference[ids].clone(),
                "sliding": self.sliding[ids].clone(),
                "detached": torch.zeros(len(ids), device=self.device, dtype=torch.bool)}

    def wrenches(self, bottle_pose: torch.Tensor, cap_pose: torch.Tensor,
                 bottle_velocity: torch.Tensor, cap_velocity: torch.Tensor) -> BatchedThreadWrenches:
        cfg = self.config
        axis, displacement, axial, lateral, wrapped = self._geometry(bottle_pose, cap_pose)
        delta = torch.atan2(torch.sin(wrapped - self.last_wrapped),
                            torch.cos(wrapped - self.last_wrapped))
        self.theta += delta
        self.last_wrapped = wrapped
        theta = self.theta_unwrapped
        limit = cfg.thread_turns * 2 * math.pi
        travel_angle = theta.clamp(-limit, limit)
        desired = self.initial_axial + cfg.thread_sign * cfg.thread_pitch * travel_angle / (2 * math.pi)
        error = axial - desired
        engagement = (1 - (axial - self.initial_axial) / (cfg.thread_pitch * cfg.thread_turns)).clamp(0, 1)
        relative_velocity = (cap_velocity[:, :3] - bottle_velocity[:, :3]
                             - torch.cross(bottle_velocity[:, 3:], displacement, dim=-1))
        axial_velocity = _dot(relative_velocity, axis)
        lateral_velocity = relative_velocity - axis * axial_velocity[:, None]
        cap_force = _clip(-cfg.lateral_stiffness * lateral
                          -cfg.lateral_damping * lateral_velocity
                          + axis * (-cfg.axial_stiffness * error
                                    -cfg.axial_damping * axial_velocity)[:, None],
                          cfg.max_constraint_force)
        cap_axis = _rotate(cap_pose[:, 3:],
                           torch.tensor([0., 0., 1.], device=self.device, dtype=self.dtype).expand(self.num_envs, 3))
        relative_omega = cap_velocity[:, 3:] - bottle_velocity[:, 3:]
        spin = _dot(relative_omega, axis)
        tilt_torque = (cfg.tilt_stiffness * torch.cross(cap_axis, axis, dim=-1)
                       -cfg.tilt_damping * (relative_omega - axis * spin[:, None]))
        lead = cfg.thread_sign * cfg.thread_pitch / (2 * math.pi)
        screw_torque = cfg.axial_stiffness * error * lead
        power = engagement.pow(cfg.engagement_resistance_exponent)
        breakaway_limit = cfg.loose_breakaway_torque + (cfg.breakaway_torque - cfg.loose_breakaway_torque) * power
        running_limit = (cfg.loose_running_resistance_torque
                         + (cfg.running_resistance_torque - cfg.loose_running_resistance_torque) * power)
        restick = self.sliding & (spin.abs() < cfg.restick_speed)
        self.sliding = self.sliding & ~restick
        self.stick_reference = torch.where(restick, theta, self.stick_reference)
        slip = (~self.sliding & ((theta - self.stick_reference).abs()
                                >= cfg.slip_angle_ratio * breakaway_limit / cfg.breakaway_stiffness)
                & (spin.abs() >= cfg.slip_speed))
        self.sliding = self.sliding | slip
        trial = -cfg.breakaway_stiffness * (theta - self.stick_reference) - cfg.breakaway_damping * spin
        breakaway = torch.where(self.sliding, torch.zeros_like(trial),
                                torch.maximum(-breakaway_limit, torch.minimum(breakaway_limit, trial)))
        resistance = -running_limit * torch.tanh(spin / cfg.friction_velocity_scale)
        excess = theta - theta.clamp(-limit, limit)
        end_stop = -cfg.end_stop_stiffness * excess - torch.where(
            spin * excess > 0, cfg.end_stop_damping * spin, torch.zeros_like(spin))
        coupling = _clip(tilt_torque + axis * (screw_torque + resistance + breakaway + end_stop)[:, None],
                         cfg.max_constraint_torque)
        anchor = bottle_pose[:, :3] + axis * (cfg.bottle_height / 2 + cfg.neck_height / 2)
        cap_torque = coupling + torch.cross(anchor - cap_pose[:, :3], cap_force, dim=-1)
        bottle_torque = -coupling + torch.cross(anchor - bottle_pose[:, :3], -cap_force, dim=-1)
        return BatchedThreadWrenches(-cap_force, bottle_torque, cap_force, cap_torque,
                                     theta, axial, desired, error,
                                     torch.linalg.vector_norm(lateral, dim=-1), axis,
                                     engagement, resistance, breakaway,
                                     breakaway_limit, running_limit)
