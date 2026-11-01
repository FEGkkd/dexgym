"""GPU-batched thread resistance for a PhysX mimic-joint screw assembly.

The mimic joint constrains pitch and lateral motion. This module supplies only
the finite angular breakaway/running resistance to its revolute DOF. It does
not apply cap root wrenches or alter rigid-body poses. Parameters match the
existing :class:`BatchedThreadCoupler` engineering example.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import BottleConfig


@dataclass(frozen=True)
class MimicResistanceTerms:
    torque_nm: torch.Tensor
    breakaway_nm: torch.Tensor
    running_nm: torch.Tensor
    breakaway_limit_nm: torch.Tensor
    engagement_fraction: torch.Tensor
    sliding: torch.Tensor


class BatchedMimicResistance:
    """Stateful, per-environment angular resistance without CPU sync.

    ``theta_rad`` is the mimic master's revolute coordinate and ``axial_m``
    is the follower prismatic coordinate, both read from the same PhysX
    articulation. ``reset`` takes the measured angle, which may be nonzero.
    """

    def __init__(self, config: BottleConfig, num_envs: int, *, device: str):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self.stick_reference_rad = torch.zeros(num_envs, device=device)
        self.sliding = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor, theta_rad: torch.Tensor) -> None:
        if theta_rad.shape != (len(env_ids),):
            raise ValueError("theta_rad must have one value per selected environment")
        self.stick_reference_rad[env_ids] = theta_rad
        self.sliding[env_ids] = False

    def step(self, theta_rad: torch.Tensor, omega_rad_s: torch.Tensor,
             axial_m: torch.Tensor) -> MimicResistanceTerms:
        if any(x.shape != (self.num_envs,) for x in (theta_rad, omega_rad_s, axial_m)):
            raise ValueError("expected one angle, speed and axial displacement per environment")
        cfg = self.config
        engagement = (1 - cfg.thread_sign * axial_m
                      / (cfg.thread_pitch * cfg.thread_turns)).clamp(0, 1)
        power = engagement.pow(cfg.engagement_resistance_exponent)
        breakaway_limit = (cfg.loose_breakaway_torque
                           + (cfg.breakaway_torque - cfg.loose_breakaway_torque) * power)
        running_limit = (cfg.loose_running_resistance_torque
                         + (cfg.running_resistance_torque
                            - cfg.loose_running_resistance_torque) * power)
        restick = self.sliding & (omega_rad_s.abs() < cfg.restick_speed)
        self.sliding = self.sliding & ~restick
        self.stick_reference_rad = torch.where(restick, theta_rad, self.stick_reference_rad)
        slip = (~self.sliding
                & ((theta_rad - self.stick_reference_rad).abs()
                   >= cfg.slip_angle_ratio * breakaway_limit / cfg.breakaway_stiffness)
                & (omega_rad_s.abs() >= cfg.slip_speed))
        self.sliding = self.sliding | slip
        trial = (-cfg.breakaway_stiffness * (theta_rad - self.stick_reference_rad)
                 - cfg.breakaway_damping * omega_rad_s)
        breakaway = torch.where(self.sliding, torch.zeros_like(trial),
                                trial.clamp(-breakaway_limit, breakaway_limit))
        running = -running_limit * torch.tanh(omega_rad_s / cfg.friction_velocity_scale)
        torque = (breakaway + running).clamp(-cfg.max_constraint_torque,
                                             cfg.max_constraint_torque)
        return MimicResistanceTerms(torque, breakaway, running,
                                    breakaway_limit, engagement, self.sliding.clone())
