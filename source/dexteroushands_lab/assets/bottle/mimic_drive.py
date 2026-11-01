"""Implicit PhysX stick/slip drive for a hard-pitch bottle articulation.

The spin DOF uses a bounded implicit spring before breakaway. During slip it
uses a zero-stiffness, zero-velocity implicit damper with a lower force cap.
This is a solver-compatible approximation of the engineering example's
running resistance, not a calibrated real closure model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import BottleConfig


@dataclass(frozen=True)
class MimicDriveState:
    sliding: torch.Tensor
    breakaway_seen: torch.Tensor
    breakaway_limit_nm: torch.Tensor
    running_limit_nm: torch.Tensor
    active_force_limit_nm: torch.Tensor


class BatchedMimicDrive:
    """Update all spin drives on GPU without explicit per-step joint torques."""

    def __init__(self, articulation_view, config: BottleConfig, *, device: str,
                 num_envs: int):
        import warp as wp

        if articulation_view.count != num_envs or num_envs < 1:
            raise ValueError("Mimic drive articulation count mismatch")
        names = list(articulation_view.shared_metatype.dof_names)
        if set(names) != {"SlideJoint", "SpinJoint"}:
            raise ValueError(f"Unexpected mimic drive DOFs: {names}")
        self.view = articulation_view
        self.config = config
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.spin_id = names.index("SpinJoint")
        # This PhysX tensor backend requires CPU articulation indices for drive
        # parameter writes even when the parameter arrays live on CUDA.
        self.ids = wp.from_torch(torch.arange(num_envs, dtype=torch.int32))
        self.target_ids = wp.from_torch(torch.arange(num_envs, dtype=torch.int32, device=device))
        self.stiffness = (wp.to_torch(articulation_view.get_dof_stiffnesses())
                          .reshape(num_envs, 2).clone())
        self.damping = (wp.to_torch(articulation_view.get_dof_dampings())
                        .reshape(num_envs, 2).clone())
        self.max_force = (wp.to_torch(articulation_view.get_dof_max_forces())
                          .reshape(num_envs, 2).clone())
        self.target = (wp.to_torch(articulation_view.get_dof_position_targets())
                       .reshape(num_envs, 2).clone())
        self.stick_reference = (wp.to_torch(articulation_view.get_dof_positions())
                                .reshape(num_envs, 2)[:, self.spin_id].clone())
        self.sliding = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.breakaway_seen = torch.zeros_like(self.sliding)
        self.last_state = None

    def reset(self, env_ids: torch.Tensor, theta_rad: torch.Tensor) -> None:
        """Reset the resistance mode for selected articulations at an episode boundary."""
        import warp as wp

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        theta = theta_rad.to(device=self.device, dtype=torch.float32)
        if theta.shape != (ids.numel(),):
            raise ValueError("Drive Reset angle shape mismatch")
        if not bool(torch.isfinite(theta).all()):
            raise ValueError("Drive Reset angle must be finite")
        self.stick_reference[ids] = theta
        self.sliding[ids] = False
        self.breakaway_seen[ids] = False
        self.stiffness[ids, self.spin_id] = self.config.breakaway_stiffness
        self.damping[ids, self.spin_id] = self.config.breakaway_damping
        self.max_force[ids, self.spin_id] = self.config.breakaway_torque
        self.target[ids, self.spin_id] = theta
        selected_cpu = wp.from_torch(ids.to(device="cpu", dtype=torch.int32))
        selected_gpu = wp.from_torch(ids.to(device=self.device, dtype=torch.int32))
        self.view.set_dof_stiffnesses(wp.from_torch(self.stiffness), selected_cpu)
        self.view.set_dof_dampings(wp.from_torch(self.damping), selected_cpu)
        self.view.set_dof_max_forces(wp.from_torch(self.max_force), selected_cpu)
        self.view.set_dof_position_targets(wp.from_torch(self.target), selected_gpu)
        self.last_state = None

    def step(self, theta_rad: torch.Tensor, spin_velocity_rad_s: torch.Tensor,
             axial_m: torch.Tensor) -> MimicDriveState:
        """Update bounded implicit drive parameters before one physics step."""
        import warp as wp

        if (theta_rad.shape != (self.num_envs,)
                or spin_velocity_rad_s.shape != (self.num_envs,)
                or axial_m.shape != (self.num_envs,)):
            raise ValueError("Mimic drive state must contain one row per environment")
        cfg = self.config
        engagement = (1.0 - axial_m / (cfg.thread_pitch * cfg.thread_turns)).clamp(0, 1)
        power = engagement.pow(cfg.engagement_resistance_exponent)
        breakaway = cfg.loose_breakaway_torque + (
            cfg.breakaway_torque - cfg.loose_breakaway_torque) * power
        running = cfg.loose_running_resistance_torque + (
            cfg.running_resistance_torque - cfg.loose_running_resistance_torque) * power
        restick = self.sliding & (spin_velocity_rad_s.abs() < cfg.restick_speed)
        self.stick_reference = torch.where(restick, theta_rad, self.stick_reference)
        self.sliding = self.sliding & ~restick
        slip_angle = cfg.slip_angle_ratio * breakaway / cfg.breakaway_stiffness
        slip = (~self.sliding & ((theta_rad - self.stick_reference).abs() >= slip_angle)
                & (spin_velocity_rad_s.abs() >= cfg.slip_speed))
        self.sliding |= slip
        self.breakaway_seen |= slip
        # Runtime articulation DOF tensors use radians for angular positions.
        # USD authored angular drive gains are degrees-based, but the tensor
        # API exposes the converted internal gains (Nm/rad, Nm s/rad).
        static_stiffness = cfg.breakaway_stiffness
        static_damping = cfg.breakaway_damping
        running_damping = running / cfg.restick_speed
        self.stiffness[:, self.spin_id] = torch.where(
            self.sliding, 0.0, static_stiffness)
        self.damping[:, self.spin_id] = torch.where(
            self.sliding, running_damping, static_damping)
        self.max_force[:, self.spin_id] = torch.where(
            self.sliding, running, breakaway)
        self.target[:, self.spin_id] = torch.where(
            self.sliding, theta_rad, self.stick_reference)
        self.view.set_dof_stiffnesses(wp.from_torch(self.stiffness), self.ids)
        self.view.set_dof_dampings(wp.from_torch(self.damping), self.ids)
        self.view.set_dof_max_forces(wp.from_torch(self.max_force), self.ids)
        self.view.set_dof_position_targets(wp.from_torch(self.target), self.target_ids)
        state = MimicDriveState(
            sliding=self.sliding.clone(), breakaway_seen=self.breakaway_seen.clone(),
            breakaway_limit_nm=breakaway, running_limit_nm=running,
            active_force_limit_nm=self.max_force[:, self.spin_id].clone())
        self.last_state = state
        return state
