"""Runtime adapter from a floating PhysX screw articulation to RL telemetry.

This backend is experimental until a contact-topology-qualified stable Reset
bank and independent policy evaluation exist. It never writes the Cap link or
applies scripted screw wrenches; PhysX solves the hard pitch and implicit drive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import BottleConfig
from .mimic_drive import BatchedMimicDrive
from .mimic_thread_view import MimicThreadView


@dataclass(frozen=True)
class MimicRuntimeTelemetry:
    axial: torch.Tensor
    constraint_error: torch.Tensor
    breakaway_limit: torch.Tensor
    running_limit: torch.Tensor


class MimicThreadRuntime:
    """Vectorized hard-pitch state, drive and selected-environment Reset."""

    def __init__(self, physics_view, config: BottleConfig, *, num_envs: int,
                 device: str):
        import warp as wp

        art = physics_view.create_articulation_view("/World/envs/env_*/Bottle")
        self.view = MimicThreadView(physics_view, art, config,
                                    device=device, num_envs=num_envs)
        self.drive = BatchedMimicDrive(art, config, device=device, num_envs=num_envs)
        self.config = config
        self.num_envs = num_envs
        self.art = art
        self._wp = wp
        self.state = self.view.read()
        self.drive.reset(torch.arange(num_envs, device=device), self.state.theta_rad)

    @property
    def theta_unwrapped(self) -> torch.Tensor:
        return self.state.theta_rad

    @property
    def sliding(self) -> torch.Tensor:
        return self.drive.sliding

    @property
    def stick_reference(self) -> torch.Tensor:
        return self.drive.stick_reference

    def reset(self, env_ids: torch.Tensor, *, bottle_pose_w: torch.Tensor,
              bottle_velocity_w: torch.Tensor, angle_rad: torch.Tensor,
              spin_velocity_rad_s: torch.Tensor | None = None,
              stick_reference_rad: torch.Tensor | None = None,
              sliding: torch.Tensor | None = None,
              breakaway_seen: torch.Tensor | None = None) -> MimicRuntimeTelemetry:
        self.view.reset(env_ids, root_pose_w=bottle_pose_w,
                        root_velocity_w=bottle_velocity_w, theta_rad=angle_rad,
                        spin_velocity_rad_s=spin_velocity_rad_s)
        self.drive.reset(env_ids, angle_rad,
                         stick_reference_rad=stick_reference_rad,
                         sliding=sliding, breakaway_seen=breakaway_seen)
        return self.update()

    def step(self) -> MimicRuntimeTelemetry:
        q = self._wp.to_torch(self.art.get_dof_positions()).reshape(self.num_envs, 2)
        qd = self._wp.to_torch(self.art.get_dof_velocities()).reshape(self.num_envs, 2)
        self.drive.step(q[:, self.view.spin_id], qd[:, self.view.spin_id],
                        q[:, self.view.slide_id])
        return self.update()

    def update(self) -> MimicRuntimeTelemetry:
        self.state = self.view.read()
        engagement = (1.0 - self.state.axial_m /
                      (self.config.thread_pitch * self.config.thread_turns)).clamp(0, 1)
        power = engagement.pow(self.config.engagement_resistance_exponent)
        breakaway = self.config.loose_breakaway_torque + (
            self.config.breakaway_torque - self.config.loose_breakaway_torque) * power
        running = self.config.loose_running_resistance_torque + (
            self.config.running_resistance_torque - self.config.loose_running_resistance_torque) * power
        return MimicRuntimeTelemetry(
            axial=self.state.axial_m,
            constraint_error=self.state.mimic_error_m + self.state.physical_error_m,
            breakaway_limit=breakaway,
            running_limit=running,
        )
