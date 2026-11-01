"""Batched PhysX state and selective Reset for a floating mimic screw.

The Bottle is the articulation root. Cap is a non-root link and must never
receive an independent RigidObject root-pose write. PhysX's installed tensor
backend requires full-size input buffers even when indices select a subset.
All per-step state calculations stay on the selected torch device.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .batched_thread import _rotate
from .config import BottleConfig


@dataclass(frozen=True)
class MimicThreadState:
    root_pose_w: torch.Tensor
    root_velocity_w: torch.Tensor
    cap_pose_w: torch.Tensor
    theta_rad: torch.Tensor
    spin_velocity_rad_s: torch.Tensor
    axial_m: torch.Tensor
    opening_rad: torch.Tensor
    mimic_error_m: torch.Tensor
    physical_error_m: torch.Tensor


class MimicThreadView:
    """Own the screw articulation's GPU tensors and per-env angle reference."""

    def __init__(self, physics_view, articulation_view, config: BottleConfig,
                 *, device: str, num_envs: int):
        if num_envs < 1 or articulation_view.count != num_envs:
            raise ValueError("Mimic articulation count must equal num_envs")
        names = list(articulation_view.shared_metatype.dof_names)
        links = list(articulation_view.shared_metatype.link_names)
        if set(names) != {"SlideJoint", "SpinJoint"} or set(links) != {
                "Bottle", "Carriage", "Cap"}:
            raise ValueError(f"Unexpected mimic articulation topology: {names}, {links}")
        if articulation_view.shared_metatype.fixed_base:
            raise ValueError("The bottle screw articulation must have a floating root")
        self.physics_view = physics_view
        self.view = articulation_view
        self.config = config
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.spin_id = names.index("SpinJoint")
        self.slide_id = names.index("SlideJoint")
        self.cap_id = links.index("Cap")
        self.local_axis = torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(
            num_envs, 3)
        import warp as wp
        self.theta_at_reset = (wp.to_torch(self.view.get_dof_positions())
                               .reshape(num_envs, -1)[:, self.spin_id].clone())

    def read(self) -> MimicThreadState:
        """Read live PhysX tensors without using potentially stale link caches."""
        import warp as wp

        root = wp.to_torch(self.view.get_root_transforms()).reshape(self.num_envs, 7).clone()
        root_velocity = (wp.to_torch(self.view.get_root_velocities())
                         .reshape(self.num_envs, 6).clone())
        cap = (wp.to_torch(self.view.get_link_transforms())
               .reshape(self.num_envs, 3, 7)[:, self.cap_id].clone())
        q = wp.to_torch(self.view.get_dof_positions()).reshape(self.num_envs, 2)
        qd = wp.to_torch(self.view.get_dof_velocities()).reshape(self.num_envs, 2)
        theta = q[:, self.spin_id].clone()
        axial = q[:, self.slide_id].clone()
        target_axial = self.config.thread_sign * self.config.thread_pitch * theta / (2 * math.pi)
        local_axis = _rotate(root[:, 3:], self.local_axis)
        physical_axial = ((cap[:, :3] - root[:, :3]) * local_axis).sum(-1)
        physical_axial -= self.config.cap_initial_z
        return MimicThreadState(
            root_pose_w=root, root_velocity_w=root_velocity, cap_pose_w=cap,
            theta_rad=theta, spin_velocity_rad_s=qd[:, self.spin_id].clone(),
            axial_m=axial, opening_rad=self.config.thread_sign * (theta - self.theta_at_reset),
            mimic_error_m=axial - target_axial,
            physical_error_m=physical_axial - axial,
        )

    def reset(self, env_ids: torch.Tensor, *, root_pose_w: torch.Tensor,
              root_velocity_w: torch.Tensor, theta_rad: torch.Tensor,
              spin_velocity_rad_s: torch.Tensor | None = None) -> None:
        """Restore selected environments without a global physics step.

        The caller supplies one row per selected environment. Both screw DOFs
        and their velocities are derived from the same continuous spin angle
        and speed. The implicit angular drive target is aligned to the reset
        angle to prevent an initial breakaway impulse.
        """
        import warp as wp

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        count = ids.numel()
        if count == 0:
            return
        if (int(ids.min()) < 0 or int(ids.max()) >= self.num_envs
                or torch.unique(ids).numel() != count):
            raise ValueError("env_ids must be distinct and in range")
        if (root_pose_w.shape != (count, 7) or root_velocity_w.shape != (count, 6)
                or theta_rad.shape != (count,)):
            raise ValueError("Mimic Reset state shape does not match selected environments")
        speed = (torch.zeros_like(theta_rad) if spin_velocity_rad_s is None
                 else spin_velocity_rad_s)
        if speed.shape != (count,):
            raise ValueError("spin_velocity_rad_s shape mismatch")
        pose = root_pose_w.to(device=self.device, dtype=torch.float32)
        velocity = root_velocity_w.to(device=self.device, dtype=torch.float32)
        theta = theta_rad.to(device=self.device, dtype=torch.float32)
        speed = speed.to(device=self.device, dtype=torch.float32)
        if (not bool(torch.isfinite(torch.cat((pose, velocity), dim=-1)).all())
                or not bool(torch.isfinite(theta).all())
                or not bool(torch.isfinite(speed).all())
                or bool((theta.abs() > 2 * math.pi * self.config.thread_turns).any())
                or bool(((pose[:, 3:].square().sum(-1) - 1).abs() > 0.01).any())):
            raise ValueError("Mimic Reset state is nonfinite, out of range or has invalid quaternion")
        root_data = (wp.to_torch(self.view.get_root_transforms())
                     .reshape(self.num_envs, 7).clone())
        root_vel_data = (wp.to_torch(self.view.get_root_velocities())
                         .reshape(self.num_envs, 6).clone())
        q = wp.to_torch(self.view.get_dof_positions()).reshape(self.num_envs, 2).clone()
        qd = wp.to_torch(self.view.get_dof_velocities()).reshape(self.num_envs, 2).clone()
        targets = (wp.to_torch(self.view.get_dof_position_targets())
                   .reshape(self.num_envs, 2).clone())
        root_data[ids] = pose
        root_vel_data[ids] = velocity
        q[ids, self.spin_id] = theta
        q[ids, self.slide_id] = self.config.thread_sign * self.config.thread_pitch * theta / (2 * math.pi)
        qd[ids, self.spin_id] = speed
        qd[ids, self.slide_id] = self.config.thread_sign * self.config.thread_pitch * speed / (2 * math.pi)
        targets[ids, self.spin_id] = theta
        selected = wp.from_torch(ids.to(dtype=torch.int32))
        self.view.set_root_transforms(wp.from_torch(root_data), selected)
        self.view.set_dof_positions(wp.from_torch(q), selected)
        self.view.set_root_velocities(wp.from_torch(root_vel_data), selected)
        self.view.set_dof_velocities(wp.from_torch(qd), selected)
        self.view.set_dof_position_targets(wp.from_torch(targets), selected)
        self.physics_view.update_articulations_kinematic()
        self.theta_at_reset[ids] = theta
