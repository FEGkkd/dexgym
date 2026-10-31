"""GPU batch of the ideal xyzw wrist command trajectory.

The caller writes the returned bounded root pose and velocity at each PhysX
substep. Fingers and bottle remain ordinary dynamic PhysX bodies.
"""

from __future__ import annotations

import torch

from .wrist_pose_control import WristPoseLimits


def _normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def _mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1),
                      aw * bw - (av * bv).sum(-1, keepdim=True)), dim=-1)


def _exp(v: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    half = angle * 0.5
    scale = torch.where(angle > 1e-12, torch.sin(half) / angle.clamp_min(1e-12),
                        0.5 - angle.square() / 48.0)
    return torch.cat((v * scale, torch.cos(half)), dim=-1)


def _error(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    inverse = torch.cat((-current[..., :3], current[..., 3:]), dim=-1)
    diff = _mul(target, inverse)
    diff = torch.where(diff[..., 3:] < 0, -diff, diff)
    radius = torch.linalg.vector_norm(diff[..., :3], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(radius, diff[..., 3:].clamp_min(0))
    return torch.where(radius > 1e-10, diff[..., :3] * angle / radius.clamp_min(1e-12),
                       torch.zeros_like(diff[..., :3]))


class BatchedWristPoseController:
    """Independent bounded SE(3) trajectories for N parallel environments."""

    def __init__(self, num_envs: int, *, device: str, dtype=torch.float32,
                 limits: WristPoseLimits | None = None):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.limits = limits or WristPoseLimits()
        self.initial_pos = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self.current_pos = self.initial_pos.clone()
        self.target_pos = self.initial_pos.clone()
        unit = torch.tensor([0., 0., 0., 1.], device=device, dtype=dtype)
        self.current_quat = unit.repeat(num_envs, 1)
        self.target_quat = self.current_quat.clone()
        self.current_velocity = torch.zeros(num_envs, 6, device=device, dtype=dtype)
        self.workspace_clip_count = torch.zeros(num_envs, device=device, dtype=torch.long)

    @property
    def current_pose(self) -> torch.Tensor:
        return torch.cat((self.current_pos, self.current_quat), dim=-1)

    @property
    def target_pose(self) -> torch.Tensor:
        return torch.cat((self.target_pos, self.target_quat), dim=-1)

    def reset(self, env_ids: torch.Tensor, position: torch.Tensor, quaternion_xyzw: torch.Tensor,
              *, target_position: torch.Tensor | None = None,
              target_quaternion: torch.Tensor | None = None) -> None:
        ids = env_ids.to(device=self.device, dtype=torch.long)
        pos = position.to(device=self.device, dtype=self.dtype)
        quat = _normalize(quaternion_xyzw.to(device=self.device, dtype=self.dtype))
        if pos.shape != (len(ids), 3) or quat.shape != (len(ids), 4):
            raise ValueError("Wrist reset tensors must match selected environment count")
        self.initial_pos[ids] = pos
        self.current_pos[ids] = pos
        self.target_pos[ids] = pos if target_position is None else target_position.to(self.device)
        self.current_quat[ids] = quat
        self.target_quat[ids] = quat if target_quaternion is None else _normalize(target_quaternion.to(self.device))
        self.current_velocity[ids] = 0
        self.workspace_clip_count[ids] = 0

    def command(self, increments: torch.Tensor) -> torch.Tensor:
        """World translation, then wrist-local rotation; return clipped-row mask."""
        if increments.shape != (self.num_envs, 6):
            raise ValueError("Expected one 6D wrist increment per environment")
        inc = increments.to(device=self.device, dtype=self.dtype)
        proposed = self.target_pos + inc[:, :3]
        proposed_z = proposed[:, 2].clone()
        proposed[:, 2] = proposed[:, 2].clamp_min(self.limits.min_height)
        delta = proposed - self.initial_pos
        radius = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = (self.limits.max_target_radius / radius.clamp_min(1e-12)).clamp(max=1)
        proposed = self.initial_pos + delta * scale
        clipped = (proposed_z < self.limits.min_height) | (radius[:, 0] > self.limits.max_target_radius)
        self.workspace_clip_count += clipped.long()
        self.target_pos = proposed
        self.target_quat = _normalize(_mul(self.target_quat, _exp(inc[:, 3:])))
        return clipped

    def step(self, physics_dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        if physics_dt <= 0:
            raise ValueError("Physics dt must be positive")
        dp = self.target_pos - self.current_pos
        distance = torch.linalg.vector_norm(dp, dim=-1, keepdim=True)
        move = dp * (self.limits.max_linear_speed * physics_dt / distance.clamp_min(1e-12)).clamp(max=1)
        dr = _error(self.target_quat, self.current_quat)
        angle = torch.linalg.vector_norm(dr, dim=-1, keepdim=True)
        rotation = dr * (self.limits.max_angular_speed * physics_dt / angle.clamp_min(1e-12)).clamp(max=1)
        self.current_pos = self.current_pos + move
        self.current_quat = _normalize(_mul(_exp(rotation), self.current_quat))
        self.current_velocity = torch.cat((move, rotation), dim=-1) / physics_dt
        return self.current_pose, self.current_velocity

    def at_target(self) -> torch.Tensor:
        return ((torch.linalg.vector_norm(self.target_pos - self.current_pos, dim=-1)
                 <= self.limits.position_tolerance)
                & (torch.linalg.vector_norm(_error(self.target_quat, self.current_quat), dim=-1)
                   <= self.limits.rotation_tolerance))
