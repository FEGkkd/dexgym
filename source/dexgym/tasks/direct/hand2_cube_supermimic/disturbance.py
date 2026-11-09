"""Reference interval velocity pushes and episode-based curriculum."""

import torch


class Disturbance:
    def __init__(self, env):
        self.env = env
        self.remaining = torch.empty(env.num_envs, device=env.device).uniform_(0.6, 1.8)
        self.adaptive_scale = 0.05
        self.success_scale = 0.0

    def reset(self, ids):
        e = self.env
        steps = e.episode_length_buf[ids].float()
        reached = steps >= 2000
        early = e.reset_terminated[ids] & ~reached
        delta = (reached.float() * 0.1 - early.float() * (1 - steps / 2000).clamp(0, 1) * 0.2).sum() / e.num_envs
        self.adaptive_scale = max(0.05, min(1.0, self.adaptive_scale + float(delta)))
        success = e.memory["count"][ids, 0] >= 3
        self.success_scale = max(
            0.0, min(1.0, self.success_scale + float((success.float() * 2 - 1).sum()) * 0.08 / e.num_envs)
        )
        self.remaining[ids] = torch.empty(len(ids), device=e.device).uniform_(0.6, 1.8)
        e.pert_direction[ids] = 0
        e.pert_velocity[ids] = 0

    def step(self):
        e = self.env
        self.remaining -= e.step_dt
        ids = (self.remaining <= 0).nonzero().flatten()
        if not len(ids):
            return
        self.remaining[ids] = torch.empty(len(ids), device=e.device).uniform_(0.6, 1.8)
        e.pert_direction[ids] = 0
        e.pert_velocity[ids] = 0
        ids = ids[e.episode_length_buf[ids] * e.step_dt >= 3.0]
        if not len(ids):
            return
        progress = max(
            0.0, min(1.0, (e.common_step_counter - 0.05 * e.max_common_steps) / max(0.75 * e.max_common_steps, 1.0))
        )
        maximum = 0.05 + 0.1 * progress * self.adaptive_scale
        direction = torch.randn(len(ids), 3, device=e.device)
        direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        velocity = direction * torch.empty(len(ids), 1, device=e.device).uniform_(0.05, maximum)
        e.pert_direction[ids] = direction
        e.pert_velocity[ids, :3] = velocity
        root_velocity = e.cube.data.root_link_vel_w.torch[ids].clone()
        root_velocity[:, :3] += velocity
        e.cube.write_root_link_velocity_to_sim_index(root_velocity=root_velocity, env_ids=ids)
