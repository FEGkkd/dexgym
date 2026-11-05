"""Stateful Newton collector for the alternating SuperMimic loop."""

import torch


class AlignedNewtonCollector:
    def __init__(self, env, actor, replay, *, action_noise_std=0.0):
        if action_noise_std < 0:
            raise ValueError("Action exploration noise must be nonnegative")
        self.env = env
        self.native = env.unwrapped
        self.actor = actor
        self.replay = replay
        self.action_noise_std = action_noise_std
        self.obs, _ = env.reset()
        self.steps = 0
        self.terminations = 0
        self.truncations = 0
        self.contact_physics_steps = 0
        self.reward_sum = 0.0

    @torch.no_grad()
    def step(self):
        """Collect one 50 Hz policy step from all Newton environments."""
        native = self.native
        q = native.robot.data.joint_pos.torch[:, native.arm_ids]
        qd = native.robot.data.joint_vel.torch[:, native.arm_ids]
        state = torch.cat((q, qd, native.command), dim=-1).detach().clone()
        history = self.obs["policy"][:, :560].reshape(native.num_envs, 10, 56).clone()
        frame = native.motion.frame.clone()
        goal = self.obs["policy"][:, 560:].reshape(native.num_envs, 10, 90)
        action = self.actor(history, goal)
        if self.action_noise_std:
            action = action + torch.randn_like(action) * self.action_noise_std
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated | truncated
        self.replay.append(
            state,
            action,
            native.last_physical_arm_state,
            history,
            frame,
            done,
            terminated=terminated,
        )
        self.steps += 1
        self.terminations += int(terminated.sum())
        self.truncations += int(truncated.sum())
        self.contact_physics_steps += int(native.last_policy_contact_counts.sum())
        self.reward_sum += float(reward.mean())
        return {
            "reward_mean": float(reward.mean()),
            "terminations": int(terminated.sum()),
            "truncations": int(truncated.sum()),
            "contact_physics_steps": int(native.last_policy_contact_counts.sum()),
        }
