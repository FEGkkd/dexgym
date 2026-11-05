"""Stateful Newton collector for the alternating SuperMimic loop."""

import torch
from dexgym.algorithms.supermimic.transition import ReferenceTimeIndex


def reference_pd_action(native, frame: torch.Tensor) -> torch.Tensor:
    """Raw action for physical reference sampling; never an Actor supervision target."""
    arrival = ReferenceTimeIndex(frame, torch.full_like(frame, native.motion.length)).next_frame()
    target = native.motion.data["q_arm"][arrival]
    normalized = ((target - native.action_map.origin) / native.action_map.scale).clamp(-0.98, 0.98)
    return torch.atanh(normalized)


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
    def step(self, *, policy: str = "actor"):
        """Collect one 50 Hz policy step from all Newton environments."""
        if policy not in ("actor", "reference_pd"):
            raise ValueError("Collection policy must be actor or reference_pd")
        native = self.native
        q = native.robot.data.joint_pos.torch[:, native.arm_ids]
        qd = native.robot.data.joint_vel.torch[:, native.arm_ids]
        state = torch.cat((q, qd, native.command), dim=-1).detach().clone()
        history = self.obs["policy"][:, :560].reshape(native.num_envs, 10, 56).clone()
        frame = native.motion.frame.clone()
        previous_action = native.last_action.detach().clone()
        goal = self.obs["policy"][:, 560:].reshape(native.num_envs, 10, 90)
        if policy == "actor":
            action = self.actor(history, goal)
        else:
            action = reference_pd_action(native, frame)
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
            truncated=truncated,
            previous_action=previous_action,
            executed_command=native.last_physical_arm_state[:, 28:42],
            source_id=torch.full_like(frame, 0 if policy == "actor" else 1),
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
            "source_id": 0 if policy == "actor" else 1,
        }
