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
        self.boundary_sign = torch.where(
            torch.rand(self.native.num_envs, 14, device=self.native.device) < 0.5,
            -torch.ones(self.native.num_envs, 14, device=self.native.device),
            torch.ones(self.native.num_envs, 14, device=self.native.device))

    @torch.no_grad()
    def step(self, *, policy: str = "actor"):
        """Collect one 50 Hz policy step from all Newton environments."""
        if policy not in ("actor", "reference_pd", "boundary_probe"):
            raise ValueError("Collection policy must be actor, reference_pd or boundary_probe")
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
        elif policy == "reference_pd":
            action = reference_pd_action(native, frame)
        else:
            action = 1.5 * self.boundary_sign
        if self.action_noise_std:
            action = action + torch.randn_like(action) * self.action_noise_std
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated | truncated
        if bool(done.any()):
            self.boundary_sign[done] = torch.where(
                torch.rand((int(done.sum()), 14), device=native.device) < 0.5, -1.0, 1.0)
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
            source_id=torch.full_like(frame, {"actor": 0, "reference_pd": 1,
                                              "boundary_probe": 2}[policy]),
            failure_reason=getattr(native, "last_failure_reason", torch.zeros_like(frame)),
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
            "source_id": {"actor": 0, "reference_pd": 1, "boundary_probe": 2}[policy],
        }
