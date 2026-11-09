"""Collect the existing standalone Newton task before DirectRLEnv auto-reset."""

import torch

from dexgym.algorithms.supermimic.core.transition import TaskState, TransitionBatch


class HandNewtonCollector:
    def __init__(self, env, robot, task, *, action_noise_std=.1):
        self.env, self.native, self.robot, self.task = env, env.unwrapped, robot, task
        self.num_envs, self.action_noise_std = self.native.num_envs, action_noise_std
        self.observation_noise = None
        if not getattr(self.native.cfg, "capture_model_transitions", False):
            raise ValueError("Enable pre-reset snapshots for model learning")
        env.reset()
        self.previous_action = torch.zeros((self.num_envs, 20), device=self.native.device)
        state = robot.physical_state(self.native)
        obs = task.observation(state, self.previous_action, task.physical_task_state(self.native))
        self.history = obs[:, None].expand(-1, task.spec.history_length, -1).clone()

    @torch.no_grad()
    def step(self, actor, *, policy="actor"):
        if policy != "actor":
            raise ValueError("Hand S0 strict-pure training collects only noisy Actor transitions")
        env, task, robot = self.native, self.task, self.robot
        state = robot.physical_state(env).detach().clone()
        old = task.physical_task_state(env)
        previous = self.previous_action.clone()
        history = self.history.clone()
        action = actor(history, task.goal(state, old))
        action = action + self.action_noise_std * torch.randn_like(action)
        _, reward, terminated, truncated, _ = self.env.step(action.tanh().clone())
        arrival = TaskState(old.task_id, old.motion_id, env.last_hand_control_frame.clone(),
                            env.last_hand_goal_data.clone())
        next_state = env.last_hand_state.clone()
        command = robot.executed_command(state, action)
        if not torch.allclose(command, next_state[:, 40:60], atol=1e-6, rtol=1e-6):
            raise RuntimeError("Hand physical/model command mismatch")
        if not torch.equal(task.hard_termination(next_state, arrival), terminated):
            raise RuntimeError("Hand physical/model terminal mismatch")
        exact_reward = task.exact_reward(next_state, old)
        if not torch.allclose(exact_reward, reward, atol=2e-3, rtol=2e-3):
            raise RuntimeError(f"Hand reward mismatch: {float((exact_reward - reward).abs().max())}")
        transition = TransitionBatch(
            state, action.clone(), previous, command, next_state, history,
            robot.dynamics_context(env).clone(), old, arrival, terminated.clone(), truncated.clone(),
            torch.zeros(self.num_envs, dtype=torch.int8, device=state.device),
            env.last_hand_failure_reason.clone(), reward.clone())
        done = terminated | truncated
        self.previous_action = torch.where(done[:, None], 0., action)
        post = robot.physical_state(env)
        obs = task.observation(post, self.previous_action, task.physical_task_state(env))
        advanced = torch.cat((self.history[:, 1:], obs[:, None]), 1)
        self.history = torch.where(done[:, None, None], obs[:, None].expand_as(advanced), advanced)
        self.last_metrics = {"mean_tip_error_m": float(env.last_hand_errors.mean()),
                             "worst_tip_mean_m": float(env.last_hand_errors.max(-1).values.mean()),
                             "reached": int(env.last_hand_reached.sum()),
                             "action_abs_mean": float(action.tanh().abs().mean())}
        return transition
