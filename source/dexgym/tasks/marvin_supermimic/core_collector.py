"""Task-injected real Newton transition collector for the SuperMimic Core."""

import torch

from dexgym.algorithms.supermimic.core.transition import TransitionBatch


class CoreNewtonCollector:
    def __init__(self, env, robot, task, *, action_noise_std=0.0):
        if action_noise_std < 0:
            raise ValueError("Action noise must be nonnegative")
        self.env, self.native, self.robot, self.task = env, env.unwrapped, robot, task
        self.num_envs = self.native.num_envs
        self.action_noise_std = action_noise_std
        env.reset()
        state = robot.physical_state(self.native)
        task_state = task.physical_task_state(self.native)
        observation = task.observation(state, self.native.last_action, task_state)
        self.history = observation[:, None].expand(-1, task.spec.history_length, -1).clone()

    @torch.no_grad()
    def step(self, actor, *, policy="actor"):
        native, task = self.native, self.task
        if policy not in ("actor", "reference_pd", "boundary_probe"):
            raise ValueError("Unknown Newton collection source")
        state = self.robot.physical_state(native).detach().clone()
        task_state = task.physical_task_state(native)
        previous_action = native.last_action.detach().clone()
        history = self.history.clone()
        if policy == "actor":
            action = actor(history, task.goal(state, task_state))
        elif policy == "reference_pd":
            if not hasattr(task, "physical_bootstrap_action"):
                raise ValueError("Task has no reference PD physical sampling policy")
            action = task.physical_bootstrap_action(native)
        else:
            action = 1.5 * torch.sign(torch.randn(self.num_envs, self.robot.spec.action_dim,
                                                   device=state.device))
        if self.action_noise_std:
            action = action + self.action_noise_std * torch.randn_like(action)
        _, native_reward, terminated, truncated, _ = self.env.step(action)
        arrival = task.advance(task_state)
        next_state = native.last_physical_arm_state.detach().clone()
        reward = task.reward(state, next_state, action, arrival, previous_action)
        if not torch.allclose(reward, native_reward, atol=5e-4, rtol=5e-3):
            raise RuntimeError("Newton and Task MDP reward time or state differs")
        transition = TransitionBatch(
            state, action.clone(), previous_action, next_state[:, 28:42].clone(),
            next_state, history, torch.empty((self.num_envs, self.robot.spec.context_dim),
                                             device=state.device),
            task_state, arrival, terminated.clone(), truncated.clone(),
            torch.full((self.num_envs,), {"actor": 0, "reference_pd": 1,
                                          "boundary_probe": 2}[policy],
                       dtype=torch.int8, device=state.device),
            native.last_failure_reason.clone().to(torch.int32), reward)
        post_state = self.robot.physical_state(native)
        post_task = task.physical_task_state(native)
        observation = task.observation(post_state, native.last_action, post_task)
        advanced = torch.cat((self.history[:, 1:], observation[:, None]), dim=1)
        self.history = torch.where((terminated | truncated)[:, None, None],
                                   observation[:, None].expand_as(advanced), advanced)
        return transition
