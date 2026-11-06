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
        context_provider = getattr(self.robot, "dynamics_context", None)
        if not callable(context_provider):
            raise ValueError("Robot Adapter must provide dynamics_context from current physics")
        context = context_provider(native)
        if (not isinstance(context, torch.Tensor)
                or context.shape != (self.num_envs, self.robot.spec.context_dim)
                or context.device != state.device or context.dtype != state.dtype
                or not bool(torch.isfinite(context).all())):
            raise ValueError("Robot dynamics_context has invalid shape, dtype, device or values")
        context = context.detach().clone()
        task_state = task.physical_task_state(native)
        previous_action = native.last_action.detach().clone()
        history = self.history.clone()
        expected_observation = task.observation(state, previous_action, task_state)
        if not torch.allclose(history[:, -1], expected_observation, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Newton and Task MDP observation history differs")
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
        # DirectRLEnv may mutate its input when a terminal environment resets.
        # Preserve the actual command source for Reward, parity and Replay.
        _, native_reward, terminated, truncated, _ = self.env.step(action.clone())
        arrival = task.advance(task_state)
        next_state = native.last_physical_arm_state.detach().clone()
        command = self.robot.executed_command(state, action).detach().clone()
        if not torch.allclose(command, next_state[:, 28:42], atol=1e-6, rtol=1e-6):
            delta = (command - next_state[:, 28:42]).abs()
            env_id, joint_id = divmod(int(delta.argmax()), delta.shape[1])
            raise RuntimeError(
                "Newton and Robot Adapter executed command differs: "
                f"env={env_id} joint={joint_id} delta={float(delta[env_id, joint_id]):.8f} "
                f"state_command={float(state[env_id, 28 + joint_id]):.8f} "
                f"action={float(action[env_id, joint_id]):.8f} "
                f"expected={float(command[env_id, joint_id]):.8f} "
                f"physics={float(next_state[env_id, 28 + joint_id]):.8f} "
                f"post_reset_command={float(native.command[env_id, joint_id]):.8f} "
                f"terminated={bool(terminated[env_id])} truncated={bool(truncated[env_id])}"
            )
        hard_termination = task.hard_termination(
            next_state, arrival, native.last_policy_contact_counts)
        if not torch.equal(hard_termination, terminated):
            raise RuntimeError("Newton and Task MDP hard termination differs")
        current_task_state = task.physical_task_state(native)
        live = ~(terminated | truncated)
        if not torch.equal(current_task_state.frame[live], arrival.frame[live]):
            raise RuntimeError("Newton and Task MDP reference time differs")
        reward = task.reward(state, next_state, action, arrival, previous_action)
        if not torch.allclose(reward, native_reward, atol=5e-4, rtol=5e-3):
            raise RuntimeError("Newton and Task MDP reward time or state differs")
        transition = TransitionBatch(
            state, action.clone(), previous_action,
            command,
            next_state, history, context,
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
