"""Known Torch physics with two different task schemas for M4.3 algorithm verification."""

import torch

from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec, RobotSpec, TaskSpec
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.transition import TaskState, TransitionBatch
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig


class ToyRobot:
    spec = RobotSpec("damped_point", 2, 1, 2, 1, "bounded_acceleration_v1",
                     "position_velocity_v1", control_dt=0.2, physics_dt=0.2)

    @staticmethod
    def checkpoint_contract():
        return {"integration": "additive_delta_v1", "command": "tanh_v1",
                "context": "per_episode_damping_v1", "physics": "damped_point_v1"}

    @staticmethod
    def integrate(state, physical_delta, action, context):
        return state + physical_delta

    @staticmethod
    def physical_delta(state, next_state):
        return next_state - state

    @staticmethod
    def executed_command(state, action):
        return action.tanh()

    @staticmethod
    def dynamics_context(native_env):
        return native_env.context.clone()

    @staticmethod
    def next_dynamics_context(context, state, action, next_state, next_task_state):
        return context


class ToyGoalTask:
    termination_mode = "analytic"
    episode_length = 64
    success_hold_steps = 1

    def __init__(self, task_id: str):
        if task_id not in ("position", "position_velocity"):
            raise ValueError("Unknown Toy task")
        self.task_id = task_id
        goal_dim = 1 if task_id == "position" else 2
        self.spec = TaskSpec(task_id, 3 if goal_dim == 1 else 4, 2,
                             goal_dim, 1, goal_dim, 2 + goal_dim,
                             f"toy_{task_id}_goal_v1")

    def checkpoint_contract(self):
        return {"reward": "exp_position_error_v1", "position_weight": 2.0,
                "velocity_weight": 0.8 if self.spec.goal_dim == 2 else 0.0,
                "action_penalty": 0.005, "termination": "none_v1",
                "episode_length": self.episode_length,
                "success_threshold": 0.12}

    def initial_task_state(self, num_envs: int, device="cpu"):
        position = torch.where(torch.arange(num_envs, device=device) % 2 == 0, 0.8, -0.8).float()
        data = position[:, None]
        if self.spec.goal_dim == 2:
            velocity = torch.where(position > 0, 0.12, -0.12)
            data = torch.stack((position, velocity), dim=-1)
        return TaskState(torch.zeros(num_envs, device=device, dtype=torch.long),
                         torch.full((num_envs,), -1, device=device, dtype=torch.long),
                         torch.zeros(num_envs, device=device, dtype=torch.long), data)

    @staticmethod
    def advance(task_state):
        return TaskState(task_state.task_id, task_state.motion_id,
                         task_state.frame + 1, task_state.data)

    def observation(self, state, previous_action, task_state):
        base = torch.cat((state, previous_action.tanh()), dim=-1)
        return (base if self.spec.goal_dim == 1
                else torch.cat((base, task_state.data[:, 1:2]), dim=-1))

    def goal(self, state, task_state):
        return task_state.data[:, None]

    def critic_features(self, state, task_state):
        return torch.cat((state, task_state.data), dim=-1)

    def reward(self, state, next_state, action, next_task_state, previous_action=None):
        position_error = next_state[:, 0] - next_task_state.data[:, 0]
        penalty = 2 * position_error.square()
        if self.spec.goal_dim == 2:
            penalty = penalty + 0.8 * (next_state[:, 1] - next_task_state.data[:, 1]).square()
        return (-penalty).exp() - 0.005 * action.tanh().square().squeeze(-1)

    @staticmethod
    def soft_termination(next_state, next_task_state):
        return torch.zeros(len(next_state), device=next_state.device)

    def valid_imagination_start(self, task_state, horizon):
        return task_state.frame + horizon <= self.episode_length

    def episode_end(self, task_state):
        return task_state.frame >= self.episode_length

    def evaluate_metrics(self, trajectory):
        if isinstance(trajectory, list):
            last = trajectory[-1]
            trajectory = {"rewards": torch.stack([item.reward for item in trajectory]),
                          "final_error": (last.next_state[:, 0] - last.next_task_state.data[:, 0]).abs()}
        return {"reward_mean": float(trajectory["rewards"].mean()),
                "final_position_error_mean": float(trajectory["final_error"].mean()),
                "success_rate": float((trajectory["final_error"] < 0.12).float().mean())}

    @staticmethod
    def success_mask(transition):
        return (transition.next_state[:, 0] - transition.next_task_state.data[:, 0]).abs() < 0.12

    @staticmethod
    def reward_upper_bound():
        return 1.0


class ToyCollector:
    def __init__(self, task: ToyGoalTask, num_envs: int, *, exploration_std: float = 0.15):
        if num_envs < 2 or exploration_std < 0:
            raise ValueError("Toy collector needs multiple environments and nonnegative exploration")
        self.task, self.num_envs, self.exploration_std = task, num_envs, exploration_std
        self.state = torch.zeros(num_envs, 2)
        self.task_state = task.initial_task_state(num_envs)
        self.previous_action = torch.zeros(num_envs, 1)
        self.context = (0.85 + 0.1 * (torch.arange(num_envs) % 3).float() / 2)[:, None]
        initial = task.observation(self.state, self.previous_action, self.task_state)
        self.history = initial[:, None].repeat(1, task.spec.history_length, 1)

    def state_dict(self):
        return {name: getattr(self, name).clone() for name in
                ("state", "previous_action", "context", "history")} | {
                    "task_state": {name: getattr(self.task_state, name).clone() for name in
                                   ("task_id", "motion_id", "frame", "data")}}

    def load_state_dict(self, record):
        for name in ("state", "previous_action", "context", "history"):
            target, value = getattr(self, name), record[name]
            if target.shape != value.shape:
                raise ValueError(f"Toy collector {name} shape differs")
            target.copy_(value)
        self.task_state = TaskState(**record["task_state"])

    @torch.no_grad()
    def step(self, actor, *, policy="actor") -> TransitionBatch:
        if policy != "actor":
            raise ValueError("Toy physical source supports Actor collection only")
        action = actor(self.history, self.task.goal(self.state, self.task_state))
        if self.exploration_std:
            action = action + self.exploration_std * torch.randn_like(action)
        command = action.tanh()
        next_velocity = self.context * self.state[:, 1:2] + 0.25 * command
        next_position = self.state[:, :1] + 0.2 * next_velocity
        next_state = torch.cat((next_position, next_velocity), dim=-1)
        arrival = self.task.advance(self.task_state)
        truncated = self.task.episode_end(arrival)
        reward = self.task.reward(self.state, next_state, action, arrival)
        transition = TransitionBatch(
            self.state.clone(), action.clone(), self.previous_action.clone(),
            command.clone(), next_state.clone(), self.history.clone(),
            self.context.clone(), self.task_state, arrival,
            torch.zeros_like(truncated), truncated.clone(),
            torch.zeros(self.num_envs, dtype=torch.int8),
            torch.zeros(self.num_envs, dtype=torch.int32), reward.clone())
        self.state = torch.where(truncated[:, None], torch.zeros_like(next_state), next_state)
        self.task_state = TaskState(arrival.task_id, arrival.motion_id,
                                    torch.where(truncated, torch.zeros_like(arrival.frame), arrival.frame),
                                    arrival.data)
        self.previous_action = torch.where(truncated[:, None], torch.zeros_like(action), action)
        observation = self.task.observation(self.state, self.previous_action, self.task_state)
        advanced = torch.cat((self.history[:, 1:], observation[:, None]), dim=1)
        self.history = torch.where(truncated[:, None, None],
                                   observation[:, None].expand_as(advanced), advanced)
        return transition


class ToyPhysicalEvaluator:
    def __init__(self, task: ToyGoalTask, *, num_envs: int = 8):
        self.task, self.num_envs = task, num_envs

    @torch.no_grad()
    def evaluate(self, actor, *, steps: int = 32):
        collector = ToyCollector(self.task, self.num_envs, exploration_std=0)
        rewards = []
        for _ in range(steps):
            rewards.append(collector.step(actor).reward)
        final_error = (collector.state[:, 0] - collector.task_state.data[:, 0]).abs()
        return self.task.evaluate_metrics({"rewards": torch.stack(rewards),
                                           "final_error": final_error})


def make_toy_trainer(task_id: str, *, num_envs: int = 16,
                     world_hidden: int = 64, world_batch: int = 32,
                     imagination_batch: int = 16):
    task, robot = ToyGoalTask(task_id), ToyRobot()
    modules = NetworkFactory.build(robot.spec, task.spec, NetworkSpec(
        world_hidden=(world_hidden, world_hidden), goal_hidden=(32,),
        actor_hidden=(64, 32), critic_hidden=(32,), latent_dim=8))
    optimizers = {
        "world_model": torch.optim.Adam(modules.world_model.parameters(), lr=2e-3),
        "actor": torch.optim.Adam(modules.actor.parameters(), lr=2e-3),
        "critic": torch.optim.Adam(modules.critic.parameters(), lr=2e-3),
    }
    train_envs = max(1, int(num_envs * 0.75))
    return SuperMimicTrainer(
        collector=ToyCollector(task, num_envs),
        replay=SchemaReplay(128, num_envs, robot.spec, task.spec, "cpu"),
        robot=robot, task=task, modules=modules, optimizers=optimizers,
        trust_evaluator=ModelTrustEvaluator(TrustConfig(
            min_distinct_starts=1, max_rmse=(0.3, 0.7, 1.5),
            require_baseline_improvement=False)),
        config=TrainerConfig(world_batch=world_batch, imagination_batch=imagination_batch,
                             world_updates_per_iteration=4, world_warmup_updates=8,
                             trust_interval=1, validation_batch=16,
                             critic_epochs=2, critic_minibatches=4),
        train_env_range=(0, train_envs), validation_env_range=(train_envs, num_envs))
