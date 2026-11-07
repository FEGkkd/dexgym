"""One alternating Core trainer drives two incompatible Goal Schemas."""

import pytest
import torch

from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec, RobotSpec, TaskSpec
from dexgym.algorithms.supermimic.core.trainer import SuperMimicTrainer, TrainerConfig
from dexgym.algorithms.supermimic.core.transition import TaskState, TransitionBatch
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig


class ToyRobot:
    spec = RobotSpec("toy", 1, 1, 1, 0, "toy_action", "toy_state")

    @staticmethod
    def integrate(state, delta, action, context):
        return state + delta

    @staticmethod
    def physical_delta(state, next_state):
        return next_state - state


class GoalTask:
    termination_mode = "analytic"

    def __init__(self, goal_dim):
        self.spec = TaskSpec(f"toy_goal_{goal_dim}", goal_dim + 1, 2, goal_dim,
                             1, goal_dim, goal_dim + 1, f"goal_{goal_dim}")

    @staticmethod
    def advance(task_state):
        return TaskState(task_state.task_id, task_state.motion_id,
                         task_state.frame + 1, task_state.data)

    def observation(self, state, previous_action, task_state):
        return torch.cat((state, task_state.data), dim=-1)

    def goal(self, state, task_state):
        return task_state.data[:, None]

    def critic_features(self, state, task_state):
        return torch.cat((state, task_state.data), dim=-1)

    def reward(self, state, next_state, action, next_task_state, previous_action=None):
        return -((next_state[:, 0] - next_task_state.data[:, 0]) ** 2)

    def soft_termination(self, next_state, next_task_state):
        return torch.zeros(len(next_state))

    def valid_imagination_start(self, task_state, horizon):
        return torch.ones_like(task_state.frame, dtype=torch.bool)


class ToyCollector:
    num_envs = 8

    def __init__(self, task):
        self.task = task
        self.state = torch.zeros(8, 1)
        self.previous_action = torch.zeros(8, 1)
        self.task_state = TaskState(torch.zeros(8, dtype=torch.long),
                                    torch.zeros(8, dtype=torch.long),
                                    torch.zeros(8, dtype=torch.long),
                                    torch.ones(8, task.spec.goal_dim))
        first = task.observation(self.state, self.previous_action, self.task_state)
        self.history = first[:, None].repeat(1, task.spec.history_length, 1)

    @torch.no_grad()
    def step(self, actor, *, policy="actor"):
        assert policy == "actor"
        goal = self.task.goal(self.state, self.task_state)
        action = actor(self.history, goal) + 0.05 * torch.randn(8, 1)
        next_state = self.state + 0.2 * action
        next_task_state = self.task.advance(self.task_state)
        reward = self.task.reward(self.state, next_state, action, next_task_state)
        transition = TransitionBatch(
            self.state.clone(), action.clone(), self.previous_action.clone(),
            action.clone(), next_state.clone(), self.history.clone(),
            torch.empty(8, 0), self.task_state, next_task_state,
            torch.zeros(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool),
            torch.zeros(8, dtype=torch.int8), torch.zeros(8, dtype=torch.int32),
            reward.clone())
        self.state = next_state
        self.task_state = next_task_state
        self.previous_action = action
        observation = self.task.observation(next_state, action, next_task_state)
        self.history = torch.cat((self.history[:, 1:], observation[:, None]), dim=1)
        return transition


def test_reset_start_fraction_is_explicit_validated_trainer_config():
    configured = TrainerConfig(imagination_initial_frame_fraction=0.5,
                               imagination_initial_frame_max=40)
    assert configured.imagination_initial_frame_fraction == 0.5
    with pytest.raises(ValueError, match="initial frame"):
        TrainerConfig(imagination_initial_frame_fraction=1.1,
                      imagination_initial_frame_max=40)
    with pytest.raises(ValueError, match="initial frame"):
        TrainerConfig(imagination_initial_frame_fraction=0.5,
                      imagination_initial_frame_max=-1)


@pytest.mark.parametrize("goal_dim", [1, 3])
def test_shared_trainer_updates_world_actor_critic_without_validation_norm_leak(goal_dim):
    torch.manual_seed(17)
    robot, task = ToyRobot(), GoalTask(goal_dim)
    modules = NetworkFactory.build(robot.spec, task.spec, NetworkSpec(
        world_hidden=(32,), goal_hidden=(16,), actor_hidden=(16,),
        critic_hidden=(16,), latent_dim=4))
    optimizers = {
        "world_model": torch.optim.Adam(modules.world_model.parameters(), lr=1e-3),
        "actor": torch.optim.Adam(modules.actor.parameters(), lr=1e-3),
        "critic": torch.optim.Adam(modules.critic.parameters(), lr=1e-3),
    }
    trainer = SuperMimicTrainer(
        collector=ToyCollector(task), replay=SchemaReplay(64, 8, robot.spec, task.spec, "cpu"),
        robot=robot, task=task, modules=modules, optimizers=optimizers,
        trust_evaluator=ModelTrustEvaluator(TrustConfig(
            min_distinct_starts=1, max_rmse=(2.0, 2.0, 2.0), require_baseline_improvement=False)),
        config=TrainerConfig(world_batch=8, imagination_batch=4,
                             world_updates_per_iteration=2, world_warmup_updates=2,
                             trust_interval=1, critic_epochs=1, critic_minibatches=2),
        train_env_range=(0, 6), validation_env_range=(6, 8))
    for _ in range(40):
        trainer.run_iteration()
    assert trainer.counts["world_updates"] > 0
    assert trainer.counts["actor_updates"] > 0
    assert trainer.counts["critic_updates"] > 0
    assert trainer.allowed_horizon == 32
    assert int(modules.world_model.input_norm.count) == 40 * 6
    assert int(modules.actor.observation_norm.count) == 40 * 6
    assert int(modules.critic.input_norm.count) == 40 * 6
    assert sum(trainer.counts[key] for key in
               ("nonfinite_world", "nonfinite_actor", "nonfinite_critic")) == 0
