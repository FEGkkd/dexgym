from dataclasses import replace

import pytest
import torch

from dexgym.algorithms.supermimic.core.imagination import ImaginationEngine
from dexgym.algorithms.supermimic.core.models import StateDeltaWorldModel
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.transition import StartBatch
from dexgym.tasks.supermimic_toy import ToyCollector, ToyGoalTask, ToyRobot


def test_world_model_and_replay_distinguish_real_contexts():
    robot, task = ToyRobot(), ToyGoalTask("position")
    collector = ToyCollector(task, 2, exploration_std=0)
    replay = SchemaReplay(4, 2, robot.spec, task.spec, "cpu")
    transition = collector.step(lambda history, goal: torch.zeros(2, 1))
    replay.append(transition)
    window = replay.window_at(torch.tensor([0, 0]), torch.tensor([0, 1]), 8)
    assert torch.equal(window.dynamics_context[:, 0], transition.dynamics_context)
    model = StateDeltaWorldModel(robot.spec, NetworkSpec(world_hidden=(), normalize=False))
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].bias.zero_()
        model.network[0].weight[0, -1] = 1.0
    state = torch.zeros(2, 2)
    action = torch.zeros(2, 1)
    contexts = torch.tensor([[1.0], [2.0]])
    prediction = model(state, action, contexts, robot)
    assert torch.equal(prediction[:, 0], contexts[:, 0])
    loss, _ = model.autoregressive_loss(window, robot)
    assert loss > 0
    loss.backward()
    assert torch.isfinite(model.network[0].weight.grad).all()
    assert model.network[0].weight.grad[0, -1].abs() > 0


class _ChangingContextRobot(ToyRobot):
    spec = replace(ToyRobot.spec, context_evolution="adapter_transition")

    @staticmethod
    def next_dynamics_context(context, state, action, next_state, task_state):
        return context + 1


class _RecordingWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.contexts = []

    def forward(self, state, action, context, robot):
        self.contexts.append(context.clone())
        return state + torch.cat((0.1 * context, torch.zeros_like(context)), dim=-1)


class _ConstantActor(torch.nn.Module):
    def forward(self, history, goal):
        return torch.zeros(len(history), 1)


class _ZeroCritic(torch.nn.Module):
    def forward(self, features):
        return features[:, 0] * 0


def test_imagination_advances_time_varying_context_and_rejects_missing_contract():
    robot, task = _ChangingContextRobot(), ToyGoalTask("position")
    task_state = task.initial_task_state(1)
    state = torch.zeros(1, 2)
    previous = torch.zeros(1, 1)
    observation = task.observation(state, previous, task_state)
    start = StartBatch(state, observation[:, None].repeat(1, 2, 1), torch.ones(1, 1),
                       task_state, previous, torch.zeros(1, dtype=torch.long),
                       torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.int8))
    world = _RecordingWorld()
    ImaginationEngine().rollout(start, _ConstantActor(), world, _ZeroCritic(),
                                robot, task, horizon=2)
    assert [float(value[0, 0]) for value in world.contexts] == [1.0, 2.0]
    robot.next_dynamics_context = None
    with pytest.raises(ValueError, match="next_dynamics_context"):
        ImaginationEngine().rollout(start, _ConstantActor(), world, _ZeroCritic(),
                                    robot, task, horizon=2)
