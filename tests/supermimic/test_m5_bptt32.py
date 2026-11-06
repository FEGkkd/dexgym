from types import SimpleNamespace

import torch

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.algorithms.supermimic.core.imagination import ImaginationEngine
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.optimization import frozen_parameters
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.transition import StartBatch
from dexgym.tasks.marvin_supermimic.core_plugins import MarvinCoreRobot
from dexgym.tasks.marvin_supermimic.se3_reach_hold import (
    MarvinBimanualSE3ReachHold, pack_dual_tcp_pose)


class KnownDifferentiablePhysics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.005))

    def forward(self, state, action, context, robot):
        delta = torch.cat((self.gain * action.tanh(), torch.zeros_like(action)), dim=-1)
        return robot.integrate(state, delta, action, context)


class ZeroCritic(torch.nn.Module):
    def forward(self, features):
        return features[:, 0] * 0


def test_se3_task_32_step_bptt_reaches_actor_while_world_model_is_frozen():
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), "cpu",
                            action_schema="arm_joint_delta_v1")
    task = MarvinBimanualSE3ReachHold(robot)
    network = NetworkSpec(world_hidden=(16,), goal_hidden=(16,),
                          actor_hidden=(32,), critic_hidden=(16,), latent_dim=8,
                          normalize=False)
    actor = NetworkFactory.build(robot.spec, task.spec, network, robot_adapter=robot).actor
    world_model = KnownDifferentiablePhysics()
    home = robot.action_map.origin[None]
    state = torch.cat((home, torch.zeros_like(home), home), dim=-1)
    witness = home.clone()
    witness[:, 0] += 0.05
    witness[:, 7] -= 0.05
    target_position, target_rotation = robot.fk(witness)
    task_state = task.task_state_from_goal(pack_dual_tcp_pose(
        target_position[:, :, 2], target_rotation[:, :, 2]))
    observation = task.observation(state, torch.zeros_like(home), task_state)
    start = StartBatch(state, observation[:, None].expand(-1, 10, -1).clone(),
                       state.new_empty((1, 0)), task_state, torch.zeros_like(home),
                       torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long),
                       torch.zeros(1, dtype=torch.int8))
    with frozen_parameters(world_model):
        rollout = ImaginationEngine().rollout(start, actor, world_model, ZeroCritic(),
                                              robot, task, horizon=32)
        rollout.actor_loss.backward()
    assert rollout.rewards.shape == (32, 1)
    assert world_model.gain.grad is None
    gradients = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
    assert gradients and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 1e-7
