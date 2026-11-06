from types import SimpleNamespace

import torch

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.tasks.marvin_supermimic.core_plugins import MarvinCoreRobot
from dexgym.tasks.marvin_supermimic.se3_reach_hold import (
    MarvinBimanualSE3ReachHold, pack_dual_tcp_pose)


def test_physical_probe_is_a_one_step_masked_transition_without_joint_goal():
    from dexgym.tasks.marvin_supermimic.se3_probe import make_probe_transition
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES),
                            "cpu", action_schema="arm_joint_delta_v1")
    task = MarvinBimanualSE3ReachHold(robot)
    q = robot.action_map.origin.expand(2, -1).clone()
    state = torch.cat((q, torch.zeros_like(q), q), dim=-1)
    position, rotation = robot.fk(q)
    goal = pack_dual_tcp_pose(position[:, :, 2], rotation[:, :, 2])
    task_state = task.task_state_from_goal(goal)
    previous_action = torch.zeros((2, 14))
    action = torch.full((2, 14), 0.1)
    next_state = robot.integrate(state, torch.zeros((2, 28)), action,
                                 torch.zeros((2, 0)))
    observation = task.observation(state, previous_action, task_state)
    history = observation[:, None].expand(-1, 10, -1).clone()
    transition = make_probe_transition(
        robot, task, state=state, action=action, previous_action=previous_action,
        next_state=next_state, history=history, task_state=task_state,
        dynamics_context=torch.zeros((2, 0)),
        terminated=torch.tensor([False, True]),
        failure_reason=torch.tensor([0, 8], dtype=torch.int32))
    transition.validate(robot.spec, task.spec, num_envs=2)
    assert transition.source_id.tolist() == [4, 4]
    assert transition.truncated.tolist() == [True, False]
    assert transition.done.tolist() == [True, True]
    assert torch.equal(transition.next_state, next_state)
    assert transition.task_state.data.shape == (2, 18)
    assert not hasattr(transition, "q_goal")
    replay = SchemaReplay(8, 2, robot.spec, task.spec, "cpu")
    replay.append(transition)
    replay.append(transition)
    window = replay.window_at(torch.tensor([0, 0]), torch.tensor([0, 1]), 8)
    assert window.mask[:, 0].tolist() == [1.0, 1.0]
    assert window.mask[:, 1:].sum() == 0
