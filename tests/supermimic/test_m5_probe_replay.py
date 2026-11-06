from types import SimpleNamespace
from dataclasses import asdict

import pytest
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


def test_paired_probe_layout_keeps_independent_start_groups():
    from dexgym.tasks.marvin_supermimic.se3_probe import paired_probe_actions
    actions, groups = paired_probe_actions(61, 14, epsilon=0.1, device="cpu")
    assert actions.shape == (61, 14)
    assert groups.tolist() == [0] * 30 + [1] * 30 + [2]
    assert torch.count_nonzero(actions[0]) == 0
    assert torch.count_nonzero(actions[29]) == 0
    assert torch.count_nonzero(actions[30]) == 0
    assert torch.count_nonzero(actions[59]) == 0
    assert torch.count_nonzero(actions[60]) == 0
    assert torch.allclose(actions[1:15], 0.1 * torch.eye(14))
    assert torch.allclose(actions[15:29], -0.1 * torch.eye(14))
    assert torch.allclose(actions[31:45], 0.1 * torch.eye(14))
    assert torch.allclose(actions[45:59], -0.1 * torch.eye(14))


def test_probe_model_data_rejects_teacher_or_different_dynamics_contract():
    from dexgym.tasks.marvin_supermimic.se3_probe import validate_probe_metadata
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES),
                            "cpu", action_schema="arm_joint_delta_v1")
    task = MarvinBimanualSE3ReachHold(robot)
    metadata = {"kind": "strict_pure_newton_paired_action_probe",
                "robot": asdict(robot.spec), "task": asdict(task.spec),
                "robot_contract": robot.checkpoint_contract(),
                "task_contract": task.checkpoint_contract(),
                "teacher_or_ik_used": False,
                "goal_joint_witness_stored": False}
    validate_probe_metadata(metadata, robot, task)
    with pytest.raises(ValueError, match="teacher or IK"):
        validate_probe_metadata({**metadata, "teacher_or_ik_used": True}, robot, task)
    with pytest.raises(ValueError, match="Robot"):
        validate_probe_metadata({**metadata, "robot": {**metadata["robot"],
                           "action_schema": "robot_fixed_v3"}}, robot, task)
    with pytest.raises(ValueError, match="Robot"):
        validate_probe_metadata({**metadata, "robot_contract": {
            **metadata["robot_contract"], "schema_version": "wrong"}}, robot, task)
