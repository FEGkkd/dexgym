"""50 Hz hold speed is distinct from Newton's 500 Hz servo velocity."""

from types import SimpleNamespace

import torch

from dexgym.tasks.marvin_supermimic.se3_reach_hold import (
    MarvinBimanualSE3ReachHold, SE3RewardConfig, pack_dual_tcp_pose)
from test_m5_se3_task import SimpleFKRobot


def _transition(task, *, delta=0.001, instantaneous_speed=0.5):
    state = torch.zeros(1, 42)
    next_state = state.clone()
    next_state[:, [0, 7]] = delta
    next_state[:, 14:28] = instantaneous_speed
    goal = pack_dual_tcp_pose(torch.zeros(1, 2, 3),
                              torch.eye(3).expand(1, 2, 3, 3))
    return SimpleNamespace(state=state, next_state=next_state,
                           next_task_state=task.task_state_from_goal(goal),
                           terminated=torch.zeros(1, dtype=torch.bool))


def test_policy_step_hold_speed_accepts_low_net_motion_and_versions_checkpoint():
    robot = SimpleFKRobot()
    old = MarvinBimanualSE3ReachHold(robot)
    new = MarvinBimanualSE3ReachHold(robot, hold_speed_mode="policy_step_displacement_v2")
    assert not bool(old.success_mask(_transition(old))[0])
    assert bool(new.success_mask(_transition(new))[0])
    assert old.checkpoint_contract() != new.checkpoint_contract()
    assert old.checkpoint_contract()["task_adapter_version"] == "marvin_bimanual_se3_reach_hold_v1"
    assert new.checkpoint_contract()["task_adapter_version"] == "marvin_bimanual_se3_reach_hold_v2"


def test_hold_reward_uses_same_velocity_contract_and_preserves_physical_qd():
    weights = SE3RewardConfig(position_weight=0, rotation_weight=0, progress_weight=0,
                              low_speed_weight=0, hold_weight=1, action_weight=0,
                              action_rate_weight=0, command_jump_weight=0,
                              joint_soft_limit_weight=0, velocity_weight=0, step_scale=1)
    old = MarvinBimanualSE3ReachHold(SimpleFKRobot(), reward_config=weights)
    new = MarvinBimanualSE3ReachHold(SimpleFKRobot(), reward_config=weights,
                                     hold_speed_mode="policy_step_displacement_v2")
    prior = _transition(new).state
    next_state = _transition(new).next_state
    goal = _transition(new).next_task_state
    action = torch.zeros(1, 14)
    assert float(new.reward(prior, next_state, action, goal)) > float(old.reward(prior, next_state, action, goal)) + 0.5
    assert float(next_state[:, 14:28].abs().max()) == 0.5


def test_invalid_hold_velocity_contract_is_rejected():
    try:
        MarvinBimanualSE3ReachHold(SimpleFKRobot(), hold_speed_mode="unversioned")
    except ValueError as exc:
        assert "hold_speed_mode" in str(exc)
    else:
        raise AssertionError("Unknown hold speed semantics were accepted")


def test_episode_hold_uses_50hz_displacement_for_all_25_steps():
    task = MarvinBimanualSE3ReachHold(SimpleFKRobot(),
                                       hold_speed_mode="policy_step_displacement_v2")
    goal = _transition(task, delta=0).next_task_state
    items = []
    for step in range(1, 201):
        state = torch.zeros(1, 42)
        next_state = state.clone()
        next_state[:, 14:28] = 0.5
        items.append(SimpleNamespace(
            state=state, next_state=next_state,
            next_task_state=task.task_state_from_goal(goal.data,
                                                       frame=torch.tensor([step])),
            action=torch.zeros(1, 14), terminated=torch.zeros(1, dtype=torch.bool),
            truncated=torch.tensor([step == 200]),
            failure_reason=torch.zeros(1, dtype=torch.int32)))
    assert task.episode_success(items, complete=True)
    assert task.evaluate_metrics(items)["longest_hold_steps"] == 200
    assert task.evaluate_metrics(items)["joint_velocity_abs_max_rad_s"] == 0.5
    assert task.evaluate_metrics(items)["hold_velocity_abs_max_rad_s"] == 0
