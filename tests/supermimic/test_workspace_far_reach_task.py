"""A distant static TCP target must be trained and scored as arrival and hold."""

from dataclasses import replace
from types import SimpleNamespace

import torch

from dexgym.tasks.marvin_supermimic.se3_reach_hold import (
    SE3SuccessConfig, rotation_6d_from_matrix)
from tests.supermimic.test_continuous_se3_follow import LinearTCPRobot


class StaticFarReference:
    names = ("far",)
    lengths = torch.tensor([41])
    hashes = {"far_goal_bank_v3": "synthetic"}

    def pose_at(self, motion_id, frame):
        target = torch.zeros(len(frame), 2, 9)
        target[:, 0, 0] = 0.4
        target[:, 1, 0] = -0.4
        target[:, :, 3:] = rotation_6d_from_matrix(torch.eye(3))
        return target.flatten(1)


def _task():
    from dexgym.tasks.marvin_supermimic.workspace_reach_hold import (
        MarvinWorkspaceSE3ReachHold, WorkspaceReachRewardConfig)

    return MarvinWorkspaceSE3ReachHold(
        StaticFarReference(), LinearTCPRobot(),
        reward_config=WorkspaceReachRewardConfig(
            progress_weight=0, low_speed_weight=0, hold_weight=0,
            action_weight=0, action_rate_weight=0, command_jump_weight=0,
            joint_soft_limit_weight=0, velocity_weight=0),
        success_config=replace(SE3SuccessConfig(), episode_steps=40,
                               arrival_deadline_steps=20))


def test_far_goal_reward_has_distance_gradient_without_progress_or_teacher():
    task = _task()
    state = torch.zeros(1, 42)
    next_state = state.clone().requires_grad_()
    arrival = task.task_state(torch.tensor([0]), torch.tensor([1]))
    reward = task.reward(state, next_state, torch.zeros(1, 14), arrival)
    reward.backward()
    assert torch.isfinite(next_state.grad).all()
    assert next_state.grad[0, 0] > 0.005
    assert next_state.grad[0, 7] < -0.005
    assert task.spec.goal_dim == task.spec.task_data_dim == 18
    assert not hasattr(arrival, "q_goal")
    torch.testing.assert_close(arrival.data, task.goal(state, arrival)[:, 0])


def test_far_goal_episode_accepts_late_hold_after_early_large_error():
    task = _task()
    target = task.task_state(torch.tensor([0]), torch.tensor([0])).data
    trajectory = []
    for step in range(1, 41):
        state = torch.zeros(1, 42)
        if step >= 16:
            state[0, 0], state[0, 7] = 0.4, -0.4
        trajectory.append(SimpleNamespace(
            state=state, next_state=state, action=torch.zeros(1, 14),
            next_task_state=task.task_state(torch.tensor([0]), torch.tensor([step])),
            terminated=torch.tensor([False]), truncated=torch.tensor([step == 40]),
            failure_reason=torch.zeros(1, dtype=torch.int32)))
    assert task.episode_success(trajectory, complete=True)
    metrics = task.evaluate_metrics(trajectory)
    assert metrics["first_complete_se3_arrival_step"] == 16
    assert metrics["longest_hold_steps"] == 25
    assert not task.episode_success(trajectory[:-1], complete=False)
    assert task.valid_imagination_start(task.task_state(torch.tensor([0]), torch.tensor([8])), 32).item()
    assert not task.valid_imagination_start(task.task_state(torch.tensor([0]), torch.tensor([9])), 32).item()


def test_arrival_deadline_applies_to_start_of_sustained_hold():
    task = _task()
    task.success_config = replace(task.success_config, hold_steps=5)
    trajectory = []
    for step in range(1, 41):
        state = torch.zeros(1, 42)
        # A single transient arrival before the deadline must not count when
        # the first sustained hold starts after that deadline.
        if step == 10 or step >= 21:
            state[0, 0], state[0, 7] = 0.4, -0.4
        trajectory.append(SimpleNamespace(
            state=state, next_state=state, action=torch.zeros(1, 14),
            next_task_state=task.task_state(torch.tensor([0]), torch.tensor([step])),
            terminated=torch.tensor([False]), truncated=torch.tensor([step == 40]),
            failure_reason=torch.zeros(1, dtype=torch.int32)))
    assert task.evaluate_metrics(trajectory)["first_complete_se3_arrival_step"] == 10
    assert not task.episode_success(trajectory, complete=True)


def test_evaluation_difficulty_filter_uses_held_out_goals_only():
    from dexgym.tasks.marvin_supermimic.teleop_trajectory_env import (
        MarvinTeleopWorkspaceEnv, MarvinTeleopWorkspaceCfg)
    cfg = MarvinTeleopWorkspaceCfg()
    cfg.evaluation_mode = True
    cfg.evaluation_split = "test"
    cfg.evaluation_max_difficulty = 0
    reference = SimpleNamespace(
        split=torch.tensor([2, 2, 2]),
        difficulty=torch.tensor([0, 1, 2]),
        names=("near", "medium", "wide"))
    env = SimpleNamespace(cfg=cfg, reference=reference, device="cpu")
    ids, frames = MarvinTeleopWorkspaceEnv._sample_reference(env, torch.arange(2))
    assert ids.tolist() == [0, 0]
    assert frames.tolist() == [0, 0]


def test_training_segment_covers_a_full_physical_episode():
    import json
    from pathlib import Path

    path = Path("configs/supermimic/workspace_reach_hold_4096_command_centered_v2.json")
    config = json.loads(path.read_text())
    physics_steps = (config["iterations_per_segment"]
                     * config["trainer"]["physics_steps_per_iteration"])
    assert physics_steps >= int(config["episode_length_s"] * 50)


def test_v5_training_checkpoint_records_goal_curriculum_level():
    from dexgym.tasks.marvin_supermimic.physics_contract import (
        marvin_teleop_workspace_physics_config)
    from dexgym.tasks.marvin_supermimic.workspace_reach_env import (
        MarvinWorkspaceSE3ReachHoldCfg)

    cfg = MarvinWorkspaceSE3ReachHoldCfg()
    cfg.reference_bank_path = (
        "source/dexgym/assets/supermimic/references/"
        "teleop_workspace_hold_curriculum_seed20261114.npz")
    cfg.reach_curriculum_version = "marvin_teleop_workspace_bank_v5"
    cfg.curriculum_level = 0
    robot = SimpleNamespace(native=SimpleNamespace(
        controller_kp_source="fixed", controller_kd_source="fixed",
        controller_damping_scale_source=1.0))
    hold = marvin_teleop_workspace_physics_config(cfg, robot)
    cfg.curriculum_level = 1
    near = marvin_teleop_workspace_physics_config(cfg, robot)
    assert hold["goal_curriculum_level"] == 0
    assert near["goal_curriculum_level"] == 1


def test_far_goal_newton_plugin_selects_reach_mdp_instead_of_tracking_mdp():
    from dexgym.tasks.marvin_supermimic.workspace_reach_env import (
        MarvinWorkspaceSE3ReachHoldEnv)

    env = SimpleNamespace(cfg=SimpleNamespace(
        reach_success_semantics="hold_start_deadline_v2"))
    task = MarvinWorkspaceSE3ReachHoldEnv._make_task(
        env, StaticFarReference(), LinearTCPRobot())
    assert task.spec.task_id == "marvin_bimanual_workspace_se3_reach_hold"
    assert task.spec.task_data_dim == 18
    assert task.success_hold_steps == 25
    assert task.checkpoint_contract()["task_adapter_version"].endswith("_v2")


def test_old_workspace_checkpoint_success_semantics_remain_explicit():
    from dexgym.tasks.marvin_supermimic.workspace_reach_hold import MarvinWorkspaceSE3ReachHold

    task = MarvinWorkspaceSE3ReachHold(
        StaticFarReference(), LinearTCPRobot(),
        success_semantics="pre_deadline_arrival_v1")
    assert task.checkpoint_contract()["task_adapter_version"] == (
        "marvin_workspace_se3_reach_hold_v1")


def test_time_aware_workspace_task_exposes_remaining_episode_to_actor_and_critic():
    from dexgym.tasks.marvin_supermimic.workspace_reach_hold import MarvinWorkspaceSE3ReachHold

    task = MarvinWorkspaceSE3ReachHold(
        StaticFarReference(), LinearTCPRobot(),
        temporal_context="normalized_time_remaining_v3")
    state = torch.zeros(1, 42)
    early = task.task_state(torch.tensor([0]), torch.tensor([0]))
    late = task.task_state(torch.tensor([0]), torch.tensor([39]))
    assert task.spec.observation_dim == 69
    assert task.spec.critic_dim == 61
    assert task.observation(state, torch.zeros(1, 14), early)[0, -1] == 1
    assert task.observation(state, torch.zeros(1, 14), late)[0, -1] == 1 / 40
    assert task.critic_features(state, early)[0, -1] == 1
    assert task.critic_features(state, late)[0, -1] == 1 / 40
    assert task.checkpoint_contract()["task_adapter_version"].endswith("_v3")


def test_finite_episode_timeout_is_terminal_for_imagined_bootstrap():
    from dexgym.tasks.marvin_supermimic.workspace_reach_hold import MarvinWorkspaceSE3ReachHold

    task = MarvinWorkspaceSE3ReachHold(
        StaticFarReference(), LinearTCPRobot(),
        temporal_context="normalized_time_remaining_v3")
    next_state = torch.zeros(1, 42)
    before_end = task.task_state(torch.tensor([0]), torch.tensor([39]))
    end = task.task_state(torch.tensor([0]), torch.tensor([40]))
    assert task.soft_termination(next_state, end).item() == 1.0
    assert task.soft_termination(next_state, before_end).item() < 1.0


def test_v3_bptt32_keeps_actor_gradient_and_cuts_terminal_value():
    from dexgym.algorithms.supermimic.core.imagination import (
        ImaginationEngine, critic_td_lambda_batch)
    from dexgym.algorithms.supermimic.core.transition import StartBatch
    from dexgym.tasks.marvin_supermimic.workspace_reach_hold import MarvinWorkspaceSE3ReachHold

    robot = LinearTCPRobot()
    robot.spec = SimpleNamespace(control_dt=0.02, context_dim=0,
                                 context_evolution="episode_constant")
    task = MarvinWorkspaceSE3ReachHold(
        StaticFarReference(), robot,
        temporal_context="normalized_time_remaining_v3")

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw = torch.nn.Parameter(torch.zeros(14))

        def forward(self, history, goal):
            return self.raw.expand(len(history), -1)

    class CommandModel(torch.nn.Module):
        def forward(self, state, action, context, robot):
            delta = torch.zeros_like(state)
            delta[:, 0] = 0.01 * action[:, 0].tanh()
            delta[:, 7] = 0.01 * action[:, 7].tanh()
            return state + delta

    class Critic(torch.nn.Module):
        def forward(self, features):
            return features[:, -1] * 0 + 100.0

    actor = Actor()
    state = torch.zeros(1, 42)
    clock = task.task_state(torch.tensor([0]), torch.tensor([8]))
    observation = task.observation(state, torch.zeros(1, 14), clock)
    start = StartBatch(state, observation[:, None].expand(-1, 10, -1).clone(),
                       torch.empty(1, 0), clock, torch.zeros(1, 14),
                       torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    rollout = ImaginationEngine().rollout(start, actor, CommandModel(), Critic(),
                                          robot, task, horizon=32)
    assert rollout.terminations[-1, 0] == 1.0
    assert rollout.survival[-1, 0] == 0.0
    _, targets = critic_td_lambda_batch(rollout)
    torch.testing.assert_close(targets[-1], rollout.rewards[-1, 0])
    rollout.actor_loss.backward()
    assert torch.isfinite(actor.raw.grad).all()
    assert actor.raw.grad[[0, 7]].abs().sum() > 0


def test_shared_training_entrypoint_selects_far_reach_without_copying_trainer():
    from scripts.supermimic.train_continuous_se3 import _task_profile

    selected = _task_profile("marvin_workspace_reach_hold_v1")
    assert selected[0].__name__ == "MarvinWorkspaceSE3ReachHoldCfg"
    assert selected[1] == "DexGym-Marvin-SuperMimic-Workspace-Reach-Hold-Direct-v0"
    assert selected[2] == "marvin_teleop_workspace_bank_v3"
    assert _task_profile("marvin_workspace_reach_hold_v2")[2] == (
        "marvin_teleop_workspace_bank_v4")
    assert _task_profile("marvin_workspace_reach_hold_v4")[2] == (
        "marvin_teleop_workspace_bank_v5")


def test_shared_evaluation_entrypoint_selects_reach_episode_metrics():
    from scripts.supermimic.evaluate_continuous_se3 import _evaluation_task_profile

    config_class, environment_id, task_id = _evaluation_task_profile("workspace_reach")
    assert config_class.__name__ == "MarvinWorkspaceSE3ReachHoldCfg"
    assert environment_id == "DexGym-Marvin-SuperMimic-Workspace-Reach-Hold-Direct-v0"
    assert task_id == "marvin_bimanual_workspace_se3_reach_hold"
