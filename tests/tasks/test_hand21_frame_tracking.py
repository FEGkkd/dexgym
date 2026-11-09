"""The Hand 2.1 PPO target follows consecutive 30 Hz reference frames."""

from types import SimpleNamespace

import torch
from dexgym.tasks.direct.hand21_ppo.env_cfg import WujiHand21TeleDexterCfg
from dexgym.tasks.direct.hand21_ppo.subgoals import HandGoals, advance_hand_goals


def _goals(frame: int, *, misses: int = 0) -> HandGoals:
    return HandGoals(
        motion_ids=torch.tensor([0]),
        goal_frame=torch.tensor([frame]),
        goal_span=torch.tensor([1]),
        dwell=torch.tensor([0]),
        dwell_required=torch.tensor([1]),
        fail_frames=torch.tensor([misses]),
        fail_budget=torch.tensor([45.0]),
        curriculum_steps=torch.tensor([100]),
        goal_age=torch.tensor([0]),
        goals_reached=torch.tensor([0]),
        completed=torch.tensor([False]),
    )


def test_target_advances_one_frame_even_when_fingertips_miss() -> None:
    cfg = WujiHand21TeleDexterCfg()
    bank = SimpleNamespace(lengths=torch.tensor([6]), fingertips=lambda ids, frames: torch.zeros((len(ids), 5, 3)))
    tips = torch.zeros((1, 5, 3))
    tips[0, 0, 0] = 0.03

    first = advance_hand_goals(tips, _goals(3), bank, cfg)
    assert first.reached.tolist() == [False]
    assert first.goals.goal_frame.tolist() == [4]
    assert first.goals.goal_span.tolist() == [1]
    assert first.goals.fail_frames.tolist() == [1]
    assert first.stride_upper.tolist() == [1]

    second = advance_hand_goals(torch.zeros_like(tips), first.goals, bank, cfg)
    assert second.reached.tolist() == [True]
    assert second.goals.goal_frame.tolist() == [5]
    assert second.goals.fail_frames.tolist() == [0]


def test_last_reference_frame_finishes_even_if_outside_threshold() -> None:
    cfg = WujiHand21TeleDexterCfg()
    bank = SimpleNamespace(lengths=torch.tensor([6]), fingertips=lambda ids, frames: torch.zeros((len(ids), 5, 3)))
    tips = torch.full((1, 5, 3), 0.03)

    result = advance_hand_goals(tips, _goals(5), bank, cfg)
    assert result.reached.tolist() == [False]
    assert result.goals.completed.tolist() == [True]
    assert result.goals.goal_frame.tolist() == [5]


def test_active_training_config_uses_one_frame_without_dwell_or_stride_curriculum() -> None:
    cfg = WujiHand21TeleDexterCfg()
    assert cfg.sim.dt * cfg.decimation == 1 / 30
    assert cfg.subgoal_stride == 1
    assert (cfg.curriculum_stride_min, cfg.curriculum_stride_max) == (1, 1)
    assert (cfg.dwell_min_steps, cfg.dwell_max_steps) == (1, 1)
    assert cfg.cross_trajectory_probability == 0.0
    assert cfg.max_goal_steps == 45
