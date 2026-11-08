"""Terminal conditions must use the fingertip transition from the current step."""

from types import SimpleNamespace

import torch
from dexgym.tasks.direct.right_hand_tracking.env import RightHandTrackingDirectEnv
from dexgym.tasks.direct.right_hand_tracking.env_cfg import RightHandTrackingDirectCfg


def test_completed_goal_and_stalled_goal_terminate_on_their_triggering_step():
    q = torch.zeros((2, 20))
    cfg = RightHandTrackingDirectCfg()
    cfg.goal_tolerance_m = 0.015
    task = SimpleNamespace(
        robot=SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=q),
                joint_vel=SimpleNamespace(torch=torch.zeros_like(q)),
            )
        ),
        bank=SimpleNamespace(
            lengths=torch.tensor([5]),
            fingertips=lambda _motion_ids, frames: torch.zeros((len(frames), 5, 3)),
        ),
        cfg=cfg,
        joint_limits=torch.tensor([[-1.0, 1.0]] * 20),
        right_ids=list(range(20)),
        motion_ids=torch.zeros(2, dtype=torch.long),
        goal_frame=torch.tensor([4, 0]),
        goal_tips=torch.zeros((2, 5, 3)),
        dwell=torch.tensor([1, 0]),
        goal_age=torch.tensor([0, 44]),
        completed=torch.zeros(2, dtype=torch.bool),
        action_delta=torch.zeros((2, 20)),
        goals_reached=torch.zeros(2, dtype=torch.long),
        episode_length_buf=torch.ones(2, dtype=torch.long),
        max_episode_length=240,
        extras={},
    )
    current = torch.zeros((2, 5, 3))
    current[1, :, 0] = 0.05
    task._current_tips = lambda: current

    terminated, truncated = RightHandTrackingDirectEnv._get_dones(task)
    reward = RightHandTrackingDirectEnv._get_rewards(task)

    assert terminated.tolist() == [True, True]
    assert truncated.tolist() == [False, False]
    assert reward[0].item() > 10.0  # final subgoal bonus is paid on the terminal step
    assert task.goals_reached.tolist() == [1, 0]
