"""The combined SE(3) reward should prioritize whichever pose component lags."""

import torch
from dexgym.tasks.marvin_s0.reward import balanced_pose_score


def test_balanced_score_prioritizes_orientation_when_position_is_good():
    score = balanced_pose_score(torch.tensor([0.013]), torch.tensor([0.65]), 0.06, 0.4)
    better_orientation = balanced_pose_score(torch.tensor([0.013]), torch.tensor([0.45]), 0.06, 0.4)
    better_position = balanced_pose_score(torch.tensor([0.008]), torch.tensor([0.65]), 0.06, 0.4)
    assert (better_orientation - score) > (better_position - score)


def test_balanced_score_prioritizes_position_when_orientation_is_good():
    score = balanced_pose_score(torch.tensor([0.07]), torch.tensor([0.05]), 0.06, 0.4)
    better_position = balanced_pose_score(torch.tensor([0.05]), torch.tensor([0.05]), 0.06, 0.4)
    better_orientation = balanced_pose_score(torch.tensor([0.07]), torch.tensor([0.02]), 0.06, 0.4)
    assert (better_position - score) > (better_orientation - score)
