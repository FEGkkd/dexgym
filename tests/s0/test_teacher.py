"""The synthetic teacher must preserve the Actor's mode and action contract."""

import os

import mujoco
import pytest
import torch
from dexgym.tasks.marvin_s0.kinematics import MarvinArmFK
from dexgym.tasks.marvin_s0.teacher import sample_fk_teacher_batch


def test_bimanual_samples_activate_both_arms_without_exposing_joint_goal():
    source = os.environ.get("DEXGYM_MARVIN_MJCF")
    if not source:
        pytest.skip("DEXGYM_MARVIN_MJCF is required for the external asset")
    fk = MarvinArmFK(mujoco.MjModel.from_xml_path(source), "cpu")
    home = torch.zeros(14)
    low = torch.full((14,), -2.0)
    high = torch.full((14,), 2.0)
    observation, label = sample_fk_teacher_batch(fk, home, low, high, 64, bimanual_fraction=1.0)
    assert observation.shape == (64, 73)
    assert label.shape == (64, 14)
    assert torch.isfinite(observation).all() and torch.isfinite(label).all()
    assert torch.all(observation[:, 68:70] == 1.0)
    assert torch.all(observation[:, 70:] == torch.tensor([0.0, 0.0, 1.0]))
    assert (label[:, :7].abs().sum(-1) > 0).all()
    assert (label[:, 7:].abs().sum(-1) > 0).all()


def test_invalid_bimanual_fraction_is_rejected():
    with pytest.raises(ValueError, match="bimanual_fraction"):
        sample_fk_teacher_batch(None, torch.zeros(14), torch.zeros(14), torch.ones(14), 1, bimanual_fraction=1.1)
