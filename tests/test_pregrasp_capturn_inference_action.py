import torch

from dexteroushands_lab.tasks.direct.bimanual_pregrasp_capturn.action_math import (
    mutable_inference_action, policy_wrist_speed,
)


def test_policy_action_can_be_modified_for_counterfactual_after_inference():
    obs = torch.ones(2, 3)
    action = mutable_inference_action(lambda x: x, obs)
    assert not action.is_inference()
    action[:, 1:] = 0
    assert action.tolist() == [[1, 0, 0], [1, 0, 0]]


def test_policy_wrist_speed_uses_full_policy_window_not_last_physx_substep():
    start_pos = torch.zeros(2, 3)
    end_pos = torch.tensor([[0.0005, 0., 0.], [0., 0., 0.]])
    start_quat = torch.tensor([[0., 0., 0., 1.], [0., 0., 0., 1.]])
    end_quat = torch.tensor([[0., 0., torch.sin(torch.tensor(.01)),
                              torch.cos(torch.tensor(.01))], [0., 0., 0., 1.]])
    linear, angular = policy_wrist_speed(start_pos, end_pos, start_quat, end_quat, .02)
    assert torch.allclose(linear, torch.tensor([.025, 0.]), atol=1e-5)
    assert torch.allclose(angular, torch.tensor([1., 0.]), atol=1e-4)
