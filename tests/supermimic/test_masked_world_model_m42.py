import torch

from dexgym.algorithms.supermimic.action import ArmActionMap
from dexgym.algorithms.supermimic.networks import DeltaWorldModel


def test_one_valid_transition_trains_eight_step_model_without_post_reset_contamination():
    model = DeltaWorldModel(hidden=32)
    mapping = ArmActionMap(torch.zeros(14), -torch.ones(14), torch.ones(14), torch.ones(14))
    windows = {
        "state": torch.zeros(2, 8, 42),
        "action": torch.zeros(2, 8, 14),
        "next_state": torch.zeros(2, 8, 42),
        "terminated": torch.zeros(2, 8, dtype=torch.bool),
        "termination_goal": torch.zeros(2, 8, 90),
        "mask": torch.zeros(2, 8),
    }
    windows["mask"][:, 0] = 1
    windows["next_state"][:, 0, :28] = 0.5
    windows["terminated"][0, 0] = True
    baseline, _ = model.autoregressive_loss(windows, mapping, update_normalizers=False)
    windows["next_state"][:, 1:] = 1e5
    windows["termination_goal"][:, 1:] = 1e5
    contaminated, _ = model.autoregressive_loss(windows, mapping, update_normalizers=False)
    torch.testing.assert_close(baseline, contaminated)
    contaminated.backward()
    assert torch.isfinite(contaminated)
    assert model.network[-1].weight.grad.abs().sum() > 0
    assert model.termination_head[-1].weight.grad.abs().sum() > 0
