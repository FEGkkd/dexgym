import torch

from dexgym.algorithms.supermimic.networks import DeltaWorldModel
from dexgym.algorithms.supermimic.termination import TaskTerminationHead


def test_shared_dynamics_has_no_fixed_goal_head_and_external_task_head_trains():
    model = DeltaWorldModel(hidden=32, legacy_termination_head=False)
    termination = TaskTerminationHead(42, 5, hidden=16)
    assert not any(key.startswith("termination_head.") for key in model.state_dict())
    window = {
        "state": torch.zeros(2, 8, 42),
        "action": torch.zeros(2, 8, 14),
        "next_state": torch.zeros(2, 8, 42),
        "mask": torch.tensor([[1.0] + [0.0] * 7, [1.0] + [0.0] * 7]),
        "terminated": torch.tensor([[True] + [False] * 7, [False] * 8]),
        "termination_goal": torch.ones(2, 8, 5),
    }
    loss, _ = model.autoregressive_loss(window, lambda action, command: command,
                                        update_normalizers=False, termination_module=termination)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in termination.parameters())
    state = torch.zeros(2, 42, requires_grad=True)
    for parameter in termination.parameters():
        parameter.requires_grad_(False)
    termination(state, torch.ones(2, 5)).sum().backward()
    assert state.grad is not None and torch.isfinite(state.grad).all()
