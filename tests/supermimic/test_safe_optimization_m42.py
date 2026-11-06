import torch

from dexgym.algorithms.supermimic.optimization import safe_gradient_step


def test_nonfinite_gradient_skips_actor_or_world_model_step():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    bad = parameter * torch.tensor(float("nan"))
    result = safe_gradient_step(bad, [parameter], optimizer)
    assert result["updated"] is False
    assert parameter.item() == 1.0
    good = (parameter - 2).square()
    result = safe_gradient_step(good, [parameter], optimizer)
    assert result["updated"] is True
    assert parameter.item() != 1.0
    assert torch.isfinite(parameter)
