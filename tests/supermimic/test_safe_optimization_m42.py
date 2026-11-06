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


def test_nonfinite_optimizer_state_rolls_back_parameters_and_resets_state():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    assert safe_gradient_step((parameter - 2).square(), [parameter], optimizer)["updated"]
    before = parameter.detach().clone()
    optimizer.state[parameter]["exp_avg_sq"].fill_(float("nan"))
    result = safe_gradient_step((parameter - 2).square(), [parameter], optimizer)
    assert result["updated"] is False and result["reason"] == "nonfinite_parameter"
    torch.testing.assert_close(parameter, before)
    assert not optimizer.state
    assert safe_gradient_step((parameter - 2).square(), [parameter], optimizer)["updated"]
