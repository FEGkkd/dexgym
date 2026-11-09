"""Paper Actor clipping order has a measurable effect on imbalanced gradients."""

import torch
from dexgym.algorithms.supermimic.core.optimization import safe_gradient_step


def test_global_norm_then_component_clip_matches_supermimic_order():
    parameter = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([parameter], lr=1.)
    gradient = torch.tensor([10., .2])
    expected = -(gradient / (gradient.norm() + 1e-6)).clamp(-.1, .1)
    result = safe_gradient_step((parameter * gradient).sum(), [parameter], optimizer,
                               grad_norm_clip=1., grad_value_clip=.1)
    assert result['updated']
    torch.testing.assert_close(parameter, expected)
