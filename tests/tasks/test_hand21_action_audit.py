"""Rollout action diagnostics use the latent and executed action spaces correctly."""

import torch
from dexgym.algorithms.action_audit import ActionAudit


def test_audit_separates_cross_goal_and_exact_span_and_resets() -> None:
    audit = ActionAudit(action_dim=2, max_span=40, device="cpu")
    mean = torch.tensor([[1.2, 0.0], [0.0, -0.5]])
    actions = torch.tensor([[0.96, 0.0], [0.8, -0.5]])
    audit.record(mean, actions, torch.tensor([0, 7]))

    rows = audit.drain(iteration=0)
    cross = rows[0]
    span_seven = rows[7]
    assert cross["env_steps"] == 1
    assert cross["latent_mean_over_limit_components"] == 1
    assert cross["gaussian_sample_over_limit_components"] == 1
    assert cross["executed_saturated_components"] == 1
    assert span_seven["env_steps"] == 1
    assert span_seven["latent_mean_over_limit_components"] == 0
    assert span_seven["gaussian_sample_over_limit_components"] == 1
    assert span_seven["executed_saturated_components"] == 0
    assert sum(row["env_steps"] for row in audit.drain(iteration=1)) == 0


def test_audit_rejects_invalid_goal_spans() -> None:
    audit = ActionAudit(action_dim=2, max_span=40, device="cpu")
    try:
        audit.record(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([41]))
    except ValueError as exc:
        assert "span" in str(exc)
    else:
        raise AssertionError("Expected invalid span to be rejected")
