"""Shared hard thresholds, differentiable soft boundary, and calibration diagnostics."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TerminationThresholds:
    tcp_error_m: float = 0.12
    joint_limit_margin_rad: float = 0.001
    speed_rad_s: float = 5.0
    tcp_softness_m: float = 0.01
    joint_softness_rad: float = 0.005
    speed_softness_rad_s: float = 0.2

    def __post_init__(self):
        if min(self.tcp_softness_m, self.joint_softness_rad, self.speed_softness_rad_s) <= 0:
            raise ValueError("Soft termination widths must be positive")


class TaskTerminationHead(nn.Module):
    """Goal-schema-specific learned head, separate from shared physical dynamics."""

    def __init__(self, state_dim: int, goal_dim: int, hidden: int = 128):
        super().__init__()
        if min(state_dim, goal_dim, hidden) < 1:
            raise ValueError("Termination head dimensions must be positive")
        self.state_dim, self.goal_dim = state_dim, goal_dim
        self.network = nn.Sequential(nn.Linear(state_dim + goal_dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.network[-1].bias, -4.0)

    def logits(self, state: torch.Tensor, current_goal: torch.Tensor) -> torch.Tensor:
        if (state.shape[-1] != self.state_dim or current_goal.shape[-1] != self.goal_dim
                or state.shape[:-1] != current_goal.shape[:-1]):
            raise ValueError("Task termination state and arrival-goal schemas differ")
        return self.network(torch.cat((state, current_goal), dim=-1)).squeeze(-1)

    def forward(self, state: torch.Tensor, current_goal: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(state, current_goal))


def analytic_soft_termination(q: torch.Tensor, qd: torch.Tensor, lower: torch.Tensor,
                              upper: torch.Tensor, tcp_error: torch.Tensor,
                              thresholds: TerminationThresholds = TerminationThresholds()) -> torch.Tensor:
    """Smooth OR of the physical TCP, joint-limit, and velocity failure predicates."""
    if q.shape != qd.shape or q.ndim != 2 or tcp_error.shape != (len(q), 2):
        raise ValueError("Expected [B,J] arm state and [B,2] TCP position errors")
    tcp = torch.sigmoid((tcp_error.float() - thresholds.tcp_error_m) / thresholds.tcp_softness_m)
    low = torch.sigmoid((lower - thresholds.joint_limit_margin_rad - q.float())
                        / thresholds.joint_softness_rad)
    high = torch.sigmoid((q.float() - upper - thresholds.joint_limit_margin_rad)
                         / thresholds.joint_softness_rad)
    speed = torch.sigmoid((qd.float().abs() - thresholds.speed_rad_s)
                          / thresholds.speed_softness_rad_s)
    risks = torch.cat((tcp, low, high, speed), dim=-1)
    return (1 - torch.prod(1 - risks, dim=-1)).clamp(0, 1)


@torch.no_grad()
def termination_calibration(probability: torch.Tensor, label: torch.Tensor,
                            tcp_error: torch.Tensor | None = None,
                            failure_reason: torch.Tensor | None = None) -> dict:
    """Metrics on real held-out Newton labels; class counts flag underdetermined results."""
    probability = probability.detach().float().flatten()
    label = label.detach().bool().flatten()
    if probability.shape != label.shape or not len(label):
        raise ValueError("Termination scores and labels must be nonempty and aligned")
    if not bool(torch.isfinite(probability).all()) or bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("Termination scores must be finite probabilities")
    predicted = probability >= 0.5
    positives = int(label.sum())
    negatives = len(label) - positives
    tp = int((predicted & label).sum())
    fp = int((predicted & ~label).sum())
    ece = 0.0
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        selected = ((probability >= low) &
                    ((probability < high) if index < 9 else (probability <= high)))
        if bool(selected.any()):
            ece += float(selected.float().mean() *
                         (probability[selected].mean() - label[selected].float().mean()).abs())
    support = positives >= 20 and negatives >= 20
    brier = float((probability - label.float()).square().mean())
    prevalence = positives / len(label)
    baseline_brier = prevalence * (1 - prevalence)
    result = {
        "positive_count": positives, "negative_count": negatives,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / positives if positives else None,
        "false_positive_rate": fp / negatives if negatives else None,
        "brier_score": brier,
        "brier_skill_score": 1 - brier / baseline_brier if baseline_brier else None,
        "ece_10_bin": ece,
        "sufficient_class_support": support,
        "calibrated": support and ece <= 0.05,
    }
    if tcp_error is not None:
        error = tcp_error.detach().float().reshape(len(label), -1).amax(-1)
        bins = (0.0, 0.04, 0.08, 0.12, float("inf"))
        result["tcp_error_bins_m"] = [
            {"range": [bins[i], bins[i + 1]], "count": int(((error >= bins[i]) & (error < bins[i + 1])).sum()),
             "mean_probability": float(probability[(error >= bins[i]) & (error < bins[i + 1])].mean())
             if bool(((error >= bins[i]) & (error < bins[i + 1])).any()) else None}
            for i in range(len(bins) - 1)
        ]
    if failure_reason is not None:
        reason = failure_reason.detach().long().flatten()
        result["failure_reason_counts"] = {
            str(int(value)): int((reason[label] == value).sum()) for value in reason[label].unique()
        }
    return result
