"""Held-out actor-data action-conditioned model prediction gate."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrustConfig:
    min_distinct_starts: int = 8
    max_rmse: tuple[float, float, float] = (0.04, 0.05, 0.1)
    require_baseline_improvement: bool = True

    def __post_init__(self):
        if self.min_distinct_starts < 1 or len(self.max_rmse) != 3 or min(self.max_rmse) <= 0:
            raise ValueError("Model trust thresholds must be positive for 1/8/32 steps")


class ModelTrustEvaluator:
    def __init__(self, config: TrustConfig = TrustConfig()):
        self.config = config

    @torch.no_grad()
    def evaluate_horizon(self, model, replay, robot, *, horizon: int,
                         batch: int = 64, env_range=None, source_id: int = 0):
        window = replay.sample_windows(batch, horizon, env_range=env_range,
                                       source_id=source_id, require_full=True)
        prediction = window.state[:, 0]
        for step in range(horizon):
            prediction = model(prediction, window.action[:, step],
                               window.dynamics_context[:, step], robot)
        truth = window.next_state[:, -1]
        error = robot.physical_delta(truth, prediction)
        baseline = robot.physical_delta(truth, window.state[:, 0])
        identity = torch.stack((window.absolute_timestamp[:, 0], window.env_id), dim=-1)
        distinct = len(torch.unique(identity, dim=0))
        return {"evaluation_kind": "action_conditioned_open_loop",
                "source_id": source_id, "horizon": horizon,
                "distinct_start_count": distinct,
                "physical_delta_rmse": float(error.square().mean().sqrt()),
                "zero_delta_baseline_rmse": float(baseline.square().mean().sqrt())}

    @torch.no_grad()
    def evaluate_available(self, model, replay, robot, *, batch=64, env_range=None,
                           source_id=0):
        report = {}
        for horizon in (1, 8, 32):
            try:
                report[str(horizon)] = self.evaluate_horizon(
                    model, replay, robot, horizon=horizon, batch=batch,
                    env_range=env_range, source_id=source_id)
            except ValueError as error:
                report[str(horizon)] = {"available": False, "reason": str(error)}
        return report

    def trusted_horizon(self, report) -> int:
        result = 0
        for index, horizon in enumerate((1, 8, 32)):
            sample = report.get(str(horizon), {})
            if (sample.get("distinct_start_count", 0) < self.config.min_distinct_starts
                    or sample.get("physical_delta_rmse", float("inf")) > self.config.max_rmse[index]
                    or (self.config.require_baseline_improvement
                        and sample.get("physical_delta_rmse", float("inf")) >=
                        sample.get("zero_delta_baseline_rmse", 0.0))):
                break
            result = horizon
        return result

