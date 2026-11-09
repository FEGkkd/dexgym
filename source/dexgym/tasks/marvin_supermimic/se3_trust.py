"""SE(3) task trust: held-out action response plus current-policy geometry."""

from dataclasses import dataclass
import math

import torch

from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig
from dexgym.algorithms.supermimic.rewards import so3_geodesic


@dataclass(frozen=True)
class SE3GeometryTrustConfig:
    # Match the task's strict 5 mm / 2 degree TCP accuracy target. These
    # bounds are for held-out model prediction, not policy success.
    max_tcp_position_p95_m: tuple[float, float, float] = (0.005, 0.005, 0.005)
    max_tcp_so3_p95_rad: tuple[float, float, float] = (0.0349066, 0.0349066, 0.0349066)
    max_joint_velocity_rmse_rad_s: tuple[float, float, float] = (0.15, 0.15, 0.15)

    def __post_init__(self):
        for name in ("max_tcp_position_p95_m", "max_tcp_so3_p95_rad",
                     "max_joint_velocity_rmse_rad_s"):
            values = getattr(self, name)
            if len(values) != 3 or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"Invalid SE(3) geometry trust thresholds: {name}")


class SE3DualDistributionTrustEvaluator:
    """Preserve action-response test on noisy data and absolute TCP test on Actor data.

    A stationary Actor has a near-perfect zero-delta baseline; requiring a
    learned model to beat it before the Actor can move creates a startup
    deadlock. The separate held-out noisy lane still must beat that baseline.
    """

    def __init__(self, noisy_config: TrustConfig, geometry_config: SE3GeometryTrustConfig,
                 *, noisy_validation_range: tuple[int, int],
                 deterministic_validation_range: tuple[int, int],
                 minimum_actor_horizon: int = 1):
        if (noisy_validation_range[0] >= noisy_validation_range[1]
                or noisy_validation_range[1] != deterministic_validation_range[0]
                or deterministic_validation_range[0] >= deterministic_validation_range[1]):
            raise ValueError("Trust needs disjoint nonempty noisy and deterministic held-out lanes")
        if not noisy_config.require_baseline_improvement:
            raise ValueError("Noisy action-response trust must beat the zero-delta baseline")
        if minimum_actor_horizon not in (1, 8, 32):
            raise ValueError("minimum_actor_horizon must be 1, 8 or 32")
        self.noisy = ModelTrustEvaluator(noisy_config)
        self.geometry = geometry_config
        self.noisy_range = noisy_validation_range
        self.deterministic_range = deterministic_validation_range
        self.minimum_actor_horizon = minimum_actor_horizon

    @torch.no_grad()
    def evaluate_available(self, model, replay, robot, *, batch=64,
                           env_range=None, source_id=0):
        if env_range != (self.noisy_range[0], self.deterministic_range[1]) or source_id != 0:
            raise ValueError("Trust validation lane contract differs")
        noisy = self.noisy.evaluate_available(model, replay, robot, batch=batch,
                                               env_range=self.noisy_range, source_id=3)
        deterministic = self.noisy.evaluate_available(
            model, replay, robot, batch=batch,
            env_range=self.deterministic_range, source_id=0)
        for horizon in (1, 8, 32):
            try:
                window = replay.sample_windows(
                    batch, horizon, env_range=self.deterministic_range,
                    source_id=0, require_full=True)
            except ValueError:
                continue
            predicted = window.state[:, 0]
            for step in range(horizon):
                predicted = model(predicted, window.action[:, step],
                                  window.dynamics_context[:, step], robot)
            truth = window.next_state[:, -1]
            predicted_position, predicted_rotation = robot.fk(predicted[:, :14])
            true_position, true_rotation = robot.fk(truth[:, :14])
            position_error = torch.linalg.vector_norm(
                predicted_position[:, :, 2] - true_position[:, :, 2], dim=-1)
            rotation_error = so3_geodesic(
                predicted_rotation[:, :, 2], true_rotation[:, :, 2])
            deterministic[str(horizon)].update({
                "tcp_position_p95_max_m": float(torch.quantile(position_error, 0.95, dim=0).max()),
                "tcp_so3_p95_max_rad": float(torch.quantile(rotation_error, 0.95, dim=0).max()),
                "joint_velocity_rmse_rad_s": float((
                    predicted[:, 14:28] - truth[:, 14:28]).square().mean().sqrt()),
            })
        return {"noisy": noisy, "deterministic": deterministic,
                "evaluation_kind": "heldout_dual_distribution_action_and_geometry"}

    def trusted_horizon(self, report) -> int:
        noisy_horizon = self.noisy.trusted_horizon(report.get("noisy", {}))
        deterministic = report.get("deterministic", {})
        geometry_horizon = 0
        for index, horizon in enumerate((1, 8, 32)):
            sample = deterministic.get(str(horizon), {})
            if (sample.get("distinct_start_count", 0) < self.noisy.config.min_distinct_starts
                    or any(not math.isfinite(sample.get(key, float("inf"))) for key in (
                        "normalized_delta_rmse", "tcp_position_p95_max_m",
                        "tcp_so3_p95_max_rad", "joint_velocity_rmse_rad_s"))
                    or sample.get("normalized_delta_rmse", float("inf"))
                        > self.noisy.config.max_rmse[index]
                    or sample.get("tcp_position_p95_max_m", float("inf"))
                        > self.geometry.max_tcp_position_p95_m[index]
                    or sample.get("tcp_so3_p95_max_rad", float("inf"))
                        > self.geometry.max_tcp_so3_p95_rad[index]
                    or sample.get("joint_velocity_rmse_rad_s", float("inf"))
                        > self.geometry.max_joint_velocity_rmse_rad_s[index]):
                break
            geometry_horizon = horizon
        admitted = min(noisy_horizon, geometry_horizon)
        return admitted if admitted >= self.minimum_actor_horizon else 0
