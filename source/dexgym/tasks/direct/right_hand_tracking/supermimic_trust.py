"""Held-out actor-transition model checks in hand geometry as well as state units."""

import math
import torch
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator
from .subgoals import HandGoals


class HandModelTrust(ModelTrustEvaluator):
    def __init__(self, config, tip_limits):
        super().__init__(config)
        self.tip_limits = tuple(tip_limits)
        if len(self.tip_limits) != 3 or any(not math.isfinite(x) or x <= 0 for x in self.tip_limits):
            raise ValueError('Expected three positive hand prediction limits')

    @torch.no_grad()
    def evaluate_horizon(self, model, replay, robot, *, horizon, batch=64, env_range=None, source_id=0):
        window = replay.sample_windows(batch, horizon, env_range=env_range, source_id=source_id, require_full=True)
        prediction = window.state[:, 0]
        for step in range(horizon):
            prediction = model(prediction, window.action[:, step], window.dynamics_context[:, step], robot)
        truth = window.next_state[:, -1]
        error = robot.physical_delta(truth, prediction)
        baseline = robot.physical_delta(truth, window.state[:, 0])
        scale = model.delta_norm.std()
        tip_errors = torch.linalg.vector_norm(robot.fk(prediction[:, :20]) - robot.fk(truth[:, :20]), dim=-1)
        identity = torch.stack((window.absolute_timestamp[:, 0], window.env_id), -1)
        old = window.task_state.index((slice(None), -1))
        before = HandGoals.unpack(old.data)
        after = HandGoals.unpack(window.next_task_state.data[:, -1])
        labels = (after.goals_reached > before.goals_reached).float()
        probabilities = model.reach_head(model.task_adapter.reach_features(truth, old)).sigmoid().squeeze(-1)
        return {'evaluation_kind': 'held_out_lane_action_conditioned_open_loop', 'horizon': horizon,
                'source_id': source_id, 'distinct_start_count': len(torch.unique(identity, dim=0)),
                'physical_delta_rmse': float(error.square().mean().sqrt()),
                'zero_delta_baseline_rmse': float(baseline.square().mean().sqrt()),
                'normalized_delta_rmse': float((error / scale).square().mean().sqrt()),
                'normalized_zero_delta_baseline_rmse': float((baseline / scale).square().mean().sqrt()),
                'mean_tip_error_m': float(tip_errors.mean()),
                'p95_worst_tip_error_m': float(torch.quantile(tip_errors.max(-1).values, .95)),
                'real_state_reach_brier': float((probabilities - labels).square().mean()),
                'reach_label_rate': float(labels.mean()), 'reach_prediction_rate': float(probabilities.mean())}

    def trusted_horizon(self, report):
        if super().trusted_horizon(report) != 32:
            return 0
        for horizon, limit in zip((1, 8, 32), self.tip_limits):
            value = report.get(str(horizon), {}).get('mean_tip_error_m', float('inf'))
            if not math.isfinite(value) or value > limit:
                return 0
        return 32
