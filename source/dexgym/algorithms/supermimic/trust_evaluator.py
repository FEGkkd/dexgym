"""Action-conditioned open-loop model validation on real physical transitions."""

import numpy as np
import torch

from .rewards import so3_geodesic
from .termination import termination_calibration
from .termination import analytic_soft_termination, TerminationThresholds


class ModelTrustEvaluator:
    def __init__(self, fk, reference_provider):
        self.fk = fk
        self.reference = reference_provider

    @torch.no_grad()
    def evaluate_horizon(self, model, replay, mapping, *, horizon: int, batch: int = 128,
                         env_range=None, source_id: int | None = None, frame_range=None,
                         termination_mode: str = "learned",
                         termination_thresholds: TerminationThresholds = TerminationThresholds()) -> dict:
        window = replay.sample_windows(batch, horizon, env_range=env_range,
                                       frame_range=frame_range, require_full=True,
                                       source_id=source_id)
        state = window["state"][:, 0]
        baseline = state.clone()
        for step in range(horizon):
            state = model(state, window["action"][:, step], mapping)
            baseline_command = (baseline[:, 28:42] if mapping is None else
                                mapping(window["action"][:, step], baseline[:, 28:42]))
            baseline = torch.cat((baseline[:, :14] + 0.02 * baseline[:, 14:28],
                                  baseline[:, 14:28], baseline_command), dim=-1)
        truth = window["next_state"][:, -1]
        pred_pos, pred_rot = self.fk(state[:, :14])
        true_pos, true_rot = self.fk(truth[:, :14])
        base_pos, _ = self.fk(baseline[:, :14])
        tcp_error = (pred_pos[:, :, 2] - true_pos[:, :, 2]).norm(dim=-1)
        orientation = so3_geodesic(pred_rot[:, :, 2], true_rot[:, :, 2])
        joint_error = state[:, :14] - truth[:, :14]
        velocity_error = state[:, 14:28] - truth[:, 14:28]
        if termination_mode == "learned":
            arrival_reference = self.reference.reference_features_at(window["next_frame"][:, -1])
            termination_probability = model.termination_probability(state, arrival_reference)
        elif termination_mode == "analytic":
            arrival = self.reference.reference_at(window["next_frame"][:, -1])
            target_tcp = arrival["link_pos"][:, :, 2]
            termination_probability = analytic_soft_termination(
                state[:, :14], state[:, 14:28], mapping.lower, mapping.upper,
                (pred_pos[:, :, 2] - target_tcp).norm(dim=-1), termination_thresholds)
        else:
            raise ValueError("Unknown termination mode")
        calibration = termination_calibration(termination_probability, window["terminated"][:, -1],
                                              tcp_error=tcp_error)
        start_id = torch.stack((window["absolute_timestamp"][:, 0], window["env_id"]), dim=-1)
        return {
            "evaluation_kind": "action_conditioned_open_loop",
            "horizon": horizon, "sample_count": batch,
            "distinct_start_count": int(torch.unique(start_id, dim=0).shape[0]),
            "source_id": source_id, "env_range": env_range, "frame_range": frame_range,
            "joint_position_rmse_rad": float(joint_error.square().mean().sqrt()),
            "joint_velocity_rmse_rad_s": float(velocity_error.square().mean().sqrt()),
            "joint_position_rmse_by_joint_rad": joint_error.square().mean(0).sqrt().cpu().tolist(),
            "joint_velocity_rmse_by_joint_rad_s": velocity_error.square().mean(0).sqrt().cpu().tolist(),
            "tcp_position_mean_mm": float(tcp_error.mean() * 1000),
            "tcp_position_p95_mm": float(torch.quantile(tcp_error.flatten(), 0.95) * 1000),
            "tcp_orientation_mean_deg": float(orientation.mean() * 180 / np.pi),
            "tcp_orientation_p95_deg": float(torch.quantile(orientation.flatten(), 0.95) * 180 / np.pi),
            "constant_velocity_tcp_mean_mm": float((base_pos[:, :, 2] - true_pos[:, :, 2]).norm(dim=-1).mean() * 1000),
            "action_abs_mean": float(window["action"].abs().mean()),
            "termination": calibration,
        }

    @torch.no_grad()
    def evaluate_available(self, model, replay, mapping, *, batch=128, env_range=None,
                           source_id=None, termination_mode="learned",
                           termination_thresholds=TerminationThresholds()) -> dict:
        result = {}
        for horizon in (1, 8, 32):
            try:
                result[str(horizon)] = self.evaluate_horizon(
                    model, replay, mapping, horizon=horizon, batch=batch,
                    env_range=env_range, source_id=source_id,
                    termination_mode=termination_mode,
                    termination_thresholds=termination_thresholds)
            except ValueError as error:
                result[str(horizon)] = {"available": False, "reason": str(error)}
        return result


def trusted_imagination_horizon(actor_validation: dict, *, min_distinct_starts: int = 8,
                                max_one_step_q_rad: float = 0.04,
                                max_one_step_tcp_mm: float = 12.0,
                                max_eight_step_tcp_mm: float = 50.0,
                                max_32_step_tcp_mm: float = 100.0) -> int:
    """Grow BPTT only when actor-collected heldout dynamics support each horizon."""
    one = actor_validation.get("1", {})
    if (one.get("distinct_start_count", 0) < min_distinct_starts
            or one.get("joint_position_rmse_rad", float("inf")) > max_one_step_q_rad
            or one.get("tcp_position_mean_mm", float("inf")) > max_one_step_tcp_mm
            or one.get("tcp_position_mean_mm", float("inf")) >=
            one.get("constant_velocity_tcp_mean_mm", 0.0)):
        return 0
    eight = actor_validation.get("8", {})
    if eight.get("distinct_start_count", 0) < min_distinct_starts or eight.get("tcp_position_mean_mm", float("inf")) > max_eight_step_tcp_mm:
        return 1
    thirty_two = actor_validation.get("32", {})
    if (thirty_two.get("distinct_start_count", 0) < min_distinct_starts
            or thirty_two.get("tcp_position_mean_mm", float("inf")) > max_32_step_tcp_mm):
        return 8
    return 32
