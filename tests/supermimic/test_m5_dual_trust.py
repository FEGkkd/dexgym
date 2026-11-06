import pytest
from dexgym.algorithms.supermimic.core.trust_evaluator import TrustConfig
from dexgym.tasks.marvin_supermimic.se3_trust import SE3DualDistributionTrustEvaluator, SE3GeometryTrustConfig


def _report():
    noisy = {}
    deterministic = {}
    for horizon in (1, 8, 32):
        noisy[str(horizon)] = {
            "distinct_start_count": 16, "normalized_delta_rmse": 0.5,
            "normalized_zero_delta_baseline_rmse": 0.8}
        deterministic[str(horizon)] = {
            "distinct_start_count": 16, "normalized_delta_rmse": 0.5,
            "normalized_zero_delta_baseline_rmse": 0.1,
            "tcp_position_p95_max_m": 0.002,
            "tcp_so3_p95_max_rad": 0.01,
            "joint_velocity_rmse_rad_s": 0.05}
    return {"noisy": noisy, "deterministic": deterministic}


def test_dual_trust_requires_action_response_and_deterministic_geometry():
    gate = SE3DualDistributionTrustEvaluator(
        TrustConfig(max_rmse=(0.9, 1.0, 1.1), metric="normalized"),
        SE3GeometryTrustConfig(), noisy_validation_range=(12, 14),
        deterministic_validation_range=(14, 16))
    report = _report()
    # A near-static zero-delta baseline wins in the deterministic lane, but
    # this is not evidence that the model is unsafe for action-conditioned BPTT.
    assert gate.trusted_horizon(report) == 32
    report["deterministic"]["8"]["tcp_position_p95_max_m"] = 0.008
    assert gate.trusted_horizon(report) == 1
    report["noisy"]["1"]["normalized_delta_rmse"] = 0.9
    assert gate.trusted_horizon(report) == 0


def test_full32_profile_never_admits_short_actor_imagination():
    gate = SE3DualDistributionTrustEvaluator(
        TrustConfig(max_rmse=(0.9, 1.0, 1.1), metric="normalized"),
        SE3GeometryTrustConfig(), noisy_validation_range=(12, 14),
        deterministic_validation_range=(14, 16), minimum_actor_horizon=32)
    report = _report()
    assert gate.trusted_horizon(report) == 32
    report["noisy"]["32"]["normalized_delta_rmse"] = 1.2
    assert gate.trusted_horizon(report) == 0
    report = _report()
    report["deterministic"]["32"]["tcp_position_p95_max_m"] = 0.006
    assert gate.trusted_horizon(report) == 0
    with pytest.raises(ValueError, match="minimum_actor_horizon"):
        SE3DualDistributionTrustEvaluator(
            TrustConfig(max_rmse=(0.9, 1.0, 1.1), metric="normalized"),
            SE3GeometryTrustConfig(), noisy_validation_range=(12, 14),
            deterministic_validation_range=(14, 16), minimum_actor_horizon=2)
