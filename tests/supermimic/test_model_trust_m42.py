import torch

from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.algorithms.supermimic.trust_evaluator import ModelTrustEvaluator, trusted_imagination_horizon


class ToyFK:
    def __call__(self, q):
        positions = torch.zeros(len(q), 2, 3, 3)
        positions[:, 0, 2, 0] = q[:, 0]
        positions[:, 1, 2, 0] = q[:, 7]
        rotations = torch.eye(3).repeat(len(q), 2, 3, 1, 1)
        return positions, rotations


class ToyWorld:
    def __init__(self, drift=0.0):
        self.drift = drift

    def __call__(self, state, action, mapping):
        next_state = state.clone()
        next_state[:, :14] += 0.01 + self.drift
        return next_state

    def termination_probability(self, state, goal):
        return torch.zeros(len(state))


class ToyReference:
    def reference_at(self, frame):
        return {"link_pos": torch.zeros(len(frame), 2, 3, 3)}

    def reference_features_at(self, frame):
        return torch.zeros(len(frame), 90)


def test_model_trust_evaluates_actor_data_separately_from_teacher_and_baseline():
    replay = MotionReplay(20, 1, "cpu")
    for step in range(16):
        state = torch.zeros(1, 42)
        state[:, :14] = step * 0.01
        next_state = state.clone()
        next_state[:, :14] += 0.01
        replay.append(state, torch.zeros(1, 14), next_state, torch.zeros(1, 10, 56),
                      torch.tensor([step]), torch.tensor([False]),
                      source_id=torch.tensor([0 if step < 8 else 1]))
    evaluator = ModelTrustEvaluator(ToyFK(), ToyReference())
    exact = evaluator.evaluate_horizon(ToyWorld(), replay, mapping=None, horizon=8,
                                       batch=16, source_id=0)
    assert exact["sample_count"] == 16 and exact["source_id"] == 0
    assert exact["joint_position_rmse_rad"] < 1e-6
    assert exact["tcp_position_mean_mm"] < 1e-3
    assert exact["constant_velocity_tcp_mean_mm"] > 1
    assert sum(item["count"] for item in exact["termination"]["tcp_error_bins_m"][1:]) > 0
    wrong = evaluator.evaluate_horizon(ToyWorld(0.02), replay, mapping=None, horizon=8,
                                       batch=16, source_id=0)
    assert wrong["joint_position_rmse_rad"] > 0.1


def test_actor_trust_grows_from_one_step_to_eight_and_thirty_two():
    trust = {"1": {"distinct_start_count": 16, "joint_position_rmse_rad": 0.01,
                   "tcp_position_mean_mm": 5.0, "constant_velocity_tcp_mean_mm": 20.0}}
    assert trusted_imagination_horizon(trust) == 1
    trust["8"] = {"distinct_start_count": 16, "tcp_position_mean_mm": 10.0}
    assert trusted_imagination_horizon(trust) == 8
    trust["32"] = {"distinct_start_count": 16, "tcp_position_mean_mm": 40.0}
    assert trusted_imagination_horizon(trust) == 32
    trust["1"]["joint_position_rmse_rad"] = 0.2
    assert trusted_imagination_horizon(trust) == 0
    trust["1"]["joint_position_rmse_rad"] = 0.01
    trust["1"]["tcp_position_mean_mm"] = 21.0
    assert trusted_imagination_horizon(trust) == 0
