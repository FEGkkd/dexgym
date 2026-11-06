from types import SimpleNamespace

import torch

from dexgym.algorithms.supermimic.core.evaluator import PhysicalEvaluator


class _Actor(torch.nn.Module):
    def forward(self, history, goal):
        return torch.zeros(len(history), 1)


class _Collector:
    num_envs = 4
    action_noise_std = 0.0

    def __init__(self):
        self.age = torch.zeros(4, dtype=torch.long)

    def step(self, actor, *, policy):
        self.age += 1
        fail = torch.tensor([self.age[0] == 3, False, False, False])
        success_end = self.age[1] == 8
        truncate = torch.tensor([False, success_end, self.age[2] == 5, False])
        reward = torch.ones(4)
        item = SimpleNamespace(reward=reward, terminated=fail, truncated=truncate,
                               failure_reason=fail.int() * 7, age=self.age.clone(),
                               success=torch.tensor([False, self.age[1] >= 7, False, False]))
        self.age = torch.where(fail | truncate, 0, self.age)
        return item


class _Task:
    success_hold_steps = 2

    @staticmethod
    def success_mask(item):
        return item.success

    @staticmethod
    def evaluate_metrics(trajectory):
        return {"last_age": float(trajectory[-1].age[0])}


def test_episode_evaluator_separates_auto_resets_and_success_hold():
    report = PhysicalEvaluator(_Collector, _Task()).evaluate(_Actor(), steps=10)
    episodes = report["episodes"]
    assert len([e for e in episodes if e["env_id"] == 0]) == 4
    assert [e["episode_length"] for e in episodes if e["env_id"] == 0] == [3, 3, 3, 1]
    assert report["hard_terminations"] == 3
    assert report["truncations"] == 3
    assert report["episode_success_rate"] == 1 / len(episodes)
    assert next(e for e in episodes if e["env_id"] == 1)["first_success_step"] == 7
    assert next(e for e in episodes if e["env_id"] == 1)["goal_hold_duration"] == 2
    assert next(e for e in episodes if e["env_id"] == 2)["truncation_reason"] == "time_limit"
    assert next(e for e in episodes if e["env_id"] == 0)["first_failure_step"] == 3


def test_independent_evaluation_uses_first_episode_per_env(tmp_path):
    trace = tmp_path / "physics.pt"
    report = PhysicalEvaluator(_Collector, _Task(),
        evaluation_kind="independent_physics_evaluation").evaluate(
            _Actor(), steps=10, one_episode_per_env=True, trace_path=trace)
    assert report["episode_count"] == 4
    assert report["episode_success_rate"] == 0.25
    assert report["hard_terminations"] == 1
    assert report["truncations"] == 2
    assert report["evaluated_transitions"] == 3 + 8 + 5 + 10
    assert trace.exists()
    saved = torch.load(trace, weights_only=False)
    assert bool(saved["evaluated_mask"][2][0])
    assert not bool(saved["evaluated_mask"][3][0])
