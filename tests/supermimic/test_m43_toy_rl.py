"""Known-dynamics Toy physics exercises the same model-based trainer twice."""

import pytest
import torch

from dexgym.tasks.supermimic_toy import ToyPhysicalEvaluator, make_toy_trainer


@pytest.mark.parametrize(("task_id", "goal_dim", "observation_dim"),
                         [("position", 1, 3), ("position_velocity", 2, 4)])
def test_toy_physics_uses_distinct_task_schemas_and_finite_training(task_id, goal_dim, observation_dim):
    torch.manual_seed(71)
    trainer = make_toy_trainer(task_id, num_envs=8, world_hidden=32,
                               world_batch=16, imagination_batch=8)
    assert trainer.task.spec.goal_dim == goal_dim
    assert trainer.task.spec.observation_dim == observation_dim
    initial = ToyPhysicalEvaluator(trainer.task).evaluate(trainer.actor, steps=24)
    for _ in range(24):
        trainer.run_iteration()
    final = ToyPhysicalEvaluator(trainer.task).evaluate(trainer.actor, steps=24)
    assert trainer.counts["world_updates"] > 0
    assert trainer.counts["actor_updates"] > 0
    assert trainer.counts["critic_updates"] > 0
    assert all(torch.isfinite(torch.tensor([initial["reward_mean"], final["reward_mean"]])))
    assert sum(trainer.counts[name] for name in
               ("nonfinite_world", "nonfinite_actor", "nonfinite_critic")) == 0
    assert int(trainer.actor.observation_norm.count) == 24 * 6
