import pytest
import torch

from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.policy import export_policy, load_policy
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.tasks.supermimic_toy import ToyGoalTask, ToyRobot


def test_task_shaped_policy_export_cpu_gpu_and_schema_guard(tmp_path):
    robot, task = ToyRobot(), ToyGoalTask("position_velocity")
    network = NetworkSpec(world_hidden=(16,), goal_hidden=(16,), actor_hidden=(16,),
                          critic_hidden=(16,), latent_dim=4)
    actor = NetworkFactory.build(robot.spec, task.spec, network).actor
    actor.update_norms(torch.rand(3, 2, 4), torch.rand(3, 1, 2))
    history, goal, state = torch.randn(3, 2, 4), torch.randn(3, 1, 2), torch.randn(3, 2)
    before = actor(history, goal)
    path = tmp_path / "policy.pt"
    metadata = export_policy(path, actor, robot.spec, task.spec, network,
                             reference_hashes={}, git_commit="test")
    loaded = load_policy(path, expected_robot=robot.spec, expected_task=task.spec)
    raw, command = loaded.act(history, goal, state, robot)
    assert metadata["policy_sha256"] == loaded.metadata["policy_sha256"]
    assert torch.equal(before, raw)
    assert torch.equal(command, robot.executed_command(state, raw))
    with pytest.raises(ValueError, match="task/goal"):
        load_policy(path, expected_task=ToyGoalTask("position").spec)
    if torch.cuda.is_available():
        loaded_gpu = load_policy(path, device="cuda", expected_task=task.spec)
        raw_gpu, _ = loaded_gpu.act(history.cuda(), goal.cuda(), state.cuda(), robot)
        assert torch.allclose(raw_gpu.cpu(), raw, atol=1e-6, rtol=1e-6)
