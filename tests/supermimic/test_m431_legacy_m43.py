import os
from pathlib import Path
import subprocess

import pytest
import torch

from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.models import GoalActor
from dexgym.algorithms.supermimic.core.specs import NetworkSpec, RobotSpec, TaskSpec


OLD_ROOT = Path("/home/ubuntu/.codex/worktrees/3d7b/dexgym")
OLD_CHECKPOINT = Path(
    "/home/ubuntu/dexgym_data/marvin_supermimic_m43/"
    "reach_smoke_557ea9f/reach_m43_smoke.pt")
PYTHON = "/home/ubuntu/isaacsim-6.0.1/bin/python"


@pytest.mark.skipif(not OLD_ROOT.exists() or not OLD_CHECKPOINT.exists(),
                    reason="Pinned original M4.3 source or checkpoint is unavailable")
def test_explicit_m43_v1_actor_load_matches_original_source(tmp_path):
    record = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
    robot, task = RobotSpec(**record["robot_spec"]), TaskSpec(**record["task_spec"])
    network = NetworkSpec(world_hidden=(64, 64), goal_hidden=(64,),
                          actor_hidden=(128, 64), critic_hidden=(64,), latent_dim=16)
    actor = GoalActor(robot, task, network)
    report = CoreCheckpoint.load_legacy_m43_v1_actor_for_evaluation(
        OLD_CHECKPOINT, actor, robot_spec=robot, task_spec=task,
        reference_hashes=record["reference_hashes"],
        physics_config=record["physics_config"], source_git_commit=record["git_commit"],
        assumed_goal_latent_tanh=True, assumed_normalize=True)
    assert report["evaluation_only"]
    torch.manual_seed(431)
    history = torch.randn(3, task.history_length, task.observation_dim)
    goal = torch.randn(3, task.goal_horizon, task.goal_dim)
    current = actor(history, goal).detach()
    fixture = tmp_path / "old_output.pt"
    old_script = """
import sys, torch
from dexgym.algorithms.supermimic.core.models import GoalActor
from dexgym.algorithms.supermimic.core.specs import NetworkSpec, RobotSpec, TaskSpec
r=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
actor=GoalActor(RobotSpec(**r['robot_spec']),TaskSpec(**r['task_spec']),
 NetworkSpec(world_hidden=(64,64),goal_hidden=(64,),actor_hidden=(128,64),
 critic_hidden=(64,),latent_dim=16))
actor.load_state_dict(r['modules']['actor'],strict=True)
torch.manual_seed(431)
h=torch.randn(3,4,48)
g=torch.randn(3,1,6)
torch.save(actor(h,g).detach(),sys.argv[2])
"""
    env = dict(os.environ, PYTHONPATH=str(OLD_ROOT / "source"))
    subprocess.run([PYTHON, "-c", old_script, str(OLD_CHECKPOINT), str(fixture)],
                   cwd=OLD_ROOT, env=env, check=True, capture_output=True, text=True)
    old = torch.load(fixture, weights_only=True)
    assert torch.allclose(current, old, rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="assumptions"):
        CoreCheckpoint.load_legacy_m43_v1_actor_for_evaluation(
            OLD_CHECKPOINT, actor, robot_spec=robot, task_spec=task,
            reference_hashes=record["reference_hashes"],
            physics_config=record["physics_config"], source_git_commit=record["git_commit"],
            assumed_goal_latent_tanh=False, assumed_normalize=True)
