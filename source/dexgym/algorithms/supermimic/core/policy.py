"""Strict PyTorch inference export for task-shaped SuperMimic Actors."""

from dataclasses import asdict
import hashlib
from pathlib import Path

import torch

from .models import GoalActor
from .specs import NetworkSpec, RobotSpec, TaskSpec


def policy_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class InferencePolicy:
    def __init__(self, actor: GoalActor, robot: RobotSpec, task: TaskSpec,
                 metadata: dict):
        self.actor, self.robot_spec, self.task_spec = actor.eval(), robot, task
        self.metadata = metadata

    @torch.no_grad()
    def act(self, history: torch.Tensor, goal: torch.Tensor, state: torch.Tensor,
            robot_adapter) -> tuple[torch.Tensor, torch.Tensor]:
        if robot_adapter.spec != self.robot_spec or state.shape != (len(history), self.robot_spec.state_dim):
            raise ValueError("Inference robot or state schema differs from export")
        raw = self.actor(history, goal)
        return raw, robot_adapter.executed_command(state, raw)


def export_policy(path: str | Path, actor: GoalActor, robot: RobotSpec,
                  task: TaskSpec, network: NetworkSpec, *, reference_hashes: dict[str, str],
                  git_commit: str) -> dict:
    if actor.robot_spec != robot or actor.task_spec != task:
        raise ValueError("Export Actor contract differs")
    if actor.network_spec != network or actor.goal_latent_tanh != network.goal_latent_tanh:
        raise ValueError("Export NetworkSpec differs from the executing Actor")
    state = actor.state_dict()
    digest = policy_sha256(state)
    metadata = {"schema": "supermimic_pytorch_policy_v1", "robot": asdict(robot),
                "task": asdict(task), "network": asdict(network),
                "reference_hashes": reference_hashes.copy(), "git_commit": git_commit,
                "control_frequency_hz": 1 / robot.control_dt,
                "asset_sha256": robot.asset_sha256, "policy_sha256": digest}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "actor": state}, path)
    return metadata


def load_policy(path: str | Path, *, device="cpu", expected_robot: RobotSpec | None = None,
                expected_task: TaskSpec | None = None) -> InferencePolicy:
    record = torch.load(path, map_location="cpu", weights_only=False)
    metadata = record["metadata"]
    if metadata.get("schema") != "supermimic_pytorch_policy_v1":
        raise ValueError("Policy export schema differs")
    robot_data = metadata["robot"].copy()
    robot_data["joint_order"] = tuple(robot_data.get("joint_order", ()))
    robot, task = RobotSpec(**robot_data), TaskSpec(**metadata["task"])
    if expected_robot is not None and expected_robot != robot:
        raise ValueError("Policy robot/action schema differs")
    if expected_task is not None and expected_task != task:
        raise ValueError("Policy task/goal schema differs")
    if policy_sha256(record["actor"]) != metadata["policy_sha256"]:
        raise ValueError("Policy SHA256 differs")
    network_data = metadata["network"].copy()
    for key in ("world_hidden", "goal_hidden", "actor_hidden", "critic_hidden",
                "termination_hidden"):
        network_data[key] = tuple(network_data[key])
    actor = GoalActor(robot, task, NetworkSpec(**network_data)).to(device)
    actor.load_state_dict(record["actor"], strict=True)
    return InferencePolicy(actor, robot, task, metadata)
