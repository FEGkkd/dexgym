"""Validated algorithm/robot/task/environment configuration boundary."""

from dataclasses import dataclass
import json
from pathlib import Path

from .specs import NetworkSpec
from .trainer import TrainerConfig
from .trust_evaluator import TrustConfig


@dataclass(frozen=True)
class CoreExperimentConfig:
    robot_id: str
    action_schema: str
    task_id: str
    reference_path: str | None
    backend: str
    num_envs: int
    iterations: int
    eval_steps: int
    seed: int
    network: NetworkSpec
    trainer: TrainerConfig
    trust: TrustConfig

    @classmethod
    def from_json(cls, path: str | Path):
        record = json.loads(Path(path).read_text())
        if set(record) != {"algorithm", "robot", "task", "environment"}:
            raise ValueError("Experiment config needs algorithm, robot, task and environment")
        robot, task, environment, algorithm = (record[key] for key in
                                               ("robot", "task", "environment", "algorithm"))
        network = algorithm["network"].copy()
        for name in ("world_hidden", "goal_hidden", "actor_hidden",
                     "critic_hidden", "termination_hidden"):
            if name in network:
                network[name] = tuple(network[name])
        trust = algorithm["trust"].copy()
        if "max_rmse" in trust:
            trust["max_rmse"] = tuple(trust["max_rmse"])
        result = cls(robot_id=robot["id"], action_schema=robot["action_schema"],
                     task_id=task["id"], reference_path=task.get("reference_path"),
                     backend=environment["backend"], num_envs=int(environment["num_envs"]),
                     iterations=int(environment["iterations"]),
                     eval_steps=int(environment["eval_steps"]), seed=int(environment["seed"]),
                     network=NetworkSpec(**network),
                     trainer=TrainerConfig(**algorithm["trainer"]),
                     trust=TrustConfig(**trust))
        if (not result.robot_id or not result.task_id or not result.action_schema
                or result.num_envs < 4 or result.iterations < 1 or result.eval_steps < 1):
            raise ValueError("Experiment IDs, environments, steps and iterations must be valid")
        return result
