"""Strict, versioned checkpoint for the injected SuperMimic Core contracts."""

from dataclasses import asdict
from pathlib import Path
import os
import random

import numpy as np
import torch


class CoreCheckpoint:
    schema_version = "supermimic_core_v1"

    def __init__(self, *, git_commit: str, reference_hashes: dict[str, str],
                 physics_config: dict | None = None):
        self.git_commit = git_commit
        self.reference_hashes = reference_hashes.copy()
        self.physics_config = (physics_config or {}).copy()

    @staticmethod
    def _modules(trainer):
        modules = {"actor": trainer.actor, "world_model": trainer.world_model,
                   "critic": trainer.critic}
        if trainer.termination_head is not None:
            modules["termination"] = trainer.termination_head
        return modules

    @staticmethod
    def _signatures(modules):
        return {name: {key: (tuple(value.shape), str(value.dtype))
                       for key, value in module.state_dict().items()}
                for name, module in modules.items()}

    def save(self, path: str | Path, trainer, *, include_replay: bool = True,
             schedulers: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        modules = self._modules(trainer)
        record = {
            "schema": self.schema_version,
            "robot_spec": asdict(trainer.robot.spec),
            "task_spec": asdict(trainer.task.spec),
            "trainer_config": asdict(trainer.config),
            "termination_mode": trainer.task.termination_mode,
            "git_commit": self.git_commit,
            "reference_hashes": self.reference_hashes,
            "physics_config": self.physics_config,
            "module_signatures": self._signatures(modules),
            "modules": {key: module.state_dict() for key, module in modules.items()},
            "optimizers": {key: value.state_dict() for key, value in trainer.optimizers.items()},
            "schedulers": {key: value.state_dict() for key, value in (schedulers or {}).items()},
            "training": {"iteration": trainer.iteration, "counts": trainer.counts.copy(),
                         "allowed_horizon": trainer.allowed_horizon,
                         "trust_report": trainer.trust_report},
            "replay": trainer.replay.state_dict() if include_replay else None,
            "collector": (trainer.collector.state_dict() if include_replay and
                          hasattr(trainer.collector, "state_dict") else None),
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                    "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(record, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def load(self, path: str | Path, trainer, *, schedulers: dict | None = None) -> dict:
        record = torch.load(path, map_location="cpu", weights_only=False)
        modules = self._modules(trainer)
        if record.get("schema") != self.schema_version:
            raise ValueError("Checkpoint algorithm schema differs; use the explicit legacy loader")
        if record["robot_spec"] != asdict(trainer.robot.spec):
            raise ValueError("Robot/action/state schema differs")
        if record["task_spec"] != asdict(trainer.task.spec):
            raise ValueError("Task schema differs")
        if record["trainer_config"] != asdict(trainer.config):
            raise ValueError("Trainer configuration differs")
        if record["termination_mode"] != trainer.task.termination_mode:
            raise ValueError("Termination schema differs")
        if record["reference_hashes"] != self.reference_hashes or record["physics_config"] != self.physics_config:
            raise ValueError("Reference or physics fingerprint differs")
        if record["module_signatures"] != self._signatures(modules):
            raise ValueError("Network shape or normalization schema differs")
        if set(record["optimizers"]) != set(trainer.optimizers):
            raise ValueError("Optimizer schema differs")
        if set(record["schedulers"]) != set(schedulers or {}):
            raise ValueError("Scheduler schema differs")
        if record["replay"] is not None:
            if record["collector"] is None or not hasattr(trainer.collector, "load_state_dict"):
                raise ValueError("Saved Replay requires a resumable physical collector")
            if (record["replay"]["capacity"] != trainer.replay.capacity or
                    record["replay"]["num_envs"] != trainer.replay.num_envs):
                raise ValueError("Replay physical environment schema differs")
        for name, module in modules.items():
            module.load_state_dict(record["modules"][name], strict=True)
        for name, optimizer in trainer.optimizers.items():
            optimizer.load_state_dict(record["optimizers"][name])
        for name, scheduler in (schedulers or {}).items():
            scheduler.load_state_dict(record["schedulers"][name])
        trainer.iteration = int(record["training"]["iteration"])
        trainer.counts = record["training"]["counts"].copy()
        # Trust is tied to the current actor distribution and held-out real
        # transitions. A resumed process must revalidate it before BPTT.
        trainer.allowed_horizon = 0
        trainer.trust_report = {}
        if record["replay"] is not None:
            trainer.replay.load_state_dict(record["replay"])
            trainer.collector.load_state_dict(record["collector"])
        torch.set_rng_state(record["rng"]["torch"])
        np.random.set_state(record["rng"]["numpy"])
        random.setstate(record["rng"]["python"])
        if record["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(record["rng"]["cuda"])
        return {"replay_restored": record["replay"] is not None,
                "requires_fresh_collection": record["replay"] is None,
                "git_commit": record["git_commit"], "schema": record["schema"]}
