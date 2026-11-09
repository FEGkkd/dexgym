"""Strict, versioned checkpoint for the injected SuperMimic Core contracts."""

import hashlib
import json
import os
import random
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


class CoreCheckpoint:
    schema_version = "supermimic_core_v2"
    # These affect the sampling protocol, not the executable robot/Actor/MDP.
    # Every other physics field remains part of the immutable inference contract.
    evaluation_protocol_fields = frozenset({
        "num_envs", "seed", "reset_joint_perturb_rad", "reset_random_reference_start",
        "active_reset_fraction", "early_reset_fraction",
        "observation_position_noise_rad", "observation_velocity_noise_rad_s",
        "reference_start_frame", "evaluation_episode_count", "evaluation_trace_path",
        "max_reference_start_frame",
        "goal_train_seed", "goal_eval_seed", "goal_curriculum_level", "evaluation_mode",
        "collection_contract",
    })

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

    @staticmethod
    def _manifest(trainer, modules, physics_config, reference_hashes):
        """Readable executable contract; state-dict shapes alone are insufficient."""
        robot_contract = getattr(trainer.robot, "checkpoint_contract", None)
        task_contract = getattr(trainer.task, "checkpoint_contract", None)
        if not callable(robot_contract) or not callable(task_contract):
            raise ValueError("Robot and Task adapters must provide checkpoint_contract")
        networks = {}
        for name, module in modules.items():
            spec = getattr(module, "network_spec", None)
            if spec is None:
                raise ValueError(f"{name} must provide NetworkSpec for checkpoint")
            norms = {}
            for norm_name in ("input_norm", "delta_norm", "observation_norm", "goal_norm"):
                norm = getattr(module, norm_name, None)
                if norm is not None:
                    norms[norm_name] = {
                        "mode": "running_fp32_v1" if hasattr(norm, "min_std") else "identity_v1",
                        "min_std": getattr(norm, "min_std", None),
                        "max_count": getattr(norm, "max_count", None),
                        "dimension": int(norm.mean.numel()) if hasattr(norm, "mean") else None,
                    }
            networks[name] = {
                "class": getattr(
                    module, "checkpoint_class_name", type(module).__module__ + "." + type(module).__qualname__
                ),
                "network_spec": asdict(spec),
                "goal_latent_tanh": getattr(module, "goal_latent_tanh", None),
                "normalizers": norms,
                "integration_version": "robot_adapter_delta_v1" if name == "world_model" else None,
                "output_activation": "linear_delta_v1" if name == "world_model" else
                                     "linear_action_v1" if name == "actor" else None,
            }
            if name == "actor" and getattr(module, "observation_noise_spec", None) is not None:
                networks[name]["observation_noise"] = module.observation_noise_spec.manifest()
        manifest = {
            "robot_spec": asdict(trainer.robot.spec),
            "robot_contract": robot_contract(),
            "task_spec": asdict(trainer.task.spec),
            "task_contract": task_contract(),
            "termination_mode": trainer.task.termination_mode,
            "networks": networks,
            "physics_config": physics_config,
            "reference_hashes": reference_hashes,
        }
        # Canonical JSON both validates that the contract is data, and makes
        # its digest stable across processes and dictionary insertion order.
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return manifest, hashlib.sha256(encoded.encode()).hexdigest()

    @staticmethod
    def _first_difference(saved, current, path="contract"):
        if isinstance(saved, dict) and isinstance(current, dict):
            for key in sorted(saved.keys() | current.keys()):
                if key not in saved or key not in current:
                    return f"{path}.{key}: missing field"
                found = CoreCheckpoint._first_difference(saved[key], current[key], f"{path}.{key}")
                if found:
                    return found
            return None
        if saved != current:
            return f"{path}: saved={saved!r}, current={current!r}"
        return None

    def save(self, path: str | Path, trainer, *, include_replay: bool | None = None,
             resume_mode: str = "fresh_physics", schedulers: dict | None = None) -> Path:
        if include_replay is not None:
            if resume_mode != "fresh_physics":
                raise ValueError("Specify either include_replay or resume_mode")
            resume_mode = "full" if include_replay else "fresh_physics"
        if resume_mode not in ("fresh_physics", "replay", "full"):
            raise ValueError("Unknown checkpoint resume_mode")
        if resume_mode == "full" and not all(callable(getattr(trainer.collector, method, None))
                                              for method in ("state_dict", "load_state_dict")):
            raise ValueError("Full resume requires a restorable collector state")
        collector_state = trainer.collector.state_dict() if resume_mode == "full" else None
        if resume_mode == "full" and (not isinstance(collector_state, dict) or not collector_state):
            raise ValueError("Full resume collector state_dict must return a nonempty state mapping")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        modules = self._modules(trainer)
        manifest, digest = self._manifest(trainer, modules, self.physics_config,
                                         self.reference_hashes)
        record = {
            "schema": self.schema_version,
            "contract_manifest": manifest,
            "contract_sha256": digest,
            "resume_mode": resume_mode,
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
            "replay": trainer.replay.state_dict() if resume_mode in ("replay", "full") else None,
            "collector": collector_state,
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                    "python": random.getstate(),
                    "observation_noise": (trainer.observation_noise.state_dict()
                                          if getattr(trainer, "observation_noise", None) is not None else None),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(record, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    @classmethod
    def export_fresh_physics(cls, source: str | Path, destination: str | Path) -> Path:
        """Publish an exact model/optimizer copy without a multi-GB physics Replay.

        The result resumes with fresh Newton collection and invalidated trust.
        The original replay checkpoint is left byte-for-byte untouched.
        """
        source, destination = Path(source), Path(destination)
        if source.resolve() == destination.resolve() or destination.exists():
            raise FileExistsError(f"Checkpoint export destination already exists: {destination}")
        record = torch.load(source, map_location="cpu", weights_only=False)
        if record.get("schema") != cls.schema_version:
            raise ValueError("Checkpoint algorithm schema differs")
        manifest = record.get("contract_manifest")
        if not isinstance(manifest, dict):
            raise ValueError("Checkpoint contract manifest is missing")
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if digest != record.get("contract_sha256"):
            raise ValueError("Checkpoint contract SHA256 is invalid")
        if (record.get("resume_mode") not in ("replay", "full")
                or record.get("replay") is None):
            raise ValueError("Export requires a Replay-bearing source checkpoint")
        source_digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                source_digest.update(chunk)
        record["export_provenance"] = {
            "kind": "fresh_physics_from_replay_v1",
            "source_sha256": source_digest.hexdigest(),
            "source_resume_mode": record["resume_mode"],
        }
        record["resume_mode"] = "fresh_physics"
        record["replay"] = None
        record["collector"] = None
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            torch.save(record, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def load(self, path: str | Path, trainer, *, schedulers: dict | None = None,
             allow_actor_schedule_migration: bool = False) -> dict:
        record = torch.load(path, map_location="cpu", weights_only=False)
        modules = self._modules(trainer)
        if record.get("schema") != self.schema_version:
            raise ValueError("Checkpoint algorithm schema differs; use the explicit legacy loader")
        if record["robot_spec"] != asdict(trainer.robot.spec):
            raise ValueError("Robot/action/state schema differs")
        if record["task_spec"] != asdict(trainer.task.spec):
            raise ValueError("Task schema differs")
        current_trainer_config = asdict(trainer.config)
        saved_trainer_config = record["trainer_config"].copy()
        saved_trainer_config.setdefault("actor_update_schedule", "model_trust")
        schedule_migration = {}
        if (allow_actor_schedule_migration
                and saved_trainer_config["actor_update_schedule"] == "model_trust"
                and current_trainer_config["actor_update_schedule"] == "every_iteration"):
            # Only the removed gate's scheduling fields may change on migration.
            # Robot, task, networks, optimizers and all other trainer fields remain strict.
            for key in ("actor_update_schedule", "world_warmup_updates", "trust_interval", "validation_batch"):
                before, after = saved_trainer_config[key], current_trainer_config[key]
                if before != after:
                    schedule_migration[key] = {"previous": before, "current": after}
                    saved_trainer_config[key] = after
        if "actor_grad_value_clip" not in record["trainer_config"]:
            if current_trainer_config["actor_grad_value_clip"] is not None:
                raise ValueError("Trainer configuration differs")
            current_trainer_config.pop("actor_grad_value_clip")
        if "actor_bootstrap_warmup_updates" not in record["trainer_config"]:
            # v2 checkpoints predate this optional schedule and used its
            # zero-warmup behavior. Do not infer a nonzero schedule on resume.
            if current_trainer_config["actor_bootstrap_warmup_updates"] != 0:
                raise ValueError("Trainer configuration differs")
            current_trainer_config.pop("actor_bootstrap_warmup_updates")
        if saved_trainer_config != current_trainer_config:
            raise ValueError("Trainer configuration differs")
        if record["termination_mode"] != trainer.task.termination_mode:
            raise ValueError("Termination schema differs")
        if record["reference_hashes"] != self.reference_hashes or record["physics_config"] != self.physics_config:
            raise ValueError("Reference or physics fingerprint differs")
        current_manifest, current_digest = self._manifest(trainer, modules, self.physics_config,
                                                          self.reference_hashes)
        stored_digest = hashlib.sha256(json.dumps(record["contract_manifest"], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if stored_digest != record["contract_sha256"]:
            raise ValueError("Checkpoint contract SHA256 is invalid")
        if current_digest != stored_digest:
            raise ValueError(self._first_difference(record["contract_manifest"], current_manifest))
        if record["module_signatures"] != self._signatures(modules):
            raise ValueError("Network shape or normalization schema differs")
        if set(record["optimizers"]) != set(trainer.optimizers):
            raise ValueError("Optimizer schema differs")
        if set(record["schedulers"]) != set(schedulers or {}):
            raise ValueError("Scheduler schema differs")
        if record["replay"] is not None:
            if record["resume_mode"] == "full" and (record["collector"] is None or
                    not callable(getattr(trainer.collector, "load_state_dict", None))):
                raise ValueError("Full resume requires a restorable physical collector")
            if (record["replay"]["capacity"] != trainer.replay.capacity or
                    record["replay"]["num_envs"] != trainer.replay.num_envs):
                raise ValueError("Replay physical environment schema differs")
        noise_record = record["rng"].get("observation_noise")
        if noise_record is not None:
            current_noise = getattr(trainer, "observation_noise", None)
            if current_noise is None or noise_record["spec"] != current_noise.spec.manifest():
                raise ValueError("Checkpoint observation noise contract differs")
        for name, module in modules.items():
            module.load_state_dict(record["modules"][name], strict=True)
        for name, optimizer in trainer.optimizers.items():
            optimizer.load_state_dict(record["optimizers"][name])
        for name, scheduler in (schedulers or {}).items():
            scheduler.load_state_dict(record["schedulers"][name])
        trainer.iteration = int(record["training"]["iteration"])
        trainer.counts = {key: record["training"]["counts"][key] for key in trainer.counts}
        # Only legacy model-trust schedules revalidate on fresh physics.
        trainer.allowed_horizon = (trainer.config.max_imagination_horizon
                                   if trainer.config.actor_update_schedule == "every_iteration" else 0)
        trainer.trust_report = {}
        if record["replay"] is not None:
            trainer.replay.load_state_dict(record["replay"],
                                           fresh_physics=record["resume_mode"] != "full")
            if record["resume_mode"] == "full":
                trainer.collector.load_state_dict(record["collector"])
        torch.set_rng_state(record["rng"]["torch"])
        if noise_record is not None:
            trainer.observation_noise.load_state_dict(noise_record)
        np.random.set_state(record["rng"]["numpy"])
        random.setstate(record["rng"]["python"])
        if record["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(record["rng"]["cuda"])
        return {"trainer_config_migration": schedule_migration,
                "replay_restored": record["replay"] is not None,
                "requires_fresh_collection": record["resume_mode"] != "full",
                "resume_mode": record["resume_mode"],
                "git_commit": record["git_commit"], "schema": record["schema"]}

    def load_actor_for_evaluation(self, path: str | Path, *, robot, task, modules,
                                  allow_reference_override: bool = False) -> dict:
        """Strict inference-only load; no optimizer, Replay or collector state is touched."""
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record.get("schema") != self.schema_version:
            raise ValueError("Evaluation requires a v2 semantic checkpoint or explicit legacy evaluator")
        changing_reference = record["reference_hashes"] != self.reference_hashes
        if changing_reference and not allow_reference_override:
            raise ValueError("Reference hashes differ")
        stored_digest = hashlib.sha256(json.dumps(record["contract_manifest"], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if stored_digest != record["contract_sha256"]:
            raise ValueError("Checkpoint contract SHA256 is invalid")
        if record["physics_config"] != record["contract_manifest"]["physics_config"]:
            raise ValueError("Checkpoint physics configuration record differs from manifest")
        active = {"actor": modules.actor, "world_model": modules.world_model,
                  "critic": modules.critic}
        if record["termination_mode"] == "learned":
            active["termination"] = modules.termination
        from types import SimpleNamespace
        current, _ = self._manifest(SimpleNamespace(robot=robot, task=task), active,
                                         self.physics_config, self.reference_hashes)
        saved = deepcopy(record["contract_manifest"])
        current = current.copy()
        target_source_override = None
        if changing_reference:
            if (not getattr(task, "inference_reference_agnostic", False)
                    or not saved["task_contract"].get("target_stream")
                    or saved["task_contract"].get("target_stream")
                        != current["task_contract"].get("target_stream")):
                raise ValueError("Task does not declare compatible causal target-stream evaluation")
            # Only the target source can change. Robot asset, controller,
            # action/goal schema, task math and network semantics remain strict.
            saved["reference_hashes"] = current["reference_hashes"]
            if "reference_hashes" in saved["task_contract"]:
                saved["task_contract"]["reference_hashes"] = current["task_contract"].get(
                    "reference_hashes")
            if "reference_asset_sha256" in saved["physics_config"]:
                saved["physics_config"]["reference_asset_sha256"] = current[
                    "physics_config"].get("reference_asset_sha256")
            target_source_override = {"training": record["reference_hashes"],
                                      "evaluation": self.reference_hashes}
        saved["physics_config"] = {key: value for key, value in saved["physics_config"].items()
                                   if key not in self.evaluation_protocol_fields}
        current["physics_config"] = {key: value for key, value in current["physics_config"].items()
                                     if key not in self.evaluation_protocol_fields}
        difference = self._first_difference(saved, current)
        if difference:
            raise ValueError(difference)
        if self._signatures(active) != record["module_signatures"]:
            raise ValueError("Network shape or normalization schema differs")
        modules.actor.load_state_dict(record["modules"]["actor"], strict=True)
        modules.actor.eval()
        protocol = {key: self.physics_config.get(key) for key in sorted(self.evaluation_protocol_fields)
                    if key in self.physics_config or key in record["physics_config"]}
        overrides = {key: value for key, value in protocol.items()
                     if record["physics_config"].get(key) != value}
        return {"schema": record["schema"], "contract_sha256": stored_digest,
                "git_commit": record["git_commit"], "policy_source": str(path),
                "evaluation_protocol": protocol, "evaluation_overrides": overrides,
                "target_source_override": target_source_override}

    @staticmethod
    def load_legacy_m43_v1_actor_for_evaluation(
            path: str | Path, actor, *, robot_spec, task_spec,
            reference_hashes: dict, physics_config: dict, source_git_commit: str,
            assumed_goal_latent_tanh: bool, assumed_normalize: bool) -> dict:
        """Explicit inference-only M4.3 migration; v1 never recorded these two semantics."""
        record = torch.load(path, map_location="cpu", weights_only=False)
        if record.get("schema") != "supermimic_core_v1":
            raise ValueError("Explicit M4.3 legacy loader requires a v1 checkpoint")
        current_robot = asdict(robot_spec)
        current_robot.pop("context_evolution", None)  # introduced by v2
        if record["robot_spec"] != current_robot or record["task_spec"] != asdict(task_spec):
            raise ValueError("Legacy Robot or Task schema differs")
        if (record["reference_hashes"] != reference_hashes
                or record["physics_config"] != physics_config
                or record["git_commit"] != source_git_commit):
            raise ValueError("Legacy source, reference or physics provenance differs")
        if (actor.goal_latent_tanh != assumed_goal_latent_tanh
                or actor.network_spec.normalize != assumed_normalize):
            raise ValueError("Legacy network assumptions differ from the requested Actor")
        signature = CoreCheckpoint._signatures({"actor": actor})["actor"]
        if signature != record["module_signatures"]["actor"]:
            raise ValueError("Legacy Actor shape or normalizer schema differs")
        actor.load_state_dict(record["modules"]["actor"], strict=True)
        actor.eval()
        return {"schema": "supermimic_core_v1", "evaluation_only": True,
                "unrecorded_semantics_assumed": {
                    "goal_latent_tanh": assumed_goal_latent_tanh,
                    "normalize": assumed_normalize,
                    "network_spec": asdict(actor.network_spec)},
                "source_git_commit": source_git_commit}
