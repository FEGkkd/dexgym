"""Bounded two-schema Torch-physics SuperMimic learning validation."""

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch

from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.policy import export_policy
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.tasks.supermimic_toy import ToyPhysicalEvaluator, make_toy_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 200:
        parser.error("Toy validation is bounded to 1-200 iterations")
    args.output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = {"git_commit": commit, "iterations": args.iterations, "seed": args.seed,
              "tasks": {}}
    for task_id in ("position", "position_velocity"):
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        trainer = make_toy_trainer(task_id, num_envs=16)
        evaluator = ToyPhysicalEvaluator(trainer.task, num_envs=16)
        before = evaluator.evaluate(trainer.actor)
        for _ in range(args.iterations):
            trainer.run_iteration()
        after = evaluator.evaluate(trainer.actor)
        manager = CoreCheckpoint(git_commit=commit, reference_hashes={})
        checkpoint = manager.save(args.output / f"{task_id}_checkpoint.pt",
                                  trainer, include_replay=True)
        network = NetworkSpec(world_hidden=(64, 64), goal_hidden=(32,),
                              actor_hidden=(64, 32), critic_hidden=(32,), latent_dim=8)
        policy_path = args.output / f"{task_id}_policy.pt"
        policy_metadata = export_policy(policy_path, trainer.actor, trainer.robot.spec,
                                        trainer.task.spec, network,
                                        reference_hashes={}, git_commit=commit)
        resumed = make_toy_trainer(task_id, num_envs=16)
        resume = manager.load(checkpoint, resumed)
        report["tasks"][task_id] = {
            "before": before, "after": after, "counts": trainer.counts,
            "trusted_horizon": trainer.allowed_horizon,
            "checkpoint": str(checkpoint), "policy": str(policy_path),
            "policy_sha256": policy_metadata["policy_sha256"],
            "resume": resume, "resumed_replay_steps": resumed.replay.steps,
            "wall_time_s": time.perf_counter() - started}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
