"""Bounded offline WM8 fit diagnostic on existing real Replay; no Actor update."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.optimization import safe_gradient_step
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.trust_evaluator import ModelTrustEvaluator, TrustConfig
from dexgym.tasks.marvin_supermimic.core_plugins import MarvinCoreRobot
from dexgym.tasks.marvin_supermimic.se3_reach_hold import MarvinBimanualSE3ReachHold
from dexgym.tasks.marvin_supermimic.se3_trust import (
    SE3DualDistributionTrustEvaluator, SE3GeometryTrustConfig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.updates <= 500:
        parser.error("Offline WM8 diagnosis is bounded to 1-500 updates")
    torch.manual_seed(5331)
    record = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if record.get("replay") is None or record["robot_spec"]["action_schema"] != "arm_joint_delta_v1":
        parser.error("Expected an M5 delta-action checkpoint with real Replay")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), device,
                            action_schema="arm_joint_delta_v1")
    task = MarvinBimanualSE3ReachHold(robot)
    network = NetworkSpec(**record["contract_manifest"]["networks"]["world_model"]["network_spec"])
    modules = NetworkFactory.build(robot.spec, task.spec, network, robot_adapter=robot)
    model = modules.world_model.to(device)
    model.load_state_dict(record["modules"]["world_model"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    optimizer.load_state_dict(record["optimizers"]["world_model"])
    saved = record["replay"]
    replay = SchemaReplay(saved["capacity"], saved["num_envs"], robot.spec, task.spec, device)
    replay.load_state_dict(saved, fresh_physics=False)
    heldout = (int(0.75 * replay.num_envs), replay.num_envs)
    collection = record.get("physics_config", {}).get("collection_contract", {})
    if collection.get("version") == "dual_source_geometry_trust_v2":
        noisy = collection["noisy_trust_config"].copy()
        noisy["max_rmse"] = tuple(noisy["max_rmse"])
        geometry = {key: tuple(value) for key, value in
                    collection["geometry_trust_config"].items()}
        noisy_start = collection["train_env_count"]
        deterministic_start = noisy_start + collection["validation_exploration_env_count"]
        evaluator = SE3DualDistributionTrustEvaluator(
            TrustConfig(**noisy), SE3GeometryTrustConfig(**geometry),
            noisy_validation_range=(noisy_start, deterministic_start),
            deterministic_validation_range=(deterministic_start, replay.num_envs))
    else:
        evaluator = ModelTrustEvaluator(TrustConfig(min_distinct_starts=8,
            max_rmse=(0.9, 1.0, 1.1), require_baseline_improvement=True, metric="normalized"))
    records = []

    def validate(step):
        report = evaluator.evaluate_available(model, replay, robot, batch=128,
                                               env_range=heldout, source_id=0)
        records.append({"additional_world_updates": step,
                        "trusted_horizon": evaluator.trusted_horizon(report),
                        "validation": report})

    validate(0)
    losses = []
    for step in range(1, args.updates + 1):
        window = replay.sample_windows(64, 8, env_range=(0, heldout[0]))
        loss, _ = model.autoregressive_loss(window, robot)
        result = safe_gradient_step(loss, model.parameters(), optimizer)
        if not result["updated"]:
            raise RuntimeError(f"Offline World Model update {step} rejected: {result['reason']}")
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == args.updates:
            validate(step)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"source_checkpoint": str(args.checkpoint),
              "diagnostic_only": True, "actor_updated": False,
              "newton_transitions_collected": 0,
              "masked_world_model_updates": args.updates,
              "train_loss_first_100_mean": sum(losses[:100]) / len(losses[:100]),
              "train_loss_last_100_mean": sum(losses[-100:]) / len(losses[-100:]),
              "records": records}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    torch.save({"source_checkpoint": str(args.checkpoint),
                "diagnostic_only": True,
                "world_model": model.state_dict()}, args.output / "world_model_diagnostic.pt")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
