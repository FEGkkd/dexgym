"""Collect fresh Newton boundary failures and audit a learned termination checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

from dexgym.algorithms.supermimic.networks import DeltaWorldModel, GoalActor
from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.algorithms.supermimic.trust_evaluator import ModelTrustEvaluator
from dexgym.algorithms.supermimic.termination import TaskTerminationHead
from dexgym.tasks.marvin_supermimic.collector import AlignedNewtonCollector
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from dexgym.tasks.marvin_supermimic.robot_adapter import DEFAULT_MJCF


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    record = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    metadata, configuration = record["metadata"], record["config"]
    if metadata["reference_hashes"].get(str(args.reference)) != _hash(args.reference):
        raise ValueError("Reference hash differs from learned checkpoint")
    if metadata["robot_asset_hashes"]["mjcf"] != _hash(DEFAULT_MJCF):
        raise ValueError("Robot asset hash differs from learned checkpoint")
    if metadata["termination_mode"] != "learned":
        raise ValueError("Checkpoint does not contain a trained learned termination mode")
    legacy_head = any(name.startswith("termination_head.")
                      for name in record["models"]["world_model"])
    model = DeltaWorldModel(hidden=configuration["world_hidden"],
                            legacy_termination_head=legacy_head).to("cuda").eval()
    model.load_state_dict(record["models"]["world_model"], strict=True)
    termination = None
    if "termination" in record["models"]:
        termination = TaskTerminationHead(42, 90).to("cuda").eval()
        termination.load_state_dict(record["models"]["termination"], strict=True)
    actor = GoalActor(goal_latent_tanh=True,
                      normalization_mode=configuration["normalization_mode"]).to("cuda").eval()
    actor.load_state_dict(record["models"]["actor"], strict=True)
    cfg = MarvinWaveDirectCfg()
    cfg.reference_path = str(args.reference)
    cfg.action_schema = metadata["action_schema"]
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.random_reference_start = False
    cfg.reset_joint_perturb_rad = 0.002
    cfg.initialize_hands_from_pd_replay = cfg.action_schema == "legacy_wave_v1"
    cfg.audit_contacts_each_physics_step = True
    cfg.episode_length_s = 28.0
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Wave-Direct-v0", cfg=cfg)
        try:
            replay = MotionReplay(args.steps, args.num_envs, "cuda")
            collector = AlignedNewtonCollector(env, actor, replay)
            for _ in range(args.steps):
                collector.step(policy="boundary_probe")
            native = env.unwrapped
            evaluator = ModelTrustEvaluator(native.fk, native.motion)
            report = evaluator.evaluate_horizon(
                model, replay, native.action_map, horizon=1,
                batch=min(4096, args.steps * args.num_envs * 2), source_id=2,
                termination_module=termination)
            report["newton_transitions"] = args.steps * args.num_envs
            report["newton_hard_terminations"] = collector.terminations
            report["newton_contact_physics_steps"] = collector.contact_physics_steps
            report["reference_sha256"] = _hash(args.reference)
            report["checkpoint"] = str(args.checkpoint)
            report["seed"] = args.seed
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
