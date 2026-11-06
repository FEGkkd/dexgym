"""Independent Newton full-Episode evaluation of a strict SE(3) Core checkpoint."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import launch_simulation

import dexgym.tasks.marvin_supermimic  # noqa: F401
from dexgym.algorithms.supermimic.core.checkpoint import CoreCheckpoint
from dexgym.algorithms.supermimic.core.evaluator import PhysicalEvaluator
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.tasks.marvin_supermimic.core_collector import CoreNewtonCollector
from dexgym.tasks.marvin_supermimic.physics_contract import marvin_se3_physics_config
from dexgym.tasks.marvin_supermimic.se3_env import MarvinSE3ReachHoldCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9501)
    parser.add_argument("--level", choices=("near", "medium", "wider_feasible"), default="near")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 64:
        parser.error("Independent evaluation is bounded to at most 64 environments")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("schema") != CoreCheckpoint.schema_version or saved["task_spec"]["task_id"] != "marvin_bimanual_se3_reach_hold":
        parser.error("Expected a semantic v2 SE(3) reach and hold checkpoint")
    network = NetworkSpec(**saved["contract_manifest"]["networks"]["actor"]["network_spec"])
    cfg = MarvinSE3ReachHoldCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.goal_train_seed = saved["physics_config"]["goal_train_seed"]
    cfg.goal_eval_seed = args.seed + 100101
    cfg.curriculum_level = args.level
    cfg.evaluation_mode = True
    cfg.audit_contacts_each_physics_step = saved["physics_config"]["audit_contacts_each_physics_step"]
    torch.manual_seed(args.seed)
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-SE3-Reach-Hold-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            robot, task = native.core_robot, native.se3_task
            modules = NetworkFactory.build(robot.spec, task.spec, network, robot_adapter=robot)
            for module in (modules.world_model, modules.actor, modules.critic, modules.termination):
                module.to(native.device)
            checkpoint = CoreCheckpoint(
                git_commit=saved["git_commit"], reference_hashes={},
                physics_config=marvin_se3_physics_config(cfg, robot))
            load = checkpoint.load_actor_for_evaluation(
                args.checkpoint, robot=robot, task=task, modules=modules)
            before = {key: value.clone() for key, value in modules.actor.state_dict().items()}
            evaluator = PhysicalEvaluator(
                lambda: CoreNewtonCollector(env, robot, task, action_noise_std=0), task,
                evaluation_kind="independent_physics_evaluation")
            args.output.mkdir(parents=True, exist_ok=True)
            report = evaluator.evaluate(modules.actor, steps=task.episode_length,
                                        one_episode_per_env=True,
                                        trace_path=args.output / "physics_trace.pt")
            if any(not torch.equal(value, modules.actor.state_dict()[key])
                   for key, value in before.items()):
                raise RuntimeError("Evaluation modified Actor weights or normalizers")
            report.update({"checkpoint_load": load, "seed": args.seed,
                           "num_envs": args.num_envs, "goal_curriculum_level": args.level,
                           "exploration_noise": 0.0,
                           "checkpoint": str(args.checkpoint)})
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({key: value for key, value in report.items()
                              if key != "episodes"}), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
