"""Real Newton WM8 + Actor32 smoke; explicitly not a learned-policy result."""

import argparse
import json
import time
from pathlib import Path

import torch
from isaaclab.app import launch_simulation

from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig
from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine

from .adapter import CubeAdapter
from .env import Hand2CubeSuperMimicEnv
from .env_cfg import Hand2CubeSuperMimicCfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--full-config", action="store_true", help="Use production batches, AMP, compilation and 1024 envs"
    )
    parser.add_argument("--num-envs", type=int, help="Override environment count for capacity validation")
    args = parser.parse_args()
    cfg = Hand2CubeSuperMimicCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args.num_envs or (1024 if args.full_config else 16)
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    recipe = SuperMimicConfig(wm_batch=32, pi_batch=16, buffer_steps=96, compile=False, compile_policy=False, amp=False)
    if args.full_config:
        recipe = SuperMimicConfig()
    with launch_simulation(cfg):
        e = Hand2CubeSuperMimicEnv(cfg)
        try:
            task = CubeAdapter(e)
            engine = SuperMimicEngine(task, recipe, args.output.parent / "smoke_run")
            print("COLLECT", flush=True)
            started = time.perf_counter()
            stats = engine.collect(64)
            collected = time.perf_counter()
            for i in range(recipe.wm_batches if args.full_config else 3):
                stats.update(engine.fit_world_model())
                print("WM", i, {k: float(v) for k, v in stats.items()}, flush=True)
            wm_done = time.perf_counter()
            wm_before = [p.detach().clone() for p in engine.wm.parameters()]
            actor_before = [p.detach().clone() for p in engine.actor.parameters()]
            print("ACTOR32", flush=True)
            stats.update(engine.policy_update())
            torch.cuda.synchronize()
            policy_done = time.perf_counter()
            assert all(torch.equal(a, b) for a, b in zip(wm_before, engine.wm.parameters())), (
                "Actor update changed world-model weights"
            )
            assert any(not torch.equal(a, b) for a, b in zip(actor_before, engine.actor.parameters())), (
                "Actor update did not change weights"
            )
            assert all(v > 0 and torch.isfinite(torch.tensor(v)) for v in engine.last_gradients.values()), (
                engine.last_gradients
            )
            with torch.no_grad():
                stats.update({"holdout/" + k: v for k, v in engine.fit_world_model(True).items()})
            engine.save(args.output.parent / "smoke_checkpoint.pt", 0)
            report = {
                "passed": True,
                "trained_policy": False,
                "steps": 64,
                "num_envs": cfg.scene.num_envs,
                "timing_seconds": {
                    "collect": collected - started,
                    "wm": wm_done - collected,
                    "policy": policy_done - wm_done,
                },
                "recipe": recipe.to_dict(),
                "stats": {k: float(v) for k, v in stats.items()},
                "gradients": engine.last_gradients,
                "dimensions": {
                    "state": task.state_dim,
                    "actor": task.policy_dim,
                    "critic": task.context_dim,
                    "contact": task.contact_dim,
                    "wm_input": task.dynamics_dim + 20,
                },
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            e.close()


if __name__ == "__main__":
    main()
