"""Compare WM8 and Actor32 updates with the pinned upstream implementation.

Includes a forced terminal replay transition and an active learned failure head.
"""

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "-C", str(args.reference), "rev-parse", "HEAD"], text=True).strip()
    if sha != "2becb7380fed86171056f9bf4338b2a6a0c14b01":
        raise ValueError("Reference revision mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import copy
    import json
    import sys
    import types

    import torch

    sys.path.insert(0, str(args.reference / "src"))
    from dex_supermimic.rl.dex_supermimic import mdp as rm
    from dex_supermimic.rl.dex_supermimic import networks as rn
    from dex_supermimic.rl.dex_supermimic import observations as ro
    from dex_supermimic.rl.dex_supermimic import state as rs
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from isaaclab.app import launch_simulation

    from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig
    from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine
    from dexgym.tasks.direct.hand2_cube_supermimic.adapter import CubeAdapter
    from dexgym.tasks.direct.hand2_cube_supermimic.env import Hand2CubeSuperMimicEnv
    from dexgym.tasks.direct.hand2_cube_supermimic.env_cfg import Hand2CubeSuperMimicCfg

    cfg = Hand2CubeSuperMimicCfg()
    cfg.seed = 42
    cfg.scene.num_envs = 16
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    c = SuperMimicConfig(wm_batch=32, pi_batch=16, buffer_steps=96, compile=False, compile_policy=False, amp=False)
    with launch_simulation(cfg):
        env = Hand2CubeSuperMimicEnv(cfg)
        try:
            task = CubeAdapter(env)
            e = SuperMimicEngine(task, c, args.output.parent / "parity_run")
            e.env.memory["cage"][0] = 25
            e.collect(64)
            for _ in range(3):
                e.fit_world_model()
            r = DexSuperMimicRunner.__new__(DexSuperMimicRunner)
            r.cfg = types.SimpleNamespace(**c.to_dict(), algorithm="world_model", variant="fixed")
            r.device = e.device
            r.env = env
            r.is_pen = False
            r.ppo_critic = None
            r.iteration = 800
            r.buffer = e.buffer
            e.iteration = 800
            env._adaptive_episode_curriculum = env.disturbance.adaptive_scale
            r.reader = types.SimpleNamespace(
                xi=task.simulator_context,
                robot=types.SimpleNamespace(data=types.SimpleNamespace(encoder_bias=env.encoder_bias)),
            )
            r.mdp = rm.DexMDP.__new__(rm.DexMDP)
            r.mdp.__dict__ = copy.copy(task.mdp.__dict__)
            r.observations = ro.TaskObservations.__new__(ro.TaskObservations)
            r.observations.__dict__ = copy.copy(task.observations.__dict__)
            r.observations.mdp = r.mdp
            r.actor = rn.Actor(207).to(e.device)
            r.wm = rn.WorldModel(113, 32, c.wm_hidden, False).to(e.device)
            r.critic = rn.Critic(211, False).to(e.device)
            r.fail_head = rn.FailureHead(211).to(e.device)
            r.optimizers = {k: torch.optim.Adam(v.parameters(), lr=c.lr) for k, v in r.modules().items()}
            r.optimizers["actor"] = torch.optim.AdamW(
                r.actor.parameters(), lr=c.lr, betas=(0.9, 0.95), weight_decay=0.0
            )
            for k, m in r.modules().items():
                m.load_state_dict(copy.deepcopy(e.modules()[k].state_dict()))
                r.optimizers[k].load_state_dict(copy.deepcopy(e.optimizers[k].state_dict()))
            r.step_wm = r.wm
            r.step_actor = r.actor
            r.reward_step = r.mdp.rewards
            r.task_features = r.observations.context
            r.wm_features = r.observations.dynamics
            r.flatten_memory = rs.flatten_memory
            r.unflatten_memory = rs.unflatten_memory

            def paired(method):
                cpu = torch.get_rng_state()
                gpu = torch.cuda.get_rng_state_all()
                a = getattr(e, method)()
                torch.set_rng_state(cpu)
                torch.cuda.set_rng_state_all(gpu)
                b = getattr(r, method)()
                metrics = {k: abs(float(a[k]) - float(b[k])) for k in a if k in b}
                parameters = {
                    k: max(float((x - y).abs().max()) for x, y in zip(m.parameters(), r.modules()[k].parameters()))
                    for k, m in e.modules().items()
                }
                print("PARITY", method, metrics, parameters, flush=True)
                assert max(metrics.values()) < 1e-5, (method, metrics)
                assert max(parameters.values()) < 1e-6, (method, parameters)
                return {"metric_max_abs_error": metrics, "parameter_max_abs_error": parameters}

            result = {
                "reference_sha": "2becb7380fed86171056f9bf4338b2a6a0c14b01",
                "world_update": paired("fit_world_model"),
                "policy_update": paired("policy_update"),
            }
            args.output.write_text(json.dumps(result, indent=2))
        finally:
            env.close()


if __name__ == "__main__":
    main()
