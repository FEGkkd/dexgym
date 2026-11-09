"""Frozen-checkpoint open-loop prediction on fresh native simulator trajectories.

Recorded controls and exogenous pushes are replayed; no future physical state is
fed to the model. This diagnoses prediction, not policy success or gradient quality.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import launch_simulation

from dexgym.algorithms.supermimic.dex_reference.networks import Actor, WorldModel

from .adapter import CubeAdapter
from .env import Hand2CubeSuperMimicEnv
from .env_cfg import Hand2CubeSuperMimicCfg
from .state import difference


def summarize(predicted, actual):
    delta = difference(predicted, actual)
    if not bool(torch.isfinite(delta).all()):
        raise FloatingPointError("Non-finite open-loop prediction error")
    groups = dict(
        joint_rmse_rad=delta[:, :20].square().mean(-1).sqrt(),
        joint_velocity_rmse_rad_s=delta[:, 20:40].square().mean(-1).sqrt(),
        object_position_m=delta[:, 40:43].norm(dim=-1),
        object_angle_rad=delta[:, 43:46].norm(dim=-1),
        object_velocity_m_s=delta[:, 46:49].norm(dim=-1),
        object_angular_velocity_rad_s=delta[:, 49:52].norm(dim=-1),
    )
    return {k: {"mean": float(v.mean()), "p95": float(v.quantile(0.95))} for k, v in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--burn-in", type=int, default=150)
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 32 or args.burn_in < 0:
        parser.error("Positive environment count, at least 32 steps, nonnegative burn-in required")
    with torch.serialization.safe_globals(
        [np.dtype, np.ndarray, np._core.multiarray._reconstruct, np.dtypes.UInt32DType]
    ):
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if payload["task_id"] != "Wuji-Hand2-Cube-Reorientation" or payload["format_version"] != 1:
        raise ValueError("Expected native cube checkpoint")
    cfg = Hand2CubeSuperMimicCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        env = Hand2CubeSuperMimicEnv(cfg)
        try:
            task = CubeAdapter(env)
            actor = Actor(task.policy_dim, task.action_dim).to(env.device).eval()
            wm = (
                WorldModel(
                    task.dynamics_dim,
                    task.contact_dim,
                    task.codec,
                    task.action_dim,
                    task.error_floor,
                    payload["config"]["wm_hidden"],
                    payload["config"]["amp"],
                )
                .to(env.device)
                .eval()
            )
            actor.load_state_dict(payload["modules"]["actor"])
            wm.load_state_dict(payload["modules"]["wm"])
            actor.requires_grad_(False)
            wm.requires_grad_(False)
            norms = {name: v.clone() for name, v in wm.named_buffers()}
            actor_norms = {name: v.clone() for name, v in actor.named_buffers()}
            obs = env.reset(seed=args.seed)[0]["policy"]
            rows = []
            with torch.inference_mode():
                for step in range(args.burn_in + args.steps):
                    x, _, aux = task.read()
                    x = x.clone()
                    aux = aux.clone()
                    sim = task.simulator_context.clone()
                    obs_dict, _, terminated, truncated, _ = env.step(actor(obs))
                    obs = obs_dict["policy"]
                    cap = env.transition
                    done = terminated | truncated
                    truth = cap["x1"].clone()
                    push = task.reader.state() - truth
                    valid = ~cap["nan"]
                    for value in (x, truth, cap["u"], aux, push, sim):
                        valid &= torch.isfinite(value).all(-1)
                    if step >= args.burn_in:
                        rows.append(
                            dict(
                                x=x,
                                x1=truth,
                                u=cap["u"].clone(),
                                aux=aux,
                                push=push.clone(),
                                sim=sim,
                                done=done.clone(),
                                valid=valid.clone(),
                            )
                        )
                    ids = done.nonzero().flatten()
                    if len(ids):
                        obs = env.reset(env_ids=ids)[0]["policy"]
                data = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
                starts = torch.arange(0, args.steps - 31, 32, device=env.device)
                pred = data["x"][starts].flatten(0, 1)
                persistent = pred.clone()
                aux = data["aux"][starts].flatten(0, 1)
                sim = data["sim"][starts].flatten(0, 1)
                valid = torch.ones(pred.shape[0], dtype=torch.bool, device=env.device)
                results = {}
                for j in range(32):
                    index = starts + j
                    valid &= data["valid"][index].flatten()
                    if j:
                        valid &= ~data["done"][index - 1].flatten()
                    target = data["x1"][index].flatten(0, 1)
                    controls = data["u"][index].flatten(0, 1)
                    pred, _ = wm(pred, controls, task.dynamics(pred, sim, aux))
                    if j + 1 in (1, 4, 8, 16, 32):
                        if not bool(valid.any()):
                            raise RuntimeError("No valid open-loop windows")
                        results[str(j + 1)] = dict(
                            samples=int(valid.sum()),
                            excluded=int((~valid).sum()),
                            world_model=summarize(pred[valid], target[valid]),
                            persistence=summarize(persistent[valid], target[valid]),
                        )
                    push = data["push"][index].flatten(0, 1)
                    pred, aux = task.replay_push(pred, push, aux)
                    persistent = persistent + push
            assert all(torch.equal(v, norms[k]) for k, v in wm.named_buffers())
            assert all(torch.equal(v, actor_norms[k]) for k, v in actor.named_buffers())
            with args.checkpoint.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            report = dict(
                checkpoint=str(args.checkpoint),
                checkpoint_sha256=digest,
                iteration=payload["iteration"],
                seed=args.seed,
                num_envs=args.num_envs,
                steps=args.steps,
                burn_in=args.burn_in,
                control_hz=50,
                scene_revision=cfg.scene_revision,
                horizons=results,
                frozen_parameters=True,
                frozen_normalization_verified=True,
                action_noise=False,
                observation_noise=True,
                interval_pushes=True,
                protocol=(
                    "Fresh seed; deterministic actor; nonoverlapping 32-step windows; "
                    "reference replay push reconstruction; mask at terminal/reset boundaries"
                ),
                limitation=(
                    "Conditioned on recorded controls and pushes, not autonomous imagination; "
                    "correlated samples; no claim of policy success or action-gradient validity"
                ),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(report), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
