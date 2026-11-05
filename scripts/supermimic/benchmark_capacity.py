"""Measure a complete Newton/replay/world-model/BPTT loop at one environment count."""

import argparse
import copy
import json
import time
from pathlib import Path

import gymnasium as gym
import torch
from dexgym.algorithms.supermimic.imagination import critic_td_lambda_loss, imagined_rollout
from dexgym.algorithms.supermimic.networks import DeltaWorldModel, GoalActor, PrivilegedCritic
from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from isaaclab.app import launch_simulation
from preflight import REFERENCE_SHA256, verify
from train_wave import collect


def benchmark(args):
    verify()
    device = torch.device("cuda")
    record = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if record["reference_sha256"] != REFERENCE_SHA256:
        raise ValueError("Checkpoint motion hash mismatch")
    model = DeltaWorldModel(hidden=1024).to(device)
    model.load_state_dict(record["world_model"])
    critic = PrivilegedCritic().to(device)
    critic.load_state_dict(record["critic"])
    actor = GoalActor().to(device)
    actor.load_state_dict(torch.load(args.actor_state, map_location=device, weights_only=True))
    prior = copy.deepcopy(actor).eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    optim_world = torch.optim.Adam(model.parameters(), lr=3e-4)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=1e-7)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=3e-4)
    cfg = MarvinWaveDirectCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.reset_joint_perturb_rad = 0.01
    cfg.seed = 140116
    torch.cuda.reset_peak_memory_stats()
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Wave-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            replay = MotionReplay(max(128, args.collect_steps + 32), args.num_envs, device)
            collection = collect(env, actor, replay, args.collect_steps, teacher_fraction=1, noise=0, teacher_noise=0)
            started = time.perf_counter()
            world_losses = []
            for _ in range(args.world_updates):
                windows = replay.sample_windows(args.world_batch, 8)
                optim_world.zero_grad()
                loss, _ = model.autoregressive_loss(windows, native.action_map)
                loss.backward()
                optim_world.step()
                world_losses.append(float(loss.detach()))
            world_seconds = time.perf_counter() - started
            started = time.perf_counter()
            actor_losses = []
            critic_losses = []
            for _ in range(args.actor_updates):
                start = replay.sample_starts(args.imag_batch)
                for module in (model, critic):
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
                optim_actor.zero_grad()
                rollout = imagined_rollout(
                    actor,
                    model,
                    critic,
                    native.fk,
                    native.action_map,
                    native.motion,
                    start,
                    horizon=32,
                    prior_actor=prior,
                    prior_weight=100,
                )
                rollout["actor_loss"].backward()
                gradient = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                if not torch.isfinite(gradient):
                    raise ValueError("BPTT Actor gradient is not finite")
                optim_actor.step()
                actor_losses.append(float(rollout["actor_loss"].detach()))
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
                optim_critic.zero_grad()
                critic_loss = critic_td_lambda_loss(critic, native.motion, rollout)
                critic_loss.backward()
                optim_critic.step()
                critic_losses.append(float(critic_loss.detach()))
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
            imagination_seconds = time.perf_counter() - started
            result = {
                "num_envs": args.num_envs,
                "reference_sha256": REFERENCE_SHA256,
                "real_transitions": collection["transitions"],
                "collection": collection,
                "replay_bytes": replay.bytes_allocated,
                "world_updates": args.world_updates,
                "world_updates_per_second": args.world_updates / world_seconds,
                "world_loss_first_last": [world_losses[0], world_losses[-1]],
                "imagination_updates": args.actor_updates,
                "imagination_updates_per_second": args.actor_updates / imagination_seconds,
                "actor_loss_first_last": [actor_losses[0], actor_losses[-1]],
                "critic_loss_first_last": [critic_losses[0], critic_losses[-1]],
                "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
                "gpu_total_bytes": torch.cuda.mem_get_info()[1],
                "physical_evaluation": "not part of the capacity benchmark",
            }
        finally:
            env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--collect-steps", type=int, default=128)
    parser.add_argument("--world-updates", type=int, default=100)
    parser.add_argument("--actor-updates", type=int, default=5)
    parser.add_argument("--world-batch", type=int, default=256)
    parser.add_argument("--imag-batch", type=int, default=64)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--actor-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(benchmark(parser.parse_args()), indent=2))
