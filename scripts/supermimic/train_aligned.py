"""Alternating Newton/WM8/BPTT32/TD(lambda) training for Marvin M4.1."""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
from dexgym.algorithms.supermimic.imagination import critic_td_lambda_loss, imagined_rollout
from dexgym.algorithms.supermimic.networks import DeltaWorldModel, GoalActor, PrivilegedCritic
from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.algorithms.supermimic.rewards import REWARD_PROFILES
from dexgym.algorithms.supermimic.training_loop import AlternatingLoop, prepare_model_windows
from dexgym.tasks.marvin_supermimic.collector import AlignedNewtonCollector
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from isaaclab.app import launch_simulation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preflight import REFERENCE_SHA256, verify  # noqa: E402
from train_wave import evaluate_full, model_errors, pd_actor_examples  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--replay-steps", type=int, default=1000)
    parser.add_argument("--world-batch", type=int, default=128)
    parser.add_argument("--imag-batch", type=int, default=64)
    parser.add_argument("--world-hidden", type=int, default=1024)
    parser.add_argument("--world-updates-per-iteration", type=int, default=1)
    parser.add_argument("--actor-updates-per-iteration", type=int, default=1)
    parser.add_argument("--min-world-updates", type=int, default=100)
    parser.add_argument("--trust-check-interval", type=int, default=50)
    parser.add_argument("--max-eight-step-tcp-mm", type=float, default=50.0)
    parser.add_argument("--max-32-step-tcp-mm", type=float, default=100.0)
    parser.add_argument("--max-eight-step-tcp-deg", type=float, default=10.0)
    parser.add_argument("--max-32-step-tcp-deg", type=float, default=15.0)
    parser.add_argument("--actor-lr", type=float, default=2e-5)
    parser.add_argument("--behavior-prior-weight", type=float, default=0.0)
    parser.add_argument("--prior-checkpoint", type=Path)
    parser.add_argument("--world-lr", type=float, default=2e-4)
    parser.add_argument("--critic-lr", type=float, default=2e-4)
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--reset-perturb-rad", type=float, default=0.03)
    parser.add_argument("--reward-profile", choices=tuple(REWARD_PROFILES), default="paper_multilink")
    parser.add_argument("--goal-latent-tanh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--observation-position-noise-rad", type=float, default=0.01)
    parser.add_argument("--observation-velocity-noise-rad-s", type=float, default=0.5)
    parser.add_argument("--aux-warmup-updates", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--final-eval", action="store_true")
    parser.add_argument("--seed", type=int, default=140116)
    parser.add_argument(
        "--output", type=Path, default=Path("/home/ubuntu/dexgym_data/marvin_supermimic_wave_m41/pure_smoke")
    )
    args = parser.parse_args()
    if args.num_envs < 2 or args.iterations < 1 or args.replay_steps < 1:
        raise ValueError("Aligned training needs at least two envs, one iteration, and one Replay slot")
    if args.min_world_updates < 1 or args.trust_check_interval < 1 or args.checkpoint_interval < 1:
        raise ValueError("Model trust and checkpoint intervals must be positive")
    if any(
        value <= 0
        for value in (
            args.max_eight_step_tcp_mm,
            args.max_32_step_tcp_mm,
            args.max_eight_step_tcp_deg,
            args.max_32_step_tcp_deg,
        )
    ):
        raise ValueError("World-model trust limits must be positive")
    if any(
        value < 0
        for value in (
            args.action_noise,
            args.reset_perturb_rad,
            args.observation_position_noise_rad,
            args.observation_velocity_noise_rad_s,
            args.aux_warmup_updates,
            args.behavior_prior_weight,
        )
    ):
        raise ValueError("Noise, perturbation, and auxiliary updates must be nonnegative")
    return args


def _checkpoint(actor, model, critic, optim_actor, optim_model, optim_critic, metrics):
    return {
        "actor": actor.state_dict(),
        "world_model": model.state_dict(),
        "critic": critic.state_dict(),
        "actor_optimizer": optim_actor.state_dict(),
        "world_optimizer": optim_model.state_dict(),
        "critic_optimizer": optim_critic.state_dict(),
        "training": metrics,
        "reference_sha256": REFERENCE_SHA256,
        "replay_saved": False,
        "resume_requires_recollection": True,
        "algorithm": "M4.1_alternating_soft_termination",
    }


def _save_checkpoint_atomic(path, actor, model, critic, optim_actor, optim_model, optim_critic, metrics):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(_checkpoint(actor, model, critic, optim_actor, optim_model, optim_critic, metrics), temporary)
    os.replace(temporary, path)


def _load_resume(path, actor, model, critic, optim_actor, optim_model, optim_critic, device):
    record = torch.load(path, map_location=device, weights_only=False)
    if record["reference_sha256"] != REFERENCE_SHA256:
        raise ValueError("Cannot resume from another reference version")
    actor.load_state_dict(record["actor"])
    missing, unexpected = model.load_state_dict(record["world_model"], strict=False)
    if unexpected or (missing and not all(name.startswith("termination_head.") for name in missing)):
        raise ValueError(f"World-model checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    critic.load_state_dict(record["critic"])
    if not missing:
        optim_actor.load_state_dict(record["actor_optimizer"])
        optim_model.load_state_dict(record["world_optimizer"])
        optim_critic.load_state_dict(record["critic_optimizer"])
    return record["training"], list(missing)


def _auxiliary_initialize(actor, motion, mapping, device, updates, optimizer):
    if not updates:
        return []
    histories, goals, targets = pd_actor_examples(motion, mapping, device)
    losses = []
    for _ in range(updates):
        indices = torch.randint(0, len(histories), (128,), device=device)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(actor(histories[indices], goals[indices]), targets[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def main():
    args = _parse_args()
    preflight = verify()
    source_commit_at_start = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    actor = GoalActor(goal_latent_tanh=args.goal_latent_tanh).to(device)
    model = DeltaWorldModel(hidden=args.world_hidden).to(device)
    critic = PrivilegedCritic().to(device)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, betas=(0.9, 0.95))
    optim_model = torch.optim.Adam(model.parameters(), lr=args.world_lr)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    prior_training, newly_initialized = None, []
    if args.resume is not None:
        prior_training, newly_initialized = _load_resume(
            args.resume, actor, model, critic, optim_actor, optim_model, optim_critic, device
        )
    cfg = MarvinWaveDirectCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.audit_contacts_each_physics_step = True
    cfg.reward_profile = args.reward_profile
    cfg.reset_joint_perturb_rad = args.reset_perturb_rad
    cfg.observation_position_noise_rad = args.observation_position_noise_rad
    cfg.observation_velocity_noise_rad_s = args.observation_velocity_noise_rad_s
    started = time.perf_counter()
    sampled_peak_gpu_used = 0
    curves = {"world_loss": [], "actor_loss": [], "critic_loss": [], "physical_reward": [], "termination_rate": []}
    trust_history = []
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Wave-Direct-v0", cfg=cfg)
        try:
            native = env.unwrapped
            motion, mapping, fk = native.motion, native.action_map, native.fk
            replay = MotionReplay(args.replay_steps, args.num_envs, device)
            aux_loss = _auxiliary_initialize(actor, motion, mapping, device, args.aux_warmup_updates, optim_actor)
            if aux_loss:
                optim_actor = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, betas=(0.9, 0.95))
            prior_actor = copy.deepcopy(actor).eval() if args.behavior_prior_weight else None
            if prior_actor is not None:
                if args.prior_checkpoint is not None:
                    prior_record = torch.load(args.prior_checkpoint, map_location=device, weights_only=False)
                    if prior_record["reference_sha256"] != REFERENCE_SHA256:
                        raise ValueError("Behavior prior uses another reference version")
                    prior_actor.load_state_dict(prior_record["actor"])
                for parameter in prior_actor.parameters():
                    parameter.requires_grad_(False)
            collector = AlignedNewtonCollector(env, actor, replay, action_noise_std=args.action_noise)
            loop = AlternatingLoop(args.world_updates_per_iteration, args.actor_updates_per_iteration)
            train_envs = max(1, int(args.num_envs * 0.8))
            trusted = False
            last_trust_check = -args.trust_check_interval
            latest_rollout = None
            pending_model_windows = None

            def collect():
                result = collector.step()
                slot = (replay.steps - 1) % replay.capacity
                model.update_norms(replay.state[slot], replay.action[slot], replay.next_state[slot])
                curves["physical_reward"].append(result["reward_mean"])
                curves["termination_rate"].append(result["terminations"] / args.num_envs)

            def replay_ready():
                nonlocal pending_model_windows
                if replay.steps < 1:
                    return False
                pending_model_windows = prepare_model_windows(
                    replay, args.world_batch, args.world_updates_per_iteration, (0, train_envs)
                )
                return pending_model_windows is not None

            def model_update():
                window = pending_model_windows.pop()
                valid = window["mask"].bool()
                next_frames = torch.where(valid, window["next_frame"], torch.zeros_like(window["next_frame"]))
                window["termination_goal"] = motion.reference_features_at(next_frames.reshape(-1)).reshape(
                    args.world_batch, 8, 90
                )
                optim_model.zero_grad()
                loss, _ = model.autoregressive_loss(window, mapping, update_normalizers=False)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite eight-step world-model loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim_model.step()
                curves["world_loss"].append(float(loss.detach()))

            def actor_ready():
                nonlocal trusted, last_trust_check
                completed_updates = loop.counts["world_updates"]
                if completed_updates < args.min_world_updates:
                    return False
                if completed_updates - last_trust_check >= args.trust_check_interval:
                    last_trust_check = completed_updates
                    try:
                        prediction = model_errors(
                            model, replay, mapping, fk, batch=32, env_range=(train_envs, args.num_envs)
                        )
                    except ValueError as error:
                        trusted = False
                        trust_history.append({"update": completed_updates, "reason": str(error), "trusted": False})
                    else:
                        trusted = (
                            prediction["8"]["tcp_position_mean_mm"] <= args.max_eight_step_tcp_mm
                            and prediction["32"]["tcp_position_mean_mm"] <= args.max_32_step_tcp_mm
                            and prediction["8"]["tcp_orientation_mean_deg"] <= args.max_eight_step_tcp_deg
                            and prediction["32"]["tcp_orientation_mean_deg"] <= args.max_32_step_tcp_deg
                        )
                        trust_history.append(
                            {"update": completed_updates, "prediction": prediction, "trusted": trusted}
                        )
                return trusted

            def actor_update():
                nonlocal latest_rollout
                start = replay.sample_starts(
                    args.imag_batch, env_range=(0, train_envs), horizon=32, clip_length=motion.length
                )
                for module in (model, critic):
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
                optim_actor.zero_grad()
                latest_rollout = imagined_rollout(
                    actor,
                    model,
                    critic,
                    fk,
                    mapping,
                    motion,
                    start,
                    horizon=32,
                    weights=REWARD_PROFILES[args.reward_profile],
                    prior_actor=prior_actor,
                    prior_weight=args.behavior_prior_weight,
                    observation_position_noise_rad=args.observation_position_noise_rad,
                    observation_velocity_noise_rad_s=args.observation_velocity_noise_rad_s,
                )
                loss = latest_rollout["actor_loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite 32-step Actor loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optim_actor.step()
                curves["actor_loss"].append(float(loss.detach()))

            def critic_update():
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
                optim_critic.zero_grad()
                loss = critic_td_lambda_loss(critic, motion, latest_rollout)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite TD(lambda) Critic loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optim_critic.step()
                curves["critic_loss"].append(float(loss.detach()))
                for parameter in model.parameters():
                    parameter.requires_grad_(True)

            for iteration in range(args.iterations):
                loop.run_one(
                    collect=collect,
                    replay_ready=replay_ready,
                    model_update=model_update,
                    actor_ready=actor_ready,
                    actor_update=actor_update,
                    critic_update=critic_update,
                )
                if (iteration + 1) % args.checkpoint_interval == 0:
                    free_gpu, total_gpu = torch.cuda.mem_get_info()
                    sampled_peak_gpu_used = max(sampled_peak_gpu_used, total_gpu - free_gpu)
                    progress = {
                        "iteration": iteration + 1,
                        **loop.counts,
                        "physical_transitions": collector.steps * args.num_envs,
                        "terminations": collector.terminations,
                        "trusted_world_model": trusted,
                        "elapsed_s": time.perf_counter() - started,
                        "gpu_vram_used_bytes": total_gpu - free_gpu,
                        "last_world_loss": curves["world_loss"][-1] if curves["world_loss"] else None,
                        "last_actor_loss": curves["actor_loss"][-1] if curves["actor_loss"] else None,
                        "last_critic_loss": curves["critic_loss"][-1] if curves["critic_loss"] else None,
                        "last_world_trust_check": trust_history[-1] if trust_history else None,
                    }
                    (args.output / "progress.json").write_text(json.dumps(progress, indent=2))
                    _save_checkpoint_atomic(
                        args.output / "latest.pt",
                        actor,
                        model,
                        critic,
                        optim_actor,
                        optim_model,
                        optim_critic,
                        {"progress": progress, "configuration": vars(args)},
                    )
                    print(json.dumps(progress), flush=True)
            metrics = {
                "algorithm": "M4.1_alternating_soft_termination",
                "reference_sha256": REFERENCE_SHA256,
                "git_commit": source_commit_at_start,
                "preflight": preflight,
                "configuration": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "resume_from": str(args.resume) if args.resume else None,
                "resume_recollected_replay": bool(args.resume),
                "resume_newly_initialized_model_keys": newly_initialized,
                "prior_training": prior_training,
                "collection": {
                    "transitions": collector.steps * args.num_envs,
                    "terminations": collector.terminations,
                    "truncations": collector.truncations,
                    "contact_physics_steps": collector.contact_physics_steps,
                    "mean_physical_reward": collector.reward_sum / collector.steps,
                },
                "world_updates": loop.counts["world_updates"],
                "actor_updates": loop.counts["actor_updates"],
                "critic_updates": loop.counts["critic_updates"],
                "auxiliary_actor_updates": len(aux_loss),
                "world_model_trust_history": trust_history,
                "replay_bytes": replay.bytes_allocated,
                "train_wall_time_s": time.perf_counter() - started,
                "peak_torch_vram_bytes": torch.cuda.max_memory_allocated(),
                "sampled_peak_gpu_vram_used_bytes": sampled_peak_gpu_used,
            }
            metrics["physical_transitions_per_s"] = metrics["collection"]["transitions"] / metrics["train_wall_time_s"]
        finally:
            env.close()
    checkpoint = _checkpoint(actor, model, critic, optim_actor, optim_model, optim_critic, metrics)
    torch.save(checkpoint, args.output / "last.pt")
    torch.save(checkpoint, args.output / "candidate.pt")
    torch.save({key: value for key, value in curves.items()}, args.output / "learning_curves.pt")
    if args.final_eval:
        metrics["evaluation"] = evaluate_full(
            actor, output_dir=args.output / "inprocess_eval", reward_profile=args.reward_profile
        )
        metrics["policy_validation_passed"] = bool(
            metrics["evaluation"]["full_motion_completed"]
            and metrics["evaluation"]["newton_contact_physics_steps"] == 0
            and metrics["evaluation"]["illegal_joint_state_frames"] == 0
            and all(
                metrics["evaluation"][key][metric] <= bound
                for key, metric, bound in (
                    ("left_position_mm", "mean", 25),
                    ("right_position_mm", "mean", 25),
                    ("left_position_mm", "p95", 50),
                    ("right_position_mm", "p95", 50),
                    ("left_orientation_deg", "mean", 8),
                    ("right_orientation_deg", "mean", 8),
                    ("left_orientation_deg", "p95", 15),
                    ("right_orientation_deg", "p95", 15),
                )
            )
        )
        checkpoint["training"] = metrics
        if metrics["policy_validation_passed"]:
            torch.save(checkpoint, args.output / "best.pt")
        else:
            torch.save(checkpoint, args.output / "candidate.pt")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({key: value for key, value in metrics.items() if key != "prior_training"}, indent=2))


if __name__ == "__main__":
    main()
