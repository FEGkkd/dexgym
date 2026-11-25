"""Train a randomly initialized PPO policy with task-space goals on Newton."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from ..marvin54_s0.physics import open_physics
from ..wholebody_s0.bag import digest
from ..wholebody_s0.ppo_task import TanhGaussian
from ..wholebody_s0.prepare import write_json
from ..wholebody_s0.train_common import atomic_checkpoint
from . import DOF, DT, TASK_ID
from .bank import TaskBank
from .cross_trajectory import add_switch_arguments, switch_config
from .finger_reward import add_finger_reward_arguments, finger_reward_config
from .control import ACTOR_ARCHITECTURE, OBSERVATION_VERSION, AbsoluteAction
from .task import BRIDGE_STEPS, HOME_FRACTION, SEGMENT_STEPS, DirectEnv
from .targets import joint_names


@torch.no_grad()
def check_reset(env):
    ids = torch.arange(env.num_envs, device=env.device)
    frames = env.bank.sample(env.num_envs, env.generator)
    env.reset(ids, frames=frames, home=torch.zeros_like(ids, dtype=torch.bool), exact=True)
    with env.backend.simulation_stream() as caller:
        actual = env.backend.state().clone()
        expected = torch.cat((env.bank.rsi_q[frames], env.bank.rsi_qd[frames]), dim=-1)
        delta = (actual - expected).abs().max()
        body_names = ("l_wrist", "r_wrist") if env.task.observe.physical_wrist else ("TCP_Link_L", "TCP_Link_R")
        wrists = [env.backend.robot.body_names.index(name) for name in body_names]
        tcp = env.backend.robot.data.body_pose_w.torch[:, wrists, :3] - env.backend.scene.env_origins[:, None]
        fk_error = (tcp - env.actual.wrist).abs().max()
        delta.record_stream(caller)
        fk_error.record_stream(caller)
    if not torch.allclose(actual, expected, rtol=0., atol=3e-5) or float(fk_error) > .0003:
        raise ValueError(f"Newton RSI parity failed: state={float(delta)}, wrist={float(fk_error)}")
    if not torch.equal(env.frame, frames + 1):
        raise ValueError("RSI target must be the next frame within the same clip")
    frame_key = "newton_physical_wrist_max_error_m" if env.task.observe.physical_wrist else "newton_tcp_max_error_m"
    return {"rsi_state_max_error": float(delta), frame_key: float(fk_error),
            "tested_worlds": env.num_envs, "next_frame_checked": True}


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = "cuda:0"
    bank = TaskBank(args.bank, "train", device)
    metadata = bank.metadata
    if digest(args.model) != metadata["teleop_model_sha256"]:
        raise ValueError("Train with the same MJCF used to build task-space targets")
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if joint_names(model) != metadata["joint_names"]:
        raise ValueError("Joint order differs from the task-space bank")
    args.output.mkdir(parents=True, exist_ok=False)
    with open_physics("newton", model, metadata["joint_names"], args.num_envs, robot_usd=args.usd) as backend:
        finger_reward = finger_reward_config(args)
        env = DirectEnv(backend, bank, model, args.seed, args.robust, switch=switch_config(args),
                        finger_reward=finger_reward)
        witness = check_reset(env)
        env.reset()
        initial = env.observation()
        count, features = initial.shape
        obs = TensorDict({"policy": initial}, batch_size=[count], device=device)
        groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = MLPModel(obs, groups, "actor", DOF).to(device)
        actor.mlp = AbsoluteAction(features, env.task.command.scale,
                                   observation_version=env.task.observe.version,
                                   action_home=env.task.command.home).to(device)
        actor.distribution = TanhGaussian(DOF, init_std=args.init_std, std_range=(.004, .12), std_type="log").to(device)
        with torch.no_grad():
            std_rad = torch.where(env.task.command.arm, args.init_std, 1.4 * args.init_std)
            actor.distribution.log_std_param.copy_((std_rad / env.task.command.scale).clamp(.004, .12).log())
        critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(512, 512, 256, 128), obs_normalization=True).to(device)
        critic.update_normalization(obs)
        rollout = RolloutStorage("rl", count, args.rollout, obs, (DOF,), device=device)
        algo = PPO(actor, critic, rollout, learning_rate=args.lr, num_learning_epochs=4, num_mini_batches=8,
                   desired_kl=.016, entropy_coef=0., schedule="fixed", device=device)
        initial_weights = {key: value.detach().clone() for key, value in actor.mlp.named_parameters()}
        provenance = {
            "task_id": TASK_ID, "bank": str(args.bank.resolve()), "bank_sha256": digest(args.bank),
            "model": str(args.model.resolve()), "model_sha256": digest(args.model),
            "usd": str(args.usd.resolve()), "usd_sha256": digest(args.usd),
            "joint_names": metadata["joint_names"], "q_default": metadata["home"],
            "action_scale": env.task.command.scale.tolist(), "control_dt": DT, "seed": args.seed,
            "num_envs": count, "features": features, "rollout": args.rollout, "iterations": args.iterations,
            "initialization": "random_no_checkpoint", "lr": args.lr, "init_std": args.init_std,
            "actor_architecture": ACTOR_ARCHITECTURE,
            "observation_version": OBSERVATION_VERSION,
            "ppo": {key: getattr(algo, key) for key in ("schedule", "gamma", "lam", "clip_param",
                    "entropy_coef", "desired_kl", "num_learning_epochs", "num_mini_batches")},
            "home_fraction": HOME_FRACTION, "bridge_steps": BRIDGE_STEPS, "segment_steps": SEGMENT_STEPS,
            "cross_trajectory_switch": env.reference_switch.config.to_dict(),
            "finger_reward": finger_reward.to_dict(),
            "thumb_projection": bank.metadata.get("thumb_projection"),
            "robust": args.robust, "nominal_envs": int((~env.domain.randomized).sum()), "witness": witness,
        }
        write_json(args.output / "provenance.json", provenance)
        print("S0_1_NEWTON", json.dumps(witness), flush=True)
        start = time.perf_counter()

        def checkpoint(name, step):
            atomic_checkpoint({"task_id": TASK_ID, "iteration": step, "features": features,
                               "actor_architecture": ACTOR_ARCHITECTURE,
                               "observation_version": OBSERVATION_VERSION,
                               "bank_sha256": provenance["bank_sha256"], "sim_steps": env.steps * count,
                               "actor": actor.mlp.state_dict(), "critic": critic.state_dict(),
                               "distribution": actor.distribution.state_dict(), "optimizer": algo.optimizer.state_dict()},
                              args.output / name)

        with (args.output / "metrics.jsonl").open("w", buffering=1, encoding="utf-8") as log:
            for step in range(1, args.iterations + 1):
                env.set_sampling_iteration(step)
                sums = {}
                with torch.inference_mode():
                    for _ in range(args.rollout):
                        reward, done, timeout, stats = env.step(algo.act(obs))
                        for key, value in stats.items():
                            sums[key] = sums.get(key, 0.) + value
                        # A sampled window can end mid-recording. Bootstrap from the
                        # continuation before reset, then stop the rollout at the reset.
                        continued = timeout.nonzero().flatten()
                        if len(continued):
                            next_obs = TensorDict({"policy": env.observation(continued)},
                                                  batch_size=[len(continued)], device=device)
                            reward[continued] += algo.gamma * critic(next_obs).squeeze(-1)
                        ids = done.nonzero().flatten()
                        if len(ids):
                            env.reset(ids)
                        obs = TensorDict({"policy": env.observation()}, batch_size=[count], device=device)
                        algo.process_env_step(obs, reward, done, {})
                    algo.compute_returns(obs)
                losses = algo.update()
                row = {key: float(value / args.rollout) for key, value in sums.items()}
                row.update(losses)
                row.update(iteration=step, transitions=env.steps * count, seconds=time.perf_counter() - start,
                           noise_std=float(actor.distribution.std.detach().mean()), segment_steps=env.segment_steps,
                           home_fraction=env.home_fraction, lr=algo.learning_rate)
                if not all(math.isfinite(value) for value in row.values()):
                    raise FloatingPointError(f"PPO update {step} is non-finite")
                log.write(json.dumps(row) + "\n")
                if step == 1 or step % 25 == 0 or step == args.iterations:
                    print("S0_1_METRIC", json.dumps(row), flush=True)
                if step % args.checkpoint_every == 0 or step == args.iterations:
                    checkpoint("model_final.pt" if step == args.iterations else f"model_{step}.pt", step)
        change = max(float((p.detach() - initial_weights[key]).abs().max()) for key, p in actor.mlp.named_parameters())
        if not math.isfinite(change) or change <= 0. or env.steps != args.rollout * args.iterations:
            raise RuntimeError("PPO did not update the policy or missed its transition budget")
        proof = {"iterations": args.iterations, "sim_transitions": env.steps * count, "num_envs": count,
                 "wall_seconds": time.perf_counter() - start, "actor_parameter_max_change": change,
                 "torch_peak_bytes": torch.cuda.max_memory_allocated(), "initialization": provenance["initialization"]}
        write_json(args.output / "training_proof.json", proof)
        print("S0_1_TRAINING_PROOF", json.dumps(proof), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("bank", "model", "usd", "output"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--rollout", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--init-std", type=float, default=.032, help="Initial exploration standard deviation in arm radians")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--robust", action="store_true")
    add_switch_arguments(parser)
    add_finger_reward_arguments(parser)
    args = parser.parse_args()
    try:
        switch_config(args)
        finger_reward_config(args)
    except ValueError as error:
        parser.error(str(error))
    if min(args.num_envs, args.iterations, args.rollout, args.checkpoint_every) < 1 or args.num_envs * args.rollout < 8:
        parser.error("Positive budgets and at least eight transitions per rollout are required")
    train(args)


if __name__ == "__main__":
    main()
