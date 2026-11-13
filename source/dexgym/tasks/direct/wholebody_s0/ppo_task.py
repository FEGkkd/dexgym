"""RSL-RL PPO on the wholebody_s0 bounded residual task."""

import json
import math
from pathlib import Path
import time

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch import nn

from .residual import N, POSE_TRACKING_SCALES, ResidualTask
from .train_common import atomic_checkpoint, run, train_parser

TASK_ID = "DexGym-WholeBody-S0-PPO-Direct-v0"


class WholebodyS0PPO(ResidualTask):
    task_id = TASK_ID
    method = "ppo"

    def hand_reward(self, reward, errors):
        # For a frozen arm, use the shared objective with the arm-only
        # terms removed from PPO credit assignment. Keep contact and joint costs.
        arm_score = sum(self.reward_weights[key] * self.group_score(errors[feature], POSE_TRACKING_SCALES[key])
                        for key, feature in (("wrist_position", "wrist"), ("wrist_rotation", "rotation"),
                                             ("relative_position", "relative"), ("relative_rotation", "relative_rotation")))
        return reward - arm_score - self.reward_weights["left_hand"] - self.reward_weights["right_hand"]


class TanhGaussian(GaussianDistribution):
    def sample(self):
        return super().sample().tanh()

    def deterministic_output(self, mlp_output):
        return mlp_output.tanh()

    def as_deterministic_output_module(self):
        return nn.Tanh()

    def log_prob(self, outputs):
        a = outputs.clamp(-1 + 1e-6, 1 - 1e-6)
        return super().log_prob(a.atanh()) - torch.log1p(-a.square()).sum(-1)


class HandTanhGaussian(TanhGaussian):
    """Sample the optimized hand actions; execute the frozen arm deterministically."""

    def __init__(self, hand, *, precision=False):
        std, limits = (.035, (.010, .075)) if precision else (.06, (.025, .16))
        super().__init__(len(hand), init_std=std, std_range=limits, std_type="log")
        self.register_buffer("hand", hand)

    def sample(self):
        action = super().sample()
        return torch.where(self.hand, action, self.mean.tanh())

    def log_prob(self, outputs):
        a = outputs.clamp(-1 + 1e-6, 1 - 1e-6)
        terms = self._distribution.log_prob(a.atanh()) - torch.log1p(-a.square())
        return (terms * self.hand).sum(-1)


def learn(args, env):
    rollout = args.rollout
    if args.control_steps % rollout:
        raise ValueError("PPO control budget must be divisible by rollout length")
    iterations = args.control_steps // rollout
    task, device, count = env.task, env.device, env.num_envs
    initial, _ = env.reset()
    obs = TensorDict(initial, batch_size=[count])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = task.make_actor().to(device)
    model = MLPModel(obs, groups, "actor", N, obs_normalization=not task.freeze_actor_norm)
    # Reuse the deterministic actor network and normalizer used by SuperMimic.
    model.mlp = actor.net
    model.obs_normalizer = actor.norm
    model.distribution = (HandTanhGaussian(~task.arm, precision=task.tip_precision)
                          if task.fingertip_priority or task.tip_precision else
                          TanhGaussian(N, init_std=.08, std_range=(.025, .20), std_type="log") if task.uses_hand_actor else
                          TanhGaussian(N, init_std=.18, std_range=(.035, .45), std_type="log"))
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(512, 512, 256, 128), obs_normalization=True)
    storage = RolloutStorage("rl", count, rollout, obs, (N,), device=device)
    algo = PPO(model, critic, storage, learning_rate=args.lr, num_learning_epochs=4,
               num_mini_batches=8, desired_kl=.016, entropy_coef=0., device=device,
               schedule="fixed" if task.tip_precision else "adaptive")
    initial_iteration = 0
    prior_steps = 0
    prior_seconds = 0.
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if payload["task_id"] != task.task_id:
            raise ValueError("PPO checkpoint task mismatch")
        initial_iteration, prior_steps = payload["iteration"], payload["sim_steps"]
        if not 0 <= initial_iteration < iterations or prior_steps != initial_iteration * rollout * count:
            raise ValueError("Resume requires a larger cumulative budget and the checkpoint's rollout shape")
        actor.load_state_dict(payload["actor"])
        critic.load_state_dict(payload["critic"])
        model.distribution.load_state_dict(payload["distribution"])
        algo.optimizer.load_state_dict(payload["optimizer"])
        algo.learning_rate = algo.optimizer.param_groups[0]["lr"]
        prior_seconds = payload["wall_seconds"]
        env.common_step_counter = prior_steps // count
        print(f"PPO resumed at iteration {initial_iteration} with {prior_steps} transitions "
              f"and learning rate {algo.learning_rate:.8g}; target iteration {iterations}", flush=True)
    elif args.arm_init:
        hand = torch.load(args.init, map_location=device, weights_only=True)
        arm = torch.load(args.arm_init, map_location=device, weights_only=True)
        actor.load_precision_weights(hand["actor"], arm["actor"])
    elif args.init:
        payload = torch.load(args.init, map_location=device, weights_only=True)
        if payload["task_id"] != task.task_id:
            raise ValueError("PPO checkpoint task mismatch")
        if args.init_v1_arm:
            actor.net.arm_policy.load_state_dict(payload["actor"])
        elif args.init_intent_actor:
            actor.load_fingertip_weights(payload["actor"])
        elif args.init_pose_actor:
            actor.load_state_dict(payload["actor"])
        else:
            actor.load_state_dict(payload["actor"])
            model.distribution.load_state_dict(payload["distribution"])
        if args.init_privileged:
            critic.load_state_dict(payload["critic"])
    elif not task.freeze_actor_norm:
        actor.norm.update(obs["policy"])
    if not (args.init_privileged or args.resume):
        critic.update_normalization(obs)
    if args.native_robot_usd:
        for param in model.mlp.hand_nets.parameters():
            param.requires_grad_(False)
        model.distribution.requires_grad_(False)
    before = [p.detach().clone() for p in actor.parameters()]
    period = max(1, math.ceil((iterations - initial_iteration) / args.checkpoints))
    start = time.perf_counter()

    def checkpoint(name, step):
        atomic_checkpoint({
            "task_id": task.task_id, "actor": actor.state_dict(), "critic": critic.state_dict(),
            "distribution": model.distribution.state_dict(), "optimizer": algo.optimizer.state_dict(),
            "sim_steps": env.common_step_counter * count, "wall_seconds": prior_seconds + time.perf_counter() - start,
            "iteration": step,
        }, args.output / name)

    with (args.output / "metrics.jsonl").open("w", buffering=1) as stream:
        for iteration in range(initial_iteration + 1, iterations + 1):
            sums = {}
            with torch.inference_mode():
                for _ in range(rollout):
                    action = algo.act(obs)
                    obs_next, reward, done, _, _ = env.step(action)
                    info = env.transition
                    step_stats = {"reward": reward.mean(), "drop_rate": info["drop"].float().mean(),
                                  **{key: value.float().mean() for key, value in info["rollout_stats"].items()},
                                  **{key: value.float().mean() for key, value in info["rollout_event_flags"].items()}}
                    for key, value in step_stats.items():
                        sums[key] = sums.get(key, 0.) + value
                    ids = done.nonzero().flatten()
                    if len(ids):
                        obs_next, _ = env.reset(env_ids=ids)
                    obs = TensorDict(obs_next, batch_size=[count])
                    learner_reward = reward
                    if args.native_robot_usd:
                        learner_reward = info["hand_learner_reward"]
                    algo.process_env_step(obs, learner_reward, done, {})
                algo.compute_returns(obs)
            losses = algo.update()
            stats = {key: float(value / rollout) for key, value in sums.items()}
            stats.update(losses)
            if not all(math.isfinite(value) for value in stats.values()):
                raise FloatingPointError(f"Nonfinite PPO update: {stats}")
            stats.update(iteration=iteration, sim_steps=env.common_step_counter * count,
                         wall_seconds=prior_seconds + time.perf_counter() - start, lr=algo.learning_rate)
            line = json.dumps(stats)
            stream.write(line + "\n")
            print(line, flush=True)
            if (iteration - initial_iteration) % period == 0 and iteration != iterations:
                checkpoint(f"model_{iteration}.pt", iteration)
    checkpoint("model_final.pt", iterations)
    change = max(float((p.detach() - old).abs().max()) for p, old in zip(actor.parameters(), before, strict=True))
    if not math.isfinite(change) or change <= 0:
        raise RuntimeError("PPO actor weights did not update")
    steps = env.common_step_counter * count
    result = {"sim_transitions": steps - prior_steps, "iterations": iterations,
              "actor_parameter_max_change": change, "learner_wall_seconds": time.perf_counter() - start}
    if args.resume:
        result.update(resume_iteration=initial_iteration, iterations_this_run=iterations - initial_iteration,
                      cumulative_sim_transitions=steps, cumulative_learner_wall_seconds=prior_seconds + result["learner_wall_seconds"])
    return result


def main():
    p = train_parser("PPO", default_control_steps=None)
    p.add_argument("--rollout", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--iterations", type=int, help="Target learner iteration, cumulative when resuming")
    p.add_argument("--resume", type=Path, help="Restore PPO networks, optimizer and counters from a checkpoint")
    args = p.parse_args()
    if args.resume and args.init:
        p.error("Choose --resume or --init")
    if args.rollout <= 0:
        p.error("--rollout must be positive")
    if args.iterations is not None:
        if args.control_steps is not None:
            p.error("Choose --iterations or --control-steps")
        args.control_steps = args.iterations * args.rollout
    if args.control_steps is None:
        if args.resume:
            p.error("Set the cumulative target with --iterations or --control-steps")
        args.control_steps = 1024
    run(args, WholebodyS0PPO, learn)


if __name__ == "__main__":
    main()
