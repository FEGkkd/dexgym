"""SuperMimic differentiable imagination on the wholebody_s0 residual task."""

import math
import time

import torch

from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig
from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine

from .residual import ResidualTask
from .train_common import run, train_parser

TASK_ID = "DexGym-WholeBody-S0-SuperMimic-Direct-v0"
STANDARD = SuperMimicConfig()


class WholebodyS0SuperMimic(ResidualTask):
    task_id = TASK_ID
    method = "supermimic"

    def collection_action(self, action, memory):
        if self.fingertip_priority or self.tip_precision:
            sigma = .035 if self.tip_precision else .06
            noise = sigma * torch.randn_like(action) * ~self.arm
            action = (action + noise).clamp(-1., 1.)
        return action

    def world_model_pose_loss(self, predicted, measured):
        hand_q = (predicted[:, :self.action_dim] - measured[:, :self.action_dim])[:, ~self.arm]
        joint_loss = (hand_q / .004).square().mean(-1)
        if self.tip_precision:
            predicted_tip = self.fingertip_feedback(predicted[:, :self.action_dim])
            measured_tip = self.fingertip_feedback(measured[:, :self.action_dim])
            tip_loss = ((predicted_tip - measured_tip) / .0005).square().sum(-1).mean((1, 2))
            return .45 * joint_loss + .55 * tip_loss
        return joint_loss


def collection_steps(iterations, utd):
    credit, steps = 0., 0
    for _ in range(iterations):
        credit += 1 / utd
        count = int(credit)
        steps += count
        credit -= count
    return steps


def learn(args, env):
    iterations = args.iterations
    task = env.task
    cfg = SuperMimicConfig(
        seed=args.seed, max_iterations=iterations, warmup_steps=args.warmup, buffer_steps=args.buffer_steps, utd=args.utd,
        wm_only_iters=args.wm_only_iters, wm_hidden=(args.wm_width,) * 3, wm_window=args.wm_window,
        wm_batches=args.wm_batches, wm_batch=args.wm_batch, pi_batch=args.actor_batch, horizon=args.horizon,
        max_step_reward=task.max_step_reward, actor_bootstrap_weight=args.actor_bootstrap_weight,
        actor_lr=args.actor_lr, action_noise=(0. if task.fingertip_priority or task.tip_precision else .06 if task.uses_hand_actor
                                             else STANDARD.action_noise), contact_weight=.20,
        contact_partial_intervals=True, contact_depth_weight=.03,
        kinematic_loss_weight=args.hand_wm_weight if task.uses_hand_actor else 0.,
        failure_warmup_iters=max(4, args.wm_only_iters // 2), failure_clip=.35,
        critic_epochs=args.critic_epochs, optimizer="fused-adam", control_error_features=True, isolate_wm_init=True,
        amp=False, compile=False, compile_policy=False, profile_training=True,
        save_interval=max(1, math.ceil(iterations / args.checkpoints)), log_interval=max(1, min(20, iterations // 8)),
    )
    engine = SuperMimicEngine(task, cfg, args.output)
    if args.arm_init:
        hand = torch.load(args.init, map_location=env.device, weights_only=False)["modules"]["actor"]
        arm = torch.load(args.arm_init, map_location=env.device, weights_only=False)
        engine.actor.load_precision_weights(hand, arm["modules"]["actor"])
        if args.init_dynamics and tuple(arm["config"]["wm_hidden"]) == tuple(cfg.wm_hidden):
            engine.wm.load_state_dict(arm["modules"]["wm"])
            engine.fail_head.load_state_dict(arm["modules"]["fail_head"])
        else:
            args.init_dynamics = False
    elif args.init:
        payload = torch.load(args.init, map_location=env.device, weights_only=False)
        if payload["task_id"] != task.task_id:
            raise ValueError("SuperMimic checkpoint task mismatch")
        if args.init_v1_arm:
            engine.actor.net.arm_policy.load_state_dict(payload["modules"]["actor"])
        if args.init_intent_actor:
            engine.actor.load_fingertip_weights(payload["modules"]["actor"])
        modules = {} if (args.init_v1_arm or args.init_intent_actor) else {"actor": engine.actor}
        if args.init_dynamics and tuple(payload["config"]["wm_hidden"]) == tuple(cfg.wm_hidden):
            modules.update(wm=engine.wm)
        elif args.init_dynamics:
            print("Initializing a new world model for the requested width", flush=True)
        if args.init_dynamics:
            modules.update(fail_head=engine.fail_head)
        if args.init_privileged:
            modules.update(critic=engine.critic)
        for key, module in modules.items():
            module.load_state_dict(payload["modules"][key])
    if not args.init_dynamics:
        # Physical first events are sparse in the screened reference library.
        # Initialize the failure prior low enough for useful early imagination.
        with torch.no_grad():
            engine.fail_head.net[-1].bias.fill_(-4.)
    before = [p.detach().clone() for p in engine.actor.parameters()]
    start = time.perf_counter()
    engine.learn()
    optimizer_steps = {
        name: max((int(state["step"]) for state in opt.state.values() if "step" in state), default=0)
        for name, opt in engine.optimizers.items()
    }
    change = max(float((p.detach() - old).abs().max()) for p, old in zip(engine.actor.parameters(), before, strict=True))
    if not all(optimizer_steps.values()) or not math.isfinite(change) or change <= 0:
        raise RuntimeError(f"SuperMimic did not update all networks: {optimizer_steps}")
    return {"sim_transitions": engine.sim_steps, "iterations": iterations,
            "optimizer_steps": optimizer_steps, "actor_parameter_max_change": change,
            "learner_wall_seconds": time.perf_counter() - start, "last_gradient_norms": engine.last_gradients}


def main():
    p = train_parser("SuperMimic", default_control_steps=None)
    p.add_argument("--iterations", type=int, help="Exact number of world-model/actor learner iterations")
    p.add_argument("--warmup", type=int, default=STANDARD.warmup_steps)
    p.add_argument("--wm-only-iters", type=int, default=STANDARD.wm_only_iters)
    p.add_argument("--wm-width", type=int, default=STANDARD.wm_hidden[0])
    p.add_argument("--wm-window", type=int, default=STANDARD.wm_window)
    p.add_argument("--wm-batches", type=int, default=STANDARD.wm_batches)
    p.add_argument("--wm-batch", type=int, default=STANDARD.wm_batch)
    p.add_argument("--actor-batch", type=int, default=STANDARD.pi_batch)
    p.add_argument("--horizon", type=int, default=STANDARD.horizon)
    p.add_argument("--actor-lr", type=float, default=5e-5)
    p.add_argument("--actor-bootstrap-weight", type=float, default=STANDARD.actor_bootstrap_weight)
    p.add_argument("--critic-epochs", type=int, default=STANDARD.critic_epochs)
    p.add_argument("--buffer-steps", type=int, default=STANDARD.buffer_steps)
    p.add_argument("--utd", type=float, default=STANDARD.utd)
    p.add_argument("--hand-wm-weight", type=float, default=.03)
    args = p.parse_args()
    if not 0 < args.utd <= 1:
        p.error("--utd must lie in (0, 1]")
    if args.control_steps is not None and args.iterations is not None:
        p.error("Choose --iterations or --control-steps")
    if args.iterations is None:
        if args.control_steps is None:
            args.iterations = STANDARD.max_iterations
        else:
            budget = args.control_steps - args.warmup
            args.iterations = math.ceil(budget * args.utd - 1e-9)
            if collection_steps(args.iterations, args.utd) != budget:
                p.error("The control budget cannot be attained with this UTD. Specify --iterations")
    if not 0 <= args.wm_only_iters < args.iterations:
        p.error("Training must include actor updates after the world-model-only stage")
    if min(args.wm_width, args.wm_window, args.wm_batches, args.wm_batch, args.actor_batch,
           args.critic_epochs, args.buffer_steps, args.horizon) <= 0 or args.buffer_steps < args.wm_window:
        p.error("Network, batch, horizon and replay sizes must be positive and replay must fit the model window")
    args.control_steps = args.warmup + collection_steps(args.iterations, args.utd)
    run(args, WholebodyS0SuperMimic, learn)


if __name__ == "__main__":
    main()
