"""Train the Hand-centric Bimanual S0 Actor by model-based RL in MuJoCo Warp."""

import argparse
import json
from pathlib import Path
import random

import mujoco
import numpy as np
import torch

from dexgym.algorithms.supermimic.dex_reference.config import SuperMimicConfig
from dexgym.algorithms.supermimic.dex_reference.engine import SuperMimicEngine

from .env import HandCentricEnv
from .io import clip_paths, write_json
from .task import HandGoalLibrary, TASK_ID
from .warp_physics import WarpPhysics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--wm-only-iters", type=int, default=400)
    parser.add_argument("--wm-width", type=int, default=2048)
    parser.add_argument("--wm-batches", type=int, default=4)
    parser.add_argument("--wm-batch", type=int, default=1024)
    parser.add_argument("--actor-batch", type=int, default=192)
    parser.add_argument("--actor-lr", type=float, default=0.00010)
    parser.add_argument("--exploration", type=float, default=0.22)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--buffer-steps", type=int, default=192)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.wm_only_iters < args.iterations or not 1 <= args.horizon <= 32:
        parser.error("Need Actor iterations after warmup and an imagination horizon between 1 and 32")
    if min(args.num_envs, args.wm_width, args.wm_batches, args.wm_batch, args.actor_batch,
           args.log_interval, args.save_interval) <= 0 or args.buffer_steps < 64:
        parser.error("Invalid batch size, replay capacity or logging interval")
    if args.actor_lr <= 0 or not 0 <= args.exploration <= 1:
        parser.error("Invalid Actor learning rate or exploration scale")
    if (args.output / "metrics.jsonl").exists():
        parser.error("Select a fresh run directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = "cuda:0"
    model = mujoco.MjModel.from_xml_path(str(args.model))
    paths = clip_paths(args.library)
    library = HandGoalLibrary(paths, model, args.model, device)
    backend = WarpPhysics(model, library.names, args.num_envs, device, graph=not args.no_cuda_graph)
    env = HandCentricEnv(backend, library)
    task = env.task
    metadata = dict(
        task_id=TASK_ID, model=str(args.model.resolve()), model_sha256=library.model_sha256,
        joint_names=library.names, library=str(args.library.resolve()), clips=len(paths),
        frames=library.frames, motion_sha256=library.sha256, num_envs=args.num_envs, seed=args.seed,
        initialization="random", actor_objective="imagined_tracking_return",
        collection="policy_exploration", action_mode="measured_joint_position_centered",
        action_scale=task.action_step.tolist(), policy_dim=task.policy_dim,
        resume=str(args.resume.resolve()) if args.resume else None,
        backend="MuJoCo Warp", mujoco_version=mujoco.__version__,
    )
    if args.resume:
        saved = json.loads((args.resume.parent / "provenance.json").read_text())
        for key in ("task_id", "model_sha256", "motion_sha256", "joint_names", "action_scale", "policy_dim"):
            if saved.get(key) != metadata[key]:
                raise ValueError(f"Checkpoint provenance differs in {key}")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "provenance.json", metadata)
    config = SuperMimicConfig(
        seed=args.seed, max_iterations=args.iterations, buffer_steps=args.buffer_steps, warmup_steps=64,
        wm_only_iters=args.wm_only_iters, wm_window=8, wm_batches=args.wm_batches, wm_batch=args.wm_batch,
        wm_hidden=(args.wm_width,) * 3, isolate_wm_init=True, control_error_features=True,
        pi_batch=args.actor_batch, horizon=args.horizon, critic_epochs=4,
        actor_lr=args.actor_lr, action_noise=args.exploration,
        failure_warmup_iters=100, failure_clip=1.0, amp=False, compile=False, compile_policy=False,
        save_interval=args.save_interval, log_interval=args.log_interval,
    )
    engine = SuperMimicEngine(task, config, args.output)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("wm_hidden", "control_error_features", "horizon"):
            if payload["config"][key] != config.to_dict()[key]:
                raise ValueError(f"Checkpoint training configuration differs in {key}")
        if payload["iteration"] >= args.iterations:
            raise ValueError("--iterations must exceed the checkpoint iteration when resuming")
        del payload
        engine.load(args.resume)
        engine.obs = env.reset()[0]["policy"]
    else:
        engine.save(args.output / "model_0.pt", 0)
    print("S0_TRAINING", json.dumps({k: v for k, v in metadata.items() if k != "motion_sha256"}), flush=True)
    engine.learn()
    optimizer_steps = {
        name: max((int(state["step"]) for state in opt.state.values() if "step" in state), default=0)
        for name, opt in engine.optimizers.items()
    }
    proof = dict(sim_transitions=engine.sim_steps, optimizer_steps=optimizer_steps,
                 gradients=engine.last_gradients, peak_torch_bytes=torch.cuda.max_memory_allocated(),
                 training_clips=len(paths), training_frames=library.frames)
    write_json(args.output / "training_proof.json", proof)
    if not all(optimizer_steps.values()) or engine.last_gradients.get("actor", 0) <= 0:
        raise RuntimeError(f"Model-based training did not update all networks: {proof}")
    print("S0_TRAINING_PROOF", json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
