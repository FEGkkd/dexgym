"""Headless/GUI DirectRLEnv smoke and reset/observation checks for Phase 1."""

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--task", choices=("grasp_hold", "cap_turn_30"), default="grasp_hold")
parser.add_argument("--thread-backend", choices=("legacy_d6", "mimic"), default="legacy_d6")
parser.add_argument("--solver-type", choices=("tgs", "pgs"), default="tgs")
parser.add_argument("--stable-train-bank", type=Path)
parser.add_argument("--right-cap-contact-gate",
                    choices=("body_only", "site_sidewall_proxy", "manifold_sidewall"),
                    default="manifold_sidewall")
parser.add_argument("--debug-state", action="store_true")
parser.add_argument("--solve-articulation-contact-last", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import (
    GraspHoldDirectEnvCfg, CapTurn30DirectEnvCfg,
)

cfg = GraspHoldDirectEnvCfg() if args.task == "grasp_hold" else CapTurn30DirectEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.thread_backend = args.thread_backend
cfg.sim.physics.solver_type = 1 if args.solver_type == "tgs" else 0
cfg.right_cap_contact_gate = args.right_cap_contact_gate
if args.stable_train_bank:
    cfg.stable_train_bank_path = str(args.stable_train_bank)
    cfg.stable_reset_probability = 1.0
elif args.thread_backend == "mimic":
    cfg.stable_reset_probability = 0.0
if args.solve_articulation_contact_last:
    cfg.sim.physics.solve_articulation_contact_last = True
env = BimanualBottleDirectEnv(cfg)
start = time.monotonic()
try:
    obs, _ = env.reset()
    print("RL_SMOKE_RESET", {key: tuple(value.shape) for key, value in obs.items()}, flush=True)
    if args.debug_state:
        print("RL_RESET_SOURCE", env._reset_sample_index.tolist(), env._source_stable.tolist(), flush=True)
        print("RL_RESET_LEFT", env.left.data.root_pose_w.torch[0].tolist(), flush=True)
        print("RL_RESET_BOTTLE", env.bottle.data.root_pose_w.torch[0].tolist(), flush=True)
    if not all(torch.isfinite(value).all() for value in obs.values()):
        raise AssertionError("Nonfinite reset observation")
    if args.thread_backend == "mimic":
        cap_error = (env.cap.data.root_pose_w.torch - env.thread.state.cap_pose_w).abs().max()
        print("RL_MIMIC_RESET_CAP_CACHE_ERROR", float(cap_error), flush=True)
        if cap_error > 1e-4:
            raise AssertionError("Reset cap cache does not match the PhysX articulation link")
    for step in range(args.steps):
        obs, reward, terminated, truncated, extras = env.step(
            torch.zeros(args.num_envs, 52, device=env.device))
        if not torch.isfinite(reward).all() or not all(torch.isfinite(value).all() for value in obs.values()):
            raise AssertionError("Nonfinite RL step")
        if step in (0, args.steps - 1):
            summary = {key: float(value) for key, value in extras.get("metrics", {}).items()}
            print("RL_SMOKE_STEP", step, "reward", float(reward.mean()),
                  "terminated", int(terminated.sum()), "truncated", int(truncated.sum()),
                  json.dumps(summary), flush=True)
            if args.debug_state:
                print("RL_CONTACT_MAX", float(env.contacts.force_magnitudes("Bottle", "left").max()),
                      float(env.contacts.force_magnitudes("Cap", "right").max()), flush=True)
                for obj, side in (("Bottle", "left"), ("Cap", "right")):
                    forces = env.contacts.force_magnitudes(obj, side)[0].tolist()
                    print("RL_CONTACT_BODIES", obj, side,
                          [(name, round(force, 4)) for name, force in
                           zip(env.contacts.body_names[side], forces) if force > 0.01], flush=True)
    elapsed = time.monotonic() - start
    print("RL_SMOKE_DONE", {"num_envs": args.num_envs, "steps": args.steps,
                            "elapsed_s": elapsed,
                            "environment_steps_per_second": args.num_envs * args.steps / elapsed,
                            "gpu_memory_allocated_mib": torch.cuda.memory_allocated() / 2**20}, flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    env.close()
    app.close()
