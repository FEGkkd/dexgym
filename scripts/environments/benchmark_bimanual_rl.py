"""Measure vectorized Phase 1 reset and step throughput on an Isaac Lab workstation.

The timed step loop has no per-step GPU synchronization or tensor-to-CPU copies.
PhysX allocations are included in CUDA free-memory measurements.
"""

import argparse
import json
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=1024)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--task", choices=("grasp_hold", "cap_turn_30"), default="grasp_hold")
parser.add_argument("--thread-backend", choices=("legacy_d6", "mimic"), default="legacy_d6")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import (
    CapTurn30DirectEnvCfg, GraspHoldDirectEnvCfg,
)

cfg = GraspHoldDirectEnvCfg() if args.task == "grasp_hold" else CapTurn30DirectEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.thread_backend = args.thread_backend
if args.thread_backend == "mimic":
    cfg.stable_reset_probability = 0.0
env = BimanualBottleDirectEnv(cfg)
action = torch.zeros(args.num_envs, 52, device=env.device)

def sync():
    if str(env.device).startswith("cuda"):
        torch.cuda.synchronize()

try:
    sync()
    start = time.monotonic()
    env.reset()
    sync()
    reset_s = time.monotonic() - start
    for _ in range(args.warmup_steps):
        env.step(action)
    sync()
    start = time.monotonic()
    for _ in range(args.steps):
        env.step(action)
    sync()
    step_s = time.monotonic() - start
    free, total = torch.cuda.mem_get_info() if str(env.device).startswith("cuda") else (0, 0)
    print("PHASE1_BENCHMARK", json.dumps({
        "task": args.task, "thread_backend": args.thread_backend,
        "num_envs": args.num_envs, "steps": args.steps,
        "physics_dt_s": cfg.sim.dt, "decimation": cfg.decimation,
        "reset_s": reset_s, "step_s": step_s,
        "environment_steps_per_second": args.num_envs * args.steps / step_s,
        "physics_steps_per_second": args.num_envs * args.steps * cfg.decimation / step_s,
        "gpu_memory_used_mib": (total - free) / 2**20,
        "gpu_memory_total_mib": total / 2**20,
    }, sort_keys=True), flush=True)
finally:
    env.close()
    app.close()
