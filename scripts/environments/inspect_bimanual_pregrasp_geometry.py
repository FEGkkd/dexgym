"""Measure live right-hand tip-site placement relative to the cap cylinder.

This is a geometry diagnostic. A tip site near the sidewall is not by itself
evidence of a collision contact or a stable three-finger pinch.
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--source", choices=("stable", "pregrasp"), default="pregrasp")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from dexteroushands_lab.assets.bottle.batched_thread import _rotate
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env import BimanualBottleDirectEnv
from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import GraspHoldDirectEnvCfg

cfg = GraspHoldDirectEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.sim.device = args.device
cfg.stable_reset_probability = 1.0 if args.source == "stable" else 0.0
env = BimanualBottleDirectEnv(cfg)
try:
    env.reset()
    cap = env.cap.data.root_pose_w.torch
    tips = env.right.data.body_link_pose_w.torch[:, env._tip_ids[1], :3]
    inverse = torch.cat((-cap[:, 3:6], cap[:, 6:7]), dim=-1)
    local = _rotate(inverse[:, None, :].expand(-1, 5, -1).reshape(-1, 4),
                    (tips - cap[:, None, :3]).reshape(-1, 3)).reshape(args.num_envs, 5, 3)
    radius = torch.linalg.vector_norm(local[:, :, :2], dim=-1)
    names = ("thumb", "index", "middle", "ring", "pinky")
    data = {}
    for index, name in enumerate(names):
        radial_error = radius[:, index] - env.bottle_cfg.cap_outer_radius
        z = local[:, index, 2]
        data[name] = {
            "radial_error_m_min": float(radial_error.min()),
            "radial_error_m_mean": float(radial_error.mean()),
            "radial_error_m_max": float(radial_error.max()),
            "local_z_m_min": float(z.min()), "local_z_m_mean": float(z.mean()),
            "local_z_m_max": float(z.max()),
        }
    print("PHASE1_PREGRASP_GEOMETRY", json.dumps({
        "source": args.source, "num_envs": args.num_envs,
        "cap_outer_radius_m": env.bottle_cfg.cap_outer_radius,
        "cap_half_height_m": env.bottle_cfg.cap_height / 2,
        "tip_sites": data,
    }, sort_keys=True), flush=True)
finally:
    env.close()
    app.close()
