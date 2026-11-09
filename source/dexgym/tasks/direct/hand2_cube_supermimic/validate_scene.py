"""Compare resolved scene options and terrain, then verify actual floor contact."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp
from isaaclab.app import launch_simulation

from .env import Hand2CubeSuperMimicEnv
from .env_cfg import Hand2CubeSuperMimicCfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference_scene.read_text())
    cfg = Hand2CubeSuperMimicCfg()
    cfg.scene.num_envs = 4
    cfg.startup_randomization = cfg.randomize_reset = cfg.observation_noise = cfg.velocity_disturbance = False
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        env = Hand2CubeSuperMimicEnv(cfg)
        try:
            env.reset()
            # Newton supplies dt to the live Warp model at the first step.
            env.step(torch.zeros(4, 20, device=env.device))
            model = env.physics.native_model
            planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
            assert len(planes) == len(reference["floors"]) == 1
            floor = int(planes[0])
            errors = {}
            for field in ("friction", "solref", "solimp", "condim", "pos", "quat"):
                actual = np.asarray(getattr(model, "geom_" + field)[floor])
                expected = np.asarray(reference["floors"][0][field])
                errors["terrain/" + field] = float(np.max(np.abs(actual - expected)))
            # Native collision masks use a different encoding. Verify contact below.
            for field, expected in reference["live_options"].items():
                value = getattr(env.physics.solver.mjw_model.opt, field)
                actual = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
                errors["solver/" + field] = float(np.max(np.abs(actual - np.asarray(expected))))
                assert np.allclose(actual, expected, rtol=1e-5, atol=1e-10), (field, actual, expected)
            assert max(errors.values()) < 1e-6, errors
            ids = torch.arange(env.num_envs, device=env.device)
            pose = torch.zeros(env.num_envs, 7, device=env.device)
            pose[:, :3] = env.scene.env_origins + pose.new_tensor([1.0, 0.0, 0.12])
            pose[:, 6] = 1
            env.cube.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(4, 6, device=env.device), env_ids=ids)
            env.sim.forward()
            contacts = torch.zeros(4, dtype=torch.bool, device=env.device)
            for _ in range(60):
                env.step(torch.zeros(4, 20, device=env.device))
                d = env.physics.solver.mjw_data
                pairs = wp.to_torch(d.contact.geom).long()
                worlds = wp.to_torch(d.contact.worldid).long()
                active = torch.arange(len(pairs), device=env.device) < wp.to_torch(d.nacon)[0]
                relevant = (pairs == floor).any(-1) & (pairs == env.physics.cube_geom).any(-1)
                valid = active & relevant & (worlds >= 0) & (worlds < 4)
                contacts[worlds[valid]] = True
            heights = env.cube.data.root_link_pose_w.torch[:, 2] - env.scene.env_origins[:, 2]
            assert bool(contacts.all()), contacts
            assert bool(((heights - 0.027).abs() < 0.002).all()), heights
            result = dict(
                passed=True,
                scene_errors=errors,
                floor_contacts=contacts.tolist(),
                cube_rest_heights_m=heights.tolist(),
                reference_scene=str(args.reference_scene),
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2))
            print(result, flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
