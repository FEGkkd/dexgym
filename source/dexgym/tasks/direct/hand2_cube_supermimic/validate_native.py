"""Validate native FK, persistent DR, partial reset, capture and interval pushes."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import json

    import torch
    import warp as wp
    from isaaclab.app import launch_simulation

    from dexgym.tasks.direct.hand2_cube_supermimic.env import Hand2CubeSuperMimicEnv
    from dexgym.tasks.direct.hand2_cube_supermimic.env_cfg import Hand2CubeSuperMimicCfg
    from dexgym.tasks.direct.hand2_cube_supermimic.rotations import mv

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = Hand2CubeSuperMimicCfg()
    cfg.seed = 11
    cfg.scene.num_envs = 16
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        e = Hand2CubeSuperMimicEnv(cfg)
        try:
            e.reset()
            p = e.physics
            s = p.solver
            sample = e.randomization_sample

            def verify_randomization():
                gain = wp.to_torch(s.mjw_model.actuator_gainprm)
                bias = wp.to_torch(s.mjw_model.actuator_biasprm)
                kp = gain[:, ::2, 0]
                kd = -bias[:, 1::2, 2]
                errors = {
                    "kp": float((kp - e.mdp.kp_default[None] * sample["kp"]).abs().max()),
                    "kd": float((kd + e.mdp.kd_default[None] * sample["kd"]).abs().max()),
                }
                b = int(s.mj_model.geom_bodyid[p.cube_geom])
                mass = wp.to_torch(s.mjw_model.body_mass)[:, b]
                errors["cube_mass"] = float((mass - 0.12 * sample["cube_mass_scale"][:, 0]).abs().max())
                size = wp.to_torch(s.mjw_model.geom_size)[:, p.cube_geom]
                errors["cube_size"] = float((size - 0.027 * sample["cube_size_scale"]).abs().max())
                assert max(errors.values()) < 1e-6, errors
                return errors

            start = verify_randomization()
            max_fk = 0.0
            push_events = 0
            for i in range(240):
                x = p.state()
                kin = p.kin.forward(x[:, :20], x[:, 20:40])[0]
                ids = [e.robot.body_names.index(name) for name in p.kin.names]
                poses = e.robot.data.body_link_pose_w.torch[:, ids, :3]
                local = mv(p.palm_rot[:, None].transpose(-1, -2), poses - p.palm_pos[:, None])
                max_fk = max(max_fk, float((kin - local).norm(dim=-1).max()))
                e.step(torch.randn(16, 20, device=e.device) * 0.1)
                if i == 1:
                    saved = e.transition["x1"].clone()
                    other = e.history[1:].clone()
                    e.reset(env_ids=torch.tensor([0], device=e.device))
                    assert torch.equal(saved, e.transition["x1"])
                    assert torch.equal(other, e.history[1:])
                    assert not torch.allclose(saved[0], p.state()[0])
                push_events += int((e.pert_velocity[:, :3].abs().sum(-1) > 0).sum())
                ids = (e.reset_terminated | e.reset_time_outs).nonzero().flatten()
                if len(ids):
                    e.reset(env_ids=ids)
            end = verify_randomization()
            assert max_fk < 1e-5, max_fk
            assert push_events > 0, push_events
            result = {
                "passed": True,
                "num_envs": 16,
                "control_steps": 240,
                "fk_max_error_m": max_fk,
                "startup_errors": start,
                "post_step_reset_errors": end,
                "push_observations": push_events,
                "pre_reset_capture_preserved": True,
                "partial_reset_preserves_other_histories": True,
            }
            args.output.write_text(json.dumps(result, indent=2))
            print(result, flush=True)
        finally:
            e.close()


if __name__ == "__main__":
    main()
