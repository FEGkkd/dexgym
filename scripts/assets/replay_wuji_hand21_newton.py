"""Directly replay one GRAB joint sequence on the standalone Hand 2.1 USD in Newton."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dexgym.assets.wuji_hand21.robot_cfg import JOINT_ORDER
from dexgym.assets.wuji_hand21.scene_cfg import WujiHand21SceneCfg, make_newton_sim_cfg
from isaaclab.app import launch_simulation
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/hand21_right_geometry_feasible/manifest.json")
    )
    parser.add_argument("--source", default="s9/spheresmall_inspect_1.npz")
    parser.add_argument("--output", type=Path, default=Path("artifacts/hand21/replay_s9_spheresmall_newton.npz"))
    args = parser.parse_args()
    manifest = args.manifest.expanduser().resolve()
    bank = json.loads(manifest.read_text())
    rows = [entry for entry in bank["motions"] if entry.get("origin_source", entry["source"]) == args.source]
    if len(rows) != 1:
        parser.error(f"Expected one uninterrupted segment for {args.source}; found {len(rows)}")
    with np.load(manifest.parent / rows[0]["file"], allow_pickle=False) as motion:
        q_ref = np.asarray(motion["q_ref"], dtype=np.float32)
        tip_ref = np.asarray(motion["tip_ref"], dtype=np.float32)
    cfg = make_newton_sim_cfg(1 / 480)
    actual = np.empty_like(q_ref)
    velocity = np.empty_like(q_ref)
    with launch_simulation(cfg):
        sim = SimulationContext(cfg)
        scene = InteractiveScene(WujiHand21SceneCfg(num_envs=1, env_spacing=2.0))
        sim.reset()
        robot = scene["robot"]
        if robot.num_joints != 20 or tuple(robot.joint_names) != JOINT_ORDER:
            raise ValueError(f"Unexpected standalone hand joint contract: {robot.joint_names}")
        initial = torch.as_tensor(q_ref[0:1], device=robot.device)
        robot.write_joint_position_to_sim_index(position=initial)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial))
        scene.write_data_to_sim()
        sim.forward()
        for frame, q in enumerate(q_ref):
            target = torch.as_tensor(q[None], device=robot.device)
            for _ in range(16):
                robot.actuators.target_command.set_position_index(value=target, full_data=True)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(cfg.dt)
            actual[frame] = robot.data.joint_pos.torch[0].cpu().numpy()
            velocity[frame] = robot.data.joint_vel.torch[0].cpu().numpy()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        q_actual=actual,
        q_ref=q_ref,
        qdot_actual=velocity,
        tip_ref=tip_ref,
        source=np.asarray(args.source),
        fps=np.float32(30),
    )
    print(
        f"states={output} frames={len(actual)} max_joint_error_rad={np.abs(actual - q_ref).max():.4f} "
        f"final_joint_speed_rad_s={np.abs(velocity[-1]).max():.4f}"
    )


if __name__ == "__main__":
    main()
