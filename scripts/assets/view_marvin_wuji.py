"""Load the whole Marvin/Wuji articulation and inspect source keyframes in Newton.

Use --viz newton_gl on a host with a display, or --viz none for a headless GPU
asset-load check. The script never connects to the physical robot.
"""

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from dexgym.assets.marvin_wuji.joint_mapping import inspect_joint_contract, qpos_to_newton
from dexgym.assets.marvin_wuji.robot_cfg import (
    JOINT_ORDER,
    PACKAGED_KEYFRAMES,
    PACKAGED_USD,
    SOURCE_SHA256,
    make_robot_cfg,
    resolve_source_mjcf,
)
from isaaclab.app import launch_simulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", help="Pinned source MJCF; defaults to DEXGYM_MARVIN_MJCF")
    parser.add_argument("--usd", type=Path, default=PACKAGED_USD, help="Converted dual_arm_marvin.usda")
    parser.add_argument("--pose", choices=("hardware_zero", "home", "measured_j5_zero", "forward_pose"), default="home")
    parser.add_argument("--viz", choices=("none", "newton_gl"), default="newton_gl")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 0:
        parser.error("--steps must be nonnegative")
    source = args.mjcf or os.environ.get("DEXGYM_MARVIN_MJCF")
    if source:
        model = mujoco.MjModel.from_xml_path(str(resolve_source_mjcf(source)))
        contract = inspect_joint_contract(model)
        if contract.names != JOINT_ORDER:
            raise ValueError("Source MJCF joint order differs from the packaged contract")
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.pose)
        if key_id < 0:
            parser.error(f"Source keyframe missing: {args.pose}")
        pose = qpos_to_newton(model.key_qpos[key_id], contract, JOINT_ORDER)
    else:
        keyframes = json.loads(PACKAGED_KEYFRAMES.read_text())
        if keyframes["source_sha256"] != SOURCE_SHA256 or tuple(keyframes["joint_order"]) != JOINT_ORDER:
            raise ValueError("Packaged keyframes do not match the pinned source model")
        pose = np.asarray(keyframes["keyframes"][args.pose], dtype=np.float64)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = make_robot_cfg(args.usd)

    sim_cfg = SimulationCfg(
        dt=0.002,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), use_cuda_graph=False),
    )
    with launch_simulation(sim_cfg, {"visualizer": [args.viz] if args.viz != "none" else []}):
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
        sim.reset()
        robot = scene["robot"]
        if robot.num_joints != 54 or tuple(robot.joint_names) != JOINT_ORDER:
            raise ValueError("Loaded Newton joint order differs from source MJCF")
        position = torch.as_tensor(pose[None, :], dtype=torch.float32, device=robot.device)
        robot.write_joint_state_to_sim_index(position=position, velocity=torch.zeros_like(position))
        sim.forward()
        scene.update(0.0)
        for side in ("L", "R"):
            body_id = robot.body_names.index(f"TCP_Link_{side}")
            tcp = robot.data.body_link_pose_w.torch[0, body_id].detach().cpu().numpy()
            print(f"{args.pose} TCP_{side} xyz={np.array2string(tcp[:3], precision=6)} xyzw={tcp[3:]}", flush=True)
        for _ in range(args.steps):
            robot.actuators.target_command.set_position_index(value=position)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_cfg.dt)
            if not torch.isfinite(robot.data.joint_pos.torch).all():
                raise ValueError("Nonfinite joint state while viewing asset")
        print(f"Loaded 54 joints and {robot.num_bodies} bodies; {args.steps} finite Newton steps", flush=True)


if __name__ == "__main__":
    main()
