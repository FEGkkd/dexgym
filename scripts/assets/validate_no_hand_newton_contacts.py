"""Check MJWarp hand contacts are absent while arm collision still works."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from dexgym.assets.marvin_wuji.joint_mapping import inspect_joint_contract, qpos_to_newton
from dexgym.assets.marvin_wuji.robot_cfg import NO_HAND_CONTACT_USD, make_robot_cfg, resolve_source_mjcf
from isaaclab.app import launch_simulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from validate_contacts import category, find_candidates, is_hand, solver_pairs, source_pairs


def arm_to_arm(pair: tuple[str, str]) -> bool:
    left, right = pair
    return {category(left), category(right)} == {"left", "right"} and not is_hand(left) and not is_hand(right)


def find_arm_collision_pose(model, contract, home: np.ndarray) -> tuple[np.ndarray, int]:
    data = mujoco.MjData(model)
    rng = np.random.default_rng(17)
    for sample in range(12000):
        q = home.copy()
        for group in ("arm_left", "arm_right"):
            for name in contract.groups[group]:
                low, high = contract.limits_by_name[name]
                q[contract.qpos_by_name[name]] = low + rng.uniform(0.02, 0.98) * (high - low)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        if any(arm_to_arm(pair) for pair in source_pairs(model, data)):
            return q, sample + 1
    raise ValueError("No legal arm-to-arm contact pose found")


def validate(mjcf: Path, usd: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    contract = inspect_joint_contract(model)
    home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()
    poses, hand_search_count = find_candidates(model, contract, home)
    poses["arm_links_cross"], arm_search_count = find_arm_collision_pose(model, contract, home)
    source_data = mujoco.MjData(model)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = make_robot_cfg(usd)

    cfg = SimulationCfg(
        dt=0.002, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), use_cuda_graph=False)
    )
    records = []
    with launch_simulation(cfg):
        sim = SimulationContext(cfg)
        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
        sim.reset()
        robot = scene["robot"]
        if robot.num_joints != 54 or robot.num_bodies != 74:
            raise ValueError("The no-hand-contact model lost robot bodies or joints")
        solver = (
            sim.physics_manager._solver
        )  # Diagnostic MJWarp contact arrays; Isaac Lab has no per-pair public view here.
        for label, q in poses.items():
            source_data.qpos[:] = q
            source_data.qvel[:] = 0
            mujoco.mj_forward(model, source_data)
            expected_pairs = source_pairs(model, source_data)
            mapped = torch.as_tensor(
                qpos_to_newton(q, contract, robot.joint_names), dtype=torch.float32, device=robot.device
            )[None, :]
            robot.write_joint_position_to_sim_index(position=mapped)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(mapped))
            sim.forward()
            robot.actuators.target_command.set_position_index(value=mapped, full_data=True)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(cfg.dt)
            observed_pairs, count = solver_pairs(solver)
            hand_contacts = sum(is_hand(left) or is_hand(right) for left, right in observed_pairs)
            arm_contacts = sum(arm_to_arm(pair) for pair in observed_pairs)
            expected_arm_contacts = sum(arm_to_arm(pair) for pair in expected_pairs)
            expected_hand_contacts = sum(is_hand(left) or is_hand(right) for left, right in expected_pairs)
            passed = hand_contacts == 0 and (label != "arm_links_cross" or arm_contacts > 0)
            if label == "hand_support":
                passed = passed and expected_hand_contacts > 0
            records.append(
                {
                    "pose": label,
                    "source_contacts": len(expected_pairs),
                    "source_hand_contacts": expected_hand_contacts,
                    "source_arm_contacts": expected_arm_contacts,
                    "newton_contacts": count,
                    "newton_hand_contacts": hand_contacts,
                    "newton_arm_contacts": arm_contacts,
                    "pass": passed,
                }
            )
    return {
        "source_hand_pose_search_samples": hand_search_count,
        "source_arm_pose_search_samples": arm_search_count,
        "joints": 54,
        "bodies": 74,
        "poses": records,
        "pass": all(record["pass"] for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf")
    parser.add_argument("--usd", type=Path, default=NO_HAND_CONTACT_USD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(resolve_source_mjcf(args.mjcf), args.usd)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
