"""Check actual MJWarp contacts for Home, crossed arms, and hand/support poses."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from dexgym.assets.marvin_wuji.joint_mapping import inspect_joint_contract, qpos_to_newton
from dexgym.assets.marvin_wuji.robot_cfg import make_robot_cfg, resolve_source_mjcf
from isaaclab.app import launch_simulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


def category(name: str) -> str:
    if name.startswith("l_") or name.endswith("_L") or "_Base_L_" in name or "_Base_L" in name:
        return "left"
    if name.startswith("r_") or name.endswith("_R") or "_Base_R_" in name or "_Base_R" in name:
        return "right"
    if name in ("pedestal", "center_mount") or name.endswith(("_pedestal", "_center_mount")):
        return "support"
    return "other"


def is_hand(name: str) -> bool:
    return name.startswith(("l_", "r_")) or "_l_mount_" in name or "_r_mount_" in name


def contact_kind(name_a: str, name_b: str) -> str | None:
    sides = {category(name_a), category(name_b)}
    if sides == {"left", "right"}:
        return "arms_cross"
    if "support" in sides and (is_hand(name_a) or is_hand(name_b)):
        return "hand_support"
    return None


def source_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[str, str]]:
    pairs = []
    for contact in data.contact[: data.ncon]:
        body_a = int(model.geom_bodyid[contact.geom1])
        body_b = int(model.geom_bodyid[contact.geom2])
        pairs.append(
            (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_a),
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_b),
            )
        )
    return pairs


def find_candidates(model, contract, home: np.ndarray, max_samples: int = 12000) -> tuple[dict, int]:
    data = mujoco.MjData(model)
    found = {"home": home.copy()}
    rng = np.random.default_rng(12)
    for sample in range(max_samples):
        q = home.copy()
        for group in ("arm_left", "arm_right"):
            for name in contract.groups[group]:
                low, high = contract.limits_by_name[name]
                q[contract.qpos_by_name[name]] = low + rng.uniform(0.02, 0.98) * (high - low)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        kinds = {contact_kind(*pair) for pair in source_pairs(model, data)}
        for kind in ("arms_cross", "hand_support"):
            if kind in kinds and kind not in found:
                found[kind] = q.copy()
        if len(found) == 3:
            return found, sample + 1
    raise ValueError(f"Could not find both source contact categories in {max_samples} legal samples: {list(found)}")


def solver_pairs(solver) -> tuple[list[tuple[str, str]], int]:
    """Read MJWarp's internal contact diagnostic; Isaac Lab's sensor buffer is lazy."""
    count = int(solver.mjw_data.nacon.numpy()[0])
    geoms = solver.mjw_data.contact.geom.numpy()[:count]
    pairs = []
    for geom_a, geom_b in geoms:
        body_a = int(solver.mj_model.geom_bodyid[geom_a])
        body_b = int(solver.mj_model.geom_bodyid[geom_b])
        pairs.append(
            (
                mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_a),
                mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_b),
            )
        )
    return pairs, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf")
    parser.add_argument("--usd", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/marvin_wuji/contacts.json"))
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(resolve_source_mjcf(args.mjcf)))
    contract = inspect_joint_contract(model)
    home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()
    candidates, searched = find_candidates(model, contract, home)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = make_robot_cfg(args.usd)

    sim_cfg = SimulationCfg(
        dt=0.002,
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), use_cuda_graph=False),
    )
    source_data = mujoco.MjData(model)
    results = []
    with launch_simulation(sim_cfg):
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
        sim.reset()
        robot = scene["robot"]
        solver = sim.physics_manager._solver  # diagnostic only; no public MJWarp per-pair accessor in this run
        for label, q_source in candidates.items():
            source_data.qpos[:] = q_source
            source_data.qvel[:] = 0
            mujoco.mj_forward(model, source_data)
            source_contact_pairs = source_pairs(model, source_data)
            mapped = qpos_to_newton(q_source, contract, robot.joint_names)
            q_newton = torch.as_tensor(mapped, dtype=torch.float32, device=robot.device)[None, :]
            robot.write_joint_position_to_sim_index(position=q_newton)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q_newton))
            sim.forward()
            robot.actuators.target_command.set_position_index(value=q_newton, full_data=True)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_cfg.dt)
            newton_contact_pairs, newton_count = solver_pairs(solver)
            source_kind_count = sum(contact_kind(*pair) == label for pair in source_contact_pairs)
            newton_kind_count = sum(contact_kind(*pair) == label for pair in newton_contact_pairs)
            passed = (
                source_data.ncon == 0 and newton_count == 0
                if label == "home"
                else source_kind_count > 0 and newton_kind_count > 0
            )
            results.append(
                {
                    "pose": label,
                    "source_contacts": source_data.ncon,
                    "newton_contacts": newton_count,
                    "source_target_kind_contacts": source_kind_count,
                    "newton_target_kind_contacts": newton_kind_count,
                    "pass": passed,
                }
            )
    result = {"random_samples_searched": searched, "candidates": results, "pass": all(x["pass"] for x in results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
