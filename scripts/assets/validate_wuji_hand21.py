"""Validate official Hand 2.1 USD structure, Newton FK, and position holds."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from dexgym.assets.wuji_hand21 import OFFICIAL_MJCF, OFFICIAL_USD
from dexgym.data.wuji_retarget import ROBOT_RIGHT_JOINT_NAMES, TIP_SITE_NAMES, RightHandFK
from pxr import Usd, UsdPhysics


def inspect_usd() -> dict:
    stage = Usd.Stage.Open(str(OFFICIAL_USD))
    if not stage or not stage.GetDefaultPrim().IsValid():
        raise ValueError("Official Hand 2.1 USD did not compose")
    variant = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics")
    joint_prims = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint)]
    names = {prim.GetName() for prim in joint_prims}
    roots = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    collision_instances = [prim for prim in stage.Traverse() if prim.IsInstance() and prim.GetName() == "collisions"]
    if (
        names != set(ROBOT_RIGHT_JOINT_NAMES)
        or len(joint_prims) != 20
        or len(roots) != 1
        or len(collision_instances) < 20
    ):
        raise ValueError(
            f"Unexpected hand USD: {len(joint_prims)} joints, {len(roots)} roots, "
            f"{len(collision_instances)} collision instances"
        )
    if variant.GetVariantSelection() != "PhysX":
        raise ValueError(f"Unexpected official USD variant: {variant.GetVariantSelection()}")
    self_collision = roots[0].GetAttribute("physxArticulation:enabledSelfCollisions").Get()
    if self_collision is not True:
        raise ValueError("Official hand USD does not enable self-collision")
    drives = [UsdPhysics.DriveAPI.Get(prim, "angular") for prim in joint_prims]
    drive_kp = [float(drive.GetStiffnessAttr().Get()) for drive in drives]
    drive_kd = [float(drive.GetDampingAttr().Get()) for drive in drives]
    return {
        "usd": str(OFFICIAL_USD),
        "joint_count": len(joint_prims),
        "articulation_roots": len(roots),
        "collision_instances": len(collision_instances),
        "variant": variant.GetVariantSelection(),
        "self_collision": self_collision,
        "authored_usd_kp_range": [min(drive_kp), max(drive_kp)],
        "authored_usd_kd_range": [min(drive_kd), max(drive_kd)],
    }


def _sample_targets(manifest: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    bank = json.loads(manifest.read_text())
    starts, goals = [], []
    subjects = [f"s{i}" for i in range(1, 11)]
    for index in range(count):
        subject = subjects[index % len(subjects)]
        rows = [
            entry
            for entry in bank["motions"]
            if entry.get("origin_source", entry["source"]).startswith(subject + "/") and entry["frames"] >= 6
        ]
        if not rows:
            raise ValueError(f"No eligible {subject} trajectory in motion bank")
        entry = rows[(index // len(subjects) + len(rows) // 2) % len(rows)]
        with np.load(manifest.parent / entry["file"], allow_pickle=False) as motion:
            q = np.asarray(motion["q_ref"], dtype=np.float32)
        frame = (len(q) - 5) // 2
        starts.append(q[frame])
        goals.append(q[frame + 4])
    return np.stack(starts), np.stack(goals)


def validate_newton(
    manifest: Path,
    count: int,
    dt: float,
    hold_seconds: float,
    *,
    disable_contacts: bool = False,
    zero_gravity: bool = False,
    hold_at_start: bool = False,
    integrator: str = "implicitfast",
) -> dict:
    import torch
    from dexgym.assets.wuji_hand21.robot_cfg import (
        JOINT_ORDER,
        OFFICIAL_DAMPING,
        OFFICIAL_EFFORT_LIMIT,
        OFFICIAL_STIFFNESS,
    )
    from dexgym.assets.wuji_hand21.scene_cfg import WujiHand21SceneCfg, make_newton_sim_cfg
    from isaaclab.app import launch_simulation
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    starts, goals = _sample_targets(manifest, count)
    fk = RightHandFK(OFFICIAL_MJCF)

    cfg = make_newton_sim_cfg(dt)
    cfg.physics.solver_cfg.disable_contacts = disable_contacts
    cfg.physics.solver_cfg.integrator = integrator
    if zero_gravity:
        cfg.gravity = (0.0, 0.0, 0.0)
    with launch_simulation(cfg):
        sim = SimulationContext(cfg)
        scene = InteractiveScene(WujiHand21SceneCfg(num_envs=count, env_spacing=2.0))
        sim.reset()
        robot = scene["robot"]
        if robot.num_joints != 20 or tuple(robot.joint_names) != JOINT_ORDER:
            raise ValueError(f"Newton joint contract mismatch: {robot.joint_names}")
        body_names = robot.body_names
        wrist = body_names.index("r_wrist")
        tips = [body_names.index(name) for name in TIP_SITE_NAMES]
        device = robot.device
        q_start = torch.as_tensor(starts, device=device)
        if hold_at_start:
            goals = starts.copy()
        q_goal = torch.as_tensor(goals, device=device)

        def teleport(q: torch.Tensor) -> np.ndarray:
            robot.write_joint_position_to_sim_index(position=q)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            scene.write_data_to_sim()
            sim.forward()
            scene.update(0.0)
            poses = robot.data.body_link_pose_w.torch
            wrist_pose = poses[:, wrist]
            relative = poses[:, tips, :3] - wrist_pose[:, None, :3]
            from isaaclab.utils.math import quat_rotate_inverse

            return quat_rotate_inverse(wrist_pose[:, None, 3:7].expand(-1, 5, -1), relative).cpu().numpy()

        actual_fk = teleport(q_goal)
        expected_fk = np.stack([fk.tips(q) for q in goals])
        fk_error_mm = np.linalg.norm(actual_fk - expected_fk, axis=-1) * 1000
        teleport(q_start)
        steps = round(hold_seconds / dt)
        speed_tail = []
        max_contacts = 0
        contact_pairs = []
        max_effort_ratio = 0.0
        first_steps = []
        effort_limits = robot.data.joint_effort_limits.torch
        if (effort_limits <= 0).any():
            raise ValueError("Newton returned nonpositive joint effort limits")
        for step in range(steps):
            robot.actuators.target_command.set_position_index(value=q_goal, full_data=True)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            max_contacts = max(max_contacts, int(sim.physics_manager._solver.mjw_data.nacon.numpy().max()))
            if not contact_pairs and max_contacts and count == 1:
                import mujoco

                solver = sim.physics_manager._solver
                geoms = solver.mjw_data.contact.geom.numpy()[:max_contacts]
                model = solver.mj_model
                contact_pairs = [
                    [
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[a])),
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[b])),
                    ]
                    for a, b in geoms
                ]
            max_effort_ratio = max(
                max_effort_ratio,
                float((robot.actuators.applied_effort.torch.abs() / effort_limits).max().item()),
            )
            speed_tail.append(float(robot.data.joint_vel.torch.abs().max().item()))
            if count == 1 and step < 10:
                q_now = robot.data.joint_pos.torch[0]
                v_now = robot.data.joint_vel.torch[0]
                worst = int(v_now.abs().argmax().item())
                first_steps.append({
                    "step": step + 1,
                    "worst_joint": JOINT_ORDER[worst],
                    "q_start": float(q_start[0, worst].item()),
                    "q_goal": float(q_goal[0, worst].item()),
                    "q_actual": float(q_now[worst].item()),
                    "velocity": float(v_now[worst].item()),
                    "effort": float(robot.actuators.applied_effort.torch[0, worst].item()),
                })
        final_q = robot.data.joint_pos.torch.cpu().numpy()
        final_qdot = robot.data.joint_vel.torch.cpu().numpy()
        actual_tips = teleport(torch.as_tensor(final_q, device=device))
        tip_error_mm = np.linalg.norm(actual_tips - expected_fk, axis=-1) * 1000
        stiffness = robot.data.joint_stiffness.torch[0].cpu().numpy()
        damping = robot.data.joint_damping.torch[0].cpu().numpy()
        effort = robot.data.joint_effort_limits.torch[0].cpu().numpy()

        def expected(rules: dict[str, float]) -> np.ndarray:
            return np.array(
                [next(value for pattern, value in rules.items() if re.fullmatch(pattern, name)) for name in JOINT_ORDER]
            )

        kp_error = float(np.abs(stiffness - expected(OFFICIAL_STIFFNESS)).max())
        kd_error = float(np.abs(damping - expected(OFFICIAL_DAMPING)).max())
        effort_error = float(np.abs(effort - expected(OFFICIAL_EFFORT_LIMIT)).max())
    return {
        "num_envs": count,
        "physics_dt_s": dt,
        "hold_seconds": hold_seconds,
        "fk_max_error_mm": float(fk_error_mm.max()),
        "fk_per_finger_max_mm": fk_error_mm.max(axis=0).tolist(),
        "hold_max_joint_error_rad": float(np.abs(final_q - goals).max()),
        "hold_max_joint_speed_rad_s": float(np.abs(final_qdot).max()),
        "hold_tail_max_joint_speed_rad_s": max(speed_tail[-max(1, round(0.2 / dt)) :]),
        "max_active_contacts_total": max_contacts,
        "first_contact_pairs": contact_pairs[:20],
        "first_steps": first_steps,
        "disable_contacts": disable_contacts,
        "zero_gravity": zero_gravity,
        "hold_at_start": hold_at_start,
        "integrator": integrator,
        "peak_effort_to_limit_ratio": max_effort_ratio,
        "hold_per_finger_max_error_mm": tip_error_mm.max(axis=0).tolist(),
        "runtime_kp_range": [float(stiffness.min()), float(stiffness.max())],
        "runtime_kd_range": [float(damping.min()), float(damping.max())],
        "runtime_kp_max_config_error": kp_error,
        "runtime_kd_max_config_error": kd_error,
        "runtime_effort_limit_max_config_error": effort_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/hand21_right_geometry_feasible/manifest.json")
    )
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--dt", type=float, default=1 / 500)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hand21/validate_newton.json"))
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--disable-contacts", action="store_true")
    parser.add_argument("--zero-gravity", action="store_true")
    parser.add_argument("--hold-at-start", action="store_true")
    parser.add_argument("--integrator", choices=["euler", "implicitfast", "rk4"], default="implicitfast")
    args = parser.parse_args()
    result = {"structure": inspect_usd()}
    if not args.static_only:
        result["newton"] = validate_newton(
            args.manifest.expanduser().resolve(), args.num_envs, args.dt, args.hold_seconds,
            disable_contacts=args.disable_contacts,
            zero_gravity=args.zero_gravity,
            hold_at_start=args.hold_at_start,
            integrator=args.integrator,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
