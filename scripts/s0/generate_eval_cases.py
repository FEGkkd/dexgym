"""Build a fixed, collision-checked S0 evaluation set from the external MJCF.

The generated JSON stays in ignored artifacts. It records q_goal for audit,
but the policy evaluation passes only TCP goals to the environment.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.assets.marvin_wuji.robot_cfg import SOURCE_SHA256, resolve_source_mjcf
from scipy.spatial.transform import Rotation


def terminal_pose(model, data, home, addresses, q):
    data.qpos[:] = home
    data.qpos[addresses] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return np.stack(
        [
            np.r_[data.xpos[bid], data.xquat[bid, [1, 2, 3, 0]]]
            for bid in (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "TCP_Link_L"),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "TCP_Link_R"),
            )
        ]
    )


def collision_free_linear_path(model, data, home, addresses, start, goal):
    for fraction in np.linspace(0.0, 1.0, 16):
        data.qpos[:] = home
        data.qpos[addresses] = start * (1.0 - fraction) + goal * fraction
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        if data.ncon:
            return False
    return True


def orientation_only_goal(model, data, home, addresses, start, side, signed_angle):
    """Offline DLS obtains a feasible orientation target at fixed TCP position."""
    offset = side * 7
    q = start.copy()
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ARM_JOINT_NAMES[offset + j]) for j in range(7)]
    velocity_ids = [int(model.jnt_dofadr[j]) for j in joint_ids]
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"TCP_Link_{'L' if side == 0 else 'R'}")
    initial = terminal_pose(model, data, home, addresses, start)[side]
    position_goal = initial[:3]
    rotation_goal = Rotation.from_rotvec([0.0, 0.0, signed_angle]) * Rotation.from_quat(initial[3:])
    for _ in range(120):
        current = terminal_pose(model, data, home, addresses, q)[side]
        residual = np.r_[
            position_goal - current[:3], (rotation_goal * Rotation.from_quat(current[3:]).inv()).as_rotvec()
        ]
        if np.linalg.norm(residual[:3]) < 1e-4 and np.linalg.norm(residual[3:]) < 5e-4:
            return q
        jac_pos = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, tcp_id)
        jacobian = np.vstack((jac_pos[:, velocity_ids], jac_rot[:, velocity_ids]))
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), residual)
        q[offset : offset + 7] = np.clip(
            q[offset : offset + 7] + np.clip(delta, -0.03, 0.03),
            model.jnt_range[joint_ids, 0] + 0.03,
            model.jnt_range[joint_ids, 1] - 0.03,
        )
    return None


def generate(mjcf: Path, count_per_category: int, seed: int):
    if hashlib.sha256(mjcf.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("The external MJCF differs from the pinned Marvin model")
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    home = model.key_qpos[key_id].copy()
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
    addresses = model.jnt_qposadr[ids]
    low, high = model.jnt_range[ids, 0] + 0.03, model.jnt_range[ids, 1] - 0.03
    home_arm = home[addresses].copy()
    rng = np.random.default_rng(seed)
    cases = []
    rejected = {}
    categories = ("local", "global", "orientation_only", "wrist", "joint_boundary", "bimanual")
    for category in categories:
        accepted = 0
        for attempt in range(5000):
            side = (attempt if category == "joint_boundary" else accepted) % 2
            mask = [1, 1] if category == "bimanual" else [int(side == 0), int(side == 1)]
            start = np.clip(home_arm + rng.uniform(-0.08, 0.08, 14), low, high)
            goal = start.copy()
            arm = slice(side * 7, side * 7 + 7)
            if category == "local":
                goal[arm] += rng.uniform(-0.12, 0.12, 7)
            elif category == "global":
                start = np.clip(home_arm + rng.uniform(-0.35, 0.35, 14), low, high)
                goal = start.copy()
                goal[arm] += rng.uniform(-0.6, 0.6, 7)
            elif category == "wrist":
                goal[side * 7 + 4 : side * 7 + 7] += rng.uniform(-0.3, 0.3, 3)
            elif category == "joint_boundary":
                joint = side * 7 + (attempt % 7)
                goal[joint] = (low[joint] + 0.02) if attempt % 4 < 2 else (high[joint] - 0.02)
            elif category == "bimanual":
                goal += rng.uniform(-0.12, 0.12, 14)
            else:
                solved = orientation_only_goal(
                    model, data, home, addresses, start, side, 0.18 if accepted % 4 < 2 else -0.18
                )
                if solved is None:
                    rejected["orientation_ik"] = rejected.get("orientation_ik", 0) + 1
                    continue
                goal = solved
            goal = np.clip(goal, low, high)
            minimum_budget_s = float(np.max(np.abs(goal - start)) / 0.8 + 0.5)
            if minimum_budget_s > 8.0:
                rejected["time_budget"] = rejected.get("time_budget", 0) + 1
                continue
            if not collision_free_linear_path(model, data, home, addresses, start, goal):
                rejected["linear_path_contact"] = rejected.get("linear_path_contact", 0) + 1
                continue
            pose = terminal_pose(model, data, home, addresses, goal)
            cases.append(
                {
                    "category": category,
                    "start_arm_q_rad": start.tolist(),
                    "goal_arm_q_rad_audit_only": goal.tolist(),
                    "goal_tcp_pose_xyzw": pose.tolist(),
                    "active_mask": mask,
                    "minimum_time_budget_s_from_command_limit": minimum_budget_s,
                    "terminal_collision_free": True,
                    "linear_joint_path_collision_free_16_samples": True,
                    "source": "pinned MJCF FK; orientation_only uses offline DLS",
                }
            )
            accepted += 1
            if accepted == count_per_category:
                break
        if accepted < count_per_category:
            raise RuntimeError(f"Only {accepted}/{count_per_category} feasible {category} cases after 5000 attempts")
    result = {
        "schema_version": 1,
        "source_mjcf_sha256": SOURCE_SHA256,
        "seed": seed,
        "count_per_category": count_per_category,
        "rejected": rejected,
        "ik_failure_hard_cases": [],
        "ik_failure_status": (
            "No recorded start/target pair has yet been independently proven collision and path feasible"
        ),
        "cases": cases,
    }
    result["case_set_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf")
    parser.add_argument("--count-per-category", type=int, default=16)
    parser.add_argument("--seed", type=int, default=24681357)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(resolve_source_mjcf(args.mjcf), args.count_per_category, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
