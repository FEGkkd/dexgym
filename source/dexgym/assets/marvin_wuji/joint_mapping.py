"""Name-based 54 DoF mapping; MuJoCo qpos and actuator ctrl order differ."""

from dataclasses import dataclass

import mujoco
import numpy as np

GROUPS = ("arm_left", "arm_right", "hand_left", "hand_right")
GROUP_SIZES = (7, 7, 20, 20)


@dataclass(frozen=True)
class JointContract:
    names: tuple[str, ...]
    qpos_by_name: dict[str, int]
    ctrl_by_name: dict[str, int]
    groups: dict[str, tuple[str, ...]]
    limits_by_name: dict[str, tuple[float, float]]

    def indices_for(self, names: list[str] | tuple[str, ...]) -> list[int]:
        return [self.qpos_by_name[name] for name in names]


def inspect_joint_contract(model: mujoco.MjModel) -> JointContract:
    """Validate names, one-to-one actuator transmission, limits, and arm/hand groups."""
    if (model.nq, model.nv, model.nu, model.njnt) != (54, 54, 54, 54):
        raise ValueError(f"Expected 54 hinge joints and controls; got nq={model.nq}, nv={model.nv}, nu={model.nu}")
    names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) for jid in range(model.njnt))
    if len(set(names)) != 54 or any(not name for name in names):
        raise ValueError("Joint names must be unique and nonempty")
    qpos = {name: int(model.jnt_qposadr[jid]) for jid, name in enumerate(names)}
    if set(qpos.values()) != set(range(54)):
        raise ValueError("Expected 54 independent scalar joint coordinates")
    ctrl: dict[str, int] = {}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        if jid < 0 or jid >= model.njnt:
            raise ValueError(f"Actuator {aid} has no joint transmission")
        name = names[jid]
        if name in ctrl:
            raise ValueError(f"Duplicate actuator transmission for {name}")
        ctrl[name] = aid
    if set(ctrl) != set(names):
        raise ValueError("Every joint must have exactly one independently mapped actuator")
    groups = {
        "arm_left": tuple(f"Joint{i}_L" for i in range(1, 8)),
        "arm_right": tuple(f"Joint{i}_R" for i in range(1, 8)),
        "hand_left": tuple(name for name in names if name.startswith("l_")),
        "hand_right": tuple(name for name in names if name.startswith("r_")),
    }
    if tuple(len(groups[group]) for group in GROUPS) != GROUP_SIZES:
        raise ValueError(f"Wrong arm/hand group sizes: {groups}")
    if {name for group in GROUPS for name in groups[group]} != set(names):
        raise ValueError("Unclassified or duplicated joint in control groups")
    limits: dict[str, tuple[float, float]] = {}
    for jid, name in enumerate(names):
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[jid]:
            raise ValueError(f"Joint {name} is not a limited hinge")
        lower, upper = map(float, model.jnt_range[jid])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"Invalid joint limit: {name} {lower} {upper}")
        limits[name] = (lower, upper)
    return JointContract(names, qpos, ctrl, groups, limits)


def qpos_to_newton(qpos: np.ndarray, contract: JointContract, newton_names: list[str]) -> np.ndarray:
    """Reorder MuJoCo generalized positions by verified joint names."""
    if len(newton_names) != 54 or set(newton_names) != set(contract.names):
        raise ValueError("Newton articulation does not expose exactly the expected 54 joint names")
    return np.asarray(qpos)[..., [contract.qpos_by_name[name] for name in newton_names]]
