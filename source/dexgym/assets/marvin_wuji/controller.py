"""Legacy Beta2-only controller experiment; not used by the Hand2.2 asset.

The pinned MJCF remains authoritative. This profile changes only runtime arm
drive damping; it does not edit the MJCF, USD, effort limits, or robot hardware.
"""

import math

import mujoco
import torch

ARM_KP_SOURCE = (400.0, 400.0, 440.0, 340.0, 200.0, 120.0, 85.0)
ARM_KD_SOURCE = (45.0, 45.0, 40.0, 34.0, 23.0, 16.0, 12.0)
ARM_JOINT_NAMES = tuple(f"Joint{joint}_{side}" for side in ("L", "R") for joint in range(1, 8))
USD_ANGULAR_GAIN_TO_SI = 180.0 / math.pi


def apply_newton_arm_tracking_profile(robot, *, damping_scale_source: float = 0.5) -> dict:
    """Reduce runtime arm Kv, preserving the imported Kp and force limits.

    The pinned USD authors the source XML's angular drive numbers directly.
    Newton's SI joint-gain buffer contains those numbers times 180/pi. The
    profile leaves that imported Kp intact and sets arm Kv to a measured
    fraction of the source XML's values. This is a simulation controller
    experiment, not a claim about real actuator gains or calibrated dynamics.
    """
    if not 0.0 < damping_scale_source <= 2.0:
        raise ValueError("damping_scale_source must be in (0, 2]")
    if len(robot.joint_names) != 54 or not set(ARM_JOINT_NAMES).issubset(robot.joint_names):
        raise ValueError("Expected the complete 54 DoF Marvin/Wuji articulation")
    joint_ids = [robot.joint_names.index(name) for name in ARM_JOINT_NAMES]
    source_kp = torch.tensor(ARM_KP_SOURCE * 2, dtype=torch.float32, device=robot.device)
    source_kd = torch.tensor(ARM_KD_SOURCE * 2, dtype=torch.float32, device=robot.device)
    expected_kp = source_kp * USD_ANGULAR_GAIN_TO_SI
    imported_kd = source_kd * USD_ANGULAR_GAIN_TO_SI
    tuned_kd = source_kd * damping_scale_source
    stiffness_before = robot.data.joint_stiffness.torch.clone()
    damping_before = robot.data.joint_damping.torch[:, joint_ids].clone()
    effort_before = robot.data.joint_effort_limits.torch.clone()
    if not torch.allclose(
        stiffness_before[:, joint_ids], expected_kp[None, :].expand(robot.num_instances, -1), rtol=1e-3, atol=1e-3
    ):
        raise ValueError("Legacy Beta2 high-gain profile is incompatible with Hand2.2 SI drives; use its default controller")
    if not (
        torch.allclose(damping_before, imported_kd[None, :].expand(robot.num_instances, -1), rtol=1e-3, atol=1e-3)
        or torch.allclose(damping_before, tuned_kd[None, :].expand(robot.num_instances, -1), rtol=1e-3, atol=1e-3)
    ):
        raise ValueError("Arm Kv is neither the pinned import nor this profile; refusing overwrite")
    target = tuned_kd[None, :].repeat(robot.num_instances, 1)
    robot.write_joint_damping_to_sim_index(damping=target, joint_ids=joint_ids)
    if not torch.allclose(robot.data.joint_damping.torch[:, joint_ids], target, rtol=1e-4, atol=1e-4):
        raise AssertionError("Newton did not accept the arm damping values")
    if not torch.equal(robot.data.joint_stiffness.torch, stiffness_before):
        raise AssertionError("Controller profile unexpectedly changed Kp")
    if not torch.equal(robot.data.joint_effort_limits.torch, effort_before):
        raise AssertionError("Controller profile unexpectedly changed effort limits")
    return {
        "profile": "newton_arm_tcp_tracking",
        "arm_joints": list(ARM_JOINT_NAMES),
        "damping_scale_source": damping_scale_source,
        "source_arm_kp": list(ARM_KP_SOURCE),
        "source_arm_kd": list(ARM_KD_SOURCE),
        "imported_arm_kp_scale_source": USD_ANGULAR_GAIN_TO_SI,
        "kp_unchanged": True,
        "effort_limits_unchanged": True,
    }


def match_mujoco_arm_gains_in_memory(model: mujoco.MjModel, *, damping_scale_source: float = 0.5) -> dict:
    """Match the compiled MuJoCo arm drives to the Newton tracking profile.

    This mutates only an in-memory ``MjModel`` for a controlled backend
    comparison. The authoritative XML and its saved values remain untouched.
    """
    if not 0.0 < damping_scale_source <= 2.0:
        raise ValueError("damping_scale_source must be in (0, 2]")
    if model.nu != 54 or model.njnt != 54:
        raise ValueError("Expected the complete 54 DoF Marvin/Wuji source model")
    for index, name in enumerate(ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Missing source joint {name}")
        actuator_ids = [aid for aid in range(model.nu) if int(model.actuator_trnid[aid, 0]) == joint_id]
        if len(actuator_ids) != 1:
            raise ValueError(f"Expected exactly one actuator for {name}")
        actuator = actuator_ids[0]
        source_kp = ARM_KP_SOURCE[index % 7]
        source_kd = ARM_KD_SOURCE[index % 7]
        if (
            not math.isclose(model.actuator_gainprm[actuator, 0], source_kp)
            or not math.isclose(model.actuator_biasprm[actuator, 1], -source_kp)
            or not math.isclose(model.actuator_biasprm[actuator, 2], -source_kd)
        ):
            raise ValueError(f"Unexpected compiled source gains for {name}; refusing overwrite")
        tuned_kp = source_kp * USD_ANGULAR_GAIN_TO_SI
        tuned_kd = source_kd * damping_scale_source
        model.actuator_gainprm[actuator, 0] = tuned_kp
        model.actuator_biasprm[actuator, 1] = -tuned_kp
        model.actuator_biasprm[actuator, 2] = -tuned_kd
    return {
        "profile": "in_memory_mujoco_match",
        "arm_joints": list(ARM_JOINT_NAMES),
        "damping_scale_source": damping_scale_source,
        "kp_scale_source": USD_ANGULAR_GAIN_TO_SI,
        "source_xml_unchanged": True,
        "effort_limits_unchanged": True,
    }
