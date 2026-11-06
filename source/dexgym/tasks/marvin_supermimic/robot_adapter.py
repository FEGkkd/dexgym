"""Versioned 42D Marvin arm dynamics and Newton position-servo action contract."""

from pathlib import Path

import mujoco
import torch

from dexgym.algorithms.supermimic.action import ArmActionMap, ArmJointDeltaActionMap
from dexgym.algorithms.supermimic.differentiable_fk import DifferentiableMarvinFK
from dexgym.assets.marvin_wuji.controller import ARM_KD_SOURCE, ARM_KP_SOURCE

DEFAULT_MJCF = Path(__file__).resolve().parents[2] / "assets/marvin_wuji/source_model/dual_arm_marvin.xml"


class MarvinArmDynamicsAdapter:
    state_dim = 42
    action_dim = 14
    physics_dt = 0.002
    physics_substeps = 10
    controller_kp_source = ARM_KP_SOURCE
    controller_kd_source = ARM_KD_SOURCE
    controller_damping_scale_source = 0.5
    fixed_v3_action_envelope_rad = 0.25
    fixed_v3_max_command_delta_rad = 0.01

    def __init__(self, motion, device: str | torch.device, *, action_schema: str = "legacy_wave_v1",
                 mjcf: str | Path = DEFAULT_MJCF):
        if action_schema not in ("legacy_wave_v1", "robot_fixed_v2", "robot_fixed_v3",
                                 "arm_joint_delta_v1"):
            raise ValueError("Unknown Marvin arm action schema")
        self.action_schema = action_schema
        self.device = torch.device(device)
        self.mjcf = Path(mjcf)
        model = mujoco.MjModel.from_xml_path(str(self.mjcf))
        names = tuple(motion.joint_names_arm)
        if len(names) != 14 or model.njnt != 54:
            raise ValueError("Marvin arm adapter requires a 54DoF source and 14 arm joints")
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        if min(joint_ids) < 0:
            raise ValueError("Motion joint order is absent from the robot asset")
        limits = torch.tensor(model.jnt_range[joint_ids].copy(), device=self.device, dtype=torch.float32)
        lower, upper = limits[:, 0] + 0.005, limits[:, 1] - 0.005
        if action_schema == "legacy_wave_v1":
            origin = motion.data["q_arm"][0]
            scale = (motion.data["q_arm"] - origin).abs().amax(dim=0) + 0.15
        else:
            home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")]
            origin = torch.tensor(home[joint_ids].copy(), device=self.device, dtype=torch.float32)
            scale = ((upper - lower) / 2 if action_schema == "robot_fixed_v2" else
                     torch.minimum((upper - lower) / 2,
                                   torch.full_like(lower, self.fixed_v3_action_envelope_rad)))
        if action_schema == "arm_joint_delta_v1":
            self.action_map = ArmJointDeltaActionMap(
                origin, lower, upper, delta_q_max=self.fixed_v3_max_command_delta_rad)
        else:
            max_delta = self.fixed_v3_max_command_delta_rad if action_schema == "robot_fixed_v3" else 0.04
            self.action_map = ArmActionMap(origin, lower, upper, scale, max_delta=max_delta)
        self.fk = DifferentiableMarvinFK(self.mjcf, device=self.device)

    def interpolate_command(self, previous: torch.Tensor, current: torch.Tensor,
                            substep: int) -> torch.Tensor:
        if not 0 <= substep < self.physics_substeps:
            raise ValueError("Physics substep is outside one policy step")
        alpha = (substep + 1) / self.physics_substeps
        return previous + alpha * (current - previous)

    @staticmethod
    def physical_state(native_env) -> torch.Tensor:
        q = native_env.robot.data.joint_pos.torch[:, native_env.arm_ids]
        qd = native_env.robot.data.joint_vel.torch[:, native_env.arm_ids]
        return torch.cat((q, qd, native_env.command), dim=-1)
