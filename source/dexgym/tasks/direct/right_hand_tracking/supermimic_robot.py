"""Standalone Hand21 bridge into the existing SuperMimic Core."""
from dataclasses import asdict
import hashlib

import torch

from dexgym.algorithms.supermimic.core.specs import RobotSpec
from dexgym.assets.wuji_hand21 import OFFICIAL_MJCF
from dexgym.data.wuji_retarget import ROBOT_RIGHT_JOINT_NAMES
from .hand_kinematics import Hand21Kinematics
from .logic import residual_joint_target


class Hand21DynamicsAdapter:
    def __init__(self, device='cpu', dtype=torch.float32):
        self.fk = Hand21Kinematics(device=device, dtype=dtype)
        self.spec = RobotSpec(
            'wuji_hand21_standalone', 60, 20, 40, 0,
            'latent_tanh_deadzone01_delta01_hand21_v1', 'q20_qd20_qcmd20_v1',
            control_dt=1/30, physics_dt=1/480,
            asset_sha256=hashlib.sha256(OFFICIAL_MJCF.read_bytes()).hexdigest(),
            command_dim=20, joint_order=ROBOT_RIGHT_JOINT_NAMES)

    @staticmethod
    def bounded_action(action):
        return action.tanh()

    def executed_command(self, state, action):
        return residual_joint_target(state[:, 40:60], self.bounded_action(action),
            self.fk.limits, scale=.1, deadzone=.1)

    def integrate(self, state, physical_delta, action, context):
        if state.shape[-1] != 60 or physical_delta.shape != (len(state), 40):
            raise ValueError('Hand state/delta dimensions differ')
        if context.shape != (len(state), 0):
            raise ValueError('Nominal hand has empty dynamics context')
        return torch.cat((state[:, :40] + physical_delta,
                          self.executed_command(state, action)), dim=-1)

    @staticmethod
    def physical_delta(state, next_state):
        return next_state[:, :40] - state[:, :40]

    @staticmethod
    def physical_state(env):
        if env.robot.num_joints != 20:
            raise ValueError('S0 must use the standalone hand articulation')
        return torch.cat((env.robot.data.joint_pos.torch[:, env.right_ids],
                          env.robot.data.joint_vel.torch[:, env.right_ids],
                          env.target_q), dim=-1)

    def dynamics_context(self, env):
        return env.target_q.new_empty((env.num_envs, 0))

    def checkpoint_contract(self):
        # Core's existing dynamics signature records these values. No arm
        # policy/world-model weights can satisfy this robot/action schema.
        return {'adapter_version': 'hand21_s0_v1', 'robot': asdict(self.spec),
                'physics': 'official_USD_Newton_MJWarp_implicitfast_full_contacts',
                'action_scale': .1, 'action_deadzone': .1,
                'action_tanh_count': 1, 'joint_limits': self.fk.limits.tolist()}
