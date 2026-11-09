# Ported from Dex-Supermimic 2becb738; native simulator adapter.
"""Differentiable clean task observations for real and imagined states.

Both critic and dynamics use only the current frame, including simulator context.
"""

import torch

from .rotations import mv
from .state import unpack

# Keep this order synchronized with wuji-mjlab's reorient
# ``dr_params_privileged`` observation term.
DR_PARAM_NAMES = (
    "friction_ratio",
    "object_mass_ratio",
    "kp_ratio",
    "kd_ratio",
    "joint_damping_ratio",
    "object_size_ratio",
)
DR_PARAM_DIM = len(DR_PARAM_NAMES)

BASE_TERMS = {
    "joint_angles": 20,
    "qpos_error": 20,
    "cube_pos_in_tag": 3,
    "cube_ori_error": 6,
    "action_history": 20,
    "true_joint_pos": 20,
    "true_joint_vel": 20,
    "fingertip_positions": 15,
}
CURRENT_TERMS = {
    "cube_pos_in_tag_clean": 3,
    "cube_ori_error_clean": 6,
    "cube_linvel": 3,
    "cube_angvel": 3,
    "pert_dir": 3,
    "pert_velocity": 6,
    "state_progress": 4,
    "cage_counter": 1,
    "dr_params": 6,
    "palm_rot_6d": 6,
}
BASE_DIM = sum(BASE_TERMS.values())
CURRENT_DIM = sum(CURRENT_TERMS.values())
LEGACY_CURRENT_TERMS = {**CURRENT_TERMS, "dr_params": 3}
LEGACY_CURRENT_DIM = sum(LEGACY_CURRENT_TERMS.values())
FRAME_DIM = BASE_DIM + CURRENT_DIM
LEGACY_FRAME_DIM = BASE_DIM + LEGACY_CURRENT_DIM


class TaskObservations:
    base_terms = BASE_TERMS

    def __init__(self, reader):
        self.reader = reader
        self.mdp = reader.mdp
        self.dr_dim = DR_PARAM_DIM
        self.frame_dim = FRAME_DIM
        self.current_terms = CURRENT_TERMS
        self.tip_ids = [self.mdp.kin.names.index(f"right_finger{i}_link4") for i in range(1, 6)]
        self.tip_scale = 5.0
        self.tip_offset = self.mdp.default.new_tensor([0.0, 0.0, 0.5])
        self.default_velocity = self.mdp.default.new_zeros(20)
        self.episode_steps = reader.env.max_episode_length
        self.context_dim = reader.env.sim_context.shape[-1] + FRAME_DIM

    @staticmethod
    def disturbance(auxiliary):
        return auxiliary[:, 12:21]

    def auxiliary(self):
        p = self.reader
        e = p.env
        p.refresh_frame()
        return torch.cat(
            (p.palm_pos - e.scene.env_origins, p.palm_rot.flatten(-2), e.pert_direction, e.pert_velocity, e.dr_params),
            -1,
        ).clone()

    def frame(self, x, memory, auxiliary):
        """Match the original critic's term frames, including frames and scaling."""
        position, rotation, direction, perturbation, dr = auxiliary.split((3, 9, 3, 6, self.dr_dim), -1)
        rotation = rotation.reshape(-1, 3, 3)
        q, qd, _, _, v, w = unpack(x)
        policy = self.mdp.observation(x, memory)
        bodies = self.mdp.kin.forward(q, qd)[0][:, self.tip_ids]
        tips = (mv(rotation[:, None], bodies) + position[:, None] - self.tip_offset).flatten(-2)
        base = torch.cat(
            (
                policy,
                q - self.mdp.default,
                qd - self.default_velocity,
                tips * self.tip_scale,
            ),
            -1,
        )
        progress = torch.cat(
            (
                memory["hold"] / self.mdp.hold_steps,
                memory["reward_window"] / self.mdp.switch_steps,
                memory["age"] / self.episode_steps,
                memory["count"],
            ),
            -1,
        )
        current = torch.cat(
            (
                policy[:, 40:49],
                mv(rotation, v),
                mv(rotation, w),
                direction,
                perturbation,
                progress,
                (memory["cage"] / self.mdp.max_outside).clamp(max=1),
                dr,
                rotation[:, 1:, :].flatten(-2),
            ),
            -1,
        )
        return torch.cat((base, current), -1)

    def context(self, x, memory, auxiliary, simulator_context):
        return torch.cat((simulator_context, self.frame(x, memory, auxiliary)), -1)

    def dynamics(self, x, simulator_context, disturbance):
        """Dynamics-only features: physics state plus startup and exogenous context.

        ``simulator_context`` includes all six reference DR ratios followed by
        exact per-joint Kp/Kd ratios, so the WM receives the complete simulation
        context rather than only the critic's DR projection.
        """
        q, qd, position, rotation, velocity, angular_velocity = unpack(x)
        return torch.cat(
            (
                q,
                qd,
                position,
                rotation.flatten(-2),
                velocity,
                angular_velocity,
                simulator_context,
                disturbance,
            ),
            -1,
        )

    def critic(self, context):
        """Move the trailing task frame ahead of simulator context."""
        return torch.cat((context[..., -self.frame_dim :], context[..., : -self.frame_dim]), -1)
