"""Isaac Lab DirectRLEnv for the first DexTrack Allegro tracking slice."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from ....assets.allegro_hand import (
    ALLEGRO_FLY_FINGER_JOINT_NAMES,
    ALLEGRO_FLY_PALM_BODY_NAME,
    ALLEGRO_FLY_TIP_BODY_NAMES,
    validate_allegro_fly_dof_contract,
)
from .allegro_hand_track_env_cfg import AllegroHandTrackDirectEnvCfg
from .reference_loader import load_grab_reference
from .tracking_math import (
    compute_object_position_error,
    compute_tracking_reward_components,
    rotation_error_xyzw,
)


class AllegroHandTrackDirectEnv(DirectRLEnv):
    """Explicit DirectTask lifecycle for one GRAB Allegro trajectory."""

    cfg: AllegroHandTrackDirectEnvCfg

    def __init__(
        self,
        cfg: AllegroHandTrackDirectEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        validate_allegro_fly_dof_contract(self.robot)

        self.reference = load_grab_reference(cfg.resolved_reference_path, self.device)
        self.actuated_joint_ids = [
            self.robot.joint_names.index(name)
            for name in (*self._root_joint_names(), *ALLEGRO_FLY_FINGER_JOINT_NAMES)
        ]
        self.joint_lower_limits = self.robot.data.joint_limits.torch[
            :, self.actuated_joint_ids, 0
        ]
        self.joint_upper_limits = self.robot.data.joint_limits.torch[
            :, self.actuated_joint_ids, 1
        ]
        self.tip_body_ids = [self.robot.body_names.index(name) for name in ALLEGRO_FLY_TIP_BODY_NAMES]
        self.palm_body_id = self.robot.body_names.index(ALLEGRO_FLY_PALM_BODY_NAME)

        self.actions = torch.zeros((self.num_envs, 22), device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.residual_target = torch.zeros_like(self.actions)
        self.reference_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reference_end = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_lost = torch.zeros_like(self.reference_end)
        self.success = torch.zeros_like(self.reference_end)
        self.joint_targets = self.robot.data.default_joint_pos.torch[
            :, self.actuated_joint_ids
        ].clone()

    @staticmethod
    def _root_joint_names() -> tuple[str, ...]:
        from ....assets.allegro_hand import ALLEGRO_FLY_ROOT_JOINT_NAMES

        return ALLEGRO_FLY_ROOT_JOINT_NAMES

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.object = RigidObject(self.cfg.object)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _reference_frame(self):
        return self.reference.frame(self.reference_indices)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.previous_actions.copy_(self.actions)
        self.actions = actions.to(self.device).clone().clamp(-1.0, 1.0)
        self.residual_target.add_(self.cfg.action_scale * self.step_dt * self.actions)
        self.residual_target.clamp_(-self.cfg.residual_limit, self.cfg.residual_limit)

    def _apply_action(self):
        reference_hand, _, _ = self._reference_frame()
        target = reference_hand + self.residual_target
        target = torch.maximum(
            torch.minimum(target, self.joint_upper_limits), self.joint_lower_limits
        )
        self.joint_targets = target
        self.robot.set_joint_position_target_index(
            target=self.joint_targets,
            joint_ids=self.actuated_joint_ids,
        )

    def _hand_state(self):
        joint_pos = self.robot.data.joint_pos.torch[:, self.actuated_joint_ids]
        joint_vel = self.robot.data.joint_vel.torch[:, self.actuated_joint_ids]
        hand_pose = self.robot.data.root_pose_w.torch.clone()
        hand_pose[:, :3] -= self.scene.env_origins
        hand_velocity = torch.cat(
            (hand_pose, self.robot.data.root_lin_vel_w.torch, self.robot.data.root_ang_vel_w.torch),
            dim=-1,
        )
        return joint_pos, joint_vel, hand_velocity

    def _object_state(self):
        object_pose = self.object.data.root_pose_w.torch.clone()
        object_pose[:, :3] -= self.scene.env_origins
        object_velocity = torch.cat(
            (object_pose, self.object.data.root_lin_vel_w.torch, self.object.data.root_ang_vel_w.torch),
            dim=-1,
        )
        return object_pose, object_velocity

    def _get_observations(self) -> dict[str, torch.Tensor]:
        joint_pos, joint_vel, hand_velocity = self._hand_state()
        object_pose, object_velocity = self._object_state()
        reference_hand, _, _ = self._reference_frame()
        goal_pos = torch.as_tensor(
            self.cfg.object_goal_pos, dtype=object_pose.dtype, device=self.device
        ).expand(self.num_envs, -1)
        object_goal_delta = goal_pos - object_pose[:, :3]
        observation = torch.cat(
            (
                joint_pos,
                joint_vel,
                hand_velocity,
                object_velocity,
                reference_hand,
                object_goal_delta,
                self.previous_actions,
            ),
            dim=-1,
        )
        if observation.shape[-1] != self.cfg.observation_space:
            raise RuntimeError(
                f"Allegro policy observation is {observation.shape[-1]}D, "
                f"expected {self.cfg.observation_space}D."
            )
        if not torch.isfinite(observation).all():
            raise RuntimeError("Allegro policy observation contains non-finite values.")
        return {"policy": torch.clamp(observation, -5.0, 5.0)}

    def _get_rewards(self) -> torch.Tensor:
        joint_pos, _, _ = self._hand_state()
        reference_hand, reference_object_pos, reference_object_quat = self._reference_frame()
        object_pose, _ = self._object_state()
        hand_error = torch.linalg.vector_norm(joint_pos - reference_hand, dim=-1)
        object_position_error = compute_object_position_error(
            object_pose[:, :3], reference_object_pos
        )
        object_orientation_error = rotation_error_xyzw(
            object_pose[:, 3:7], reference_object_quat
        )
        body_positions = self.robot.data.body_link_pose_w.torch[:, :, :3]
        palm_position = body_positions[:, self.palm_body_id]
        palm_distance = torch.linalg.vector_norm(
            palm_position - (object_pose[:, :3] + self.scene.env_origins), dim=-1
        )
        fingertip_positions = body_positions[:, self.tip_body_ids]
        fingertip_distance = torch.linalg.vector_norm(
            fingertip_positions
            - (object_pose[:, :3] + self.scene.env_origins).unsqueeze(1),
            dim=-1,
        ).mean(dim=-1)
        goal_pos = torch.as_tensor(
            self.cfg.object_goal_pos, dtype=object_pose.dtype, device=self.device
        ).expand(self.num_envs, -1)
        goal_distance = compute_object_position_error(object_pose[:, :3], goal_pos)
        self.success = (goal_distance < 0.05) & (hand_error < 0.5)
        self.object_lost = (palm_distance > self.cfg.object_loss_distance) | (
            object_pose[:, 2] < self.cfg.object_loss_height
        )
        self.reference_end = self.reference_indices >= self.reference.num_frames - 1

        components = compute_tracking_reward_components(
            hand_error=hand_error,
            object_position_error=object_position_error,
            object_orientation_error=object_orientation_error,
            palm_distance=palm_distance,
            fingertip_distance=fingertip_distance,
            actions=self.actions,
            residual=self.residual_target,
            success=self.success,
            object_lost=self.object_lost,
            hand_scale=self.cfg.hand_reward_scale,
            object_position_scale=self.cfg.object_position_reward_scale,
            object_orientation_scale=self.cfg.object_orientation_reward_scale,
            contact_scale=self.cfg.contact_reward_scale,
            action_penalty_scale=self.cfg.action_penalty_scale,
            residual_penalty_scale=self.cfg.residual_penalty_scale,
            success_reward=self.cfg.success_reward,
        )
        reward = torch.stack(tuple(components.values()), dim=0).sum(dim=0) * self.step_dt
        log = self.extras.setdefault("log", {})
        for name, value in components.items():
            log[name] = value.mean().item()
        log["tracking/reference_frame"] = self.reference_indices.float().mean().item()
        log["termination/reference_end"] = self.reference_end.float().mean().item()
        log["termination/object_lost"] = self.object_lost.float().mean().item()
        log["tracking/success"] = self.success.float().mean().item()
        self.reference_indices.add_(1).clamp_(max=self.reference.num_frames - 1)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self.object_lost
        time_out = self.reference_end | (
            self.episode_length_buf >= self.max_episode_length - 1
        )
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.reference_indices[env_ids] = 0
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.residual_target[env_ids] = 0.0
        self.reference_end[env_ids] = False
        self.object_lost[env_ids] = False
        self.success[env_ids] = False

        default_pos = self.robot.data.default_joint_pos.torch[env_ids][:, self.actuated_joint_ids]
        default_vel = self.robot.data.default_joint_vel.torch[env_ids][:, self.actuated_joint_ids]
        self.joint_targets[env_ids] = default_pos
        self.robot.set_joint_position_target_index(
            target=default_pos, joint_ids=self.actuated_joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=default_pos, joint_ids=self.actuated_joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=default_vel, joint_ids=self.actuated_joint_ids, env_ids=env_ids
        )

        object_pose = self.object.data.default_root_pose.torch[env_ids].clone()
        object_pose[:, :3] = torch.as_tensor(
            self.cfg.object_goal_pos, dtype=object_pose.dtype, device=self.device
        ) + self.scene.env_origins[env_ids]
        object_pose[:, 3:7] = torch.as_tensor(
            self.cfg.object_goal_rot_xyzw, dtype=object_pose.dtype, device=self.device
        )
        object_velocity = torch.zeros_like(self.object.data.default_root_vel.torch[env_ids])
        self.object.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
        self.object.write_root_velocity_to_sim_index(
            root_velocity=object_velocity, env_ids=env_ids
        )
