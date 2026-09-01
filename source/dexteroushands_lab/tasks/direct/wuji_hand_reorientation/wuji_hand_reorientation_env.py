"""Isaac Sim DirectTask for Wuji Hand in-hand reorientation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
    random_orientation,
    quat_from_euler_xyz,
    sample_uniform,
)

from ....assets.wuji_hand import (
    WUJI_ISAAC_BODY_NAMES,
    WUJI_ISAAC_JOINT_NAMES,
    WUJI_ISAAC_NON_TIP_BODY_NAMES,
    WUJI_ISAAC_TIP_BODY_NAMES,
)
from .wuji_hand_reorientation_env_cfg import WujiHandReorientationDirectEnvCfg


class WujiHandReorientationDirectEnv(DirectRLEnv):
    """Explicit Isaac lifecycle for the official native WUJI USD."""

    cfg: WujiHandReorientationDirectEnvCfg

    def __init__(
        self,
        cfg: WujiHandReorientationDirectEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        self._contact_body_names = list(self.robot.body_names)
        if self.contact_sensor.body_names != ["Object"]:
            raise RuntimeError(
                "Wuji contact sensor must use Object as its sensor body, got "
                f"{self.contact_sensor.body_names}"
            )
        filter_count = self.contact_sensor.contact_view.filter_count
        expected_filter_count = self.num_envs * len(self._wuji_contact_body_paths)
        if filter_count != expected_filter_count:
            raise RuntimeError(
                "Wuji contact filter count does not match the expanded exact-filter "
                f"contract: {filter_count} != {expected_filter_count}"
            )

        self._contact_filter_body_names = [
            path.rsplit("/", 1)[-1] for path in self._wuji_contact_body_paths
        ]
        if len(set(self._contact_filter_body_names)) != len(self._contact_filter_body_names):
            raise RuntimeError(
                "Wuji contact filters contain duplicate rigid-body names: "
                f"{self._contact_filter_body_names}"
            )
        missing_filter_bodies = sorted(
            set(self._contact_body_names) - set(self._contact_filter_body_names)
        )
        unexpected_filter_bodies = sorted(
            set(self._contact_filter_body_names) - set(self._contact_body_names)
        )
        if missing_filter_bodies or unexpected_filter_bodies:
            raise RuntimeError(
                "Wuji contact filters and articulation bodies do not describe the "
                f"same rigid bodies. missing={missing_filter_bodies}, "
                f"unexpected={unexpected_filter_bodies}"
            )

        # RigidContactView preserves the exact filter list order. The native
        # USD traversal order is not guaranteed to match articulation body
        # order, so build an explicit name-based permutation first. Filters
        # are generated env-major and each environment owns one contiguous
        # block in the PhysX matrix.
        filter_body_to_local_index = {
            name: index
            for index, name in enumerate(self._contact_filter_body_names)
        }
        local_indices = torch.tensor(
            [filter_body_to_local_index[name] for name in self._contact_body_names],
            dtype=torch.long,
            device=self.device,
        )
        env_offsets = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            * len(self._contact_filter_body_names)
        )
        self._contact_filter_indices = env_offsets[:, None] + local_indices[None, :]

        expected_body_names = {name for name in WUJI_ISAAC_BODY_NAMES}
        actual_body_names = {name for name in self._contact_body_names}
        missing_body_names = sorted(expected_body_names - actual_body_names)
        unexpected_body_names = sorted(actual_body_names - expected_body_names)
        if missing_body_names or unexpected_body_names:
            raise RuntimeError(
                "Official WUJI USD body mapping mismatch: "
                f"missing={missing_body_names}, unexpected={unexpected_body_names}"
            )

        body_name_to_id = {name: index for index, name in enumerate(self._contact_body_names)}
        self._tip_body_ids = [body_name_to_id[name] for name in WUJI_ISAAC_TIP_BODY_NAMES]
        self._distal_body_ids = list(self._tip_body_ids)
        self._palm_body_ids = [
            body_name_to_id[name] for name in WUJI_ISAAC_NON_TIP_BODY_NAMES
        ]
        self._cage_body_ids = list(range(len(self._contact_body_names)))
        if len(self._tip_body_ids) != 5 or len(self._palm_body_ids) != 21:
            raise RuntimeError(
                "Official WUJI contact mapping expected five tip and 21 non-tip bodies, "
                f"got tips={len(self._tip_body_ids)} non_tips={len(self._palm_body_ids)}"
            )

        self.actuated_joint_ids = [
            self.robot.joint_names.index(name) for name in cfg.actuated_joint_names
        ]
        self.joint_lower_limits = self.robot.data.joint_limits.torch[
            :, self.actuated_joint_ids, 0
        ]
        self.joint_upper_limits = self.robot.data.joint_limits.torch[
            :, self.actuated_joint_ids, 1
        ]
        self.num_actuated_joints = len(self.actuated_joint_ids)
        if self.num_actuated_joints != self.cfg.action_space:
            raise RuntimeError("The Wuji action dimension does not match the joint mapping.")
        if len(cfg.initial_joint_pos) != self.num_actuated_joints:
            raise RuntimeError("The Wuji initial pose does not match the joint mapping.")

        self.default_joint_pos = self.robot.data.default_joint_pos.torch[
            :, self.actuated_joint_ids
        ].clone()
        # Torque normalization follows the limits authored in the official USD.
        self.effort_limits = self.robot.data.joint_effort_limits.torch[
            :, self.actuated_joint_ids
        ].clone().clamp_min(1.0e-3)

        self.actions = torch.zeros(
            (self.num_envs, self.num_actuated_joints), device=self.device
        )
        self.previous_actions = torch.zeros_like(self.actions)
        self.previous_previous_actions = torch.zeros_like(self.actions)
        self.joint_targets = self.default_joint_pos.clone()

        # Isaac Lab pose buffers use xyzw; the task contract names this explicitly.
        self.tag_in_palm_pos = torch.tensor(
            cfg.tag_in_palm_pos, device=self.device
        ).view(1, 3)
        self.tag_in_palm_quat = torch.tensor(
            cfg.tag_in_palm_quat_xyzw, device=self.device
        ).view(1, 4)

        self.goal_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.goal_rot[:, 3] = 1.0

        self.hold_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.goal_timer = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.goal_reach_count = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.in_success_window = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.window_timer = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reward_hold_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reward_window_timer = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reward_hold_counter_snapshot = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reward_in_success_window = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.success_achieved = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.goal_switched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.within_threshold = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.cage_penalty_counter = torch.zeros(
            self.num_envs, device=self.device
        )

        self.observation_history = torch.zeros(
            (
                self.num_envs,
                self.cfg.observation_history_length,
                self.cfg.observation_frame_dim,
            ),
            device=self.device,
        )
        self.critic_shared_history = torch.zeros_like(self.observation_history)
        self.critic_joint_pos_history = torch.zeros(
            (self.num_envs, self.cfg.observation_history_length, self.num_actuated_joints),
            device=self.device,
        )
        self.critic_joint_vel_history = torch.zeros_like(self.critic_joint_pos_history)
        self.critic_fingertip_history = torch.zeros(
            (self.num_envs, self.cfg.observation_history_length, 15),
            device=self.device,
        )
        self.observation_history_reset = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._episode_control_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._initial_cage_escape = torch.zeros(self.num_envs, device=self.device)
        self._initial_tip_contact_count = torch.zeros(
            self.num_envs, device=self.device
        )
        self._audit_runtime_physics()

    def _audit_runtime_physics(self):
        """Audit that the loaded articulation still uses the official USD data."""
        from pxr import Usd, UsdPhysics, UsdShade

        root_path = "/World/envs/env_0/WujiHand"
        root = self.sim.stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            raise RuntimeError(f"Official WUJI USD root does not exist: {root_path}")
        official_root = self.sim.stage.GetPrimAtPath(f"{root_path}/root_joint")
        if not official_root.IsValid() or not official_root.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError("Official WUJI USD /root_joint articulation root is missing.")

        body_names = set()
        body_masses = {}
        collision_count = 0
        collision_bodies = set()
        resolved_material_paths = set()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_names.add(prim.GetName())
                mass = prim.GetAttribute("physics:mass").Get()
                body_masses[prim.GetName()] = float(mass) if mass is not None else 0.0
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_count += 1
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                if material and material.GetPrim().IsValid():
                    resolved_material_paths.add(str(material.GetPath()))
                body = prim
                while body.IsValid() and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    body = body.GetParent()
                if body.IsValid():
                    collision_bodies.add(body.GetName())

        missing_bodies = sorted(set(WUJI_ISAAC_BODY_NAMES) - body_names)
        extra_bodies = sorted(body_names - set(WUJI_ISAAC_BODY_NAMES))
        if missing_bodies or extra_bodies:
            raise RuntimeError(
                "Official WUJI USD rigid-body mapping mismatch: "
                f"missing={missing_bodies}, extra={extra_bodies}"
            )
        if collision_count != len(WUJI_ISAAC_BODY_NAMES) or collision_bodies != body_names:
            raise RuntimeError(
                "Official WUJI USD collision coverage mismatch: "
                f"count={collision_count}, bodies={sorted(collision_bodies)}"
            )
        expected_material_path = "/World/Physics/WujiHandPhysicsMaterial"
        if resolved_material_paths != {expected_material_path}:
            raise RuntimeError(
                "Official WUJI collision materials are not fully inherited: "
                f"resolved={sorted(resolved_material_paths)}, "
                f"expected={expected_material_path}"
            )

        revolute_names = [
            prim.GetName()
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if prim.IsA(UsdPhysics.RevoluteJoint)
        ]
        if tuple(revolute_names) != WUJI_ISAAC_JOINT_NAMES:
            raise RuntimeError(
                "Official WUJI USD joint mapping mismatch: "
                f"actual={revolute_names}, expected={list(WUJI_ISAAC_JOINT_NAMES)}"
            )

        data = self.robot.data
        for label, value in (
            ("joint_limits", data.joint_limits.torch),
            ("joint_vel_limits", data.joint_vel_limits.torch),
            ("joint_effort_limits", data.joint_effort_limits.torch),
            ("joint_stiffness", data.joint_stiffness.torch),
            ("joint_damping", data.joint_damping.torch),
            ("body_mass", data.body_mass.torch),
        ):
            if not torch.isfinite(value).all() or value.numel() == 0:
                raise RuntimeError(f"Official WUJI USD produced invalid {label} data.")

        filter_pairs = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            relation = prim.GetRelationship("physics:filteredPairs")
            if relation:
                filter_pairs.extend(
                    (prim.GetName(), target.name) for target in relation.GetTargets()
                )
        print(
            "WUJI_OFFICIAL_USD_AUDIT "
            f"status=aligned asset={self.cfg.robot.spawn.usd_path} "
            f"default_prim={self.sim.stage.GetDefaultPrim().GetPath()} "
            f"bodies={len(body_names)} joints={len(revolute_names)} "
            f"collision_prims={collision_count} mass_sum={sum(body_masses.values()):.8f} "
            f"physics_material={sorted(resolved_material_paths)} "
            f"filtered_pairs={filter_pairs} "
            "drives=read_from_usd",
            flush=True,
        )
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.object = RigidObject(self.cfg.object)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # The official USD exposes one rigid body per link and fingertip.
        # Prepare each source body before cloning so PhysX can report contacts.
        self._prepare_wuji_contact_bodies()
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object

        # PhysX supports one sensor body against many filtered bodies. Put the
        # sensor on Object and keep the filter columns in articulation order.
        sensor_cfg = self.cfg.contact_sensor.replace(
            prim_path="/World/envs/env_.*/Object",
            filter_prim_paths_expr=self._expanded_wuji_filter_paths,
        )
        self.contact_sensor = ContactSensor(sensor_cfg)
        self.contact_sensors = [self.contact_sensor]
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(
            intensity=1800.0, color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/Light", light_cfg)

    def _prepare_wuji_contact_bodies(self):
        """Prepare contact reporting and friction on every official WUJI body."""
        from pxr import Usd, UsdPhysics, UsdShade
        from isaaclab.sim.schemas import schemas as sim_schemas

        root_path = "/World/envs/env_0/WujiHand"
        root_prim = self.sim.stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Wuji source prim does not exist: {root_path}")

        # The USD collision meshes are instance proxies and cannot receive a
        # relationship authored directly. Bind on the non-instance rigid
        # body parents instead; physics-purpose material bindings inherit into
        # the collision prototype while preserving the official USD asset.
        material_path = "/World/Physics/WujiHandPhysicsMaterial"
        self.cfg.wuji_hand_physics_material.func(
            material_path, self.cfg.wuji_hand_physics_material
        )
        material = UsdShade.Material(self.sim.stage.GetPrimAtPath(material_path))
        if not material:
            raise RuntimeError(f"Failed to create WUJI physics material: {material_path}")

        body_paths = []
        for prim in Usd.PrimRange(root_prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if prim.IsInstance() or prim.IsInstanceProxy():
                    raise RuntimeError(
                        "Official WUJI rigid body unexpectedly became an instance: "
                        f"{prim.GetPath()}"
                    )
                material_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                material_binding_api.Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
                sim_schemas.activate_contact_sensors(
                    str(prim.GetPath()), stage=self.sim.stage
                )
                body_paths.append(str(prim.GetPath()))

        expected_body_count = len(self.cfg.actuated_joint_names) + 1
        if len(body_paths) < expected_body_count:
            raise RuntimeError(
                "Wuji USD body/contact mapping expected "
                f"at least {expected_body_count} rigid bodies, got {len(body_paths)}: "
                f"{body_paths}"
            )
        if len(body_paths) != len(WUJI_ISAAC_BODY_NAMES):
            raise RuntimeError(
                "Official WUJI physics material binding expected one binding per "
                f"body, got {len(body_paths)}"
            )
        self._wuji_physics_material_path = material_path
        self._wuji_contact_body_paths = body_paths
        self._expanded_wuji_filter_paths = [
            f"/World/envs/env_{env_id}/WujiHand/{body_path.removeprefix(root_path + "/")}"
            for env_id in range(self.cfg.scene.num_envs)
            for body_path in body_paths
        ]

    def _pre_physics_step(self, actions: torch.Tensor):
        # Keep raw action history independent from the processed target.
        self.previous_previous_actions.copy_(self.previous_actions)
        self.previous_actions.copy_(self.actions)
        self.actions = actions.to(self.device).clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        raw_target = self.default_joint_pos + self.cfg.action_scale * self.actions
        raw_target = torch.clamp(
            raw_target, self.joint_lower_limits, self.joint_upper_limits
        )

        ema_target = (
            self.cfg.ema_alpha * raw_target
            + (1.0 - self.cfg.ema_alpha) * self.joint_targets
        )
        elapsed = self.episode_length_buf * self.step_dt
        warmup = (elapsed < self.cfg.warmup_time_s).unsqueeze(-1)
        self.joint_targets = torch.where(
            warmup, self.default_joint_pos, ema_target
        )
        self.robot.set_joint_position_target_index(
            target=self.joint_targets, joint_ids=self.actuated_joint_ids
        )

    def _palm_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pose = self.robot.data.root_pose_w.torch
        palm_pos = root_pose[:, :3] - self.scene.env_origins
        palm_quat = root_pose[:, 3:7]
        return palm_pos, palm_quat

    def _tag_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        palm_pos, palm_quat = self._palm_pose()
        tag_pos = palm_pos + quat_apply(
            palm_quat, self.tag_in_palm_pos.expand(self.num_envs, -1)
        )
        tag_quat = quat_mul(
            palm_quat, self.tag_in_palm_quat.expand(self.num_envs, -1)
        )
        return tag_pos, tag_quat

    def _object_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_pose = self.object.data.root_pose_w.torch
        return (
            object_pose[:, :3] - self.scene.env_origins,
            object_pose[:, 3:7],
        )

    def _joint_pos_normalized(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos.torch[:, self.actuated_joint_ids]
        center = 0.5 * (self.joint_lower_limits + self.joint_upper_limits)
        half_range = 0.5 * (self.joint_upper_limits - self.joint_lower_limits)
        return ((joint_pos - center) / (half_range + 1.0e-6)).clamp(-1.0, 1.0)

    def _build_policy_frame(self, apply_noise: bool | None = None) -> torch.Tensor:
        joint_angles = self._joint_pos_normalized()
        qpos_error = joint_angles - self._normalize_target(self.joint_targets)

        tag_pos, tag_quat = self._tag_pose()
        object_pos, object_quat = self._object_state()
        cube_pos_in_tag = quat_apply_inverse(
            tag_quat, object_pos - tag_pos
        )

        tag_quat_inv = quat_conjugate(tag_quat)
        cube_quat_in_tag = quat_mul(tag_quat_inv, object_quat)
        goal_quat_in_tag = quat_mul(tag_quat_inv, self.goal_rot)
        quat_error_tag = quat_mul(
            cube_quat_in_tag, quat_conjugate(goal_quat_in_tag)
        )
        cube_ori_error = matrix_from_quat(quat_error_tag).reshape(
            self.num_envs, 9
        )[:, 3:]

        if apply_noise is None:
            apply_noise = self.cfg.observation_noise_enabled
        if apply_noise:
            joint_angles = joint_angles + sample_uniform(
                -0.06, 0.06, joint_angles.shape, self.device
            )
            cube_pos_in_tag = cube_pos_in_tag + sample_uniform(
                -0.008, 0.008, cube_pos_in_tag.shape, self.device
            )
            cube_ori_error = cube_ori_error + 0.05 * torch.randn_like(
                cube_ori_error
            )

            injection_mask = (
                torch.rand(self.num_envs, device=self.device) < 0.02
            ).unsqueeze(-1)
            injected_quat = random_orientation(
                self.num_envs, str(self.device)
            )
            injected_ori = matrix_from_quat(injected_quat).reshape(
                self.num_envs, 9
            )[:, 3:]
            cube_ori_error = torch.where(
                injection_mask, injected_ori, cube_ori_error
            )

        frame = torch.cat(
            (
                joint_angles,
                qpos_error,
                cube_pos_in_tag,
                cube_ori_error,
                self.previous_actions,
            ),
            dim=-1,
        )
        if frame.shape[-1] != self.cfg.observation_frame_dim:
            raise RuntimeError(
                f"Wuji policy frame is {frame.shape[-1]}D, expected "
                f"{self.cfg.observation_frame_dim}D."
            )
        return frame

    def _normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        center = 0.5 * (self.joint_lower_limits + self.joint_upper_limits)
        half_range = 0.5 * (self.joint_upper_limits - self.joint_lower_limits)
        return ((target - center) / (half_range + 1.0e-6)).clamp(-1.0, 1.0)

    def _update_history(
        self, history: torch.Tensor, frame: torch.Tensor
    ) -> None:
        """Append one frame, backfilling the first post-reset frame."""
        if history.ndim != 3 or frame.ndim != 2:
            raise RuntimeError(
                "Wuji history expects [num_envs, history, dim] and [num_envs, dim], "
                f"got {tuple(history.shape)} and {tuple(frame.shape)}"
            )
        if history.shape[0] != frame.shape[0] or history.shape[-1] != frame.shape[-1]:
            raise RuntimeError(
                "Wuji history/frame shape mismatch: "
                f"{tuple(history.shape)} versus {tuple(frame.shape)}"
            )

        history[:, :-1] = history[:, 1:].clone()
        history[:, -1] = frame
        if self.observation_history_reset.any():
            reset_ids = self.observation_history_reset.nonzero(as_tuple=False).flatten()
            history[reset_ids] = frame[reset_ids].unsqueeze(1)


    @staticmethod
    def _flatten_term_history(
        history: torch.Tensor, term_dims: tuple[int, ...]
    ) -> torch.Tensor:
        """Flatten history as term-major, oldest-to-newest."""
        if history.ndim != 3:
            raise RuntimeError(
                f"Wuji history must be rank 3, got {tuple(history.shape)}"
            )
        if sum(term_dims) != history.shape[-1]:
            raise RuntimeError(
                "Wuji term dimensions do not cover the frame: "
                f"sum={sum(term_dims)} frame={history.shape[-1]}"
            )

        parts = []
        start = 0
        for dim in term_dims:
            parts.append(
                history[:, :, start : start + dim].reshape(history.shape[0], -1)
            )
            start += dim
        return torch.cat(parts, dim=-1)

    def _build_critic_observation(self) -> torch.Tensor:
        """Build the privileged asymmetric critic observation."""
        clean_frame = self._build_policy_frame(apply_noise=False)
        self._update_history(self.critic_shared_history, clean_frame)

        joint_pos = self.robot.data.joint_pos.torch[:, self.actuated_joint_ids]
        joint_vel = self.robot.data.joint_vel.torch[:, self.actuated_joint_ids]
        joint_pos_rel = joint_pos - self.default_joint_pos
        joint_vel_rel = joint_vel - self.robot.data.default_joint_vel.torch[
            :, self.actuated_joint_ids
        ]
        self._update_history(self.critic_joint_pos_history, joint_pos_rel)
        self._update_history(self.critic_joint_vel_history, joint_vel_rel)

        fingertip_pose = self.robot.data.body_link_pose_w.torch[:, self._tip_body_ids]
        root_position = self.robot.data.root_pose_w.torch[:, :3].unsqueeze(1)
        fingertip_pos = (5.0 * (fingertip_pose[:, :, :3] - root_position)).reshape(
            self.num_envs, 15
        )
        self._update_history(self.critic_fingertip_history, fingertip_pos)

        tag_pos, tag_quat = self._tag_pose()
        object_pos, object_quat = self._object_state()
        clean_cube_pos = quat_apply_inverse(tag_quat, object_pos - tag_pos)
        tag_quat_inv = quat_conjugate(tag_quat)
        cube_quat_in_tag = quat_mul(tag_quat_inv, object_quat)
        goal_quat_in_tag = quat_mul(tag_quat_inv, self.goal_rot)
        quat_error_tag = quat_mul(
            cube_quat_in_tag, quat_conjugate(goal_quat_in_tag)
        )
        clean_cube_ori = matrix_from_quat(quat_error_tag).reshape(
            self.num_envs, 9
        )[:, 3:]

        state_progress = torch.stack(
            (
                self.hold_counter.float() / max(self.cfg.success_hold_steps, 1),
                self.reward_window_timer.float() / max(self.cfg.goal_switch_delay, 1),
                self.episode_length_buf.float() / max(self.max_episode_length, 1),
                self.goal_reach_count.float(),
            ),
            dim=-1,
        )
        cage_counter = (
            self.cage_penalty_counter
            / float(max(self.cfg.cage_max_outside_steps, 1))
        ).clamp(max=1.0).unsqueeze(-1)

        # Isaac currently has no Wuji disturbance or model-parameter DR events.
        # Keep these fields explicit and finite while their physical
        # randomization is migrated separately.
        perturbation_direction = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        perturbation_velocity = torch.zeros(
            (self.num_envs, 6), device=self.device
        )
        dr_params = torch.ones((self.num_envs, 6), device=self.device)

        palm_quat = self.robot.data.root_pose_w.torch[:, 3:7]
        palm_rot = matrix_from_quat(palm_quat).reshape(self.num_envs, 9)[:, 3:]

        critic = torch.cat(
            (
                self._flatten_term_history(
                    self.critic_shared_history, (20, 20, 3, 6, 20)
                ),
                self._flatten_term_history(
                    self.critic_joint_pos_history, (self.num_actuated_joints,)
                ),
                self._flatten_term_history(
                    self.critic_joint_vel_history, (self.num_actuated_joints,)
                ),
                self._flatten_term_history(self.critic_fingertip_history, (15,)),
                clean_cube_pos,
                clean_cube_ori,
                self.object.data.root_lin_vel_w.torch,
                self.object.data.root_ang_vel_w.torch,
                perturbation_direction,
                perturbation_velocity,
                state_progress,
                cage_counter,
                dr_params,
                palm_rot,
            ),
            dim=-1,
        )
        expected_dim = self.cfg.state_space
        if critic.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Wuji critic observation is {critic.shape[-1]}D, expected {expected_dim}D."
            )
        if not torch.isfinite(critic).all():
            raise RuntimeError("Wuji critic observation contains non-finite values.")
        return critic

    def _get_observations(self) -> dict[str, torch.Tensor]:
        frame = self._build_policy_frame()
        self._update_history(self.observation_history, frame)
        critic = self._build_critic_observation()
        self.observation_history_reset.zero_()

        obs = self._flatten_term_history(
            self.observation_history, (20, 20, 3, 6, 20)
        )
        return {
            "policy": torch.clamp(obs, -5.0, 5.0),
            "critic": critic,
        }

    def _sample_goal_rotation(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        palm_quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        relative_goal = random_orientation(len(env_ids), str(self.device))
        if palm_quat is None:
            root_pose = self.robot.data.root_pose_w.torch[env_ids]
            palm_quat = root_pose[:, 3:7]
        tag_quat = quat_mul(
            palm_quat, self.tag_in_palm_quat.expand(len(env_ids), -1)
        )
        return quat_mul(tag_quat, relative_goal)

    def _update_goal_state(self, orientation_error: torch.Tensor):
        goal_timer = self.goal_timer + 1
        within = orientation_error < self.cfg.success_threshold
        approaching = ~self.in_success_window

        hold_counter = torch.where(
            approaching & within,
            self.hold_counter + 1,
            torch.where(
                approaching,
                torch.zeros_like(self.hold_counter),
                self.hold_counter,
            ),
        )
        just_succeeded = approaching & (
            hold_counter >= self.cfg.success_hold_steps
        )

        in_success_window = self.in_success_window | just_succeeded
        window_timer = torch.where(
            just_succeeded,
            torch.zeros_like(self.window_timer),
            self.window_timer,
        )
        hold_counter = torch.where(
            just_succeeded, torch.zeros_like(hold_counter), hold_counter
        )
        self.goal_reach_count += just_succeeded.to(torch.long)

        window_timer = torch.where(
            in_success_window,
            window_timer + 1,
            window_timer,
        )
        reward_hold_counter = torch.where(
            in_success_window & within,
            self.reward_hold_counter + 1,
            self.reward_hold_counter,
        )
        reward_hold_counter = torch.where(
            just_succeeded,
            torch.zeros_like(reward_hold_counter),
            reward_hold_counter,
        )

        should_switch = in_success_window & (
            window_timer >= self.cfg.goal_switch_delay
        )
        switch_ids = should_switch.nonzero(as_tuple=False).flatten()
        if switch_ids.numel() > 0:
            self.goal_rot[switch_ids] = self._sample_goal_rotation(switch_ids)

        self.reward_window_timer = window_timer.clone()
        self.reward_hold_counter_snapshot = reward_hold_counter.clone()
        self.reward_in_success_window = in_success_window.clone()
        self.success_achieved = just_succeeded
        self.goal_switched = should_switch
        self.within_threshold = within

        self.goal_timer = torch.where(
            should_switch, torch.zeros_like(goal_timer), goal_timer
        )
        self.hold_counter = torch.where(
            should_switch, torch.zeros_like(hold_counter), hold_counter
        )
        self.in_success_window = torch.where(
            should_switch,
            torch.zeros_like(in_success_window),
            in_success_window,
        )
        self.window_timer = torch.where(
            should_switch, torch.zeros_like(window_timer), window_timer
        )
        self.reward_hold_counter = torch.where(
            should_switch,
            torch.zeros_like(reward_hold_counter),
            reward_hold_counter,
        )

    def _compute_cage_escape(self) -> torch.Tensor:
        body_pose = self.robot.data.body_link_pose_w.torch[:, self._cage_body_ids]
        palm_pose = self.robot.data.root_pose_w.torch
        palm_pos = palm_pose[:, :3]
        palm_quat = palm_pose[:, 3:7]

        hand_rel = body_pose[:, :, :3] - palm_pos.unsqueeze(1)
        hand_in_palm = quat_apply_inverse(
            palm_quat.unsqueeze(1).expand_as(body_pose[:, :, 3:7]),
            hand_rel,
        )
        object_pos, _ = self._object_state()
        cube_rel = object_pos + self.scene.env_origins - palm_pos
        cube_in_palm = quat_apply_inverse(palm_quat, cube_rel)

        xy_min = hand_in_palm[:, :, :2].amin(dim=1) - self.cfg.cage_margin
        xy_max = hand_in_palm[:, :, :2].amax(dim=1) + self.cfg.cage_margin
        z_min = hand_in_palm[:, :, 2].amin(dim=1) - self.cfg.cage_margin
        z_max = hand_in_palm[:, :, 2].amax(dim=1) + self.cfg.cage_z_upper_margin

        dx = torch.relu(xy_min[:, 0] - cube_in_palm[:, 0]) + torch.relu(
            cube_in_palm[:, 0] - xy_max[:, 0]
        )
        dy = torch.relu(xy_min[:, 1] - cube_in_palm[:, 1]) + torch.relu(
            cube_in_palm[:, 1] - xy_max[:, 1]
        )
        dz = torch.relu(z_min - cube_in_palm[:, 2]) + torch.relu(
            cube_in_palm[:, 2] - z_max
        )
        return dx + dy + dz

    def _update_cage_counter(self):
        escape_distance = self._compute_cage_escape()
        outside = escape_distance > 0.0
        self.cage_penalty_counter = torch.where(
            outside,
            self.cage_penalty_counter + 1.0,
            (self.cage_penalty_counter - self.cfg.cage_counter_decay).clamp(
                min=0.0
            ),
        ).clamp(max=15.0)
        return outside, escape_distance

    def _contact_matrix_with_object(
        self, matrix_proxy, label: str
    ) -> torch.Tensor:
        """Return a filtered Object/hand matrix in articulation body order."""
        if matrix_proxy is None:
            raise RuntimeError(f"Wuji contact sensor has no filtered {label} matrix.")
        force_matrix = matrix_proxy.torch
        expected_shape = (
            self.num_envs,
            1,
            self.num_envs * len(self._contact_filter_body_names),
            3,
        )
        if tuple(force_matrix.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected Wuji Object-centric {label} shape: "
                f"{tuple(force_matrix.shape)} != {expected_shape}"
            )
        force_matrix = force_matrix[:, 0, :, :]
        env_indices = torch.arange(self.num_envs, device=self.device).unsqueeze(1)
        env_indices = env_indices.expand_as(self._contact_filter_indices)
        return force_matrix[env_indices, self._contact_filter_indices]

    def _contact_forces_with_object(self) -> torch.Tensor:
        """Return normal Object forces against hand bodies in articulation order."""
        return self._contact_matrix_with_object(
            self.contact_sensor.data.force_matrix_w, "normal-force"
        )

    def _contact_friction_forces_with_object(self) -> torch.Tensor:
        """Return friction Object forces against hand bodies in articulation order."""
        return self._contact_matrix_with_object(
            self.contact_sensor.data.friction_forces_w, "friction-force"
        )

    def _get_rewards(self) -> torch.Tensor:
        object_pose = self.object.data.root_pose_w.torch
        object_quat = object_pose[:, 3:7]
        orientation_error = quat_error_magnitude(object_quat, self.goal_rot)
        self._update_goal_state(orientation_error)
        outside, escape_distance = self._update_cage_counter()

        # Capture the first post-reset state per environment. This exposes
        # initialization regressions without changing the policy contract.
        first_control_step = self._episode_control_steps == 0
        if torch.any(first_control_step):
            self._initial_cage_escape[first_control_step] = escape_distance[
                first_control_step
            ]

        orientation_alignment = torch.where(
            orientation_error <= self.cfg.success_threshold,
            torch.ones_like(orientation_error),
            (
                1.0
                - (orientation_error - self.cfg.success_threshold) / math.pi
            ).clamp(min=0.0),
        )
        joint_pos = self.robot.data.joint_pos.torch[:, self.actuated_joint_ids]
        hand_pose_penalty = torch.sum(
            torch.square(joint_pos - self.default_joint_pos), dim=-1
        )
        first_order = torch.sum(
            torch.square(self.actions - self.previous_actions), dim=-1
        )
        second_order = torch.sum(
            torch.square(
                self.actions
                - 2.0 * self.previous_actions
                + self.previous_previous_actions
            ),
            dim=-1,
        )
        action_rate_penalty = first_order + second_order

        applied_torque = self.robot.data.applied_torque.torch[
            :, self.actuated_joint_ids
        ]
        torque_penalty = torch.sum(
            torch.square(applied_torque / self.effort_limits.clamp_min(1.0e-3)),
            dim=-1,
        )
        object_contact_forces = self._contact_forces_with_object()
        object_contact_mag = torch.linalg.norm(object_contact_forces, dim=-1)
        object_contact_mask = object_contact_mag > self.cfg.contact_force_threshold
        tip_contact = object_contact_mask[:, self._tip_body_ids]
        distal_contact = object_contact_mask[:, self._distal_body_ids]
        palm_contact = object_contact_mask[:, self._palm_body_ids]
        if torch.any(first_control_step):
            self._initial_tip_contact_count[first_control_step] = tip_contact[
                first_control_step
            ].float().sum(dim=-1)
        tip_velocity = self.robot.data.body_link_vel_w.torch[:, self._tip_body_ids, :3]
        object_velocity = self.object.data.root_lin_vel_w.torch.unsqueeze(1)
        tip_slide_penalty = torch.sum(
            torch.linalg.norm(tip_velocity - object_velocity, dim=-1) * tip_contact.float(),
            dim=-1,
        )
        palm_detach_reward = (
            (~torch.any(palm_contact, dim=-1)) & torch.any(distal_contact, dim=-1)
        ).float()

        cage_penalty = (
            outside.float()
            * self.cage_penalty_counter
            / float(self.cfg.cage_max_outside_steps)
            * 4.0
        )
        hold_reward = (
            self.reward_in_success_window
            & self.within_threshold
        ).float() * self.reward_hold_counter_snapshot.float()

        control_dt = self.cfg.sim.dt * self.cfg.decimation
        reward = control_dt * (
            15.0 * orientation_alignment
            - 0.2 * hand_pose_penalty
            - 1.0 * action_rate_penalty
            - 24.0 * torque_penalty
            - 500.0 * cage_penalty
            - 0.3 * tip_slide_penalty
            + 0.5 * palm_detach_reward
            + 11.4 * hold_reward
        )

        log = self.extras.setdefault("log", {})
        log["Metrics/orientation_error_rad"] = orientation_error.mean().item()
        log["Metrics/goal_reach_count"] = self.goal_reach_count.float().mean().item()
        log["Metrics/success_rate"] = self.success_achieved.float().mean().item()
        log["Metrics/cage_escape_distance"] = escape_distance.mean().item()
        log["Metrics/cage_outside_rate"] = outside.float().mean().item()
        log["Metrics/torque_penalty"] = torque_penalty.mean().item()
        log["Metrics/tip_contact_count"] = tip_contact.float().sum(dim=-1).mean().item()
        log["Metrics/tip_slide_penalty"] = tip_slide_penalty.mean().item()
        log["Metrics/palm_detach_rate"] = palm_detach_reward.mean().item()
        log["Metrics/reset_initial_cage_escape"] = self._initial_cage_escape.mean().item()
        log["Metrics/reset_initial_tip_contact_count"] = (
            self._initial_tip_contact_count.mean().item()
        )
        self._episode_control_steps += 1
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cage_drop = self.cage_penalty_counter >= self.cfg.cage_max_outside_steps
        terminated = cage_drop
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(
        self, env_ids: Sequence[int] | torch.Tensor | None
    ):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        pitch = sample_uniform(-0.4, 0.1, (len(env_ids),), self.device)
        zero = torch.zeros_like(pitch)
        root_pose = default_root_pose.clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 3:7] = quat_mul(
            default_root_pose[:, 3:7], quat_from_euler_xyz(zero, pitch, zero)
        )
        root_velocity = default_root_vel.clone()
        root_velocity.zero_()
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids
        )

        default_pos = self.default_joint_pos[env_ids]
        offsets = torch.zeros_like(default_pos)
        offset_ids = torch.tensor(
            [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19],
            device=self.device,
            dtype=torch.long,
        )
        offsets[:, offset_ids] = sample_uniform(
            -0.3, -0.1, (len(env_ids), len(offset_ids)), self.device
        )
        joint_pos = torch.clamp(
            default_pos + offsets,
            self.joint_lower_limits[env_ids],
            self.joint_upper_limits[env_ids],
        )
        self.robot.set_joint_position_target_index(
            target=default_pos, joint_ids=self.actuated_joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=joint_pos, joint_ids=self.actuated_joint_ids, env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=torch.zeros_like(joint_pos),
            joint_ids=self.actuated_joint_ids,
            env_ids=env_ids,
        )

        object_pose = self.object.data.default_root_pose.torch[env_ids].clone()
        object_velocity = self.object.data.default_root_vel.torch[env_ids].clone()
        object_pose[:, :3] = torch.tensor(
            self.cfg.object_start_pos,
            device=self.device,
            dtype=object_pose.dtype,
        )
        object_pose[:, :3] += sample_uniform(
            -0.01, 0.01, object_pose[:, :3].shape, self.device
        )
        object_pose[:, :3] += self.scene.env_origins[env_ids]
        object_pose[:, 3:7] = random_orientation(len(env_ids), str(self.device))
        object_velocity.zero_()
        self.object.write_root_pose_to_sim_index(
            root_pose=object_pose, env_ids=env_ids
        )
        self.object.write_root_velocity_to_sim_index(
            root_velocity=object_velocity, env_ids=env_ids
        )

        self.goal_rot[env_ids] = self._sample_goal_rotation(
            env_ids, palm_quat=root_pose[:, 3:7]
        )
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.previous_previous_actions[env_ids] = 0.0
        self.joint_targets[env_ids] = default_pos

        self.hold_counter[env_ids] = 0
        self.goal_timer[env_ids] = 0
        self.goal_reach_count[env_ids] = 0
        self.in_success_window[env_ids] = False
        self.window_timer[env_ids] = 0
        self.reward_hold_counter[env_ids] = 0
        self.reward_window_timer[env_ids] = 0
        self.reward_hold_counter_snapshot[env_ids] = 0
        self.reward_in_success_window[env_ids] = False
        self.success_achieved[env_ids] = False
        self.goal_switched[env_ids] = False
        self.within_threshold[env_ids] = False
        self.cage_penalty_counter[env_ids] = 0.0
        self.observation_history_reset[env_ids] = True
        self.critic_shared_history[env_ids] = 0.0
        self.critic_joint_pos_history[env_ids] = 0.0
        self.critic_joint_vel_history[env_ids] = 0.0
        self.critic_fingertip_history[env_ids] = 0.0
        self._episode_control_steps[env_ids] = 0
        self._initial_cage_escape[env_ids] = 0.0
        self._initial_tip_contact_count[env_ids] = 0.0
