"""Shared physically coupled GraspHold/CapTurn30 Isaac Lab DirectRLEnv.

The two world-anchored official hand articulations receive bounded wrist root
trajectories and ordinary 20-DoF finger position targets. Bottle and cap are
free dynamic bodies; only equal-and-opposite screw wrenches are applied.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
import tempfile

import torch
from isaaclab.envs import DirectRLEnv

from dexteroushands_lab.assets.batched_wrist_pose import (
    BatchedWristPoseController, _error, _mul, _normalize,
)
from dexteroushands_lab.assets.bottle import BottleConfig
from dexteroushands_lab.assets.bottle.batched_thread import BatchedThreadCoupler, _rotate
from dexteroushands_lab.assets.bottle.generate_usd import build_bottle_usd, author_thread_joint
from dexteroushands_lab.assets.bottle.mimic_thread import author_mimic_thread
from dexteroushands_lab.assets.bottle.mimic_runtime import MimicThreadRuntime
from dexteroushands_lab.assets.bottle.mimic_contract import mimic_reset_geometry
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec
from dexteroushands_lab.scenes.batched_contacts import BatchedContactMonitor
from dexteroushands_lab.scenes.bimanual_bottle import make_bimanual_bottle_scene_cfg
from dexteroushands_lab.scenes.grasp_init import grasp_sample_from_dict

from .task_contract import ACTOR_FIELDS, CRITIC_FIELDS
from .reward_terms import (
    digit_contact_score, grasp_contact_reward, physically_supported_turn_reward,
    suspended_bottle, terminal_failure_penalty, thread_opening_progress,
    physically_valid_grasp, thread_error_penalty, cap_turn_physics_valid,
)
from .reset_sampler import split_bank_indices, split_pregrasp_indices
from .contact_math import surface_slip_speed


def _relative_pose(parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
    q = _normalize(parent[:, 3:])
    inverse = torch.cat((-q[:, :3], q[:, 3:]), dim=-1)
    return torch.cat((_rotate(inverse, child[:, :3] - parent[:, :3]),
                      _normalize(_mul(inverse, _normalize(child[:, 3:])))), dim=-1)


def _bank_tensor(samples, field, device):
    return torch.tensor([field(sample) for sample in samples], device=device, dtype=torch.float32)


class BimanualBottleDirectEnv(DirectRLEnv):
    """One 52D actor controls both hands; task_kind selects the curriculum."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        from isaaclab_physx.physics import PhysxManager

        if cfg.thread_backend not in ("legacy_d6", "mimic"):
            raise ValueError("Unknown bottle thread backend")
        if cfg.thread_backend == "mimic" and cfg.stable_reset_probability != 0:
            raise ValueError("Legacy stable grasp bank violates hard-pitch mimic geometry; use pregrasp resets")
        self._mimic_backend = cfg.thread_backend == "mimic"

        self.bottle_cfg = BottleConfig(
            allow_detachment=False,
            axial_stiffness=cfg.thread_axial_stiffness,
        )
        self._asset_dir = Path(tempfile.mkdtemp(prefix="dexgym-phase1-bottle-"))
        build_bottle_usd(self.bottle_cfg, self._asset_dir)
        cfg.scene = make_bimanual_bottle_scene_cfg(
            num_envs=cfg.scene.num_envs, asset_dir=self._asset_dir,
            bottle=self.bottle_cfg, wrist_control_mode="pose_command_wrist",
            contact_debug=False, bottle_center_z=0.5,
        )
        # The shared standalone scene's broad hand sensors also match visual
        # child prims at scale. The exact object/body batch views below supply
        # all Phase 1 contact data, so omit those redundant sensors here.
        cfg.scene.left_hand_contact = None
        cfg.scene.right_hand_contact = None
        cfg.scene.cap_contact = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        # Every substep needs a newly interpolated wrist pose and screw wrench.
        self._physics_handles_decimation = False
        self.left, self.right = self.scene["left_hand"], self.scene["right_hand"]
        self.bottle, self.cap = self.scene["bottle_body"], self.scene["bottle_cap"]
        self.specs = (load_hand2_spec(variant="beta1", side="left"),
                      load_hand2_spec(variant="beta1", side="right"))
        self._all_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._joint_reorder = []
        self._joint_limits = []
        self._tip_ids = []
        for hand, spec, prefix in ((self.left, self.specs[0], "l"),
                                   (self.right, self.specs[1], "r")):
            if len(hand.joint_names) != 20 or set(hand.joint_names) != set(spec.joint_names):
                raise RuntimeError("Official Beta 1 hand joint mapping changed")
            order = [spec.joint_names.index(name) for name in hand.joint_names]
            self._joint_reorder.append(order)
            self._joint_limits.append(torch.tensor(
                [[spec.joints[i].lower, spec.joints[i].upper] for i in order],
                device=self.device, dtype=torch.float32))
            self._tip_ids.append([hand.body_names.index(f"{prefix}_{digit}_tip")
                                  for digit in ("thumb", "index_finger", "middle_finger",
                                                "ring_finger", "pinky")])
        self.wrists = (BatchedWristPoseController(self.num_envs, device=self.device),
                       BatchedWristPoseController(self.num_envs, device=self.device))
        self.thread = (MimicThreadRuntime(PhysxManager.get_physics_sim_view(),
                                          self.bottle_cfg, num_envs=self.num_envs,
                                          device=self.device)
                       if self._mimic_backend else
                       BatchedThreadCoupler(self.num_envs, self.bottle_cfg,
                                            device=self.device))
        self.contacts = BatchedContactMonitor(
            PhysxManager.get_physics_sim_view(), self.num_envs, *self.specs,
            device=self.device,
        )
        self._load_banks()
        self._finger_targets = [torch.zeros(self.num_envs, 20, device=self.device) for _ in range(2)]
        self._last_action = torch.zeros(self.num_envs, 52, device=self.device)
        self._last_raw_action = self._last_action.clone()
        self._action_clip_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._joint_clip_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._stable_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._success_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._breakaway_seen = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._maximum_opening = torch.zeros(self.num_envs, device=self.device)
        self._source_stable = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._reset_sample_index = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._theta_at_reset = torch.zeros(self.num_envs, device=self.device)
        self._axial_at_reset = torch.zeros(self.num_envs, device=self.device)
        self._bottle_at_reset = torch.zeros(self.num_envs, 7, device=self.device)
        self._previous_open = torch.zeros(self.num_envs, device=self.device)
        self._previous_proximity = torch.zeros(self.num_envs, 2, device=self.device)
        self._proximity_fresh = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._latest_thread = None
        self._metrics = None
        self._reward_sums = {name: torch.zeros(self.num_envs, device=self.device)
                             for name in ("contact", "approach", "holding", "progress",
                                          "action", "drift", "constraint", "success", "failure")}

    def _setup_scene(self):
        if self._mimic_backend:
            self.thread_joints = [author_mimic_thread(
                self.sim.stage, f"/World/envs/env_{env}", self.bottle_cfg,
                cap_local_pose_xyzw=(0.0, 0.0,
                                     0.5 + self.bottle_cfg.cap_initial_z,
                                     0.0, 0.0, 0.0, 1.0))
                for env in range(self.scene.num_envs)]
        else:
            self.thread_joints = [author_thread_joint(
                self.sim.stage, f"/World/envs/env_{env}", self.bottle_cfg)
                for env in range(self.scene.num_envs)]

    def _load_banks(self):
        root = Path(__file__).resolve().parents[5]
        stable_path = root / "docs/assets/phase0e_bimanual_grasp_init_bank.json"
        external_stable = self.cfg.stable_eval_bank_path
        external_train = self.cfg.stable_train_bank_path
        if external_stable is not None and external_train is not None:
            raise ValueError("Choose one external stable bank per run")
        if external_train is not None:
            if self.cfg.evaluation_mode:
                raise ValueError("External training bank cannot be used for held-out evaluation")
            stable_path = Path(external_train)
            if not stable_path.is_absolute():
                stable_path = root / stable_path
        if external_stable is not None:
            if not self.cfg.evaluation_mode:
                raise ValueError("External stable bank is reserved for evaluation")
            stable_path = Path(external_stable)
            if not stable_path.is_absolute():
                stable_path = root / stable_path
        pre_path = root / "docs/assets/phase0_pregrasp_reset_bank.json"
        stable = [grasp_sample_from_dict(row) for row in json.loads(stable_path.read_text())]
        pre = [grasp_sample_from_dict(row) for row in json.loads(pre_path.read_text())]
        if not stable or not pre or any(not x.validated for x in stable):
            raise RuntimeError("Stable and geometric pregrasp banks are required")
        if any(x.validated or not x.metadata.get("geometry_qualified") for x in pre):
            raise RuntimeError("Pregrasp bank contains invalid training candidates")
        if external_stable is None and external_train is None:
            stable_ids, pre_ids = split_bank_indices(
                len(stable), len(pre), evaluation=self.cfg.evaluation_mode)
        else:
            stable_ids = tuple(range(len(stable)))
            pre_ids = split_pregrasp_indices(
                len(pre), evaluation=external_stable is not None)
        stable = [stable[i] for i in stable_ids]
        pre = [pre[i] for i in pre_ids]
        if self._mimic_backend:
            for sample in pre:
                geometry = mimic_reset_geometry(sample.bottle.root_pose,
                                                 sample.cap.root_pose,
                                                 sample.thread_initial_angle_rad,
                                                 self.bottle_cfg)
                if (abs(geometry["pitch_error_m"]) > 5e-5
                        or geometry["lateral_m"] > 1e-4
                        or abs(geometry["twist_error_rad"]) > 0.01):
                    raise RuntimeError("Pregrasp Reset bank is incompatible with the mimic screw")
        self._stable_count = len(stable)
        samples = stable + pre
        for sample in samples:
            if (sample.metadata.get("asset_variant") != "beta1"
                    or sample.metadata.get("left_usd_sha256") != self.specs[0].usd_sha256
                    or sample.metadata.get("right_usd_sha256") != self.specs[1].usd_sha256):
                raise RuntimeError("Reset bank asset version differs from loaded official USD")
        self._bank = {}
        for name in ("left", "right", "bottle", "cap"):
            for field in ("root_pose", "root_velocity"):
                self._bank[f"{name}_{field}"] = _bank_tensor(
                    samples, lambda s, n=name, f=field: getattr(getattr(s, n), f), self.device)
        for name in ("left", "right"):
            for field in ("joint_position", "joint_velocity", "joint_target", "wrist_target_pose"):
                self._bank[f"{name}_{field}"] = _bank_tensor(
                    samples, lambda s, n=name, f=field: getattr(getattr(s, n), f), self.device)
        self._bank["thread_angle"] = _bank_tensor(samples,
                                                   lambda s: s.thread_initial_angle_rad, self.device)
        self._bank["snapshot_mask"] = torch.tensor(
            ["thread_coupler_state" in s.metadata for s in samples], device=self.device, dtype=torch.bool)
        template = {"last_wrapped_rad": 0., "theta_unwrapped_rad": 0.,
                    "initial_theta_rad": 0., "initial_axial_m": 0.,
                    "stick_reference_rad": 0., "sliding": False, "detached": False}
        for key in template:
            values = [s.metadata.get("thread_coupler_state", template)[key] for s in samples]
            self._bank[f"snapshot_{key}"] = torch.tensor(values, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if len(ids) == 0:
            return
        n = len(ids)
        stable = torch.rand(n, device=self.device) < self.cfg.stable_reset_probability
        indices = torch.where(
            stable,
            torch.randint(self._stable_count, (n,), device=self.device),
            self._stable_count + torch.randint(len(self._bank["thread_angle"]) - self._stable_count,
                                               (n,), device=self.device),
        )
        self._source_stable[ids] = stable
        self._reset_sample_index[ids] = indices
        origin = self.scene.env_origins[ids]

        def restore_root(asset, name):
            pose = self._bank[f"{name}_root_pose"][indices].clone()
            pose[:, :3] += origin
            velocity = self._bank[f"{name}_root_velocity"][indices]
            asset.reset(env_ids=ids)
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
            asset.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)
            return pose

        left_pose = restore_root(self.left, "left")
        right_pose = restore_root(self.right, "right")
        bottle_pose = restore_root(self.bottle, "bottle")
        if self._mimic_backend:
            # Cap is an articulation link; an independent root write would
            # conflict with the PhysX screw and can inject an initial impulse.
            self.cap.reset(env_ids=ids)
            cap_pose = self._bank["cap_root_pose"][indices].clone()
            cap_pose[:, :3] += origin
        else:
            cap_pose = restore_root(self.cap, "cap")
        for side, (hand, name, pose) in enumerate(((self.left, "left", left_pose),
                                                  (self.right, "right", right_pose))):
            reorder = self._joint_reorder[side]
            hand.write_joint_position_to_sim_index(
                position=self._bank[f"{name}_joint_position"][indices][:, reorder], env_ids=ids)
            hand.write_joint_velocity_to_sim_index(
                velocity=self._bank[f"{name}_joint_velocity"][indices][:, reorder], env_ids=ids)
            target = self._bank[f"{name}_joint_target"][indices][:, reorder]
            hand.set_joint_position_target_index(target=target, env_ids=ids)
            self._finger_targets[side][ids] = target
            wrist_target = self._bank[f"{name}_wrist_target_pose"][indices].clone()
            wrist_target[:, :3] += origin
            self.wrists[side].reset(ids, pose[:, :3], pose[:, 3:],
                                    target_position=wrist_target[:, :3],
                                    target_quaternion=wrist_target[:, 3:])
        if self.cfg.no_right_cap_contact_counterfactual:
            # Episode-boundary-only ablation. The right hand remains a normal
            # articulation but begins far enough from the cap to remove its
            # contact path; no object state or per-step pose is altered.
            right_pose = right_pose.clone()
            right_pose[:, :3] = origin + torch.tensor([0.35, 0.0, 0.65], device=self.device)
            self.right.write_root_pose_to_sim_index(root_pose=right_pose, env_ids=ids)
            self.right.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(n, 6, device=self.device), env_ids=ids)
            self.wrists[1].reset(ids, right_pose[:, :3], right_pose[:, 3:])
        if self._mimic_backend:
            self._latest_thread = self.thread.reset(
                ids, bottle_pose_w=bottle_pose,
                bottle_velocity_w=self._bank["bottle_root_velocity"][indices],
                angle_rad=self._bank["thread_angle"][indices])
            self.bottle.update(0.0)
            self.cap.update(0.0)
        else:
            self.thread.reset_from_pose(ids, bottle_pose, cap_pose,
                                        initial_angle=self._bank["thread_angle"][indices])
            snap_mask = self._bank["snapshot_mask"][indices]
            if torch.any(snap_mask):
                snap = {key: self._bank[f"snapshot_{key}"][indices[snap_mask]]
                        for key in ("last_wrapped_rad", "theta_unwrapped_rad",
                                    "initial_theta_rad", "initial_axial_m",
                                    "stick_reference_rad", "sliding", "detached")}
                self.thread.reset_from_pose(ids[snap_mask], bottle_pose[snap_mask], cap_pose[snap_mask],
                                            snapshot=snap)
            # All bank states are engaged; no per-step stage mutation is required.
            for env in ids.detach().cpu().tolist():
                self.thread_joints[env].GetJointEnabledAttr().Set(True)
                self.thread_joints[env].CreateCollisionEnabledAttr(False)
        self.contacts.reset(ids)
        self._theta_at_reset[ids] = self.thread.theta_unwrapped[ids]
        self._bottle_at_reset[ids] = bottle_pose
        self._previous_open[ids] = 0
        self._previous_proximity[ids] = 0
        self._proximity_fresh[ids] = False
        self._stable_steps[ids] = 0
        self._success_steps[ids] = 0
        self._breakaway_seen[ids] = False
        self._maximum_opening[ids] = 0
        self._last_action[ids] = 0
        self._last_raw_action[ids] = 0
        self._action_clip_count[ids] = 0
        self._joint_clip_count[ids] = 0
        self._metrics = None
        for values in self._reward_sums.values():
            values[ids] = 0
        if not self._mimic_backend:
            partial = len(ids) < self.num_envs and self._latest_thread is not None
            if partial:
                peer_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
                peer_mask[ids] = False
                peer_thread = {name: getattr(self.thread, name)[peer_mask].clone()
                               for name in ("last_wrapped", "theta", "initial_theta",
                                            "initial_axial", "stick_reference", "sliding")}
                previous_result = self._latest_thread
            result = self.thread.wrenches(
                self.bottle.data.root_pose_w.torch, self.cap.data.root_pose_w.torch,
                self.bottle.data.root_vel_w.torch, self.cap.data.root_vel_w.torch)
            if partial:
                for name, values in peer_thread.items():
                    getattr(self.thread, name)[peer_mask] = values
                for field in fields(result):
                    value = getattr(result, field.name)
                    prior = getattr(previous_result, field.name)
                    value[peer_mask] = prior[peer_mask]
            self._latest_thread = result
        self._axial_at_reset[ids] = self._latest_thread.axial[ids]

    def _pre_physics_step(self, actions: torch.Tensor):
        # Deterministic RSL-RL playback may hand us an inference-mode tensor.
        # Reset writes into this buffer, so own a normal mutable copy.
        self._last_raw_action = actions.detach().clone()
        clipped = actions.clamp(-1, 1)
        self._action_clip_count += (actions != clipped).any(dim=-1).long()
        self._last_action = clipped
        for side, start in ((0, 0), (1, 26)):
            increments = torch.cat((clipped[:, start:start+3] * self.cfg.wrist_translation_scale,
                                    clipped[:, start+3:start+6] * self.cfg.wrist_rotation_scale), dim=-1)
            self.wrists[side].command(increments)
            proposed = self._finger_targets[side] + (
                clipped[:, start+6:start+26] * self.cfg.finger_increment_scale)
            lower, upper = self._joint_limits[side][:, 0], self._joint_limits[side][:, 1]
            target = torch.maximum(lower, torch.minimum(upper, proposed))
            self._joint_clip_count += (target != proposed).any(dim=-1).long()
            self._finger_targets[side] = target

    def _apply_action(self):
        for side, hand in enumerate((self.left, self.right)):
            pose, velocity = self.wrists[side].step(self.physics_dt)
            hand.write_root_pose_to_sim_index(root_pose=pose, env_ids=self._all_ids)
            hand.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=self._all_ids)
            hand.set_joint_position_target_index(target=self._finger_targets[side], env_ids=self._all_ids)
        if self._mimic_backend:
            self._latest_thread = self.thread.step()
        else:
            result = self.thread.wrenches(
                self.bottle.data.root_pose_w.torch, self.cap.data.root_pose_w.torch,
                self.bottle.data.root_vel_w.torch, self.cap.data.root_vel_w.torch)
            self._latest_thread = result
            for obj, force, torque in ((self.bottle, result.bottle_force, result.bottle_torque),
                                       (self.cap, result.cap_force, result.cap_torque)):
                obj.permanent_wrench_composer.reset()
                obj.permanent_wrench_composer.add_forces_and_torques_index(
                    forces=force[:, None, :], torques=torque[:, None, :], is_global=True)

    def _update_metrics(self):
        if self._mimic_backend:
            self._latest_thread = self.thread.update()
        self.contacts.update(self.physics_dt)
        bp = self.bottle.data.root_pose_w.torch
        cp = self.cap.data.root_pose_w.torch
        bv = self.bottle.data.root_vel_w.torch
        left_forces = self.contacts.force_magnitudes("Bottle", "left")
        right_forces = self.contacts.force_magnitudes("Cap", "right")
        left_score = digit_contact_score(
            left_forces, self.contacts.body_names["left"],
            ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"))
        right_score = digit_contact_score(
            right_forces, self.contacts.body_names["right"],
            ("thumb", "index_finger", "middle_finger"), aggregation="balanced")

        def proximity(hand, tip_ids, obj_pose, radius, half_height):
            tips = hand.data.body_link_pose_w.torch[:, tip_ids, :3]
            q = obj_pose[:, 3:]
            inverse = torch.cat((-q[:, :3], q[:, 3:]), dim=-1)
            local = _rotate(inverse[:, None, :].expand(-1, 5, -1).reshape(-1, 4),
                            (tips - obj_pose[:, None, :3]).reshape(-1, 3)).reshape(self.num_envs, 5, 3)
            radial = torch.linalg.vector_norm(local[:, :, :2], dim=-1)
            radial_score = torch.exp(-torch.abs(radial - radius) / 0.025)
            axial_score = torch.exp(-torch.relu(local[:, :, 2].abs() - half_height) / 0.025)
            return (radial_score * axial_score).mean(dim=-1)

        left_proximity = proximity(self.left, self._tip_ids[0], bp,
                                   self.bottle_cfg.bottle_radius, self.bottle_cfg.bottle_height * 0.35)
        right_proximity = proximity(self.right, self._tip_ids[1], cp,
                                    self.bottle_cfg.cap_outer_radius, self.bottle_cfg.cap_height * 0.5)

        def finger_active(forces, names, digit):
            mask = torch.tensor([digit in name for name in names], device=self.device)
            return (forces[:, mask].amax(dim=-1) > self.contacts.force_floor_n)

        left_names = self.contacts.body_names["left"]
        right_names = self.contacts.body_names["right"]
        left_thumb = finger_active(left_forces, left_names, "thumb")
        left_other = sum(finger_active(left_forces, left_names, finger).long()
                         for finger in ("index_finger", "middle_finger", "ring_finger", "pinky"))
        right_thumb = finger_active(right_forces, right_names, "thumb")
        right_index = finger_active(right_forces, right_names, "index_finger")
        right_middle = finger_active(right_forces, right_names, "middle_finger")
        right_tripod = right_thumb & right_index & right_middle
        left_contact = left_thumb & (left_other >= 2)
        right_contact = right_tripod
        # Both object bodies are free; the hold metric is evaluated in world space.
        displacement = torch.linalg.vector_norm(bp[:, :3] - self._bottle_at_reset[:, :3], dim=-1)
        suspended = suspended_bottle(
            bp[:, :3], bp[:, 3:], self._bottle_at_reset[:, :3],
            self._bottle_at_reset[:, 3:], self.scene.env_origins)
        low_speed = (torch.linalg.vector_norm(bv[:, :3], dim=-1) < 0.2)
        grasp_ok = physically_valid_grasp(
            left_contact, right_contact, suspended, low_speed,
            self._latest_thread.constraint_error)
        # Rotation is credited only while the actual bilateral grasp gate is
        # met. The looser partial-contact gate admitted left-only bottle
        # rotation and right-only cap contact in measured counterfactuals.
        turn_support = grasp_ok
        self._stable_steps = torch.where(grasp_ok, self._stable_steps + 1,
                                         torch.zeros_like(self._stable_steps))
        sign = self.bottle_cfg.thread_sign
        opening, axial_travel, axial_goal = thread_opening_progress(
            self.thread.theta_unwrapped, self._latest_thread.axial,
            self._theta_at_reset, self._axial_at_reset,
            sign=sign, pitch=self.bottle_cfg.thread_pitch)
        cap_ok = cap_turn_physics_valid(
            opening, axial_travel, axial_goal, self._latest_thread.constraint_error,
            grasp_ok, goal_rad=self.cfg.cap_goal_rad,
            pitch_tolerance_m=self.cfg.cap_pitch_tolerance_m,
            constraint_tolerance_m=self.cfg.cap_constraint_tolerance_m)
        self._success_steps = torch.where(cap_ok, self._success_steps + 1,
                                          torch.zeros_like(self._success_steps))
        self._breakaway_seen |= self.thread.sliding
        self._maximum_opening = torch.maximum(self._maximum_opening, opening)
        self._metrics = {"left_contact": left_contact, "right_contact": right_contact,
                         "left_thumb": left_thumb, "left_other_count": left_other,
                         "right_thumb": right_thumb, "right_index": right_index,
                         "right_middle": right_middle,
                         "left_contact_score": left_score, "right_contact_score": right_score,
                         "left_proximity": left_proximity, "right_proximity": right_proximity,
                         "grasp_ok": grasp_ok, "opening": opening,
                         "axial_travel": axial_travel, "axial_goal": axial_goal,
                         "turn_support": turn_support,
                         "suspended": suspended, "cap_ok": cap_ok,
                         "bottle_displacement": displacement,
                         "bottle_velocity": torch.linalg.vector_norm(bv[:, :3], dim=-1)}

    def _get_dones(self):
        self._update_metrics()
        bp = self.bottle.data.root_pose_w.torch
        cp = self.cap.data.root_pose_w.torch
        finite = torch.isfinite(torch.cat((bp, cp), dim=-1)).all(dim=-1)
        failure = (bp[:, 2] < 0.14) | (cp[:, 2] < 0.12) | ~finite
        ground = self.contacts.ground_contact("Bottle") | self.contacts.ground_contact("Cap")
        constraint_failure = self._latest_thread.constraint_error.abs() > 0.02
        failure |= ground | constraint_failure
        if self.cfg.task_kind == "grasp_hold":
            success = self._stable_steps >= 25
        else:
            success = self._success_steps >= 5
        timeout = self.episode_length_buf >= self.max_episode_length
        self.extras["phase1_episode"] = {
            "success": success.clone(), "failure": failure.clone(),
            "episode_steps": self.episode_length_buf.clone(),
            "bottle_z_m": bp[:, 2].clone(), "cap_z_m": cp[:, 2].clone(),
            "left_bottle_contact_ratio": self.contacts.duration_ratio("Bottle", "left").clone(),
            "right_cap_contact_ratio": self.contacts.duration_ratio("Cap", "right").clone(),
            "opening_rad": self._metrics["opening"].clone(),
            "maximum_opening_rad": self._maximum_opening.clone(),
            "axial_travel_m": self._metrics["axial_travel"].clone(),
            "constraint_error_m": self._latest_thread.constraint_error.clone(),
            "breakaway_seen": self._breakaway_seen.clone(),
            "ground_contact": ground.clone(),
            "constraint_failure": constraint_failure.clone(),
            "source_stable": self._source_stable.clone(),
            "left_contact": self._metrics["left_contact"].clone(),
            "right_contact": self._metrics["right_contact"].clone(),
        }
        return failure | success, timeout

    def _get_rewards(self):
        m = self._metrics
        left = m["left_contact"].float()
        right = m["right_contact"].float()
        holding = m["grasp_ok"].float()
        movement = self._last_action.square().mean(dim=-1)
        proximity = torch.stack((m["left_proximity"], m["right_proximity"]), dim=-1)
        approach = (proximity - self._previous_proximity).sum(dim=-1).clamp(-0.2, 0.2)
        approach = torch.where(self._proximity_fresh, approach, torch.zeros_like(approach))
        self._previous_proximity = proximity
        self._proximity_fresh[:] = True
        drift_penalty = (m["bottle_displacement"] - 0.02).clamp_min(0).clamp(max=0.1) * 20
        zero = torch.zeros_like(holding)
        if self.cfg.task_kind == "grasp_hold":
            contact_reward = grasp_contact_reward(m["left_contact_score"],
                                                   m["right_contact_score"])
            progress_reward = zero
            constraint_penalty = thread_error_penalty(self._latest_thread.constraint_error)
            holding_reward = holding
            success_bonus = 5.0 * (self._stable_steps == 25).float()
        else:
            contact_reward = 0.2 * m["left_contact_score"] + 0.2 * m["right_contact_score"]
            progress_reward = physically_supported_turn_reward(
                m["opening"], self._previous_open,
                m["axial_travel"] - m["axial_goal"],
                m["turn_support"] & (m["axial_travel"] > 0),
                axial_tolerance_m=self.cfg.cap_pitch_tolerance_m)
            constraint_penalty = -10.0 * self._latest_thread.constraint_error.abs().clamp(max=0.01)
            holding_reward = 0.5 * holding
            success_bonus = 25.0 * (self._success_steps == 5).float()
        failure_penalty = terminal_failure_penalty(self.extras["phase1_episode"]["failure"])
        terms = {"contact": contact_reward, "approach": 0.3 * approach,
                 "holding": holding_reward, "progress": progress_reward,
                 "action": -0.05 * movement, "drift": -drift_penalty,
                 "constraint": constraint_penalty, "success": success_bonus,
                 "failure": failure_penalty}
        reward = sum(terms.values())
        self._previous_open = m["opening"].clone()
        self.extras["metrics"] = {
            "left_contact_rate": left.mean(), "right_contact_rate": right.mean(),
            "left_thumb_rate": m["left_thumb"].float().mean(),
            "left_other_count_mean": m["left_other_count"].float().mean(),
            "right_thumb_rate": m["right_thumb"].float().mean(),
            "right_index_rate": m["right_index"].float().mean(),
            "right_middle_rate": m["right_middle"].float().mean(),
            "grasp_rate": holding.mean(), "opening_mean_rad": m["opening"].mean(),
            "turn_support_rate": m["turn_support"].float().mean(),
            "opening_max_rad": m["opening"].max(),
            "left_contact_score": m["left_contact_score"].mean(),
            "right_contact_score": m["right_contact_score"].mean(),
            "approach_reward": approach.mean(),
            "progress_reward": progress_reward.mean(),
            "positive_progress_reward": progress_reward.clamp_min(0).mean(),
            "drift_penalty": drift_penalty.mean(),
            "terminal_failure_penalty": failure_penalty.mean(),
            "action_clip_fraction": (self._action_clip_count.float() /
                                      (self.episode_length_buf.float() + 1)).mean(),
            # The counter increments once per *hand* with any clipped joint.
            # Normalize by two hand commands per policy step so this remains a fraction.
            "joint_clip_fraction": (self._joint_clip_count.float() /
                                     (2 * (self.episode_length_buf.float() + 1))).mean(),
            "stable_reset_fraction": self._source_stable.float().mean(),
            "action_element_saturation_fraction": (self._last_raw_action.abs() > 1).float().mean(),
            "left_wrist_action_abs_mean": self._last_raw_action[:, :6].abs().mean(),
            "left_finger_action_abs_mean": self._last_raw_action[:, 6:26].abs().mean(),
            "right_wrist_action_abs_mean": self._last_raw_action[:, 26:32].abs().mean(),
            "right_finger_action_abs_mean": self._last_raw_action[:, 32:].abs().mean(),
        }
        self.extras["log"] = {f"phase1/{key}": value for key, value in self.extras["metrics"].items()}
        for name, value in terms.items():
            self._reward_sums[name] += value
            self.extras["log"][f"reward/{name}_mean"] = value.mean()
            self.extras["log"][f"reward/{name}_sum_mean"] = self._reward_sums[name].mean()
            self.extras["log"][f"reward/{name}_max_abs"] = value.abs().max()
        return reward

    def _get_observations(self):
        bp = self.bottle.data.root_pose_w.torch
        cp = self.cap.data.root_pose_w.torch
        body_vel = torch.cat((self.bottle.data.root_vel_w.torch,
                              self.cap.data.root_vel_w.torch), dim=-1)
        lp = self.left.data.root_pose_w.torch
        rp = self.right.data.root_pose_w.torch
        ltip = self.left.data.body_link_pose_w.torch[:, self._tip_ids[0], :3]
        rtip = self.right.data.body_link_pose_w.torch[:, self._tip_ids[1], :3]
        actor = {}
        for side, (hand, name) in enumerate(((self.left, "left"), (self.right, "right"))):
            q = hand.data.joint_pos.torch
            qd = hand.data.joint_vel.torch
            actor[f"{name}_joint_position"] = q / torch.pi
            actor[f"{name}_joint_velocity"] = qd / 10
            actor[f"{name}_joint_target_error"] = (self._finger_targets[side] - q) / torch.pi
            root = hand.data.root_pose_w.torch
            actor[f"{name}_wrist_pose"] = torch.cat(((root[:, :3] - self.scene.env_origins) / 0.5,
                                                      root[:, 3:]), dim=-1)
            actor[f"{name}_wrist_target_error"] = torch.cat(
                ((self.wrists[side].target_pos - root[:, :3]) / 0.1,
                 _error(self.wrists[side].target_quat, root[:, 3:])), dim=-1)
            actor[f"{name}_wrist_velocity"] = hand.data.root_vel_w.torch / torch.tensor(
                [0.15, 0.15, 0.15, 1., 1., 1.], device=self.device)
        actor["left_fingertips_relative_to_bottle"] = (
            _rotate(_normalize(torch.cat((-bp[:, None, 3:6].expand(-1, 5, -1),
                                          bp[:, None, 6:].expand(-1, 5, -1)), dim=-1).reshape(-1, 4)),
                    (ltip - bp[:, None, :3]).reshape(-1, 3)).reshape(self.num_envs, 15) / 0.1)
        actor["right_fingertips_relative_to_cap"] = (
            _rotate(_normalize(torch.cat((-cp[:, None, 3:6].expand(-1, 5, -1),
                                          cp[:, None, 6:].expand(-1, 5, -1)), dim=-1).reshape(-1, 4)),
                    (rtip - cp[:, None, :3]).reshape(-1, 3)).reshape(self.num_envs, 15) / 0.1)
        actor["bottle_relative_to_left_wrist"] = _relative_pose(lp, bp)
        actor["cap_relative_to_right_wrist"] = _relative_pose(rp, cp)
        actor["cap_relative_to_bottle"] = _relative_pose(bp, cp)
        actor["bottle_and_cap_velocity"] = body_vel / 5
        opening = self.bottle_cfg.thread_sign * (self.thread.theta_unwrapped - self._theta_at_reset)
        actor["thread_progress_and_goal"] = torch.stack(
            (opening / torch.pi, self._latest_thread.axial / 0.2,
             (self.cfg.cap_goal_rad - opening) / torch.pi,
             self._latest_thread.constraint_error / 0.01), dim=-1)
        actor["previous_action"] = self._last_action
        for key, width in ACTOR_FIELDS.items():
            if actor[key].shape != (self.num_envs, width):
                raise RuntimeError(f"Actor observation field {key} has wrong shape")
        policy = torch.cat([actor[key] for key in ACTOR_FIELDS], dim=-1)
        critic = {}
        for key, obj, side in (("left_bottle_forces", "Bottle", "left"),
                               ("right_cap_forces", "Cap", "right"),
                               ("left_cap_forces", "Cap", "left"),
                               ("right_bottle_forces", "Bottle", "right")):
            critic[key] = self.contacts.force_magnitudes(obj, side).clamp(max=100) / 10
        critic["finger_applied_torque"] = torch.cat(
            (self.left.data.applied_torque.torch, self.right.data.applied_torque.torch), dim=-1) / 2
        critic["thread_internal_state"] = torch.stack(
            (self.thread.sliding.float(), self.thread.stick_reference / torch.pi,
             self._latest_thread.breakaway_limit / 0.02,
             self._latest_thread.running_limit / 0.006), dim=-1)
        critic["object_mass"] = torch.tensor(
            [self.bottle_cfg.bottle_mass, self.bottle_cfg.cap_mass],
            device=self.device).expand(self.num_envs, 2)
        def slip(hand, tip_ids, obj, side):
            forces = self.contacts.force_magnitudes(obj, side)
            names = self.contacts.body_names[side]
            weights = torch.stack([
                forces[:, [i for i, body in enumerate(names) if digit in body]].sum(-1)
                for digit in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
            ], dim=-1)
            object_asset = self.bottle if obj == "Bottle" else self.cap
            return surface_slip_speed(
                hand.data.body_link_pose_w.torch[:, tip_ids, :3],
                hand.data.body_link_vel_w.torch[:, tip_ids, :3],
                object_asset.data.root_pose_w.torch,
                object_asset.data.root_vel_w.torch, weights)

        critic["relative_slip"] = torch.stack(
            (slip(self.left, self._tip_ids[0], "Bottle", "left"),
             slip(self.right, self._tip_ids[1], "Cap", "right")), dim=-1)
        critic["full_body_velocity"] = body_vel
        for key, width in CRITIC_FIELDS.items():
            if critic[key].shape != (self.num_envs, width):
                raise RuntimeError(f"Critic observation field {key} has wrong shape")
        privileged = torch.cat((policy, *[critic[key] for key in CRITIC_FIELDS]), dim=-1)
        return {"policy": torch.nan_to_num(policy), "critic": torch.nan_to_num(privileged)}
