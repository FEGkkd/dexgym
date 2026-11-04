"""Episode-boundary, selective reset of the standalone bimanual PhysX scene.

No physics step or per-step teleport occurs here. Isaac Lab 3.0's indexed
state/target APIs write only the selected environment rows.
Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/how-to/write_articulation_cfg.html
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
import math

from dexteroushands_lab.assets.bottle import ThreadCoupler
from dexteroushands_lab.scenes.grasp_init import GraspInitSample
from dexteroushands_lab.scenes.pregrasp_reset import (
    has_pregrasp_geometry_evidence, validate_pregrasp_state)


class BimanualResetManager:
    """Restore physical and episode-local state in selected environments."""

    def __init__(self, scene, specs, left_controllers, right_controllers,
                 thread_couplers, thread_joints, contact_monitor,
                 control_mode: str = "floating_wrench_pd"):
        self.scene = scene
        self.specs = specs
        self.controllers = (left_controllers, right_controllers)
        self.couplers = thread_couplers
        self.joints = thread_joints
        self.contact_monitor = contact_monitor
        self.num_envs = len(thread_couplers)
        self.control_mode = control_mode
        if not (len(left_controllers) == len(right_controllers) == len(thread_joints)
                == self.num_envs):
            raise ValueError("Reset manager requires one controller/coupler/joint per environment")
        self.init_valid = [False] * self.num_envs
        # Distinct from init_valid, which means a gravity-on hold was validated.
        self.pregrasp_geometry_qualified = [False] * self.num_envs
        self.sample_seed = [None] * self.num_envs
        self.fixed_anchors = (
            (scene["left_hand"].data.root_pose_w.torch.detach().cpu().tolist(),
             scene["right_hand"].data.root_pose_w.torch.detach().cpu().tolist())
            if control_mode == "fixed_wrist" else None
        )

    def reset(self, env_ids: Sequence[int], init_samples: Sequence[GraspInitSample]) -> None:
        import torch

        ids = tuple(int(env) for env in env_ids)
        samples = tuple(init_samples)
        if len(ids) != len(samples) or len(set(ids)) != len(ids):
            raise ValueError("Provide exactly one init sample for each distinct environment")
        if any(env < 0 or env >= self.num_envs for env in ids):
            raise IndexError("Reset environment index out of range")
        for env, sample in zip(ids, samples):
            if sample.mode == "geometry_pregrasp":
                validate_pregrasp_state(sample, self.couplers[0].config,
                                        *self.specs)
            sample_mode = sample.metadata.get("wrist_control_mode")
            if sample_mode is not None and sample_mode != self.control_mode:
                raise ValueError("Init sample wrist control mode differs from fixture")
            if self.control_mode == "fixed_wrist":
                origin = self.scene.env_origins[env].detach().cpu().tolist()
                for side, hand in enumerate((sample.left, sample.right)):
                    pose = hand.root_pose
                    anchored = self.fixed_anchors[side][env]
                    position = [pose[i] + origin[i] for i in range(3)]
                    if (math.dist(position, anchored[:3]) > 2e-4
                            or abs(sum(a*b for a, b in zip(pose[3:7],
                                                             anchored[3:7]))) < 0.999999):
                        raise ValueError("Fixed-wrist sample differs from the authored root anchor")
            if (sample.metadata.get("asset_variant") != self.specs[0].variant
                    or sample.metadata.get("left_joint_order") != list(self.specs[0].joint_names)
                    or sample.metadata.get("right_joint_order") != list(self.specs[1].joint_names)
                    or sample.metadata.get("left_usd_sha256") != self.specs[0].usd_sha256
                    or sample.metadata.get("right_usd_sha256") != self.specs[1].usd_sha256):
                raise ValueError("Init sample asset identity or canonical joint order mismatch")
            if sample.metadata.get("bottle_parameters") != asdict(self.couplers[0].config):
                raise ValueError("Init sample bottle profile or thread engagement mismatch")
            for state in (sample.left, sample.right, sample.bottle, sample.cap):
                if (len(state.root_pose) != 7 or len(state.root_velocity) != 6
                        or not all(math.isfinite(x) for x in (*state.root_pose,
                                                                *state.root_velocity))):
                    raise ValueError("Init sample has invalid root state")
                if abs(math.sqrt(sum(x * x for x in state.root_pose[3:7])) - 1.0) > 1e-3:
                    raise ValueError("Init sample root quaternion is not unit length")
            for hand in (sample.left, sample.right):
                if (any(len(values) != 20 for values in (hand.joint_position,
                                                        hand.joint_velocity,
                                                        hand.joint_target))
                        or len(hand.wrist_target_pose) != 7
                        or not all(math.isfinite(x) for values in (
                            hand.joint_position, hand.joint_velocity,
                            hand.joint_target, hand.wrist_target_pose) for x in values)):
                    raise ValueError("Init sample has invalid finger or wrist target state")
                if (abs(math.sqrt(sum(x * x for x in hand.wrist_target_pose[3:7])) - 1.0)
                        > 1e-3):
                    raise ValueError("Init sample wrist target quaternion is not unit length")
            probe = ThreadCoupler(self.couplers[0].config)
            snapshot = sample.metadata.get("thread_coupler_state")
            restore_kwargs = (sample.bottle.root_pose[:3], sample.bottle.root_pose[3:7],
                              sample.cap.root_pose[:3], sample.cap.root_pose[3:7])
            thread = (probe.restore_state(snapshot, *restore_kwargs) if snapshot is not None
                      else probe.reset_from_pose(
                          *restore_kwargs,
                          initial_angle_unwrapped=sample.thread_initial_angle_rad))
            expected_engagement = (self.couplers[0].config.initial_engagement -
                                   (thread.axial - self.couplers[0].config.cap_initial_z))
            if (not math.isfinite(sample.thread_initial_engagement_m)
                    or not 0 <= expected_engagement <= self.couplers[0].config.cap_height
                    or not math.isclose(sample.thread_initial_engagement_m,
                                        expected_engagement, abs_tol=1e-4)
                    or abs(thread.theta_unwrapped - sample.thread_initial_angle_rad) > 1e-4):
                raise ValueError("Init sample cap engagement or thread angle mismatch")
        if not ids:
            return
        env_tensor = torch.tensor(ids, device=self.scene["bottle_body"].device, dtype=torch.long)

        def root_state(asset, states):
            positions = []
            velocities = []
            for env, state in zip(ids, states):
                origin = self.scene.env_origins[env].detach().cpu().tolist()
                pose = state.root_pose
                positions.append((*tuple(pose[j] + origin[j] for j in range(3)), *pose[3:7]))
                velocities.append(state.root_velocity)
            asset.reset(env_ids=env_tensor)
            asset.write_root_pose_to_sim_index(
                root_pose=torch.tensor(positions, device=asset.device, dtype=torch.float32),
                env_ids=env_tensor,
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=torch.tensor(velocities, device=asset.device, dtype=torch.float32),
                env_ids=env_tensor,
            )
            return positions

        left_states = [sample.left for sample in samples]
        right_states = [sample.right for sample in samples]
        bottle_states = [sample.bottle for sample in samples]
        cap_states = [sample.cap for sample in samples]
        left_pose = root_state(self.scene["left_hand"], left_states)
        right_pose = root_state(self.scene["right_hand"], right_states)
        bottle_pose = root_state(self.scene["bottle_body"], bottle_states)
        cap_pose = root_state(self.scene["bottle_cap"], cap_states)

        for side, states in ((0, left_states), (1, right_states)):
            asset = self.scene["left_hand" if side == 0 else "right_hand"]
            canonical = self.specs[side].joint_names
            if set(asset.joint_names) != set(canonical) or len(asset.joint_names) != 20:
                raise RuntimeError(f"Isaac joint names differ from the pinned {self.specs[side].side} asset")
            reorder = [canonical.index(name) for name in asset.joint_names]
            for field, writer, key in (
                ("joint_position", asset.write_joint_position_to_sim_index, "position"),
                ("joint_velocity", asset.write_joint_velocity_to_sim_index, "velocity"),
                ("joint_target", asset.set_joint_position_target_index, "target"),
            ):
                values = [[getattr(state, field)[index] for index in reorder] for state in states]
                writer(**{key: torch.tensor(values, device=asset.device, dtype=torch.float32),
                          "env_ids": env_tensor})

        for index, env in enumerate(ids):
            for side, pose, state in ((0, left_pose[index], left_states[index]),
                                      (1, right_pose[index], right_states[index])):
                target = state.wrist_target_pose
                origin = self.scene.env_origins[env].detach().cpu().tolist()
                target_position = tuple(target[j] + origin[j] for j in range(3))
                self.controllers[side][env].reset_to(
                    pose[:3], pose[3:7], target_pos=target_position,
                    target_quat=target[3:7],
                )
            restore_kwargs = (bottle_pose[index][:3], bottle_pose[index][3:7],
                              cap_pose[index][:3], cap_pose[index][3:7])
            snapshot = samples[index].metadata.get("thread_coupler_state")
            if snapshot is not None:
                self.couplers[env].restore_state(snapshot, *restore_kwargs)
            else:
                self.couplers[env].reset_from_pose(
                    *restore_kwargs,
                    initial_angle_unwrapped=samples[index].thread_initial_angle_rad)
            self.joints[env].GetJointEnabledAttr().Set(True)
            self.joints[env].CreateCollisionEnabledAttr(False)
            self.init_valid[env] = samples[index].validated
            self.pregrasp_geometry_qualified[env] = (
                samples[index].mode == "geometry_pregrasp" and
                samples[index].metadata.get("geometry_qualified") is True and
                has_pregrasp_geometry_evidence(samples[index]))
            self.sample_seed[env] = samples[index].seed
        self.contact_monitor.reset(ids)
