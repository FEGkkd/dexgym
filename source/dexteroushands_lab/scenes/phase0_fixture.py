"""Standalone gravity-on dual-hand fixture shared by Phase 0 GPU checks.

Construct only after AppLauncher has started Isaac Sim. The fixture uses
normal finger position drives. Wrist mode is selected explicitly: the older
floating wrench controller, a fixed root, or a bounded trajectory of a
world-anchored root. Bottle and cap remain dynamic.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from dexteroushands_lab.assets.bottle import BottleConfig, ThreadCoupler
from dexteroushands_lab.assets.bottle.generate_usd import build_bottle_usd, author_thread_joint
from dexteroushands_lab.assets.wrist_6dof import Wrist6DoFController
from dexteroushands_lab.assets.wrist_pose_control import (WristPoseController,
                                                          WristPoseLimits,
                                                          split_bimanual_action_52)
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec
from dexteroushands_lab.scenes.bimanual_bottle import make_bimanual_bottle_scene_cfg
from dexteroushands_lab.scenes.contact_monitor import BimanualContactMonitor, ContactMonitorConfig
from dexteroushands_lab.scenes.grasp_init import BimanualGraspInitializer
from dexteroushands_lab.scenes.reset_manager import BimanualResetManager


class BimanualPhase0Fixture:
    def __init__(self, *, num_envs: int, device: str, config: BottleConfig | None = None,
                 variant: str = "beta1", contact_debug: bool = False,
                 include_hand_hand: bool = True, wrist_limits=None,
                 temporary_bottle_support: bool = False,
                 control_mode: str = "floating_wrench_pd",
                 left_initial_pose=None, right_initial_pose=None,
                 hand_self_collisions: bool = True):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab_physx.physics import PhysxManager
        from pxr import Gf, Sdf, UsdPhysics

        self.config = config or BottleConfig()
        self.num_envs = num_envs
        self.control_mode = control_mode
        if control_mode == "pose_command_wrist" and (
            wrist_limits is not None and not isinstance(wrist_limits, WristPoseLimits)
        ):
            raise TypeError("pose_command_wrist requires WristPoseLimits, not wrench PD limits")
        self.dt = 0.0001 if self.config.source_profile == "pco1881_reference" else 0.002
        self.specs = tuple(load_hand2_spec(variant=variant, side=side)
                           for side in ("left", "right"))
        # Runtime USD derivatives live outside the source tree. This fixture
        # must not rewrite a user's checked-in or locally edited bottle model.
        asset_dir = Path(tempfile.mkdtemp(prefix="dexgym-phase0-bottle-"))
        self.asset_dir = asset_dir
        build_bottle_usd(self.config, asset_dir)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=self.dt, device=device))
        self.scene = InteractiveScene(make_bimanual_bottle_scene_cfg(
            num_envs=num_envs, asset_dir=asset_dir, bottle=self.config,
            variant=variant, contact_debug=contact_debug, bottle_center_z=0.5,
            wrist_control_mode=control_mode,
            left_initial_pose=left_initial_pose, right_initial_pose=right_initial_pose,
            hand_self_collisions=hand_self_collisions,
        ))
        self.joints = [author_thread_joint(
            self.sim.stage, f"/World/envs/env_{env}", self.config)
            for env in range(num_envs)]
        self.temporary_support_joint = None
        if temporary_bottle_support:
            if num_envs != 1:
                raise ValueError("Temporary grasp support is only implemented for env 0")
            support = UsdPhysics.FixedJoint.Define(
                self.sim.stage, "/World/envs/env_0/TemporaryGraspSupport")
            support.CreateBody1Rel().SetTargets([Sdf.Path("/World/envs/env_0/Bottle")])
            support.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.5))
            support.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            support.CreateJointEnabledAttr(True)
            self.temporary_support_joint = support
        for env in range(num_envs):
            origin = self.scene.env_origins[env].detach().cpu().tolist()
            for side, pose in (("LeftHand", left_initial_pose or
                                (-0.30, 0.0, 0.62, 0.0, -0.7071067811865475,
                                 0.0, 0.7071067811865476)),
                               ("RightHand", right_initial_pose or
                                (0.30, 0.0, 0.62, 0.0, 0.7071067811865475,
                                 0.0, 0.7071067811865476))):
                path = f"/World/envs/env_{env}/{side}/root_joint"
                joint = UsdPhysics.Joint(self.sim.stage.GetPrimAtPath(path))
                enabled = joint.GetJointEnabledAttr().Get() if joint else None
                anchored = control_mode in ("fixed_wrist", "pose_command_wrist")
                if enabled is not anchored:
                    raise RuntimeError(f"Unexpected hand root joint state: {path}={enabled}")
                if anchored:
                    joint.CreateLocalPos0Attr(Gf.Vec3f(*(pose[i] + origin[i]
                                                       for i in range(3))))
                    joint.CreateLocalRot0Attr(Gf.Quatf(pose[6], *pose[3:6]))
        self.sim.reset()
        self.assets = tuple(self.scene[name] for name in
                            ("left_hand", "right_hand", "bottle_body", "bottle_cap"))
        self.left, self.right, self.bottle, self.cap = self.assets
        for hand, spec in zip((self.left, self.right), self.specs):
            if set(hand.joint_names) != set(spec.joint_names):
                raise RuntimeError(f"Pinned {spec.side} joint order mismatch")
        controller_type = (WristPoseController if control_mode == "pose_command_wrist"
                           else Wrist6DoFController)
        self.controllers = tuple(
            [controller_type(tuple(p[:3]), tuple(p[3:7]),
                             **({"limits": wrist_limits} if wrist_limits is not None
                                else {}))
             for p in hand.data.root_pose_w.torch.detach().cpu().tolist()]
            for hand in (self.left, self.right)
        )
        self.couplers = [ThreadCoupler(self.config) for _ in range(num_envs)]
        self.monitor = BimanualContactMonitor(
            PhysxManager.get_physics_sim_view(), num_envs, *self.specs,
            config=ContactMonitorConfig(include_hand_hand=include_hand_hand),
        )
        self.initializer = BimanualGraspInitializer(self.config, *self.specs)
        self.reset_manager = BimanualResetManager(
            self.scene, self.specs, self.controllers[0], self.controllers[1],
            self.couplers, self.joints, self.monitor, control_mode=control_mode,
        )
        self.root_body_ids = (
            self.left.body_names.index(self.specs[0].root_body),
            self.right.body_names.index(self.specs[1].root_body),
        )

    def command_action52(self, env_ids, action_rows):
        """Apply per-env 6D wrist increments and 20+20 physical finger targets.

        Fixed-wrist mode accepts finger targets only and requires all wrist
        increments to be zero. The legacy wrench mode remains an experiment.
        """
        import math
        import torch

        ids = tuple(int(env) for env in env_ids)
        rows = tuple(split_bimanual_action_52(row) for row in action_rows)
        if (len(ids) != len(rows) or len(set(ids)) != len(ids)
                or any(env < 0 or env >= self.num_envs for env in ids)):
            raise ValueError("Command rows must match distinct valid environment IDs")
        if self.control_mode not in ("fixed_wrist", "pose_command_wrist"):
            raise ValueError("52D ideal-wrist command is unavailable in wrench PD mode")
        for env, sides in zip(ids, rows):
            for side, (translation, rotation, fingers) in enumerate(sides):
                if self.control_mode == "fixed_wrist" and any(
                    abs(value) > 0 for value in (*translation, *rotation)
                ):
                    raise ValueError("Fixed wrist cannot accept pose increments")
                for target, joint in zip(fingers, self.specs[side].joints):
                    if not joint.lower <= target <= joint.upper:
                        raise ValueError(f"Finger target exceeds official limit: {joint.name}")
                if self.control_mode == "pose_command_wrist":
                    controller = self.controllers[side][env]
                    proposed = tuple(a+b for a,b in zip(controller.target_pos,
                                                        translation))
                    if (proposed[2] < controller.limits.min_height
                            or math.dist(proposed, controller.initial_pos)
                            > controller.limits.max_target_radius):
                        raise ValueError("Pose increment exceeds wrist workspace")
        for env, sides in zip(ids, rows):
            for side, (translation, rotation, _) in enumerate(sides):
                if self.control_mode == "pose_command_wrist":
                    self.controllers[side][env].command_relative(translation, rotation)
        for side, hand in enumerate((self.left, self.right)):
            reorder = [self.specs[side].joint_names.index(name)
                       for name in hand.joint_names]
            values = [[row[side][2][index] for index in reorder] for row in rows]
            if values:
                hand.set_joint_position_target_index(
                    target=torch.tensor(values, device=hand.device, dtype=torch.float32),
                    env_ids=torch.tensor(ids, device=hand.device, dtype=torch.long))

    def step(self, *, bottle_support_wrenches=None):
        """Advance one step; optional bounded bottle wrench is synthesis-only.

        ``bottle_support_wrenches`` maps environment IDs to world-frame
        ``(force, torque)`` pairs. The normal hold fixture passes none.
        """
        import torch

        if self.control_mode == "pose_command_wrist":
            # The official world fixed root_joint remains enabled. PhysX's
            # indexed root write updates that anchor, so contact reactions do
            # not integrate into wrist drift. Each write advances a bounded
            # SE(3) substep, and hand finger links remain dynamic.
            # Source: https://isaac-sim.github.io/IsaacLab/v3.0.0/source/api/lab_physx/isaaclab_physx.assets.html
            env_ids = torch.arange(self.num_envs, device=self.left.device, dtype=torch.long)
            for side, hand in enumerate((self.left, self.right)):
                commands = [controller.step(self.dt) for controller in self.controllers[side]]
                hand.write_root_pose_to_sim_index(
                    root_pose=torch.tensor([pose for pose, _ in commands],
                                           device=hand.device, dtype=torch.float32),
                    env_ids=env_ids,
                )
                hand.write_root_velocity_to_sim_index(
                    root_velocity=torch.tensor([velocity for _, velocity in commands],
                                               device=hand.device, dtype=torch.float32),
                    env_ids=env_ids,
                )

        def rows(asset, field):
            return getattr(asset.data, field).torch.detach().cpu().tolist()

        poses = [rows(asset, "root_pose_w") for asset in self.assets]
        velocities = [rows(asset, "root_vel_w") for asset in self.assets]
        wrenches = [[] for _ in self.assets]
        thread_states = []
        for env in range(self.num_envs):
            for side in (0, 1):
                pose = poses[side][env]
                vel = velocities[side][env]
                wrenches[side].append(
                    self.controllers[side][env].wrench(
                        pose[:3], pose[3:7], vel[:3], vel[3:6])
                    if self.control_mode == "floating_wrench_pd"
                    else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            bp, cp = poses[2][env], poses[3][env]
            bv, cv = velocities[2][env], velocities[3][env]
            coupled = self.couplers[env].wrenches(
                bp[:3], bp[3:7], bv[:3], bv[3:6],
                cp[:3], cp[3:7], cv[:3], cv[3:6],
            )
            if coupled.state.detached:
                self.joints[env].GetJointEnabledAttr().Set(False)
                self.joints[env].CreateCollisionEnabledAttr(True)
            support = (bottle_support_wrenches or {}).get(env, ((0.0,) * 3, (0.0,) * 3))
            wrenches[2].append((
                tuple(a + b for a, b in zip(coupled.bottle_force, support[0])),
                tuple(a + b for a, b in zip(coupled.bottle_torque, support[1])),
            ))
            wrenches[3].append((coupled.cap_force, coupled.cap_torque))
            thread_states.append(coupled.state)
        for index, (asset, pairs) in enumerate(zip(self.assets, wrenches)):
            asset.permanent_wrench_composer.reset()
            if index < 2 and self.control_mode != "floating_wrench_pd":
                continue
            kwargs = {}
            if index < 2 and self.control_mode == "floating_wrench_pd":
                kwargs["body_ids"] = torch.tensor([self.root_body_ids[index]],
                                                   device=asset.device, dtype=torch.long)
                # A translation wrench applied at the wrist link COM creates
                # an unintended moment about the whole free hand's COM. Place
                # the force at the mass-weighted articulation COM; the
                # composer adds the equivalent root-link moment in PhysX.
                masses = asset.data.body_mass.torch
                body_com = asset.data.body_com_pos_w.torch
                articulation_com = (
                    (body_com * masses[..., None]).sum(dim=1) /
                    masses.sum(dim=1)[:, None]
                )
                kwargs["positions"] = articulation_com[:, None, :]
            asset.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.tensor([pair[0] for pair in pairs], device=asset.device,
                                    dtype=torch.float32)[:, None, :],
                torques=torch.tensor([pair[1] for pair in pairs], device=asset.device,
                                     dtype=torch.float32)[:, None, :],
                is_global=True, **kwargs,
            )
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.dt)
        self.monitor.update_after_physics_step(self.dt)
        for asset in self.assets:
            if (not torch.isfinite(asset.data.root_pose_w.torch).all().item()
                    or not torch.isfinite(asset.data.root_vel_w.torch).all().item()):
                raise RuntimeError("Nonfinite Phase 0 rigid/articulation state")
        return thread_states
