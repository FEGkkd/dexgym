"""Standalone gravity-on dual-hand fixture shared by Phase 0 GPU checks.

Construct only after AppLauncher has started Isaac Sim. The fixture uses
normal finger position drives and bounded floating-wrist wrenches. Bottle
and cap receive only equal/opposite thread coupling; no hold fixture exists.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

from dexteroushands_lab.assets.bottle import BottleConfig, ThreadCoupler
from dexteroushands_lab.assets.bottle.generate_usd import build_bottle_usd, author_thread_joint
from dexteroushands_lab.assets.wrist_6dof import Wrist6DoFController
from dexteroushands_lab.assets.wuji_hand2_beta import load_hand2_spec
from dexteroushands_lab.scenes.bimanual_bottle import make_bimanual_bottle_scene_cfg
from dexteroushands_lab.scenes.contact_monitor import BimanualContactMonitor, ContactMonitorConfig
from dexteroushands_lab.scenes.grasp_init import BimanualGraspInitializer
from dexteroushands_lab.scenes.reset_manager import BimanualResetManager


class BimanualPhase0Fixture:
    def __init__(self, *, num_envs: int, device: str, config: BottleConfig | None = None,
                 variant: str = "beta1", contact_debug: bool = False,
                 include_hand_hand: bool = True):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab_physx.physics import PhysxManager
        from pxr import UsdPhysics

        self.config = config or BottleConfig()
        self.num_envs = num_envs
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
        ))
        self.joints = [author_thread_joint(
            self.sim.stage, f"/World/envs/env_{env}", self.config)
            for env in range(num_envs)]
        for side in ("LeftHand", "RightHand"):
            path = f"/World/envs/env_0/{side}/root_joint"
            joint = UsdPhysics.Joint(self.sim.stage.GetPrimAtPath(path))
            if not joint or joint.GetJointEnabledAttr().Get() is not False:
                raise RuntimeError(f"Hand root must float: {path}")
        self.sim.reset()
        self.assets = tuple(self.scene[name] for name in
                            ("left_hand", "right_hand", "bottle_body", "bottle_cap"))
        self.left, self.right, self.bottle, self.cap = self.assets
        for hand, spec in zip((self.left, self.right), self.specs):
            if set(hand.joint_names) != set(spec.joint_names):
                raise RuntimeError(f"Pinned {spec.side} joint order mismatch")
        self.controllers = tuple(
            [Wrist6DoFController(tuple(p[:3]), tuple(p[3:7]))
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
            self.couplers, self.joints, self.monitor,
        )
        self.root_body_ids = (
            self.left.body_names.index(self.specs[0].root_body),
            self.right.body_names.index(self.specs[1].root_body),
        )

    def step(self):
        """Advance exactly one physical step without any root-pose writes."""
        import torch

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
                wrenches[side].append(self.controllers[side][env].wrench(
                    pose[:3], pose[3:7], vel[:3], vel[3:6],
                ))
            bp, cp = poses[2][env], poses[3][env]
            bv, cv = velocities[2][env], velocities[3][env]
            coupled = self.couplers[env].wrenches(
                bp[:3], bp[3:7], bv[:3], bv[3:6],
                cp[:3], cp[3:7], cv[:3], cv[3:6],
            )
            if coupled.state.detached:
                self.joints[env].GetJointEnabledAttr().Set(False)
                self.joints[env].CreateCollisionEnabledAttr(True)
            wrenches[2].append((coupled.bottle_force, coupled.bottle_torque))
            wrenches[3].append((coupled.cap_force, coupled.cap_torque))
            thread_states.append(coupled.state)
        for index, (asset, pairs) in enumerate(zip(self.assets, wrenches)):
            asset.permanent_wrench_composer.reset()
            kwargs = {}
            if index < 2:
                kwargs["body_ids"] = torch.tensor([self.root_body_ids[index]],
                                                   device=asset.device, dtype=torch.long)
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
