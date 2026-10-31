"""Selective reset contract without a physics step or global state writes."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from dexteroushands_lab.assets.bottle import BottleConfig, ThreadCoupler
from dexteroushands_lab.assets.wrist_6dof import Wrist6DoFController
from dexteroushands_lab.scenes.grasp_init import BimanualGraspInitializer
from dexteroushands_lab.scenes.reset_manager import BimanualResetManager


def _spec(side):
    names = tuple(f"{side}_{i}" for i in range(20))
    return SimpleNamespace(
        side=side, variant="beta1", source_commit="commit", usd_sha256=f"{side}-usd",
        asset_tree_sha256=f"{side}-tree", joint_names=names,
        joints=tuple(SimpleNamespace(name=name, lower=-1.0, upper=1.0) for name in names),
    )


class _Asset:
    def __init__(self, with_joints=False):
        self.device = "cpu"
        self.pose = torch.tensor([[9., 9., 9., 0, 0, 0, 1]] * 2)
        self.velocity = torch.ones(2, 6)
        self.joint_names = [f"left_{i}" for i in range(20)] if with_joints else []
        self.joint_position = torch.ones(2, 20)
        self.joint_velocity = torch.ones(2, 20)
        self.target = torch.ones(2, 20)
        self.reset_ids = []

    def reset(self, env_ids):
        self.reset_ids.append(tuple(env_ids.tolist()))

    def write_root_pose_to_sim_index(self, *, root_pose, env_ids):
        self.pose[env_ids] = root_pose

    def write_root_velocity_to_sim_index(self, *, root_velocity, env_ids):
        self.velocity[env_ids] = root_velocity

    def write_joint_position_to_sim_index(self, *, position, env_ids):
        self.joint_position[env_ids] = position

    def write_joint_velocity_to_sim_index(self, *, velocity, env_ids):
        self.joint_velocity[env_ids] = velocity

    def set_joint_position_target_index(self, *, target, env_ids):
        self.target[env_ids] = target


class _Attr:
    def __init__(self, value):
        self.value = value

    def Set(self, value):
        self.value = value

    def Get(self):
        return self.value


class _Joint:
    def __init__(self):
        self.enabled = _Attr(False)
        self.collision = _Attr(True)

    def GetJointEnabledAttr(self):
        return self.enabled

    def CreateCollisionEnabledAttr(self, value=None):
        if value is not None:
            self.collision.Set(value)
        return self.collision


class _Monitor:
    def __init__(self):
        self.reset_ids = []

    def reset(self, env_ids):
        self.reset_ids.append(tuple(env_ids))


class _Scene(dict):
    def __init__(self, assets):
        super().__init__(assets)
        self.env_origins = torch.tensor([[0., 0., 0.], [1.5, 0., 0.]])


def test_reset_only_env_zero_restores_detached_joint_targets_and_thread():
    cfg = BottleConfig()
    specs = (_spec("left"), _spec("right"))
    initializer = BimanualGraspInitializer(cfg, *specs)
    sample = initializer.generate(mode="near_contact", seed=4)
    left, right, bottle, cap = (_Asset(True), _Asset(True), _Asset(), _Asset())
    right.joint_names = list(specs[1].joint_names)
    scene = _Scene({"left_hand": left, "right_hand": right,
                    "bottle_body": bottle, "bottle_cap": cap})
    couplers = [ThreadCoupler(cfg), ThreadCoupler(cfg)]
    couplers[0]._detached = True
    couplers[0]._sliding = True
    joints = [_Joint(), _Joint()]
    monitor = _Monitor()
    left_ctrl = [Wrist6DoFController((0, 0, 0.5), (0, 0, 0, 1)) for _ in range(2)]
    right_ctrl = [Wrist6DoFController((0, 0, 0.5), (0, 0, 0, 1)) for _ in range(2)]
    manager = BimanualResetManager(scene, specs, left_ctrl, right_ctrl, couplers, joints, monitor)
    bad_cap = replace(sample.cap, root_pose=(*sample.cap.root_pose[:3], 0, 0, 1, 0))
    with pytest.raises(ValueError, match="thread angle"):
        manager.reset((0,), (replace(sample, cap=bad_cap),))
    assert torch.all(bottle.pose[:, :3] == 9)
    manager.reset((0,), (sample,))
    assert torch.allclose(bottle.pose[0], torch.tensor(sample.bottle.root_pose))
    assert torch.all(bottle.pose[1, :3] == 9)
    assert torch.all(left.joint_position[1] == 1)
    assert torch.allclose(left.target[0], left.joint_position[0])
    assert not couplers[0].detached and not couplers[0]._sliding
    assert joints[0].enabled.Get() and not joints[0].collision.Get()
    assert not joints[1].enabled.Get() and joints[1].collision.Get()
    assert monitor.reset_ids == [(0,)]
    assert left_ctrl[0].target_pos == sample.left.root_pose[:3]
