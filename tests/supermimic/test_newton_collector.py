"""Real Newton transitions must keep executed actions and pre-reset next states."""

from types import SimpleNamespace

import torch
from dexgym.algorithms.supermimic.replay_buffer import MotionReplay
from dexgym.tasks.marvin_supermimic.collector import AlignedNewtonCollector


class ZeroActor:
    def __call__(self, history, goal):
        return torch.zeros(len(history), 14)


class ResettingEnv:
    def __init__(self):
        self.resets = 0
        self.steps = 0
        q = torch.zeros(2, 54)
        qd = torch.zeros_like(q)
        self.unwrapped = SimpleNamespace(
            num_envs=2,
            device="cpu",
            arm_ids=torch.arange(14),
            robot=SimpleNamespace(
                data=SimpleNamespace(joint_pos=SimpleNamespace(torch=q), joint_vel=SimpleNamespace(torch=qd))
            ),
            command=torch.zeros(2, 14),
            motion=SimpleNamespace(frame=torch.tensor([2, 3]), length=10, data={"q_arm": torch.zeros(10, 14)}),
            action_map=SimpleNamespace(origin=torch.zeros(14), scale=torch.ones(14)),
            last_physical_arm_state=torch.zeros(2, 42),
            last_policy_contact_counts=torch.zeros(2, dtype=torch.int32),
        )

    def reset(self):
        self.resets += 1
        return {"policy": torch.zeros(2, 1460)}, {}

    def step(self, action):
        self.steps += 1
        self.unwrapped.last_physical_arm_state = torch.full((2, 42), 77.0)
        self.unwrapped.robot.data.joint_pos.torch.zero_()  # automatic post-step reset in env 0
        return (
            {"policy": torch.zeros(2, 1460)},
            torch.ones(2),
            torch.tensor([True, False]),
            torch.tensor([False, False]),
            {},
        )


def test_stateful_collector_keeps_prereset_physics_and_hard_termination_labels():
    env = ResettingEnv()
    replay = MotionReplay(40, 2, "cpu")
    collector = AlignedNewtonCollector(env, ZeroActor(), replay, action_noise_std=0.0)
    collector.step()
    collector.step()
    assert env.resets == 1
    assert env.steps == 2
    assert replay.steps == 2
    assert replay.terminated[:2, 0].all()
    assert not replay.terminated[:2, 1].any()
    assert replay.next_state[:2, :, 0].eq(77).all()
    assert replay.action[:2].eq(0).all()
