"""Batched thread dynamics retain scalar breakaway and local-axis semantics."""

import math

import pytest
import torch

from dexteroushands_lab.assets.bottle import BottleConfig, ThreadCoupler
from dexteroushands_lab.assets.bottle.batched_thread import BatchedThreadCoupler


def _input(angle=0.0, z=0.0):
    config = BottleConfig()
    bottle = torch.tensor([[0., 0., .5, 0., 0., 0., 1.]], dtype=torch.float64)
    cap = torch.tensor([[0., 0., .5 + config.cap_initial_z + z,
                         0., 0., math.sin(angle / 2), math.cos(angle / 2)]], dtype=torch.float64)
    velocity = torch.zeros(1, 6, dtype=torch.float64)
    return bottle, cap, velocity


def test_batched_thread_matches_scalar_wrenches_and_partial_reset():
    cfg = BottleConfig()
    batch = BatchedThreadCoupler(2, cfg, device="cpu", dtype=torch.float64)
    scalar = ThreadCoupler(cfg)
    bottle, cap, velocity = _input()
    bp, cp, vel = bottle.repeat(2, 1), cap.repeat(2, 1), velocity.repeat(2, 1)
    batch.reset_from_pose(torch.arange(2), bp, cp)
    scalar.reset_from_pose(bottle[0, :3], bottle[0, 3:], cap[0, :3], cap[0, 3:])
    for angle in (0.001, 0.01, 0.03, 0.06):
        _, cap, _ = _input(angle, cfg.thread_pitch * angle / (2 * math.pi))
        cp[:] = cap
        got = batch.wrenches(bp, cp, vel, vel)
        expected = scalar.wrenches(bp[0, :3], bp[0, 3:], vel[0, :3], vel[0, 3:],
                                   cp[0, :3], cp[0, 3:], vel[0, :3], vel[0, 3:])
        assert got.cap_force[0].tolist() == pytest.approx(expected.cap_force, abs=1e-9)
        assert got.cap_torque[0].tolist() == pytest.approx(expected.cap_torque, abs=1e-9)
        assert got.theta_unwrapped[0].item() == pytest.approx(expected.state.theta_unwrapped, abs=1e-10)
    before = batch.theta_unwrapped[1].clone()
    bottle, cap, _ = _input()
    batch.reset_from_pose(torch.tensor([0]), bottle, cap)
    assert batch.theta_unwrapped[0].item() == pytest.approx(0.0)
    assert torch.equal(batch.theta_unwrapped[1], before)


def test_batched_thread_tracks_local_axis_under_tilt():
    cfg = BottleConfig()
    batch = BatchedThreadCoupler(1, cfg, device="cpu", dtype=torch.float64)
    bottle, cap, vel = _input()
    q = torch.tensor([0., math.sin(.4), 0., math.cos(.4)], dtype=torch.float64)
    bottle[0, 3:] = q
    cap[0, 3:] = q
    axis = torch.tensor([math.sin(.8), 0., math.cos(.8)], dtype=torch.float64)
    cap[0, :3] = bottle[0, :3] + axis * cfg.cap_initial_z
    batch.reset_from_pose(torch.tensor([0]), bottle, cap)
    result = batch.wrenches(bottle, cap, vel, vel)
    assert result.axial[0].item() == pytest.approx(cfg.cap_initial_z, abs=1e-10)
    assert result.lateral_error[0].item() == pytest.approx(0.0, abs=1e-10)


def test_batched_breakaway_and_restick_match_scalar_sequence():
    cfg = BottleConfig()
    batch = BatchedThreadCoupler(1, cfg, device="cpu", dtype=torch.float64)
    scalar = ThreadCoupler(cfg)
    bp, cp, velocity = _input()
    batch.reset_from_pose(torch.tensor([0]), bp, cp)
    scalar.reset_from_pose(bp[0, :3], bp[0, 3:], cp[0, :3], cp[0, 3:])
    for angle, spin in ((.001, .0), (.02, .2), (.06, .2), (.1, .2),
                        (.1, .0), (.09, -.2), (.02, -.2)):
        _, cp, _ = _input(angle, cfg.thread_pitch * angle / (2 * math.pi))
        cap_velocity = velocity.clone()
        cap_velocity[0, 5] = spin
        got = batch.wrenches(bp, cp, velocity, cap_velocity)
        expected = scalar.wrenches(bp[0, :3], bp[0, 3:], velocity[0, :3], velocity[0, 3:],
                                   cp[0, :3], cp[0, 3:], cap_velocity[0, :3], cap_velocity[0, 3:])
        assert got.cap_torque[0].tolist() == pytest.approx(expected.cap_torque, abs=1e-9)
        assert batch.sliding[0].item() == scalar.snapshot_state()["sliding"]
