"""Synthetic, independent and native-physics contract checks for the V1-B task."""

from pathlib import Path
import unittest

import numpy as np
import torch

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from dexgym.tasks.direct.wholebody_s05.v1.task.geometry import ResidualKinematics
from dexgym.tasks.direct.wholebody_s05.v1.task.observation import (CausalHistory, ObservationBuilder,
                                                                  geometry_mask, velocity_mask)
from dexgym.tasks.direct.wholebody_s05.v1.task.physics import ServoProfile
from dexgym.tasks.direct.wholebody_s05.v1.task.safety import NativeSweep, SafetyProjection
from dexgym.tasks.direct.wholebody_s05.v1.task.smoke import make_task, target_frame

XML = Path(__file__).resolve().parents[3] / (
    "source/dexgym/assets/marvin_wuji/hand2_2/generated/dual_arm_marvin.xml"
)


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RobotModel(XML)

    def test_native_fk_matches_differentiable_frames_and_gradients(self):
        fk = ResidualKinematics(self.robot)
        numpy_home = self.robot.home
        q = torch.tensor(numpy_home[None], dtype=torch.float32, requires_grad=True)
        got = fk(q)
        native = self.robot.geometry(numpy_home)
        for x, y in ((got.wrist, native.wrists), (got.rotation, native.rotations),
                     (got.local, native.local), (got.tips, native.tips)):
            np.testing.assert_allclose(x[0].detach().numpy(), y, atol=1e-6)
        loss = got.tips.square().sum() + got.local.square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(torch.count_nonzero(q.grad).item(), 40)

    def test_history_replaces_no_future_and_cuts_switched_episode_and_gap(self):
        history = CausalHistory(1, 3, "cpu")
        valid = torch.tensor([True])
        t0 = torch.tensor([40_000_000])
        g0 = torch.tensor([0])
        prefix = history.append(torch.tensor([[2., 3., 1.]]), g0, t0, valid)
        self.assertEqual(prefix[0, :12].count_nonzero(), 0)
        result = history.append(torch.tensor([[4., 5., 1.]]), g0, t0 + 20_000_000, valid)
        self.assertEqual(result[0, -6:].tolist(), [2., 3., 1., 4., 5., 1.])
        self.assertEqual(prefix[0, -3:].tolist(), [2., 3., 1.])
        cut = history.append(torch.tensor([[6., 7., 1.]]), g0 + 1, t0 + 40_000_000, valid)
        self.assertEqual(cut[0, :12].count_nonzero(), 0)
        gap = history.append(torch.tensor([[6., 7., 1.]]), g0 + 1, t0 + 80_000_000, valid)
        self.assertEqual(gap[0, :12].count_nonzero(), 0)

    def test_compact_approved_columns(self):
        full = ObservationBuilder()
        compact = ObservationBuilder(compact=True)
        self.assertEqual(full.columns.sum(), 1703)
        self.assertEqual(compact.columns.sum(), 1428)
        self.assertFalse(compact.columns[1236:1461].any())
        self.assertFalse(compact.columns[1647:1697].any())

    def test_exact_servo_interval(self):
        initial = torch.zeros((1, 54))
        target = torch.ones((1, 54))
        profile = ServoProfile()
        used = profile.applied_interval(initial, target)
        self.assertEqual(tuple(used.shape), (1, 10, 54))
        expected = torch.as_tensor(1 - np.exp(-np.arange(1, 11) * .002 / .016)).to(used)
        torch.testing.assert_close(used[0, :, 0], expected)
        torch.testing.assert_close(used[0, :, 14], torch.ones(10))

    def test_native_sweep_proves_home_and_rejects_crossing(self):
        sweep = NativeSweep(self.robot)
        home = self.robot.home
        self.assertTrue(sweep.permits(home, home))
        q = home.copy()
        q[0] -= .005
        self.assertTrue(sweep.permits(home, q))
        # Failure to resolve a large swept interval cannot become a proof.
        distant = np.zeros_like(home)
        self.assertFalse(NativeSweep(self.robot, max_depth=0).permits(home, distant))

    def test_batched_minimum_gap_preserves_signed_exact_queries(self):
        guard = SafetyProjection(self.robot, device="cpu", max_step=[.027] * 14 + [.10] * 40)
        home = torch.tensor(self.robot.home, dtype=torch.float32)
        contact = home.clone()
        contact[self.robot.names.index("l_index_finger_mcp_abd")] += .275
        poses = torch.stack((home, contact, contact, home))
        observed = guard.minimum_gaps(poses, torch.tensor([True, True, True, False]))
        expected = [guard.sweep.pose_gap(q.numpy()) for q in (home, contact, contact)]
        torch.testing.assert_close(observed, torch.tensor([*expected, 0.], dtype=poses.dtype))
        self.assertLess(observed[1].item(), 0.)

    def test_native_advances_and_b0_masks_every_axis(self):
        task, q, reset = make_task(self.robot, 1, ServoProfile(), "B0", "cpu")
        obs = task.observe(target_frame(task, q, 0))
        result = task.step(torch.full_like(q, .9))
        self.assertEqual(tuple(obs.shape), (1, 8515))
        self.assertTrue(result.valid[0])
        self.assertEqual(result.substeps[0].item(), 10)
        torch.testing.assert_close(result.c, q)
        torch.testing.assert_close(result.action, torch.zeros_like(q))
        self.assertEqual(tuple(result.engine_capture()["torque"].shape), (1, 54))
        state = result.x1_pre_reset.clone()
        reset()
        torch.testing.assert_close(result.x1_pre_reset, state)

    def test_invalid_input_has_no_training_reward(self):
        task, q, _ = make_task(self.robot, 1, ServoProfile(), "B54", "cpu")
        task.observe(target_frame(task, q, 0, permit=False))
        result = task.step(torch.zeros_like(q))
        self.assertTrue(result.fallback[0])
        self.assertFalse(result.valid[0])
        self.assertEqual(result.reward[0].item(), 0.)


    def test_illegal_nominal_falls_back_identically_for_b0_and_b54(self):
        results = []
        for mode in ("B0", "B54"):
            task, q, _ = make_task(self.robot, 1, ServoProfile(), mode, "cpu")
            target = target_frame(task, q, 0)
            illegal = target.nominal.clone()
            illegal[0, 0] += 20
            task.observe(target._replace(nominal=illegal))
            results.append(task.step(torch.zeros_like(q)))
        a, b = results
        torch.testing.assert_close(a.c, b.c)
        self.assertEqual(a.events.tolist(), b.events.tolist())
        self.assertTrue(a.fallback.all() and b.fallback.all())

    def test_reward_single_finger_penalty_and_termination_mask(self):
        task, q, _ = make_task(self.robot, 1, ServoProfile(), "B54", "cpu")
        target = target_frame(task, q, 0)
        valid = torch.ones(1, dtype=torch.bool)
        zero = torch.zeros_like(q)
        def score(geometry, visible):
            return task.reward(geometry, target.velocity, geometry_mask(valid), velocity_mask(valid),
                               target, zero, zero, zero, task.guard.scale, valid, valid,
                               torch.zeros(1), torch.full((1,), .005), visible)
        perfect = score(target.geometry, valid)
        self.assertAlmostEqual(perfect.value.item(), 1., places=6)
        disturbed = q.clone().requires_grad_(True)
        offset = torch.zeros_like(q)
        offset[0, 17] = .1
        worse = score(task.geometry(disturbed + offset), valid)
        self.assertLess(worse.terms["global_tip"].item(), perfect.terms["global_tip"].item())
        worse.value.sum().backward()
        self.assertTrue(torch.isfinite(disturbed.grad).all())
        self.assertEqual(score(target.geometry, ~valid).value.item(), 0.)


if __name__ == "__main__":
    unittest.main()
