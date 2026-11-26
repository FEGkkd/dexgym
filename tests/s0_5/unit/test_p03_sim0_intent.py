"""Independent tests of bimanual geometry and causal reference transitions."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.sim0.intent import IntentMapper


def rotations(v):
    return Rotation.from_rotvec(v).as_matrix()


def config(kind="T1", **parts):
    return {
        "kind": kind,
        "shared_rotation": rotations([.12, -.28, .32]),
        "shared_translation_m": [.21, -.12, .26],
        "wrist_mount_rotations": rotations([[.38, .05, -.14], [-.19, .28, .07]]),
        "midpoint_center_m": [.10, -.02, .26],
        "midpoint_limits_m": [.32, .22, .17],
        "dt_s": .02,
        "rejoin_duration_s": .08,
        "recovery_valid_frames": 2,
        **parts,
    }


def pose(left=(.34, -.05, .21), right=(.15, .40, .63), twist=None):
    if twist is None:
        twist = [[.08, .31, -.14], [-.29, .16, .25]]
    return np.c_[np.array([left, right]), Rotation.from_rotvec(twist).as_quat()]


class BimanualGeometryTest(unittest.TestCase):
    def test_t1_rigid_and_signed_orientation_mounts(self):
        c = config()
        x = pose()
        out = IntentMapper(c).step(x, True)
        shared = c["shared_rotation"]
        expected_p = np.stack([shared @ row[:3] + c["shared_translation_m"] for row in x])
        expected_r = np.stack([shared @ Rotation.from_quat(x[side, 3:]).as_matrix()
                               @ c["wrist_mount_rotations"][side] for side in range(2)])
        np.testing.assert_allclose(out.wrist_xyz_m, expected_p, rtol=0, atol=2e-15)
        np.testing.assert_allclose(out.wrist_rotation, expected_r, rtol=0, atol=2e-15)
        np.testing.assert_allclose(out.wrist_xyz_m[1] - out.wrist_xyz_m[0],
                                   shared @ (x[1,:3] - x[0,:3]), rtol=0, atol=2e-15)
        self.assertAlmostEqual(np.linalg.norm(out.wrist_xyz_m[1] - out.wrist_xyz_m[0]),
                               np.linalg.norm(x[1,:3] - x[0,:3]), places=13)

    def test_t2_changes_only_common_midpoint_and_is_nonlinear(self):
        c = config("T2")
        x = pose(left=(1.0,-.37,.94), right=(.68,.43,1.40))
        p2, r2 = IntentMapper(c).batch(np.stack([x, x, x]))
        p1, r1 = IntentMapper(config()).batch(np.stack([x, x, x]))
        np.testing.assert_allclose(p2[:,1] - p2[:,0], p1[:,1] - p1[:,0], rtol=0, atol=3e-15)
        np.testing.assert_allclose(p2[:,1] - p1[:,1], p2[:,0] - p1[:,0], rtol=0, atol=3e-15)
        np.testing.assert_array_equal(r1,r2)
        center, limits = np.asarray(c["midpoint_center_m"]), np.asarray(c["midpoint_limits_m"])
        expected = center + limits * np.tanh((p1.mean(axis=1) - center) / limits)
        np.testing.assert_allclose(p2.mean(axis=1), expected, rtol=0, atol=2e-15)
        self.assertGreater(np.linalg.norm(p2.mean(axis=1)[0] - p1.mean(axis=1)[0]), .2)
        perturb = np.array([.16, .0, .0])
        series = np.array([x.copy(), x.copy(), x.copy()])
        series[:, :, :3] += perturb * np.array([0., 1., 2.])[:,None,None]
        midpoints = IntentMapper(c).batch(series)[0].mean(axis=1)
        self.assertGreater(np.linalg.norm(midpoints[2] - 2*midpoints[1] + midpoints[0]), 1e-5)
        one = IntentMapper(c).step(x, [True, True])
        np.testing.assert_allclose(one.wrist_xyz_m, p2[0], rtol=0, atol=2e-15)
        np.testing.assert_allclose(one.wrist_rotation, r2[0], rtol=0, atol=2e-15)

    def test_posture_ablation_frames_and_equivalence_condition(self):
        seed = rotations([[-.17, -.25, .10], [.36, .05, .08]])
        c = config(orientation_mode="RELATIVE", wrist_seed_rotations=seed)
        relative = IntentMapper(c)
        x = pose()
        y = pose(twist=[[.30, -.11, .18],[-.20, .07, .01]])
        self.assertEqual(relative.step(x, False).mode, "HOLD")
        start = relative.step(x, True)
        later = relative.step(y, True)
        np.testing.assert_allclose(start.wrist_rotation, seed, rtol=0, atol=2e-15)
        h_ref = Rotation.from_quat(x[:,3:]).as_matrix()
        h_now = Rotation.from_quat(y[:,3:]).as_matrix()
        mounts = c["wrist_mount_rotations"]
        a = seed @ mounts.transpose(0,2,1) @ h_ref.transpose(0,2,1)
        fixed = IntentMapper(config(orientation_mode="FIXED_ABSOLUTE", absolute_calibration_rotations=a))
        fixed_p, fixed_r = fixed.batch(np.stack([x, y]))
        np.testing.assert_allclose(fixed_r[0], start.wrist_rotation, rtol=0, atol=3e-15)
        np.testing.assert_allclose(fixed_r[1], later.wrist_rotation, rtol=0, atol=3e-15)
        np.testing.assert_allclose(later.wrist_rotation, a @ h_now @ mounts, rtol=0, atol=3e-15)
        full = IntentMapper(config()).step(y, True)
        self.assertGreater(np.linalg.norm(full.wrist_rotation - later.wrist_rotation), .1)
        np.testing.assert_allclose(later.wrist_xyz_m, full.wrist_xyz_m, rtol=0, atol=2e-15)
        np.testing.assert_allclose(fixed_p[1], full.wrist_xyz_m, rtol=0, atol=2e-15)
        signs_flipped = y.copy()
        signs_flipped[0, 3:] *= -1
        check = relative.step(signs_flipped, True)
        np.testing.assert_allclose(check.wrist_rotation, later.wrist_rotation, rtol=0, atol=2e-15)

    def test_signed_p02_contract_keys_and_right_handed_validation(self):
        c=config()
        signed = dict(kind="T1", world_rotation=c["shared_rotation"],
                      world_translation_m=c["shared_translation_m"],
                      tracker_to_tcp_orientation_mount=c["wrist_mount_rotations"])
        signed_result = IntentMapper(signed).step(pose(), True)
        regular = IntentMapper(c).step(pose(), True)
        np.testing.assert_array_equal(signed_result.wrist_xyz_m, regular.wrist_xyz_m)
        np.testing.assert_array_equal(signed_result.wrist_rotation, regular.wrist_rotation)
        signs = np.diag([1., -1., 1.])
        with self.assertRaisesRegex(ValueError, "right-handed"):
            IntentMapper(config(shared_rotation=signs))
        with self.assertRaisesRegex(ValueError, "positive"):
            IntentMapper(config("T2", midpoint_limits_m=[.1, 0., .2]))


class CausalClutchTest(unittest.TestCase):
    def setUp(self):
        self.a = pose()
        self.b = pose(left=(.12,-.43,-.2),right=(.29,.24,.20),
                      twist=[[.39,-.25,.19], [.14,-.08,-.48]])

    def test_explicit_clutch_relocations_keep_persistent_common_offset(self):
        c = config("T3")
        mapper = IntentMapper(c)
        first = mapper.step(self.a, True)
        holding = mapper.step(self.b, True, "CLUTCH_ON")
        self.assertEqual((holding.mode, holding.valid, holding.input_valid), ("HOLD", False, True))
        np.testing.assert_array_equal(holding.wrist_xyz_m, first.wrist_xyz_m)
        self.assertEqual(mapper.step(self.b, [False,True], "CLUTCH_OFF").reason,
                         "CLUTCH_RELEASE_REQUIRES_VALID_PAIR")
        self.assertEqual(mapper.step(self.b, True).reason, "CLUTCH_HELD")
        release = mapper.step(self.b, True, "CLUTCH_OFF")
        self.assertEqual((release.mode, release.generation, release.rejoin_weight), ("REJOINING", 1, 0.))
        np.testing.assert_array_equal(release.wrist_xyz_m, first.wrist_xyz_m)
        np.testing.assert_array_equal(release.wrist_rotation, first.wrist_rotation)
        self.assertGreater(release.relative_vector_error_m, .1)
        self.assertGreater(release.relative_orientation_error_rad, .1)

        p2, r2 = IntentMapper(config("T2")).batch(self.b[None])
        offset = first.wrist_xyz_m.mean(0) - p2[0].mean(0)
        np.testing.assert_allclose(release.reference_offset_m, offset, rtol=0, atol=2e-15)
        path = [mapper.step(self.b, True) for _ in range(4)]
        self.assertEqual([p.rejoin_weight for p in path], [.15625, .5, .84375, 1.])
        self.assertEqual([p.mode for p in path], ["REJOINING"]*3 + ["ACTIVE"])
        for step in path:
            np.testing.assert_allclose(step.reference_offset_m, offset, rtol=0, atol=2e-15)
            expected_p = (1 - step.rejoin_weight) * first.wrist_xyz_m + step.rejoin_weight * (p2[0] + offset)
            np.testing.assert_allclose(step.wrist_xyz_m, expected_p, rtol=0, atol=2e-15)
            self.assertAlmostEqual(step.relative_vector_error_m,
                                   (1 - step.rejoin_weight) * release.relative_vector_error_m, places=13)
        np.testing.assert_allclose(path[-1].wrist_rotation, r2[0], rtol=0, atol=2e-15)
        moved = self.b.copy()
        moved[:,:3] += [.18, .02, -.03]
        new = mapper.step(moved, True)
        base = IntentMapper(config("T2")).step(moved, True)
        self.assertEqual(new.generation, 1)
        np.testing.assert_allclose(new.wrist_xyz_m, base.wrist_xyz_m + offset, rtol=0, atol=2e-15)
        np.testing.assert_allclose(new.wrist_xyz_m[1] - new.wrist_xyz_m[0],
                                   c["shared_rotation"] @ (moved[1,:3] - moved[0,:3]), rtol=0, atol=2e-15)

    def test_invalid_hold_recovers_with_fresh_pairs_without_reference_relocation(self):
        mapper = IntentMapper(config("T3"))
        empty = mapper.step(None, [True, False])
        self.assertEqual((empty.mode, empty.valid, empty.wrist_xyz_m), ("HOLD", False, None))
        prior = mapper.step(self.a, True)
        invalid = self.b.copy()
        invalid[0,3:] = 0.
        held = mapper.step(invalid, True)
        np.testing.assert_array_equal(held.wrist_xyz_m, prior.wrist_xyz_m)
        self.assertFalse(held.input_valid)
        wait = mapper.step(self.b, True)
        self.assertEqual((wait.mode,wait.reason), ("HOLD","RECOVERY_WAIT_VALID_PAIR"))
        mapper.step(self.b, [True, False])
        self.assertEqual(mapper.step(self.b, True).mode, "HOLD")
        recovered = mapper.step(self.b, True)
        self.assertEqual((recovered.mode, recovered.generation, recovered.rejoin_weight), ("REJOINING",0,0.))
        np.testing.assert_array_equal(recovered.wrist_xyz_m, prior.wrist_xyz_m)
        meanwhile = mapper.step(self.b, True)
        self.assertEqual(meanwhile.mode, "REJOINING")
        held_again = mapper.step(None, False)
        np.testing.assert_array_equal(held_again.wrist_xyz_m, meanwhile.wrist_xyz_m)
        self.assertEqual(mapper.step(self.b, True).mode, "HOLD")
        np.testing.assert_array_equal(mapper.step(self.b, True).wrist_xyz_m, meanwhile.wrist_xyz_m)
        for _ in range(4):
            last = mapper.step(self.b, True)
        base_b = IntentMapper(config("T2")).step(self.b, True)
        self.assertEqual((last.mode,last.generation), ("ACTIVE",0))
        np.testing.assert_array_equal(last.reference_offset_m, np.zeros(3))
        np.testing.assert_allclose(last.wrist_xyz_m, base_b.wrist_xyz_m, rtol=0, atol=2e-15)

    def test_prefix_equivalence_and_output_mutation_isolation(self):
        sequence = [(self.a,True,None), (self.b,True,"CLUTCH_ON"), (self.b,False,None),
                    (self.a,True,"CLUTCH_OFF"), (self.b,True,None), (self.b,False,None)]
        x = IntentMapper(config("T3",orientation_mode="RELATIVE"))
        y = IntentMapper(config("T3",orientation_mode="RELATIVE"))
        left = [x.step(*row) for row in sequence]
        y.step(None, False)  # Startup invalid data cannot establish an orientation or relocation reference.
        right = [y.step(*row) for row in sequence]
        untrusted_future_a = pose((99.0, -.3, .2),(99.15,.59,.63))
        untrusted_future_b = pose((-99.0, -.3, .2),(-99.15,.59,.63))
        x.step(untrusted_future_a, True)
        y.step(untrusted_future_b, True)
        for i, j in zip(left,right):
            self.assertEqual((i.mode,i.valid,i.generation,i.reason),(j.mode,j.valid,j.generation,j.reason))
            np.testing.assert_array_equal(i.wrist_xyz_m,j.wrist_xyz_m)
            np.testing.assert_array_equal(i.wrist_rotation,j.wrist_rotation)
        out = x.step(None, False)
        intact_position = out.wrist_xyz_m.copy()
        intact_rotation = out.wrist_rotation.copy()
        out.wrist_xyz_m[0] = 1e3
        out.wrist_rotation[0] = 1e3
        out.reference_offset_m[:] = 1e3
        next_hold = x.step(None,False)
        np.testing.assert_array_equal(next_hold.wrist_xyz_m,intact_position)
        np.testing.assert_array_equal(next_hold.wrist_rotation,intact_rotation)
        self.assertLess(np.linalg.norm(next_hold.reference_offset_m), 1.)


if __name__ == "__main__":
    unittest.main()
