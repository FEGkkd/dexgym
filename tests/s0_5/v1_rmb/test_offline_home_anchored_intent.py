"""Independent target geometry controls for the offline simulation diagnosis."""

from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.v1.rmb.home_anchored_intent import (
    SignedHumanTick, SimulationIntent, lock_registered_start,
)


def pair(positions, axes):
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[:, :3, 3] = positions
    poses[:, :3, :3] = Rotation.from_rotvec(axes).as_matrix()
    return poses


class HomeAnchoredIntentTests(unittest.TestCase):
    def setUp(self):
        self.start = pair([[.2, .6, -.1], [.7, .8, .3]], [[1.2, 0., 0.], [.2, .3, -.4]])
        self.home = pair([[.1, .4, .75], [-.3, .4, .80]], [[0., 0., 0.], [0., 0., .1]])
        shape = np.zeros((5, 4, 3))
        shape[:, :, 0] = .04 + np.arange(4) * .015
        shape[:, :, 1] = np.arange(5)[:, None] * .02
        self.model = SimpleNamespace(local=np.tile(shape, (2, 1, 1, 1)),
                                     palm=self.home[:, :3, 3], palm_rotation=self.home[:, :3, :3],
                                     wrists=self.home[:, :3, 3] + [0., 0., .02], rotations=self.home[:, :3, :3])
        self.human = self.model.local * 1.2
        self.human[:, :, 2:, 2] = .009

    def test_rigid_start_preserves_orientation_and_interhand_transform(self):
        for yaw in (False, True):
            fit = lock_registered_start(self.start, self.home, align_yaw=yaw)
            target = fit @ self.start
            np.testing.assert_allclose(target[:, :3, 3].mean(axis=0), self.home[:, :3, 3].mean(axis=0), atol=1e-12)
            np.testing.assert_allclose(np.linalg.inv(target[0]) @ target[1],
                                       np.linalg.inv(self.start[0]) @ self.start[1], atol=1e-12)
            np.testing.assert_allclose(target[:, :3, :3], fit[:3, :3] @ self.start[:, :3, :3], atol=1e-12)
            if yaw:
                line = target[1, :2, 3] - target[0, :2, 3]
                home_line = self.home[1, :2, 3] - self.home[0, :2, 3]
                np.testing.assert_allclose(line / np.linalg.norm(line), home_line / np.linalg.norm(home_line), atol=1e-12)

    def test_degenerate_horizontal_baseline_blocks_yaw(self):
        start = self.start.copy()
        start[1, :3, 3] = start[0, :3, 3] + [0., .12, 0.]
        with self.assertRaisesRegex(ValueError, "Yaw undefined"):
            lock_registered_start(start, self.home, align_yaw=True)
        lock_registered_start(start, self.home)

    def test_attachment_is_explicit_and_never_qualifies_physical_intent(self):
        locked = lock_registered_start(self.start, self.home)
        row = SignedHumanTick(None, self.start, tuple(self.human))
        naked = SimulationIntent(locked, self.model).map(row)
        self.assertIsNone(naked["simulation_scoring_tcp"])
        self.assertIsNone(naked["robot_morphology_target"])
        self.assertEqual(naked["anatomical_mount_status"], "UNKNOWN")
        self.assertFalse(naked["physical_calibration_qualified"])
        bridge = np.tile(np.eye(4), (2, 1, 1))
        mapper = SimulationIntent(locked, self.model, proxy_from_model_palm=bridge)
        attached = mapper.map(row)
        np.testing.assert_allclose(naked["sim_mapped_wrist_proxy"], attached["sim_mapped_wrist_proxy"], atol=1e-12)
        np.testing.assert_array_equal(attached["raw_manus_local"], self.human)
        local = attached["robot_morphology_target"]["local_landmark_xyz"]
        np.testing.assert_allclose(np.linalg.norm(np.diff(local, axis=2), axis=-1),
                                   np.linalg.norm(np.diff(self.model.local, axis=2), axis=-1), atol=1e-12)
        np.testing.assert_allclose(attached["robot_morphology_target"]["wrist_xyz"],
                                   (locked @ self.start @ mapper.model_palm_from_scoring_tcp)[:, :3, 3], atol=1e-12)
        partial = mapper.map(SignedHumanTick(None, self.start, (self.human[0], None)))
        self.assertIsNotNone(partial["simulation_scoring_tcp"])
        self.assertIsNone(partial["robot_morphology_target"])


if __name__ == "__main__":
    unittest.main()
