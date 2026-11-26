import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from dexgym.tasks.direct.wholebody_s05.v1.rmb.nominal import matrix
from dexgym.tasks.direct.wholebody_s05.v1.rmb.probe_nominal import (
    PHASE1_BACKOFF_SCALES,
    PHASE1_DT_S,
    PHASE1_ROTATION_RATE_RAD_S,
    PHASE1_TRANSLATION_RATE_M_S,
    phase1_cartesian_trials,
)


class Phase1CartesianProbeTests(unittest.TestCase):
    def test_caps_are_applied_before_each_backoff(self):
        origin = matrix([1, 2, 3], Rotation.from_euler("z", .1).as_matrix())
        projected = matrix([1, 3, 3], Rotation.from_euler("z", .6).as_matrix())
        original_target = projected.copy()
        trials = list(phase1_cartesian_trials(origin, projected))
        self.assertEqual(tuple(scale for scale, _ in trials), PHASE1_BACKOFF_SCALES)
        for scale, attempt in trials:
            displacement = attempt[:3, 3] - origin[:3, 3]
            angle = Rotation.from_matrix(attempt[:3, :3] @ origin[:3, :3].T).as_rotvec()
            np.testing.assert_allclose(displacement, [0, scale * PHASE1_TRANSLATION_RATE_M_S * PHASE1_DT_S, 0])
            np.testing.assert_allclose(angle, [0, 0, scale * PHASE1_ROTATION_RATE_RAD_S * PHASE1_DT_S])
        np.testing.assert_array_equal(projected, original_target)

    def test_unclipped_goal_at_full_scale(self):
        origin = matrix([0, 0, 0], np.eye(3))
        projected = matrix([0.0001, 0, 0], Rotation.from_euler("z", .001).as_matrix())
        scale, attempt = next(phase1_cartesian_trials(origin, projected))
        self.assertEqual(scale, 1.0)
        np.testing.assert_allclose(attempt, projected, atol=1e-12)
