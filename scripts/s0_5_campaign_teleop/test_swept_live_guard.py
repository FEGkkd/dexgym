"""CPU regression and fail-closed controls for the diagnostic swept guard."""

from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory
from time import monotonic_ns
import unittest

import numpy as np

from dexgym.tasks.direct.wholebody_s05.p02.model import RobotModel
from swept_live_guard import Guard, MAGIC, REQUEST, RESPONSE, Reason


ASSETS = Path(__file__).resolve().parents[2] / "source/dexgym/assets/marvin_wuji/hand2_2/generated"
CROSSING_START = [
    2.0038192622259023, -1.7030077784150977, 0.08271551079576334, -1.171557993422256,
    -0.793011680797775, 0.008821386564273892, 0.11535095787539351, 1.641805372981656,
    0.33026121973681155, -2.02466757174886, 0.19650864043443184, 0.41170090761211364,
    0.39026995512154516, 0.8188602358969781,
]
CROSSING_END = [
    0.9635915092175189, -1.2733464043080827, 0.42761809979446275, -0.9610012588363388,
    2.6301190538155925, -0.2950483871982478, -0.4918113497485199, -2.22951571408942,
    -1.8380720955475534, 1.8158688428742948, 0.5390065619947735, -1.4994237177877119,
    0.6507907841146281, -0.7997250219810643,
]


def packet(sequence, group, issued, measured, proposal, magic=MAGIC):
    return REQUEST.pack(magic, sequence, group, *np.concatenate((issued, measured, proposal)))


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Native Windows MuJoCo cannot load assets from a non-ASCII path.
        cls.directory = TemporaryDirectory(prefix="s05_sweep_cpu_")
        cls.addClassCleanup(cls.directory.cleanup)
        root = copytree(ASSETS, Path(cls.directory.name) / "assets")
        cls.robots = tuple(RobotModel(root / "dual_arm_marvin.xml") for _ in range(2))
        cls.home = cls.robots[0].home

    def setUp(self):
        self.guard = Guard(self.robots)

    def check(self, sequence, group, issued, measured, proposal, budget_ns=100_000_000):
        request = packet(sequence, group, issued, measured, proposal)
        seq, reason, returned_group, states = self.guard.evaluate(request, monotonic_ns() + budget_ns)
        self.assertEqual((seq, returned_group), (sequence, group))
        self.guard.publish_decision(reason, states)
        return reason

    def test_bounded_pose_and_group_update(self):
        self.assertEqual((REQUEST.size, RESPONSE.size), (1316, 24))
        arm = self.home.copy()
        arm[0] += .001
        self.assertEqual(self.check(1, 0, self.home, self.home, arm), Reason.PERMIT)
        # A lost successful reply leaves Home as the last published snapshot.
        self.assertEqual(self.check(2, 0, self.home, self.home, arm), Reason.PERMIT)
        left = arm.copy()
        left[15] += .001
        self.assertEqual(self.check(3, 1, arm, self.home, left), Reason.PERMIT)
        self.assertEqual(self.check(4, 2, arm, self.home, left), Reason.GROUP_MISMATCH)
        self.assertEqual(self.check(4, 1, arm, self.home, left), Reason.SEQUENCE)

    def test_start_chain_rejects_state_swap(self):
        other = self.home.copy()
        other[0] += .01
        self.assertEqual(self.check(1, 0, other, self.home, other), Reason.START_UNBOUND)
        self.assertEqual(self.check(2, 0, self.home, self.home, self.home), Reason.PERMIT)
        self.assertEqual(self.check(3, 0, other, other, other), Reason.START_UNBOUND)

    def test_nonfinite_limits_coupling_and_expiry(self):
        request = self.home.copy()
        request[0] = np.nan
        self.assertEqual(self.check(1, 0, self.home, self.home, request), Reason.NONFINITE)
        request[0] = self.robots[0].bounds[0, 1] + 1e-4
        self.assertEqual(self.check(2, 0, self.home, self.home, request), Reason.LIMIT_OR_COUPLING)
        request = self.home.copy()
        request[5:7] = self.robots[0].bounds[5:7, 1]
        self.assertEqual(self.check(3, 0, self.home, self.home, request), Reason.LIMIT_OR_COUPLING)
        self.assertEqual(self.check(4, 0, self.home, self.home, self.home, -1), Reason.TIME_BUDGET)

    def test_legal_endpoints_with_intermediate_collision(self):
        start, end = self.home.copy(), self.home.copy()
        start[:14], end[:14] = CROSSING_START, CROSSING_END
        native = self.guard.envelopes[0].native
        self.assertGreater(native.pose_gap(start), native.clearance)
        self.assertGreater(native.pose_gap(end), native.clearance)
        self.assertLess(native.pose_gap((start + end) * .5), 0)
        self.assertEqual(self.check(1, 0, start, start, end), Reason.COLLISION)

    def test_wrong_wire_version_and_truncated_packet(self):
        request = packet(1, 0, self.home, self.home, self.home, magic=b"S05XXXX1")
        self.assertEqual(self.guard.evaluate(request, monotonic_ns())[1], Reason.BAD_REQUEST)
        self.assertEqual(self.guard.evaluate(request[:-1], monotonic_ns())[1], Reason.BAD_REQUEST)


if __name__ == "__main__":
    unittest.main()
