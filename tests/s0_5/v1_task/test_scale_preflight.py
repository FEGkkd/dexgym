import unittest

import torch

from dexgym.tasks.direct.wholebody_s05.v1.task import JOINTS
from dexgym.tasks.direct.wholebody_s05.v1.task.scale_preflight import AMPLITUDE, fixture_actions, validate_workload


class ScalePreflightTests(unittest.TestCase):
    def test_deterministic_fixture_requests_every_axis_in_both_directions(self):
        home = torch.zeros((64, JOINTS))
        action = fixture_actions(home, 0)
        self.assertTrue(torch.equal(action, fixture_actions(home, 0)))
        self.assertFalse(torch.equal(action, fixture_actions(home, 1)))
        self.assertTrue(torch.all((action != 0).sum(dim=1) == 4))
        self.assertTrue(torch.all((action < 0).any(dim=0)))
        self.assertTrue(torch.all((action > 0).any(dim=0)))
        torch.testing.assert_close(action.abs().max(), torch.tensor(AMPLITUDE))

    def test_workload_is_checked_before_rollout(self):
        validate_workload(64, 8, 512)
        validate_workload(4096, 6, 25_000)
        for arguments in ((64, 8, 511), (4096, 7, 25_000), (4097, 1, 25_000), (1, 1, 25_001), (1, 0, 25_000)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_workload(*arguments)
