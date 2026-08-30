import sys
import unittest
from pathlib import Path

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_ROOT))

from dexteroushands_lab.tasks.manager_based.wuji_arm_hand.mdp import (  # noqa: E402
    build_arm_hand_observation,
    split_arm_hand_action,
)


class ManagerContractTest(unittest.TestCase):
    def test_action_split_preserves_arm_hand_dimensions(self):
        action = torch.zeros((3, 9))
        arm, hand = split_arm_hand_action(action, arm_dof=7, hand_dof=2)
        self.assertEqual(tuple(arm.shape), (3, 7))
        self.assertEqual(tuple(hand.shape), (3, 2))

    def test_observation_blocks_concatenate(self):
        blocks = [torch.zeros((2, size)) for size in (7, 20, 14, 7, 20)]
        observation = build_arm_hand_observation(*blocks)
        self.assertEqual(tuple(observation.shape), (2, 68))


if __name__ == "__main__":
    unittest.main()
