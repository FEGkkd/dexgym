import sys
import unittest
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_ROOT))

from dexteroushands_lab.tasks.direct.shadow_hand_reorientation.task_contract import (  # noqa: E402
    REORIENTATION_CONTRACT,
)
from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.task_contract import (  # noqa: E402
    WUJI_REORIENTATION_CONTRACT,
)
from dexteroushands_lab.tasks.task_manifest import get_task_record  # noqa: E402


class DataContractTest(unittest.TestCase):
    def test_reorientation_dimensions_are_explicit(self):
        self.assertEqual(REORIENTATION_CONTRACT.action_dim, 20)
        self.assertEqual(REORIENTATION_CONTRACT.observation_dim, 84)

    def test_joint_order_is_explicit(self):
        self.assertEqual(REORIENTATION_CONTRACT.joint_order[0], "robot0_WRJ1")
        self.assertEqual(REORIENTATION_CONTRACT.joint_order[-1], "robot0_THJ0")
        self.assertEqual(REORIENTATION_CONTRACT.quaternion_convention, "wxyz")

    def test_task_manifest_records_workflow(self):
        record = get_task_record("DexterousHands-Shadow-Hand-Reorientation-Direct-v0")
        self.assertEqual(record.workflow, "direct")
        self.assertEqual(record.source_task, "ShadowHandReOrientation")

    def test_wuji_contract_matches_official_usd_action_width(self):
        self.assertEqual(WUJI_REORIENTATION_CONTRACT.action_dim, 20)
        self.assertEqual(WUJI_REORIENTATION_CONTRACT.frame_observation_dim, 69)
        self.assertEqual(WUJI_REORIENTATION_CONTRACT.observation_history_length, 3)
        self.assertEqual(WUJI_REORIENTATION_CONTRACT.observation_dim, 207)
        self.assertEqual(WUJI_REORIENTATION_CONTRACT.critic_observation_dim, 413)
        self.assertEqual(
            WUJI_REORIENTATION_CONTRACT.joint_order[0], "right_finger1_joint1"
        )
        self.assertEqual(
            WUJI_REORIENTATION_CONTRACT.joint_order[-1], "right_finger5_joint4"
        )


if __name__ == "__main__":
    unittest.main()
