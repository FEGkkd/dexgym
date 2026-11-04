import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SOURCE_ROOT))

from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.task_contract import (  # noqa: E402
    WUJI_REORIENTATION_CONTRACT,
)
from dexteroushands_lab.utils.policy_metadata import (  # noqa: E402
    build_policy_artifacts,
    runtime_from_env_cfg,
)


class PolicyMetadataTest(unittest.TestCase):
    def test_artifacts_match_wuji_contract(self):
        artifacts = build_policy_artifacts(
            WUJI_REORIENTATION_CONTRACT,
            runtime={"action_scale": 0.5, "ema_alpha": 0.5},
            checkpoint="model_0.pt",
            exported_files=["exported/policy.pt"],
        )
        self.assertEqual(
            set(artifacts),
            {
                "policy_metadata.yaml",
                "observation_schema.yaml",
                "action_schema.yaml",
                "normalization.yaml",
            },
        )
        self.assertEqual(artifacts["policy_metadata.yaml"]["observation_dim"], 207)
        self.assertEqual(artifacts["policy_metadata.yaml"]["critic_observation_dim"], 413)
        self.assertEqual(artifacts["action_schema.yaml"]["dim"], 20)
        self.assertEqual(
            artifacts["action_schema.yaml"]["joint_order"][0], "right_finger1_joint1"
        )
        self.assertEqual(
            artifacts["observation_schema.yaml"]["policy_input"]["layout"],
            "term-major, oldest-to-newest frames",
        )
        self.assertEqual(artifacts["observation_schema.yaml"]["critic_input"]["dim"], 413)
        self.assertEqual(
            artifacts["observation_schema.yaml"]["critic_input"]["layout"],
            "207 clean shared history + 206 privileged state terms",
        )

    def test_history_is_term_major_and_chronological(self):
        configured_root = os.environ.get("WUJI_DESCRIPTION_ROOT")
        if not configured_root:
            sibling_root = Path(__file__).resolve().parents[2] / "wuji-description"
            if sibling_root.is_dir():
                configured_root = str(sibling_root)
        usd_path = (
            Path(configured_root) / "hand/body/usd/right/wujihand.usd"
            if configured_root
            else None
        )
        if usd_path is None or not usd_path.is_file():
            self.skipTest("official WUJI USD is not available for simulator import")
        os.environ["WUJI_DESCRIPTION_ROOT"] = configured_root
        from dexteroushands_lab.tasks.direct.wuji_hand_reorientation.wuji_hand_reorientation_env import (
            WujiHandReorientationDirectEnv,
        )

        history = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0, 5.0], [11.0, 12.0, 13.0, 14.0, 15.0]]]
        )
        flattened = WujiHandReorientationDirectEnv._flatten_term_history(
            history, (2, 3)
        )
        expected = torch.tensor(
            [[1.0, 2.0, 11.0, 12.0, 3.0, 4.0, 5.0, 13.0, 14.0, 15.0]]
        )
        self.assertTrue(torch.equal(flattened, expected))

    def test_runtime_extracts_control_period(self):
        class Sim:
            dt = 1.0 / 120.0

        class EnvCfg:
            sim = Sim()
            decimation = 6
            action_scale = 0.5

        runtime = runtime_from_env_cfg(EnvCfg())
        self.assertAlmostEqual(runtime["control_dt"], 0.05)
        self.assertEqual(runtime["decimation"], 6)

    def test_other_task_does_not_inherit_wuji_critic_layout(self):
        contract = replace(WUJI_REORIENTATION_CONTRACT, task_id="Other-Direct-v0",
                           critic_observation_dim=220)
        schema = build_policy_artifacts(contract)["observation_schema.yaml"]
        self.assertEqual(schema["critic_input"]["dim"], 220)
        self.assertEqual(schema["critic_input"]["layout"],
                         "task-specific critic input; see task contract")
        self.assertNotIn("privileged_fields", schema["critic_input"])


if __name__ == "__main__":
    unittest.main()
