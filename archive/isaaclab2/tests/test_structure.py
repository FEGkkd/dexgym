import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StructureTest(unittest.TestCase):
    def test_both_task_workflows_exist(self):
        self.assertTrue((ROOT / "source/dexteroushands_lab/tasks/direct").is_dir())
        self.assertTrue((ROOT / "source/dexteroushands_lab/tasks/manager_based").is_dir())

    def test_training_entrypoints_exist(self):
        self.assertTrue((ROOT / "scripts/rsl_rl/train.py").is_file())
        self.assertTrue((ROOT / "scripts/rsl_rl/play.py").is_file())

    def test_wuji_runtime_has_no_legacy_asset_conversion_references(self):
        forbidden = (
            "wuji_" + "mj" + "cf_reference",
            "WUJI_" + "ISAAC_ASSET_PATH",
            "wuji_" + "isaac_generated",
            "right_" + "merged",
        )
        runtime_roots = (ROOT / "source", ROOT / "scripts")
        offenders = []
        for runtime_root in runtime_roots:
            for path in runtime_root.rglob("*.py"):
                text = path.read_text()
                for token in forbidden:
                    if token in text:
                        offenders.append(f"{path.relative_to(ROOT)}: {token}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
