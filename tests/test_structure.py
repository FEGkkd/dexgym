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


if __name__ == "__main__":
    unittest.main()
