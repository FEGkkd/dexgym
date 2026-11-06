import ast
from pathlib import Path


TASKS = Path(__file__).resolve().parents[2] / "source/dexgym/tasks/marvin_supermimic"


def _class_base(path, class_name):
    source = (TASKS / path).read_text()
    tree = ast.parse(source)
    found = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return [ast.unparse(base) for base in found.bases], source


def test_tracking_and_reach_share_only_robot_physics_base():
    tracking, _ = _class_base("env.py", "MarvinWaveDirectEnv")
    reach, reach_source = _class_base("reach_env.py", "MarvinReachDirectEnv")
    base, base_source = _class_base("base_env.py", "MarvinArmNewtonBaseEnv")
    assert tracking == reach == ["MarvinArmNewtonBaseEnv"]
    assert base == ["DirectRLEnv"]
    assert "MotionManager" not in reach_source + base_source
    assert "WaveDirectEnv" not in reach_source + base_source
