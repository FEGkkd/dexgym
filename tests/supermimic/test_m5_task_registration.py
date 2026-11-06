import ast
from pathlib import Path

from dexgym.tasks.marvin_supermimic import SE3_REACH_HOLD_TASK_ID
from dexgym.tasks.marvin_supermimic.register_core import register_marvin
from dexgym.algorithms.supermimic.registry import PluginRegistry


def test_new_se3_task_is_independent_and_registered():
    assert SE3_REACH_HOLD_TASK_ID == "DexGym-Marvin-SuperMimic-SE3-Reach-Hold-Direct-v0"
    registry = PluginRegistry()
    register_marvin(registry)
    assert "marvin_bimanual_se3_reach_hold" in registry._tasks
    # Isaac Lab's DirectRLEnv import requires the Newton runtime bootstrap;
    # inspect the inheritance AST in this CPU-only schema test.
    source = Path(__file__).parents[2] / "source/dexgym/tasks/marvin_supermimic/se3_env.py"
    classes = {node.name: node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef)}
    bases = [ast.unparse(base) for base in classes["MarvinSE3ReachHoldEnv"].bases]
    assert bases == ["MarvinArmNewtonBaseEnv"]
