"""Guard the Core dependency direction and dimension-driven contracts."""

import ast
from pathlib import Path
import re


CORE = (Path(__file__).resolve().parents[2] / "source/dexgym/algorithms/supermimic/core")


def test_core_has_no_marvin_wave_motion_or_fixed_network_dimensions():
    files = tuple(CORE.glob("*.py"))
    assert files
    for path in files:
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("dexgym.tasks", "dexgym.motion")), path
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(("dexgym.tasks", "dexgym.motion"))
                           for alias in node.names), path
        executable = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
        assert not re.search(r"\b(?:14|42|90|146|560|900|1357)\b", executable), path


def test_task_owned_legacy_physical_evaluator_keeps_historical_function_body():
    root = Path(__file__).resolve().parents[2]
    historical = root / "scripts/supermimic/train_wave.py"
    task_owned = root / "source/dexgym/tasks/marvin_supermimic/legacy_evaluator.py"
    def function(path):
        return next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "evaluate_full")
    assert ast.dump(function(historical), include_attributes=False) == ast.dump(
        function(task_owned), include_attributes=False)
    assert "from train_wave import evaluate_full" not in (
        root / "scripts/supermimic/evaluate.py").read_text()
