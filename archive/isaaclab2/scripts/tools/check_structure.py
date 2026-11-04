"""Static check for the two required task workflows."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = [
    ROOT / "source/dexteroushands_lab/tasks/direct",
    ROOT / "source/dexteroushands_lab/tasks/manager_based",
    ROOT / "scripts/rsl_rl/train.py",
    ROOT / "scripts/rsl_rl/play.py",
    ROOT / "scripts/tools/validate_policy_export.py",
]

missing = [str(path) for path in REQUIRED if not path.exists()]
if missing:
    raise SystemExit("Missing training architecture paths:\n" + "\n".join(missing))

print("DexGym Isaac Lab training architecture: OK")
