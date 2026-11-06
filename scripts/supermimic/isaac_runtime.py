"""Launch an M4.2 script with the validated Isaac Sim 6 / Newton import order.

Usage: python isaac_runtime.py train [train.py arguments]
Set PYTHONPATH to the pinned Newton overlay and this checkout's source directory.
"""

import os
import runpy
import sys
from pathlib import Path

import torch  # noqa: F401
import mujoco  # noqa: F401
import warp  # noqa: F401
import newton  # noqa: F401
import mujoco_warp  # noqa: F401


DEFAULT_VENV_SITE = "/home/ubuntu/dexgym/.venv/lib/python3.12/site-packages"
sys.path.append(os.environ.get("DEXGYM_VENV_SITE", DEFAULT_VENV_SITE))

ALLOWED = {"train", "evaluate", "validate_dynamics", "validate_termination"}
if len(sys.argv) < 2 or sys.argv[1] not in ALLOWED:
    raise SystemExit(f"Select one of {', '.join(sorted(ALLOWED))}")
entry = sys.argv.pop(1)
sys.argv[0] = str(Path(__file__).with_name(entry + ".py"))
runpy.run_path(sys.argv[0], run_name="__main__")
