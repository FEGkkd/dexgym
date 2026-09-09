"""Compatibility import for the project-level RSL-RL CLI helpers."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.cli_args import add_rsl_rl_args, parse_rsl_rl_cfg

__all__ = ["add_rsl_rl_args", "parse_rsl_rl_cfg"]
