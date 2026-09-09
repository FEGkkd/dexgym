"""Compatibility wrapper for :mod:`rsl_rl.base.train`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rsl_rl.base.train import main


if __name__ == "__main__":
    main()
