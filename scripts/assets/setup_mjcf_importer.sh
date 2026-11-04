#!/usr/bin/env bash
set -euo pipefail

# Isaac Lab 3.0's official standalone MJCF importer has an isolated-wheel
# usd-exchange pin that requires the project's published uv override set.
# Run after `uv sync --locked --group dev` from the DexGym repository root.
uv pip install --python .venv/bin/python 'isaaclab[importers]==3.0.0' \
  --overrides 'https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt' \
  --index 'https://pypi.nvidia.com' \
  --torch-backend cu130 \
  --index-strategy unsafe-best-match
