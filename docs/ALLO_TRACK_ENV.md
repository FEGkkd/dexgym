# `allo_track` Environment

The migration's isolated Conda environment is named `allo_track` and is
located at `/home/ubuntu/miniconda3/envs/allo_track`. It contains Python 3.12,
Torch 2.10.0+cu128, NumPy 1.26.4, pytest, Gymnasium, RSL-RL 5.0.1, and
TensorBoard.

Activate it from the repository root:

```bash
source scripts/activate_allo_track.sh
```

Run simulator-independent tests with ROS pytest plugins disabled:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The environment specification is in `environment-allo-track.yml`. Isaac Sim
and Isaac Lab remain external runtime dependencies; this environment owns the
task code, contract tests, PPO library, and TensorBoard tooling without
reusing the `isaacsim-6.0.1` Conda environment.
