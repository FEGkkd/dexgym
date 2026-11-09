# Hand S0 environment on the Ubuntu host

Working directory: `/home/ubuntu/dexgym/worktrees/supermimic`.
The existing PPO runs in a different worktree and is not modified.

Run verbatim from this directory:

```bash
bash scripts/supermimic/hand_s0_host.sh --witness
bash scripts/supermimic/hand_s0_host.sh --pytest -q tests/supermimic/test_hand_s0_core.py
```

Expected: `S0_KERNEL_WITNESS (8, 8) NVIDIA GeForce RTX 5090`, followed by normal
exit0; nine hand Core regression tests pass. Seed42 output was8.112417221069336
for the CUDA8x8 product sum. This is a kernel witness, not a policy result.

The launcher uses Isaac Sim6.0.1's Python3.12 and torch2.11.0+cu130. It appends
the shared venv only after torch import, avoiding that venv's reproducible
SIGBUS135 at process exit. The original venv and running PPO are untouched.
It uses the existing process-local numpy/IsaacLab/scipy/omniclient/USD overlays
under /tmp and the Newton1.6.1 / MuJoCo3.12 / Warp1.17 runtime overlay in
~/dexgym_data. These paths must exist; they are not silently installed.

Executor validation: CUDA witness exit0; targeted25 tests pass; complete
tests/supermimic suite230 passed in18.99s after adding repository root to
the runtime sys.path so tests importing other test modules can resolve.
Fresh independent agent-following-document validation passed: witness exit0 and 9 Core tests in0.24s, separate SSH session. Newton simulation also completed the F2 smoke with exit0.
