# SuperMimic task consolidation, 2026-11-10

The supported SuperMimic task is now only
`DexGym-Wuji-Hand2-Cube-Reorientation-SuperMimic-Direct-v0`, implemented in
`source/dexgym/tasks/direct/hand2_cube_supermimic/` and trained by
`source/dexgym/algorithms/supermimic/dex_reference/`.
Independent PPO tasks, including Hand21 TeleDexter, remain registered.

The cleanup removes the early Marvin Wave/Reach/SE3/teleoperation experiments,
the older object-free Hand21 SuperMimic task, legacy Core and M4 training engines,
their configurations, forwarding modules, experiment scripts, tests, reports,
reference motions and four published Marvin checkpoints. Their previous contents
are recoverable from Git at parent commit `fa69b1988e5f3f0ee7af3ab8d463182e5fc98852`.
Shared Marvin/Wuji robot assets remain because the original right-hand PPO task
still uses them. Generic human-motion and GRAB preprocessing are retained.
External training artifacts and the completed Cube model/logs were not deleted.

Model Trust Gate, progressive 1/8/32 admission and terminal Critic bootstrap
weight warmup existed in the removed historical code. They were not used by the
completed 4096-environment, 8000-iteration Cube run. The retained recipe has
64 initial collection steps, 500 WM-only iterations, WM8 and fixed Actor32;
ordinary survival-weighted terminal bootstrap and TD(lambda) remain intact.
The pinned reference's failure-head training schedule is also unchanged.

The Cube task, Cube assets, reference engine and Hand21 PPO Python files are
byte-for-byte unchanged by this cleanup. In the older right-hand environment,
only an unconfigured SuperMimic-only reward/FK branch and obsolete import aliases
were removed; the PPO reward branch is retained. The host launchers are renamed
from `hand_s0_host.sh` / `hand_s0_runtime.py` to `host.sh` / `runtime.py`.
Cached binary overlay paths on this Ubuntu host keep their historical directory
names; those directories supply libraries, not retired task implementations.

Validation on the Ubuntu host:

- `bash scripts/supermimic/host.sh --pytest tests -q -rs`: **64 passed, 8 skipped**.
  Skips require external robot-source settings or local GRAB/MANO fixtures.
- [Native contract](cube_reorient_validation/cleanup_native.json): 16 environments,
  240 control steps, FK maximum error 6.59e-7 m; pre-reset capture, partial reset,
  randomization and push observations passed.
- [Reference algorithm parity](cube_reorient_validation/cleanup_algorithm.json):
  WM/failure-head losses and parameters, Actor/Critic losses and parameters,
  and survival all match the pinned Dex-Supermimic update with maximum error 0.
- [Bounded training smoke](cube_reorient_validation/cleanup_smoke.json): 16 environments,
  64 real collection steps, WM8 fitting and an Actor32/Critic update passed;
  all four modules have finite nonzero gradients. This is a small-batch,
  non-AMP/non-compiled smoke, not a new full-scale training result.
- A wheel built from a clean source staging directory contains Cube and PPO,
  excludes retired task/algorithm/reference packages, and loads the Cube
  configuration and sole SuperMimic registration independently of the checkout.

No large-scale training was launched. Prior Cube learning evidence and its
remaining performance gap are in the [final evaluation](cube_reorient_final_evaluation.md).
