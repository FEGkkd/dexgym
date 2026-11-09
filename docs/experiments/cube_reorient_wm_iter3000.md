# Frozen world-model diagnostic at iteration 3000

This is a diagnostic of the ongoing native scene-v2 run, not a task-success result.
The actor and WM are loaded from the same native checkpoint; no optimizer or
normalization update occurs. The run used 64 new seed-321 environments, 150
burn-in steps and 256 recorded steps. Startup/reset randomization, observation
noise and interval pushes remain enabled. Actions are deterministic.

The 512 initial windows are nonoverlapping within each environment, but are
correlated trajectory samples. Terminal states are scored; samples after a
terminal/reset boundary are excluded cumulatively. At horizon 32, 510 remain.
The WM receives recorded controls and replayed exogenous pushes, not future
physical states. This is easier than autonomous imagined control and says
nothing directly about the correctness of action gradients.

| Horizon | Seconds | WM joint RMSE (rad) | WM cube position (mm) | WM cube angle (deg) | Persistence cube angle (deg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.02 | 0.00205 | 0.203 | 0.339 | 0.451 |
| 4 | 0.08 | 0.00591 | 0.752 | 1.252 | 1.507 |
| 8 | 0.16 | 0.00685 | 1.375 | 2.307 | 2.499 |
| 16 | 0.32 | 0.00747 | 2.687 | 4.034 | 3.469 |
| 32 | 0.64 | 0.00774 | 4.005 | 7.003 | 3.495 |

At horizon 32 the WM predicts joint positions better than persistence, but its
cube position and angle errors exceed persistence. The persistence angle error
also shows that the cube has relatively little net motion in this dataset.
The diagnostic therefore does not establish an accurate model of successful
reorientation. It reveals accumulated object-prediction drift under the current
policy; it does not isolate the cause of poor policy learning.

No training change follows from this diagnostic alone. The unchanged run continues
to its scheduled 4000/8000 behavior evaluations. Machine-readable means, p95,
checkpoint hash and sample counts are in
[cube_reorient_validation/wm_iter3000_seed321.json](cube_reorient_validation/wm_iter3000_seed321.json).
The task documentation gives the reproduction command.
