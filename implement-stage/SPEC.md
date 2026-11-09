# Hand-only S0 reproduction

Base commit: `cf2cddeeddaee7551313ec03a88ee1c45c0a9bc5`.

Extend the existing `right_hand_tracking` task and SuperMimic Core in place on
branch `supermimic`, in `~/dexgym/worktrees/supermimic`. Use the existing official
standalone 20-DoF Wuji Hand 2.1 USD, full self-collisions and GRAB library.
Do not substitute the Marvin arm or a kinematic simulator for training.

Preserve TeleDexter consecutive subgoals, random dwell, failure budgets, RSI
and cross-trajectory switching. Train a causal, current-goal policy by fitting
real-simulator transitions with WM8 and optimizing BPTT32 imagined returns
with survival weighting and TD(lambda). No PPO updates, behavior cloning,
reference-joint action supervision, future-goal input or expert warm start in
the S0 model-based run. Existing PPO remains the baseline.

Outputs for the current phase: reproducible commands/configs, CPU and Newton
checks, a bounded actual-training/checkpoint/evaluation demonstration,
source-to-implementation fidelity table, reproduction report and commits pushed
to `origin/supermimic`.

User scope update, 2026-11-09: this phase must establish correctness of the
architecture, task definition and model-based RL training method, and readiness
to begin larger training later. It does NOT require a converged/high-accuracy
policy. Do not continue large training to chase policy precision in this phase.
Small physical tests and actual optimizer updates remain necessary evidence;
a runnable stub does not pass. Hardware deployment, objects and tactile sensing
remain outside S0. The original user PPO process must remain untouched.
