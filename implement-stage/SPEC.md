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

Outputs: reproducible commands/configs, CPU and Newton checks, trained actor,
independent held-out metrics, source-to-implementation fidelity table, complete
reproduction report and commits pushed to `origin/supermimic`.

Success requires the full training and accuracy gates in ASSUMPTIONS.md, not
merely a smoke test or a runnable framework. Hardware deployment, object
interaction and tactile sensing are outside this requested S0 stage.
