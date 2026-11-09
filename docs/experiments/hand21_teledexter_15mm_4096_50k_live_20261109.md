# Hand 2.1 TELEDEXTER 15 mm PPO run (started 2026-11-09)

The active `DexGym-Wuji-Hand21-TeleDexter-Direct-v0` task now requires all five
fingertips to remain within **15 mm** for its sampled 5–15 policy-step hold.
The clean diagnostic task retains its historical 20 mm setting, and the
20 mm replay reports remain historical baselines. All other task rules and
the Tanh Gaussian policy configuration are unchanged: 4,096 environments,
30 Hz actions, latent initial Std 1.0, Entropy coefficient 0.005, and the
deadzone residual joint-target action.

The run uses seed 42, a **50,000-iteration maximum**, and an independent
experiment directory:

`logs/rsl_rl/dexgym_wuji_hand21_teledexter_15mm_tanh_std1_entropy005/2026-11-09_13-10-47_hand21_15mm_4096_50k_s42/`

It was launched through `scripts/train_hand21_action_audit.py`, which leaves
PPO updates unchanged and also writes `action_audit_exact_span.csv`. TensorBoard
serves the experiment directory on port **6006**. The host's LAN address at
launch was `192.168.1.165`, so a client on that LAN can open
`http://192.168.1.165:6006/`. Training and TensorBoard run in tmux sessions
`hand21_15mm_50k` and `hand21_15mm_tensorboard`. The raw training log is local
at `artifacts/hand21/ppo_15mm_4096_50k_s42/train.log` and is not committed.

Before launch, the 15 mm Newton smoke check completed 154 goals and 24
cross-trajectory switches in 16 environments over 120 policy steps. The
TensorBoard endpoint returned HTTP 200 and listed the new run after PPO began.
This note records a live run, not a final training outcome.
