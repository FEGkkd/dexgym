# Hand 2.1 TELEDEXTER 15 mm PPO run (2026-11-09)

The active `DexGym-Wuji-Hand21-TeleDexter-Direct-v0` task now requires all five
fingertips to remain within **15 mm** for its sampled 5–15 policy-step hold.
The clean diagnostic task retains its historical 20 mm setting, and the
20 mm replay reports remain historical baselines. All other task rules and
the Tanh Gaussian policy configuration are unchanged: 4,096 environments,
30 Hz actions, latent initial Std 1.0, Entropy coefficient 0.005, and the
deadzone residual joint-target action.

The run used seed 42, a **50,000-iteration maximum**, and an independent
experiment directory:

`logs/rsl_rl/dexgym_wuji_hand21_teledexter_15mm_tanh_std1_entropy005/2026-11-09_13-10-47_hand21_15mm_4096_50k_s42/`

It was launched through `scripts/train_hand21_action_audit.py`, which leaves
PPO updates unchanged and also writes `action_audit_exact_span.csv`. TensorBoard
continues to serve the experiment directory on port **6006**. The host's LAN address at
launch was `192.168.1.165`, so a client on that LAN can open
`http://192.168.1.165:6006/`. TensorBoard runs in tmux session
`hand21_15mm_tensorboard`; the training session has exited. The raw training log is local
at `artifacts/hand21/ppo_15mm_4096_50k_s42/train.log` and is not committed.

Before launch, the 15 mm Newton smoke check completed 154 goals and 24
cross-trajectory switches in 16 environments over 120 policy steps. The
TensorBoard endpoint returned HTTP 200 and listed the new run after PPO began.

Training was stopped at **iteration 3812** on the user's request. The latest
saved checkpoint is `model_3800.pt`; checkpoints were saved every 50
iterations. The 200-iteration mean reward rose from about 4,115 at iterations
1900–2099 to about 4,169 at 3500–3699, but episode length and mean goals
reached stayed in roughly the same range (about 750–800 policy steps and
25–26 goals). This is a fluctuating training plateau, not evidence of a
verified optimum. The checkpoint has not yet been evaluated on a separate
held-out rollout under the 15 mm rules, so no validation completion rate is
claimed in this initial run record. A later
[checkpoint diagnosis](hand21_15mm_training_diagnosis_20261109.md) compares
held-out target throughput and failure counts for three checkpoints.
