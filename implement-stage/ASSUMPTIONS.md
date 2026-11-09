# Assumption ledger — Hand S0
ASK=never. Meaning-changing decisions are recorded before dependent code.

| ID | Unspecified detail | Choice | Class | Source |
|---|---|---|---|---|
| A01 | Meaning of S0 | Fixed-wrist, no-object, simulated standalone Hand 2.1 controller; hardware success is not claimed | semantic | user/context |
| A02 | Dataset split | Existing GRAB s1–s7 train, s8–s10 validation; no validation transitions for optimization or normalizers | semantic | default |
| A03 | High precision | Prespecified target: held-out fixed-clock 30 Hz mean tip error <=5 mm, P95 worst-finger error <=15 mm; >=95% settled-goal success under full current-rule curriculum. Report 5/10/15 mm coverage, delay, failures and censored targets. Do not loosen targets after observing results | semantic | default |
| A04 | Statistical coverage | At least three held-out evaluation seeds, >=256 environments per seed for subgoal tests; fixed-clock evaluation covers all eligible held-out clips, including errors before early failure | semantic | default |
| A05 | Tolerance and reference rate | Keep user's current 15 mm threshold, 30 Hz reference/control, span upper 20→40 over 25600 per-env steps, dwell 5–15, cross probability .1, fail multiplier 1.5 and cross budget300. Paper uses30 mm and quotes60Hz span units; list these adaptations explicitly | semantic | user/context |
| A06 | Reward fidelity | Preserve sparse reached-goal score/time cost; restore paper hand dense terms including thumb/fingers and intermediate knuckles from FK(q_ref), removing only object terms. Existing PPO simplified-dense reward remains a labelled historical baseline; algorithm comparison uses matched task settings | semantic | default |
| A07 | Actor information | Causal current fingertip target and deployable hand state/history; no q_ref or future reference in Actor. Critic may observe task counters/budgets. Ground-truth q_ref allowed only RSI, reward geometry and privileged diagnostic | interface | default |
| A08 | Core reuse | Extend current Core with state-dependent task transitions; preserve legacy task advance behavior. Separate control clock from reference index; cross-motion target changes do not mean physical reset | interface | default |
| A09 | Discontinuous task gradients | Real task events remain exact. Use model-based differentiable reward/termination heads fitted to true transition labels where hard threshold/dwell prevents pathwise gradients. Any learned approximation is evaluated against true rewards/events; no straight-through fake gradients or teacher action loss | semantic | default |
| A10 | Physical model | Existing official standalone USD, SI drives, implicitfast480Hz/30Hz, full collisions. Learned state initially [q20,qd20,qcmd20]; known command update analytic. Do not reuse arm-model weights | interface | user |
| A11 | Final robustness scope | Primary S0 accuracy benchmark uses nominal dynamics and fixed wrist, matching existing PPO. Paper action masking/randomization and deployment differences must be audited and explicitly separated from nominal algorithm evidence | semantic | default |
| A12 | Existing GPU job | Keep current PPO running during CPU development. Do not launch a large competing GPU job without resource planning; preserve its saved checkpoints/logs if later scheduling requires a change | interface | default |
| A13 | Artifacts and publishing | Keep data/replay/large checkpoints outside Git; commit reproducible code/configs/reports and store selected policy through existing LFS conventions. Push only new supermimic branch | interface | user |
| A14 | Hand intermediate knuckle geometry | lvl1 uses each official proximal body origin, lvl2 the middle body origin, all relative to r_wrist. Sites come from the official MJCF, checked against Newton USD; this is the explicit Wuji embodiment mapping of the paper knuckle reward | semantic | default |
| A15 | Model action coordinates | Core Actor emits latent action; apply tanh exactly once before the unchanged PPO deadzone/command map. Replay stores latent executed sample including exploration noise; Actor history stores bounded executed action. This preserves the same physical action support as PPO | interface | default |
| A16 | Dense knuckle aggregation | Apply each lvl1/lvl2 exponential kernel to the mean Euclidean knuckle error over five fingers, weighted .6/.4; appendix says knuckles are averaged across fingers but does not supply source code. Report this interpretation explicitly | semantic | default |
| A17 | Reward event model | Fit a small reach-probability head on real transitions, with exact pre-goal counters and arrival fingertip errors. Use analytic dense and sparse score multiplied by predicted reach probability during imagination; physical rewards/events remain exact. Head is owned/frozen/checkpointed with world model; no label reweighting so probabilities retain event calibration | interface | default |
| A18 | Training scale and trust | Initial run uses 256 real Newton environments, 1000-step replay, smaller hand networks, WM8 and Actor BPTT32 only. Actor starts after 200 world updates and held-out-lane 1/8/32 predictions beat zero-delta baseline with mean tip error <=1.5/4/7.5 mm. Separate lanes use training subjects; final GRAB validation remains untouched. This is an added engineering gate, not a SuperMimic paper claim | semantic | default |
| A19 | Optimizer and resume | Actor Adam betas(.9,.95), global gradient norm clip1 followed by value clip.1; world/Actor/Critic initial lr2e-4; gamma.99 lambda.95; Critic5epochs4minibatches. Cosine LR floor.03 at250000iterations. Intermediate saves use explicit fresh-physics resume, preserve model/optimizer/normalizers and curriculum counter but recollect replay | interface | default |

## Interpretation notes
A03 is an engineering acceptance target, not a paper-reported result or a promise
that every reference is feasible at that speed. If unmet, report the failure
and continue diagnosis; a weaker check does not complete the goal.
A09 is required by the actual hybrid task, not an excuse to replace it with
fixed-time motion imitation. Exact physical event labels remain authoritative.
