# SuperMimic algorithm source comparison

## Sources actually inspected

| Source | Version | License | Mechanism inspected |
| --- | --- | --- | --- |
| [SuperMimic paper](https://arxiv.org/abs/2610.28480) | arXiv `2610.28480`, PDF retrieved for this work | Paper | Eight-step autoregressive world model, 32-step differentiable imagination, survival weighted policy return, TD(λ) critic, soft termination |
| [TrackerLab SuperMimic](https://github.com/Renforce-Dynamics/trackerlab-supermimic) | `e2db2dea0e81c91ebca815c895b60ee9aa58237a` | MIT, including SuperMimic subdirectory | `runner.py`, `networks.py`, `buffer.py`, tracking MDP, motion library, kinematics and release validation |
| [supermimic-mjlab](https://github.com/YHFone/supermimic-mjlab) | `9aab0a588c6941bf9862a2d444b711fd73e2b832` | Apache-2.0 | `algo/trainer.py`, `buffer.py`, `reward.py`, `state.py` |

The paper specifies the algorithmic equations. TrackerLab and mjlab are independent engineering implementations; neither is presented as the paper authors' official training code.

## Matching mathematical core

`DeltaWorldModel.autoregressive_loss` unrolls eight transitions; all state and termination losses are divided by the number of valid real elements. `imagined_rollout` unrolls up to 32 steps with `c[t+1]=c[t](1-d[t+1])`. The Actor return is the survival weighted reward sum plus `γ^H c[H] V(s_H)`. The Critic target uses `G_t = r_t + γ(1-d_(t+1))[(1-λ)V_(t+1)+λG_(t+1)]`, with γ=0.99 and λ=0.95. A hand-calculated return test, finite difference gradient tests, and model freezing tests live under `tests/supermimic`.

TrackerLab uses analytic soft termination for its tracking task; mjlab trains a learned termination predictor. M4.2 supports both. The learned label comes from Newton's hard termination of the actual pre-reset arrival state and its **current** arrival reference. The old future-goal offset at `next_frame` is corrected. Analytic mode shares the hard threshold configuration for TCP error, arm joint limits, and arm speed.

## Intentional Marvin differences

The fixed-base Marvin robot exposes 42D arm state and 14D arm actions while retaining all 54 joints in Newton. Its known position-servo command update remains analytic within the world model; only physical `Δq` and `Δqd` are learned. Differentiable Marvin FK computes the six links and two TCPs. The Reward has both a paper-style multi-link profile and the M4.1 TCP priority profile. The existing Legacy Actor has no Actor/Goal/Critic normalization buffers and no goal-latent `tanh`; compatibility mode preserves those outputs exactly.

The M4.2 startup schedule admits 1, 8, then 32 step BPTT as held-out actor-data model accuracy permits. One-step startup temporarily omits terminal critic bootstrap to avoid immediate exploitation of an untrained value function. This is an engineering warmup choice, not a claim that the paper prescribes it. New v3 action scaling is explicit and versioned because the initial robot-wide v2 scale caused real Newton arm-speed failures. No physical safety threshold, controller gain, or robot asset was changed.

## Remaining parity limits

The paper's broad locomotion and task-conditioning results have not been reproduced for Marvin. M4.2 currently verifies a second procedural six-link tracking reference, not a different observation or object-contact MDP. A split task termination head was evaluated on independently collected TCP-boundary failures, but arm-speed, limit, and contact failure coverage remains unproven. The safe v3 Actor data itself contains no positive terminal labels. BF16 and `torch.compile` are not correctness dependencies. Results and failed gates are reported in `m42_validation.md`.
