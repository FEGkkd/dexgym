# Phase 1 hard-pitch PPO and contact diagnosis — 2026-11-01

This is an experimental research round on `codex/bimanual-bottlecap-rl-v0`, based
on `8987c9c`. The hard-pitch backend code was committed at `42c45f8`. The
official Beta 1 USD source is `wuji-description` commit
`c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`. The engineering bottle and
resistance values are examples, not measurements of a real closure.

## Measured training and independent evaluation

| Item | Result |
|---|---|
| Task | `GraspHold`, `thread_backend=mimic`, natural geometric pregrasp only |
| Runtime | Isaac Sim 6.0.1, Isaac Lab 3.0, RTX 5090 32 GiB |
| Seed, environments, updates | 41, 1024, 25 |
| Physics / policy rate | 500 / 50 Hz (`dt=0.002`, decimation 10) |
| Transitions, wall time, last throughput | 819,200; about 122.7 s; 6,678 environment steps/s |
| Last sampled PPO iteration | Mean reward −22.64; left contact rate 0.3387; right tripod rate 0.0002; grasp rate 0 |
| PPO health | Approximate KL 0.0144; clip fraction 0.2066; explained variance 0.7411; value loss 25.0584; grouped action std 0.2487–0.2506 |
| Action bounds | Last action clip fraction 0.0161, joint clip fraction 0.0737 |
| Deterministic held-out evaluation | **0/200** pregrasp grasp successes; final left and right contact rates 0; no breakaway |
| Zero-action counterfactual | **0/200** pregrasp grasp successes; no correct contacts |
| Random-action counterfactual | **0/200** pregrasp grasp successes; rare transient correct contact |
| CapTurn30 zero-action counterfactual | **0/200** pregrasp opening successes; maximum opening ≈0 rad, no breakaway |

The held-out evaluation uses the 8 reserved pregrasp bank rows, reused over
200 episodes; it does not establish cross-family generalization. The sampled
left-contact increase during PPO disappeared under deterministic action, so
neither the reward nor the checkpoint establishes a learned grasp. The model
is in ignored
`logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_11-43-08_mimic_pregrasp_seed41_25iter/model_24.pt`.
The complete stdout and evaluation JSON are in ignored `artifacts/phase1/`
under `mimic_grasp_seed41_25iter_*` and `mimic_capturn_zero_heldout_200.*`.

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
scripts/run_isaaclab.sh scripts/rsl_rl/train.py \
  --task DexterousHands-WujiHand2-Bimanual-GraspHold-Direct-v0 \
  --num_envs 1024 --max_iterations 25 --stable_reset_probability 0 \
  --thread_backend mimic --seed 41 --run_name mimic_pregrasp_seed41_25iter --headless
scripts/run_isaaclab.sh scripts/environments/evaluate_bimanual_rl.py \
  --task grasp_hold --policy learned --source pregrasp \
  --thread-backend mimic --held-out --num-envs 64 --episodes 200 \
  --seed 1001 --headless --checkpoint \
  logs/rsl_rl/dexteroushands_bimanual_bottle_phase1/2026-11-01_11-43-08_mimic_pregrasp_seed41_25iter/model_24.pt
```

## Targeted physical contact diagnosis

`scripts/environments/probe_pregrasp_contact_jacobian.py` starts from bank
row 3 and finds one collision-gap-reducing joint direction per finger using
the official MJCF geometry. It then sends bounded **scripted** 52D actions
through the real DirectRLEnv. The wrists remain pose-commanded, the bottle
and cap are free, gravity and finger drives are on, and the screw is hard
pitch. These runs are diagnosis, **not policy evaluation or validated Reset
bank entries**. `--max-bottle-displacement-m` stops an unsafe candidate.

| Controlled ablation | Measured result |
|---|---|
| Contact-gated closing, fixed right wrist, 1 s | All five left digits contacted the bottle for 100% of the last 0.5 s; bottle displacement 19.1 mm. Right thumb and middle contacted, right index did not. |
| Right wrist commanded down 10 mm plus right index MCP flex, 1 s | Left five digits 100% last 0.5 s; right thumb and index 100%, middle 0%; bottle maximum displacement 21.3 mm. |
| Add unbounded right middle abduction, 1 s | Left five digits 100%; right thumb and middle 100%, index 0%. At the final frame, index–middle distal self-contact was 60.9 N, separation −0.065 mm, and index MCP drive was at its 2 N m limit. |
| Cap right-middle PIP target at 1.40 rad, no middle abduction, 1 s | Right thumb and index 100% last 0.5 s, middle 0%; no final index–middle self-contact. The middle tip was 6.6 mm radially outside the cap at the final frame. |
| Also cap middle abduction target at −0.10 rad, 1 s | Right thumb 100%, index 96%, middle 0% last 0.5 s; no final self-contact; bottle maximum displacement 19.3 mm. |
| Follow measured cap translation with bounded right wrist commands | Bottle moved 227.5 mm in 1 s, so this naive Oracle tracking experiment was rejected. It is not a valid grasp method. |

The 70 mm diameter bottle is reachable by multiple left finger collision
bodies in this scripted fixture, but this does not prove left-only hold or
Reset replay. The right tripod is the measured blocker: one finger loses the
cap when its PIP curls beyond the sidewall, while stronger middle abduction
blocks the index through finger–finger contact and drive saturation. The
right wrist/cap motion also changes the geometry while the left hand closes.
Continuing the same PPO with essentially no right tripod experience would
consume transitions without addressing that physical bottleneck.

The default batched contact monitor reports the correct object and hand body,
but **does not classify the cap contact point as sidewall or top/edge**.
`--contact-points` enables an expensive one-environment PhysX manifold
inspection. In the 15-step variant, the last-frame cap-local sidewall points
gave right-thumb 4.34 N and right-middle 0.43 N, with no right-index point;
two index–middle self-contact points totalled 51.45 N. The sidewall rule in
this probe requires `|z_cap| ≤ cap_height/2 − 1 mm` and radial position within
4 mm of the example cap radius. It is a diagnostic rule, not a calibrated
reward gate. A longer point-query run emitted PhysX's incomplete-contact-data
warning for the current `max_contact_data_count=8` view and did not produce
an output JSON. Therefore a complete, scalable sidewall contact filter is
**not verified**; this point query must not be silently installed in PPO.

```bash
scripts/run_isaaclab.sh scripts/environments/probe_pregrasp_contact_jacobian.py \
  --sample-index 3 --steps 50 --maintain-contact \
  --right-wrist-down-steps 5 --right-index-mcp-flex \
  --right-middle-pip-target-max 1.40 --right-middle-mcp-abd \
  --right-middle-abd-target-min=-0.10 --headless \
  --output artifacts/phase1/pregrasp_jacobian_middle_bounded_1s.json
scripts/run_isaaclab.sh scripts/environments/probe_pregrasp_contact_jacobian.py \
  --sample-index 3 --steps 15 --maintain-contact --contact-points \
  --right-wrist-down-steps 5 --right-index-mcp-flex \
  --right-middle-mcp-abd --headless \
  --output artifacts/phase1/pregrasp_jacobian_sidewall_probe.json
```

## Gate status after this round

| Gate | Status | Evidence / remaining work |
|---|---|---|
| 1024 environment Reset, contact, thread and PPO update | **Pass for experimental Mimic pregrasp mode** | 100 selective Resets with zero peer change, finite 1024-env PPO, hard-pitch error <0.1 µm in smoke. Stable-bank curriculum remains unavailable. |
| Physical contact-driven right tripod | **Fail** | No sustained right thumb–index–middle cap contact in the tested scripted fixture; the PPO had right tripod rate 0.0002. |
| GraspHold held-out deterministic ≥60% | **Fail** | 0/200 pregrasp episodes. |
| CapTurn30 stable-start ≥30% | **Not Tested on Mimic** | Existing four stable rows violate the hard-pitch Reset geometry by ≈1.2–1.26 mm. |
| CapTurn30 pregrasp ≥10% | **Not Trained on Mimic** | Zero-action control 0/200; no contact-qualified curriculum. |
| Sidewall-specific contact/reward | **Not Verified** | Single-env manifold points work briefly; current point buffer can overflow and batched point filtering has not been benchmarked. |
| Independent cross-family evaluation | **Not Tested** | Four old stable rows form one grasp family; no compatible independent stable family exists. |

The focused Mimic, reward, RL contract and contact-math regression command
returned **20 passed**. The earlier full repository suite returned **161
passed, 1 skipped, 2 failed**; those two failures are in user-owned Allegro
asset/MANO fixture tests and are not reported as Phase 1 passes. The
diagnostic script passed Python compilation and its 15-step point probe and
50-step force-only probes produced finite JSON. One longer point probe
reported an incomplete PhysX contact buffer and produced no JSON; that run
is a failed diagnostic, not a passing test.

The next smallest useful experiment is a constrained right-wrist and
index/middle-abduction search with explicit self-collision and actual cap
sidewall contact-point constraints. A candidate must then be replayed from
Reset with the free bottle and cap before it can become a stable Mimic bank
row. The contact monitor needs a tested batched sidewall classifier before
CapTurn reward or success can use its body-only force as proof of a correct
cap grip. This round intentionally did not start further PPO training.
