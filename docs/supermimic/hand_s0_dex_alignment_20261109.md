# Hand S0 v3: Dex-Supermimic alignment (2026-11-09)

> Historical report. Current default is v4; see [v4 alignment and the unresolved Actor deadzone](hand_s0_dex_v4_alignment_20261109.md). Do not infer current training readiness from this older report.

The user requested matching collection/update ratio, separate LR decay, network
sizes, Actor initialization, model/policy batches, initial collection and a fixed
500-iteration world-model-only phase. Reference:
[CumulusAlpha/Dex-Supermimic at 2becb738](https://github.com/CumulusAlpha/Dex-Supermimic/tree/2becb7380fed86171056f9bf4338b2a6a0c14b01).
This implements those selected changes, not every mechanism in that repository.

## Effective default

| Item | Hand S0 v3 |
| --- | --- |
| Physical task | Independent standalone Hand21, 30 Hz control, fixed wrist, no objects/tactile observations |
| Environment count | 256, retained from the existing Hand setup |
| World network | 2048/2048/2048, LayerNorm + SiLU, zero delta-output initialization |
| Actor | Direct ELU MLP 512/256/128; small orthogonal output weights, gain 0.01, zero bias |
| Actor input | Ten 125D observation frames + current 15D goal, concatenated; no future goals or separate goal encoder |
| Actor action | Pre-tanh network output; existing hand adapter applies tanh once, deadzone and accumulated target control |
| Critic | ELU MLP 512/512/256/128, existing 162D privileged features |
| Data ratio | UTD 0.4; collect 2,3 vectorized physics steps alternately |
| Initial collection | 64 vectorized physics steps, without optimizer updates |
| WM-only period | First 500 iterations; Actor/Critic begin on iteration 501 |
| WM per iteration | 10 optimizer steps, each sampling 2048 windows, window length 8 |
| Actor | 1024 imagined trajectories, horizon 32, one optimizer step per active iteration |
| Critic | TD(lambda), 5 epochs x 4 shuffled minibatches |
| LR | 2e-4; cosine over 250000 iterations, then constant 0.03 floor |
| Default total budget | 250000 iterations; independent of LR decay length and CLI segment length |

New task-local modules are `supermimic_networks.py` and `supermimic_schedule.py`.
The PPO task and generic Core implementation are unchanged. Existing Hand running
normalizers, reward event head, termination head and physical task rules remain.
No failure replay mixture, contact predictor, soft task counters, failure-probability
cap, or model-error Actor gate was imported from the reference repository.

The 500-round delay is iteration-based: it corresponds to 5000 WM optimizer steps
at the default 10 batches/round. Initial collection is separate and does not count
as any of these 500 iterations. Fractional collection is calculated from absolute
iteration, so resume preserves its phase without a floating-point accumulator.

## Compatibility

The v3 network architecture cannot directly resume old v1/v2 weights. Start v3
from scratch. `agents/hand_s0_v2.json` preserves the old architecture for explicit
legacy training/evaluation. V1 JSON still migrates only to v2. V3 checkpoint
configuration strictly binds UTD, warmup, precollection, and LR schedule; mismatches
are rejected. On fresh-physics resume, 64 new collection steps rebuild replay,
while the saved iteration determines WM-only status and LR. Curriculum progression
uses total recorded physical transitions, including collection-only steps.

## Verification and limits

- 293 regression tests passed in 19.79 seconds; 35 existing torch.jit warnings.
- Tests verified the exact 2/3 collection sequence, 25000 steps across 10000 rounds,
  LR floor beyond 250000, current-goal gradients, ELU architecture, and small
  orthogonal initialization.
- A toy boundary fixture starts at iteration 499, verifies iteration 500 leaves
  Actor/Critic unchanged, saves/restores, and verifies iteration 501 updates them.
  This is an explicit boundary fixture, not 500 completed physics-training rounds.
- Default physical run: 256 environments, full networks/batches, 64-step prefill
  and 2 iterations; 17664 real transitions, 20 WM updates, zero Actor/Critic updates.
- Diagnostic physical run changes only `wm_only_iters` from 500 to 1. Two iterations
  perform 20 WM, 1 Actor and 20 Critic updates. Its resume adds 64 collection steps
  and one iteration: 10 further WM, 1 Actor and 20 Critic updates. All nonfinite
  counters remain zero, and the Actor uses BPTT32 with batch 1024.
- The resulting checkpoint loads through the strict evaluation entry point and
  completes 16 environments x 32 steps on validation subjects. This checks execution,
  not convergence or tracking quality.

All physical commands exited 0 on the Ubuntu RTX 5090 host. No large-scale
training was launched. Machine-readable evidence is in
[hand_s0_dex_alignment_20261109.json](../experiments/hand_s0_dex_alignment_20261109.json).
Raw logs/checkpoints remain at `/home/ubuntu/dexgym/artifacts/hand_s0/dex_align/`.
The formal default remains 500 WM-only rounds; the diagnostic override is not
part of the packaged default.
