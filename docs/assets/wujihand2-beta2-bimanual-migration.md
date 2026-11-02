# WujiHand2 Beta 2 bimanual asset migration (2026-11-02)

## Scope and identity

The GraspHold, CapTurn30 and unified PregraspCapTurn30 DirectRLEnv configurations now load the pinned official WujiHand2 Beta 2 left and right USDs by default. Each side still has 20 finger joints and a 2 N·m native USD drive limit. The 52D joint actor, ideal pose-command wrists, dynamic bottle/cap, gravity, and selected thread backend remain task controlled. The hand asset is from `wuji-description` tag `v2026.9.11`, commit `c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95`; the loader checks both USD and complete source-tree hashes.

| Asset | USD SHA-256 | Complete asset tree SHA-256 |
| --- | --- | --- |
| Beta 2 left | `f8094cff005b5b069d55cb3f36402f98aba5e5983f583e9968be2763a2d0909b` | `c4c7ae7f784b041ed3fd138548995896f08c4b02817f65fa215789056157ec10` |
| Beta 2 right | `839fc0d0d0c06a2d7d463851b087a2dfe69bc883bc240ead47230536cbd54b20` | `7b7d4d56684407d980ba689b8ea84daa94127b9fa013dbac294776f8b8976a95` |

Beta 2 adds five `*_tip_sensor_frame` bodies per hand. The actor still reads the official `*_tip` rigid-body positions. The privileged critic discards the extra sensor-frame force slots, preserving its 421D checkpoint interface and the 277D actor interface. No tactile measurements enter the actor.

## Reset admission and limitations

All four historical Beta 1 stable grasp rows were treated as unvalidated Beta 2 pose proposals and replayed for 1 s with gravity and no support. **Zero of four passed**: right-cap contact ratios were approximately 0.23–0.50, below the old 0.8 criterion. These rows were not relabeled as a Beta 2 stable bank. The Beta 2 defaults for the older GraspHold and CapTurn30 tasks therefore use pregrasp-only resets (`stable_reset_probability=0`). Requesting Beta 2 stable resets without a separate physically validated bank fails explicitly.

The Beta 1 Phase 1B pregrasp rows were copied only as *unqualified geometric proposals*. Their Beta 1 PhysX evidence was removed. Native Beta 2 PGS Mimic probes then screened 51 original candidates and 32, 128 and 256 local hand-pose perturbations around the one reachable source. The four screens admitted 1, 1, 2 and 8 rows respectively. Admission required thread geometry and solver consistency, bounded force and separation, no excessive cross-object contact, active bilateral contact within 0.3 s, and no bilateral contact in the zero-action probe. The resulting [Beta 2 Reset bank](beta2_phase1b_pregrasp_bank.json) has 12 Level A near-contact rows; its SHA-256 is `6811aa8bd8f58454ddd4f1f1ae780c10a4cdf1532b03fdb353a830041a2a9d63`. Their active probe made first bilateral contact at policy step 1, maximum correct-object force was 47.71 N, and maximum Mimic constraint error was 6.11 μm.

The [frozen split](beta2_phase1b_pregrasp_split.json) reserves 9 local hand-shape seeds for training and 3 for evaluation. All 12 rows descend from **one relative tripod template and one natural source seed (1625)**. This supports only a within-template test. Weak-contact Level B, original-distance natural pregrasps, Beta 2 stable grasp, actual PPO success, and cross-family generalization remain unverified. The earlier PGS Mimic angular-velocity discrepancy and contact signed-separation interpretation also remain open; the asset change does not resolve them.

## Reproduce

Set `WUJI_DESCRIPTION_ROOT` to the pinned official checkout and use the Isaac Lab Python environment. The exact generation scripts and seeds are recorded below. Raw candidates, native probe reports, admitted intermediate rows, USD/contact smoke logs and stable-grasp replay traces are saved in the ignored local directory `logs/beta2_migration_2026-11-02/`. The frozen manifest records each input SHA.

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
export PYTHONPATH=/home/ubuntu/dexgym/source:/home/ubuntu/dexgym
"$ISAACLAB_PYTHON" scripts/environments/prepare_beta2_reset_candidates.py \
  --source docs/assets/phase1b_pregrasp_bank.json \
  --output /tmp/dexgym_beta2_phase1b_candidates.json
"$ISAACLAB_PYTHON" scripts/environments/generate_beta2_local_pregrasps.py \
  --source /tmp/dexgym_beta2_phase1b_candidates.json --parent-seed 66003 \
  --count 32 --seed 9101 --output /tmp/dexgym_beta2_local32_candidates.json
"$ISAACLAB_PYTHON" scripts/environments/generate_beta2_local_pregrasps.py \
  --source /tmp/dexgym_beta2_phase1b_candidates.json --parent-seed 66003 \
  --count 128 --seed 9102 --output /tmp/dexgym_beta2_local128_candidates.json
"$ISAACLAB_PYTHON" scripts/environments/generate_beta2_local_pregrasps.py \
  --source /tmp/dexgym_beta2_phase1b_candidates.json --parent-seed 66003 \
  --count 256 --seed 9103 --wrist-limit-m 0.0002 --joint-limit-rad 0.005 \
  --output /tmp/dexgym_beta2_local256_candidates.json
```

Run `scripts/environments/validate_beta2_pregrasp_pgs.py` on each candidate file with `--count` 51, 32, 128 or 256, `--steps 15 --probe-fingers stable --probe-order active_first`, writing paired `--output` and `--admitted-bank` files. Then pass all four candidate/report/admitted triples to `scripts/environments/freeze_beta2_pregrasp_bank.py` to regenerate the committed bank and split. The native validator uses the current 500 Hz PGS Mimic solver; its scripted probe is only a Reset feasibility experiment, not an action schedule in the RL task. The stable Beta 1 joint targets in that probe are a feasibility reference; no Beta 1 hand asset or Reset certificate is loaded into the Beta 2 task.

```bash
for case in phase1b:51 local32:32 local128:128 local256:256; do
  name=${case%%:*}
  count=${case##*:}
  input=/tmp/dexgym_beta2_${name}_candidates.json
  if [ "$name" = phase1b ]; then input=/tmp/dexgym_beta2_phase1b_candidates.json; fi
  scripts/run_isaaclab.sh scripts/environments/validate_beta2_pregrasp_pgs.py \
    --native-bank "$input" --count "$count" --steps 15 \
    --probe-fingers stable --probe-order active_first \
    --output "/tmp/dexgym_beta2_${name}_report.json" \
    --admitted-bank "/tmp/dexgym_beta2_${name}_admitted.json" --viz none
done
"$ISAACLAB_PYTHON" scripts/environments/freeze_beta2_pregrasp_bank.py \
  --candidate /tmp/dexgym_beta2_phase1b_candidates.json \
  --report /tmp/dexgym_beta2_phase1b_report.json \
  --admitted /tmp/dexgym_beta2_phase1b_admitted.json \
  --candidate /tmp/dexgym_beta2_local32_candidates.json \
  --report /tmp/dexgym_beta2_local32_report.json \
  --admitted /tmp/dexgym_beta2_local32_admitted.json \
  --candidate /tmp/dexgym_beta2_local128_candidates.json \
  --report /tmp/dexgym_beta2_local128_report.json \
  --admitted /tmp/dexgym_beta2_local128_admitted.json \
  --candidate /tmp/dexgym_beta2_local256_candidates.json \
  --report /tmp/dexgym_beta2_local256_report.json \
  --admitted /tmp/dexgym_beta2_local256_admitted.json \
  --bank docs/assets/beta2_phase1b_pregrasp_bank.json \
  --manifest docs/assets/beta2_phase1b_pregrasp_split.json
```

Native smoke commands:

```bash
scripts/run_isaaclab.sh scripts/environments/smoke_phase1b_env.py --hand-variant beta2 --num-envs 16 --steps 10 --viz none
scripts/run_isaaclab.sh scripts/environments/smoke_phase1b_env.py --hand-variant beta2 --evaluation --num-envs 2 --steps 5 --viz none
scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_rl.py --hand-variant beta2 --task grasp_hold --num-envs 1 --steps 3 --viz none
scripts/run_isaaclab.sh scripts/environments/smoke_bimanual_rl.py --hand-variant beta2 --task cap_turn_30 --num-envs 1 --steps 3 --viz none
```

Historical checkpoint replay is explicit: pass `--hand_variant beta1` to `rsl_rl/base/play.py` or `rsl_rl/base/train.py`, or `--hand-variant beta1` to the evaluation and smoke scripts. This restores the original Beta 1 stable Reset proportions and Phase 1B bank/split. A Beta 1 checkpoint has not been validated as a Beta 2 policy; changing the asset variant changes its physical test condition.

## Verification on this migration

- Official asset static audit and native contact probe: passed. One-env Beta 2 standalone scene ran 20 steps with finite state; contact probe measured left/right active contact and zero wrong-side contact. These are asset probes, not task success.
- Beta 2 unified task: 1 and 16 training-mode environments, 2 held-out evaluation-mode environments, finite Reset/Step and 277D/421D observation checks passed. In the 16-env zero-action smoke, right contact rate was 0.0625 and maximum opening was 0.0421 rad, so no CapTurn30 learning is implied.
- Beta 2 GraspHold and CapTurn30: one-environment Reset/Step passed for 3 zero-action steps each. Historical Beta 1 CapTurn30 and unified-task explicit-variant smoke checks passed.
- Focused unit tests: variant routing, Beta 2 fingertip frames, force-schema width, candidate certificate removal, local perturbation reproducibility and split checks passed.

The existing Phase 1B training objective was paused for this asset migration. PPO training and 200-episode independent evaluation must be rerun with Beta 2 before any end-to-end success claim.
