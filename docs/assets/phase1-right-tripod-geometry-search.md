# Phase 1 right-cap tripod geometry and dynamic replay — 2026-11-01

This experiment follows [the first hard-pitch PPO round](phase1-mimic-ppo-round.md).
It uses the official WujiHand2 Beta 1 MJCF only for **offline candidate
screening**, then tests candidates in the native USD Isaac Sim 6.0.1 / Isaac
Lab 3.0 PhysX scene. The 70 mm example bottle and free threaded cap remain
dynamic under gravity. Every wrist movement in the PhysX probe uses the
bounded `pose_command_wrist` trajectory; finger targets use normal 2 N m
limited articulation drives. These scripted probes are not learned policies.

## Why the initial three-body-contact result was insufficient

For pregrasp bank row 3, separate index/middle MCP abduction to about −0.2 rad
each, right-wrist descent of 10 mm and contact-gated finger closing produced
one second of free-bottle simulation. During the **last 0.5 s**, left thumb,
index, middle, ring and pinky each contacted the bottle on 100% of sampled
policy steps. Right thumb and index each contacted the cap on 100%, and right
middle on 96%. Maximum bottle displacement was 19.9 mm. The final frame had
one right index–thumb self-contact at 5.41 N with −0.071 mm separation.

An independent `max_contact_data_count=256` single-environment PhysX contact
view resolved the actual manifold positions. The right thumb and middle had
sidewall contact on 100% and 96% of the last 0.5 s, respectively. The right
index had **zero** sidewall steps: its distal contact point stayed at
approximately −12.5 mm on the cap-local axis, the bottom rim of the 25 mm
cap. The production body-level `right_contact` signal would count this as a
tripod even though the requested cap-sidewall topology is absent. The
single-environment 256-point view avoided the earlier 8-point overflow; its
memory and throughput at 1024 environments are **not verified**.

Targeted changes did not correct the contact topology:

| Physical variant | Right cap-sidewall ratios over final 0.5 s | Other result |
|---|---|---|
| Baseline separated MCP abduction, 10 mm wrist descent | Thumb 100%, index 0%, middle 96% | Bottle maximum drift 19.9 mm; one 5.41 N index–thumb self-contact at final frame. |
| Wrist descent reduced to 8 mm | Thumb 100%, index 0%, middle 0% | Index remained at bottom rim; middle detached. |
| Index PIP target capped at 1.6 rad | Thumb 92%, index 0%, middle 0% | Bottle maximum drift 25.8 mm; index MCP target saturated at its upper limit. |
| Index DIP target raised by 0.2 rad | Thumb 20%, index 0%, middle 0% | Index still at rim; final index–thumb self-contact included 16.75 N and 5.03 N points. |

## Local official-mesh optimization and PhysX falsification

`scripts/environments/optimize_right_tripod_mjcf.py` reads the measured
right-wrist, finger, bottle and cap state from a PhysX trace. It optimizes a
bounded 5 mm wrist translation, 0.12 rad rotation and small official-limit
finger offsets against an **interior** shortened cap-sidewall cylinder. It
penalizes cap penetration, right-hand cross-digit overlap and right-bottle
collision. Its output is explicitly `MJCF_ONLY_REQUIRES_NATIVE_USD_PHYSX_REPLAY`.
It never inserts a bank row or marks a grasp validated.

With the cap pose at the end of the initial 1 s scripted closure, seed 41
reduced the offline objective from 2.567 to 0.064 in 4,707 evaluations. The
three target mesh gaps were 0.475–0.533 mm; target cross-digit gaps were at
least 0.944 mm. The candidate changed the wrist by about 4.1, 0.7 and 3.9 mm
and 0.059 rad, and changed seven finger joints within their official limits.
**PhysX rejected it**: all three right cap contacts vanished during the
bounded transition, despite the wrist and finger joints reaching their
targets. The initially loaded cap angle was about 0.096 rad; as the fingers
released it, the implicit breakaway spring returned it to about zero. That
changed the cap orientation by roughly 0.09 rad while the free bottle moved
less than 1 mm. The static optimization had used a transient loaded cap pose.

A second optimization used the **measured post-springback** cap pose, seed 43.
Its offline objective fell from 2.768 to 0.047 in 6,861 evaluations; target
mesh gaps were 0.497–0.508 mm and target self-gaps at least 0.547 mm. In
PhysX, only the right middle retained sidewall contact during the subsequent
1 s hold; right thumb and index each had zero contact. Moving the commanded
right wrist an additional 2 mm toward the cap also ended with zero right cap
contact. Both optimized candidates therefore **failed** dynamic tripod
validation and were not saved as grasp-bank samples.

The pitch constraint remained numerically tight in these runs (maximum
reported axial error about 0.70 µm), and no bottle was teleported or fixed.
The failed result is about contact continuity and the cap's loaded-to-unloaded
rotation; small static mesh gaps do not prove a dynamically stable contact
patch. This does not prove that a three-sidewall-digit pose is impossible.

## Reproduction and status

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
scripts/run_isaaclab.sh scripts/environments/probe_pregrasp_contact_jacobian.py \
  --sample-index 3 --steps 50 --maintain-contact --contact-points \
  --right-wrist-down-steps 5 --right-index-mcp-flex \
  --right-index-abd-target-min=-0.2 --right-middle-mcp-abd \
  --right-middle-abd-target-min=-0.2 --right-middle-pip-target-max 1.6 \
  --headless --output artifacts/phase1/pregrasp_jacobian_split_abd_fullstate_1s.json
PYTHONPATH=source "$ISAACLAB_PYTHON" \
  scripts/environments/optimize_right_tripod_mjcf.py \
  --trace artifacts/phase1/pregrasp_jacobian_split_abd_fullstate_1s.json \
  --seed 41 --maxiter 60 \
  --output artifacts/phase1/optimized_right_tripod_mjcf_41.json
scripts/run_isaaclab.sh scripts/environments/probe_pregrasp_contact_jacobian.py \
  --sample-index 3 --steps 50 --maintain-contact --contact-points \
  --right-wrist-down-steps 5 --right-index-mcp-flex \
  --right-index-abd-target-min=-0.2 --right-middle-mcp-abd \
  --right-middle-abd-target-min=-0.2 --right-middle-pip-target-max 1.6 \
  --candidate-json artifacts/phase1/optimized_right_tripod_mjcf_41.json \
  --candidate-transition-steps 10 --candidate-hold-steps 50 --headless \
  --output artifacts/phase1/optimized_right_tripod_physx_41.json
PYTHONPATH=source "$ISAACLAB_PYTHON" \
  scripts/environments/optimize_right_tripod_mjcf.py \
  --trace artifacts/phase1/pregrasp_jacobian_split_abd_fullstate_1s.json \
  --cap-pose-trace artifacts/phase1/optimized_right_tripod_physx_41.json \
  --seed 43 --maxiter 80 \
  --output artifacts/phase1/optimized_right_tripod_settled_43.json
scripts/run_isaaclab.sh scripts/environments/probe_pregrasp_contact_jacobian.py \
  --sample-index 3 --steps 50 --maintain-contact --contact-points \
  --right-wrist-down-steps 5 --right-index-mcp-flex \
  --right-index-abd-target-min=-0.2 --right-middle-mcp-abd \
  --right-middle-abd-target-min=-0.2 --right-middle-pip-target-max 1.6 \
  --candidate-json artifacts/phase1/optimized_right_tripod_settled_43.json \
  --candidate-transition-steps 10 --candidate-hold-steps 50 --headless \
  --output artifacts/phase1/optimized_right_tripod_settled_physx_43.json
```

All raw JSON traces and logs are ignored under `artifacts/phase1/`; the code,
commands and summaries are committed. The optimizer's SciPy termination flag
was false at its iteration limit; its **candidate objective** improved, but
this is not an optimizer convergence or PhysX success claim. Updated scripts
passed `py_compile`, `git diff --check`, a seed-41 rerun that reproduced the
same offline objective and gaps, and a short PhysX safety-bound smoke.

**Gate status:** no topology-qualified Mimic stable bank, no independent
Reset replay, no sustained right thumb–index–middle sidewall grip, and no
learned CapTurn30 success. The first reproducible right *object-body* tripod
contact is useful diagnostic progress only. The next experiment should
optimize **contact continuity under the actual cap spring and free-object
dynamics**, with actual cap-local manifold points as constraints, then replay
any candidate from Reset for at least 1 s before allowing it into the stable
bank. Only then is further CapTurn PPO justified.
