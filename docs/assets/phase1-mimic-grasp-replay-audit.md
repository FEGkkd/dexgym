# Phase 1 Mimic tripod contact and fresh Reset replay — 2026-11-01

This is a targeted physical diagnosis after the strict-hold audit. It uses
the 70 mm engineering bottle, official WujiHand2 Beta 1 candidate USD,
gravity-on free bottle/cap, hard-pitch Mimic screw, bounded pose-command
wrists and normal 2 N m finger drives. Every finger command below is a
scripted **candidate-generation diagnostic**, not a learned policy. No
scripted cap torque, support fixture, object teleport or relaxed contact
physics was used. Ignored raw traces are under `artifacts/phase1/`.

## Contact topology and its physical limit

Pregrasp bank row 9 previously reached true right thumb/index/middle
cap-sidewall contact for only 4/50 policy steps. The index then contacted
the bottom rim around cap-local z = -11.6 mm, outside the accepted interior
sidewall band. With every other scripted command unchanged, backing the
right index MCP flex target off by **0.05 rad** after the initial closure
lifted its actual distal contact point to about z = -7.9 to -8.0 mm.
The following 50/50 policy steps had real three-digit sidewall contact.
Left thumb, index, ring and pinky contacted the bottle for 50/50 steps;
left middle did not. No right-hand cross-object force or right-index
self-contact was observed at the final sample. The Mimic pitch error stayed
below 0.7 micrometres.

This was **not a stable grasp**. The bottle moved only about 0.056 mm
between the first and last *50 Hz sampled poses* of that window, but the
captured instantaneous bottle linear and angular speeds were 0.191 m/s and
6.54 rad/s. The cap spin was 1.85 rad/s; left index contact reached about
63 N. The sampled displacement had concealed high-frequency dynamics.
The candidate remains `validated=false`.

## Fresh-process Reset falsification

The probe can now save the live hands, 40 joint states and targets, wrist
targets, free-bottle and cap states, screw angle and spin, and Mimic drive
stick reference/sliding/breakaway state. The separate
`replay_mimic_grasp_candidate.py` restores that candidate through the
DirectRLEnv `_reset_idx` path, then runs one second with zero actions.
The replay deliberately prevents auto-Reset on task termination so the
entire physical trajectory is measured. It never marks a bank row valid.

The first replay with only screw angle/spin restored failed: right sidewall
tripod ratio 0/50. Inspection revealed that the original Reset reset the
Mimic stick reference to the current angle, while the captured candidate's
stick reference was 0 rad. `BatchedMimicDrive.reset` and
`MimicThreadRuntime.reset` now accept an optional complete drive state,
recompute the selected joint's bounded PhysX drive parameters, and leave
other environments unchanged. Pregrasp bank rows without this metadata
still use a fresh, nonsliding stick state.

Repeating the candidate capture and replay with **all** drive state restored
still gave right sidewall tripod ratio **0/50** and no strict GraspHold
success. The bottle and cap pose Reset errors were 0 and 0.48 micrometres;
joint position, screw angle/spin, stick reference, sliding and breakaway
state errors were zero. Initial contact separation was -0.078 mm, bottle
drift during the replay 1.95 mm, rotation 0.0155 rad, thread error below
0.26 micrometres, and no ground contact. Nevertheless, cap spin jumped to
6.54 rad/s on the first policy step and right-sidewall contact vanished.
The candidate depended on a rapidly moving contact configuration and
cannot be reproduced as an episode start. It is not placed in a stable bank.

## Controlled dynamics ablations

The same scripted trajectory at 16 instead of 8 PhysX position iterations
reduced captured bottle angular speed from 6.54 to 1.38 rad/s and cap spin
from 1.85 to 0.94 rad/s, but the final 50-step right tripod rate fell from
50/50 to **0/50**. Merely increasing solver iterations is not a stable-grasp
fix.

At 8 iterations, contact-force feedback that backed off left finger targets
above 20 N reduced the final-window left peak from about 63.2 to 14.5 N
and bottle angular speed to 1.46 rad/s. It also reduced the final-window
right tripod to **0/50**. The free bottle's motion changes the narrow right
contact geometry; relieving the left hand alone does not yield a replayable
bilateral state. These are two controlled diagnostic changes, not a broad
parameter search or evidence of RL learning.

The installed Isaac Lab 3.0 PhysX configuration explicitly says that TGS
with `enable_external_forces_every_iteration=False` can yield noisy
velocities. Holding the original 8 solver iterations and scripted targets
fixed, enabling this flag reduced the final bottle angular speed from
6.54 to **1.21 rad/s** and the left contact peak from 63.2 to **10.5 N**.
It did not restore a grasp: right thumb/middle had interior sidewall contact
for 50/50 steps, while the index had **0/50**. A single follow-up bounded
world-wrist translation of approximately 2 mm toward the measured index
cap-local radial gap also left the index at **0/50**, with bottle angular
speed 1.15 rad/s. At that stage, no TGS candidate was replayed or added
to a stable bank. The TGS flag is available only in the diagnostic probe
and replay CLI; the RL environment default was not changed by this failed
ablation. These results rule out the solver warning as a sufficient fix.

## Exact-target tripod replay and screw-velocity inconsistency

The TGS probe's right index was already on the real cap sidewall at closing
step 50. The earlier diagnostic then reduced its MCP target from 0.557 to
0.507 rad at step 51, and the index contact disappeared immediately. An
intermediate 0.537 rad target reduced the distal gap but still lost contact.
Holding the **exact** closing targets retained thumb/index/middle sidewall
contact for **50/50** subsequent policy steps. Final index contact was on
the distal body at cap-local z approximately -8.8 mm, with about 2.4 N
normal force and -0.029 mm separation. This was scripted candidate
generation, not policy learning.

Fresh-process, gravity-on Reset replay of that captured state produced
**50/50 left-bottle** and **50/50 right-cap-sidewall tripod** contact steps.
Maximum bottle drift was 0.400 mm, rotation 0.00372 rad, initial minimum
contact separation -0.084 mm, peak left/right body forces 11.20/2.33 N,
and maximum thread error 0.239 micrometres. There was no ground contact or
task failure. Captured root poses, 40 joint positions, screw angle/spin,
stick reference and slip state were restored within the reported numerical
tolerances. This establishes a reproducible **contact topology**, but the
state does **not** pass strict GraspHold or qualify for the validated bank.

The blocker is the screw's contact-rich velocity state. Over 500 physics
substeps at 0.002 s, PhysX reported a mean SpinJoint speed of **+3.532
rad/s**. Independently read cap-versus-bottle link angular velocity agreed
within 2.15e-6 rad/s. Yet the SpinJoint angle changed only **-0.00854
rad** over the same second, an average finite difference of **-0.00856
rad/s**. RMS speed-versus-angle-difference error was **3.558 rad/s**. The
joint stayed in stick mode throughout. The strict hold limit is 0.2 rad/s
cap spin, so the gate correctly remains false. Replacing the velocity with
an angle-difference filter before resolving the solver discrepancy could
hide a physical inconsistency.

Two controlled replay ablations did not fix it. Setting only the Reset spin
velocity to zero yielded 3.795 rad/s mean reported speed, with the same
50/50 tripod and no strict success. Scaling the entire engineering-example
stick/slip torque and drive-gain family by 10 yielded 4.665 rad/s, again no
strict success. This stronger resistance is an uncalibrated diagnostic, not
a new bottle specification. An isolated no-hand screw with the same TGS
setting had maximum speed below 5e-10 rad/s; a separate 1.19 N cap-contact
proxy reached at most 0.0084 rad/s. The flag alone or a simple single
contact does not reproduce the multi-hand contact discrepancy. Its exact
cause within the contact-rich articulation solver remains unresolved.

A further selective ablation moved only the right hand away at Reset while
leaving the captured left-bottle contact, bottle/cap dynamics and initial
3.60 rad/s screw velocity intact. Right-cap contact fell to 0/50, while
left-bottle contact stayed 50/50. The screw's mean reported speed fell to
-0.063 rad/s over the full second and averaged about -0.031 rad/s over the
last 100 physics substeps; its angle changed -0.0434 rad. The full-run
speed-versus-angle-difference RMS error fell from 3.558 to 0.185 rad/s.
Thus the persistent multi-rad/s inconsistency requires the active right
hand/cap contact path in this candidate. The result does not identify which
finger pair or solver constraint is responsible. This ablation is invalid
as a grasp, and the replay tool cannot mark it as a passing bank sample.

Digit ablations used the same captured thumb/middle-contact state but
restored one digit at a time to its collision-screened pregrasp joint pose.
With only the **middle** contacting the cap (mean/peak body force
1.03/1.26 N), reported screw speed averaged **0.860 rad/s** while actual
angle-difference speed averaged -0.0246 rad/s. With only the **thumb**
contacting (1.61/1.67 N), these values were **2.055** and **-0.0178
rad/s**. With both contacting, reported speed averaged **3.292 rad/s**
while angle-difference speed averaged -0.00814 rad/s. Both single-contact
finger states still show a discrepancy, unlike the separate low-force
proxy. No single digit alone explains the failure. The contact-load and
solver-coupling hypothesis is stronger, but exact internal PhysX cause
remains unverified. These one-digit states are diagnostic Reset ablations,
not valid grasp starts.

```bash
scripts/run_isaaclab.sh scripts/environments/probe_pregrasp_contact_jacobian.py \
  --sample-index 9 --steps 50 --maintain-contact --contact-points \
  --right-cap-contact-gate manifold_sidewall --right-index-mcp-flex \
  --right-index-abd-target-min=-0.2 --right-middle-mcp-abd \
  --right-middle-abd-target-min=-0.2 --right-middle-pip-target-max 1.6 \
  --hold-after-close --candidate-transition-steps 10 --candidate-hold-steps 50 \
  --capture-candidate --external-forces-every-iteration --headless \
  --output artifacts/phase1/row9_tgs_exact_target_hold_probe.json
scripts/run_isaaclab.sh scripts/environments/replay_mimic_grasp_candidate.py \
  --candidate-result artifacts/phase1/row9_tgs_exact_target_hold_probe.json \
  --steps 50 --external-forces-every-iteration --trace-thread-substeps --headless \
  --output artifacts/phase1/row9_tgs_exact_target_hold_replay_summary.json
```

The optional `--spin-reset-override 0`, `--resistance-scale 10`,
`--ablate-right-cap-contact`, and `--ablate-right-digit thumb` (or
`middle_finger`) replay flags are diagnostic and cannot produce a passing
bank candidate. The
isolated no-hand and contact-proxy fixtures use `probe_thread_mimic.py`
with zero torque and axial force, real geometry, the implicit running drive,
and `--external-forces-every-iteration`; the contact comparison additionally
uses `--contact-proxy`. Raw traces remain ignored under
`artifacts/phase1/`. No new PPO ran and the candidate stays
`validated=false`.

## Code, reproduction and status

The probe adds optional live-state capture, force-relief and solver-iteration
measurements. The replay tool emits per-policy-step bottle velocity, cap
spin, contact, drift and constraint data. The production Mimic Reset now
restores optional spin velocity and stick/slip state from bank metadata;
the existing pregrasp rows default to zero spin and fresh stick state.
Isaac's installed PhysX backend can store drive parameter tensors on CPU
while simulation state tensors are CUDA, so selected drive writes use each
buffer's actual device. The 16-environment, 100-reset regression compared
both drive buffers and simulation caches, with maximum peer-state error 0.
The final 1024-environment, three-step Mimic smoke completed with finite
state, zero cap-cache Reset error and 5,715 environment steps/s during the
short measured stepping interval. This is a runtime check, not a physical
grasp or learned-policy success test. `py_compile` and `git diff --check`
passed for the touched files. The full repository suite returned **168
passed, 1 skipped, 2 failed**; the two failures remain in concurrently
user-edited Allegro/MANO test areas (`test_allegro_hand_track_config.py` and
the untracked `test_checkpoint1_helpers.py`). No new Phase 1 test failure
was reported. The full log is ignored at
`artifacts/phase1/mimic_drive_restore_final_full_pytest.log`.

```bash
export WUJI_DESCRIPTION_ROOT=/tmp/wuji-description-v2026.9.11
export ISAACLAB_PYTHON=/home/ubuntu/miniconda3/envs/allo_track/bin/python
scripts/run_isaaclab.sh scripts/environments/probe_pregrasp_contact_jacobian.py \
  --sample-index 9 --steps 50 --maintain-contact --contact-points \
  --right-cap-contact-gate manifold_sidewall --right-wrist-down-steps 0 \
  --right-index-mcp-flex --right-index-abd-target-min=-0.2 \
  --right-middle-mcp-abd --right-middle-abd-target-min=-0.2 \
  --right-middle-pip-target-max 1.6 \
  --post-close-index-mcp-delta=-0.05 --candidate-transition-steps 10 \
  --candidate-hold-steps 50 --capture-candidate --headless \
  --output artifacts/phase1/row9_index_mcp_relax_005_fullstate_capture.json
scripts/run_isaaclab.sh scripts/environments/replay_mimic_grasp_candidate.py \
  --candidate-result artifacts/phase1/row9_index_mcp_relax_005_fullstate_capture.json \
  --steps 50 --headless \
  --output artifacts/phase1/row9_index_mcp_relax_005_fullstate_fresh_replay.json
scripts/run_isaaclab.sh scripts/environments/validate_bimanual_rl_reset.py \
  --thread-backend mimic --num-envs 16 --resets 100 --headless
```

The TGS velocity ablation reruns the first probe command with
`--external-forces-every-iteration` and output
`artifacts/phase1/row9_mcp_relax_tgs_forces_each_iter.json`. The one
geometric correction adds `--post-close-world-translation 0.0016 0.0010
-0.0007` and writes `artifacts/phase1/row9_tgs_forces_inward2mm.json`.
If a candidate generated with that flag is replayed, the replay command
must receive the same flag; the tool rejects mismatched solver settings.

**Status:** the requested >=60% natural-pregrasp GraspHold and CapTurn30
learning targets remain unmet. No Mimic stable bank, 30-degree learned turn
or cross-family stable evaluation is claimed. The next useful experiment
must form the tripod at lower bottle and cap velocity, then capture and
replay a quiet state. The right MCP relaxation identifies one reachable
contact direction; it does not solve dynamic stability.
