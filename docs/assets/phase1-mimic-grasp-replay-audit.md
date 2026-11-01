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
speed 1.15 rad/s. No candidate from these TGS runs was replayed or added
to a stable bank. The TGS flag is available only in the diagnostic probe
and replay CLI; the RL environment default was not changed by this failed
ablation. These results rule out the solver warning as a sufficient fix.

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
