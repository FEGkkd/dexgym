# Bimanual geometric pregrasp Reset (Phase 0)

This module defines **pregrasp starting states**, not already successful grasps.
The two official WujiHand2 Beta 1 native USD hands retain their 20 physical
finger joints. The 70 mm engineering bottle and threaded cap remain dynamic.
There is no reward, observation, PPO task, or one-second hold requirement in
this admission pipeline. Learning to close, support and unscrew is left to a
future policy.

## Two templates

| Template | Object frame | Topology | Initial geometric target |
|---|---|---|---|
| `left_cylindrical_pregrasp` | bottle | Thumb opposite the four long fingers; pads approach the middle to upper body, following `left.png` | Each of five digits within 0.5–12 mm of the 35 mm radius body; four long-finger contact directions within a 75° half arc and thumb at least 125° opposite |
| `right_cap_tripod_pregrasp` | cap | Thumb, index and middle fingertips surround the cap sidewall, following `right.png`; ring and pinky are curled clear | Three actual fingertip sites 1–12 mm radially outside the sidewall and within its axial height; thumb at least 125° opposite the index/middle site mean; ring and pinky sites at least 20 mm clear |

The templates are [left_cylindrical_pregrasp.json](../../source/dexteroushands_lab/scenes/presets/left_cylindrical_pregrasp.json)
and [right_cap_tripod_pregrasp.json](../../source/dexteroushands_lab/scenes/presets/right_cap_tripod_pregrasp.json).
Their wrist positions are expressed in the named object's local frame. All
quaternions use `xyzw`, metres and radians. The left center was opened from
the previously measured `left.png` inspired cylindrical pose. The right center
was refined using the pinned official right MJCF fingertip sites and collision
meshes. The earlier body-near-cap pose left actual fingertips far from the
sidewall; this revision brings the thumb, index and middle sites around the
cap and curls the ring and pinky away. These centers are geometric
presets; they are not claims that the user's physical Beta hardware matches
the Beta 1 candidate or that a grasp can hold the bottle.

The left wrist translation is sampled within ±4 mm per axis, rotation within
±0.06, ±0.06, ±0.08 rad, and its finger joints within ±0.08 rad each. The
right wrist translation is sampled within ±2.5 mm per axis, rotation within
±0.04, ±0.04, ±0.05 rad, and its finger joints within ±0.04 rad each.
Out-of-limit draws are
rejected rather than clipped. `PregraspResetGenerator.draw(seed=...)` is
deterministic and marks `geometry_qualified=false`. Its output must not be
used directly as an admissible training Reset.

## Admission and runtime sampling

The offline first stage uses the pinned official Beta 1 MJCF collision meshes
and an engineering cylinder proxy for the bottle/neck/cap. It measures each
digit's nearest distance, cap sidewall point, angular topology, cross-object
clearance, cross-digit self clearance and hand-to-hand clearance. For the
right hand it also checks the five **actual fingertip sites** and requires a
three-site tripod around the cap; distal body proximity alone cannot pass. It only
proposes samples. The second stage resets the **official native USD** in
Isaac/PhysX, advances one 2 ms physics step to refresh contact data, and
rejects reported penetration below -0.1 mm, contact peaks of 50 N or more,
and root drift of 3 mm or more. The negative control deliberately penetrates
the bottle and a separate control crosses the right index/middle fingers to
prove that both object and self-contact views report overlap. This one-step
probe is a geometry diagnostic, not a suspended hold test. The MJCF positive
margin plus native USD collision probe are both required for bank admission.

The separate [phase0_pregrasp_reset_bank.json](phase0_pregrasp_reset_bank.json)
stores the accepted Reset states and per-sample evidence. Every bank entry
retains `validated=false`; that existing flag continues to mean a gravity-on
one-second hold. `BimanualResetManager.pregrasp_geometry_qualified[env]`
reports geometric eligibility separately from `init_valid[env]`.

At runtime, `PregraspResetBank.draw_runtime(seed=...)` selects one screened
local wrist/joint preshape and uniformly samples bottle X/Y within ±20 mm, Z
within ±10 mm, and world yaw within ±π. It transforms the complete assembly
rigidly, including both wrist targets, cap and thread frame, preserving all
internal collision distances. There is no new local joint or wrist jitter
after screening. To expand local ranges, regenerate and recheck the bank.
The whole assembly is kept above the floor. Runtime selection is deterministic
for a seed. `BimanualResetManager.reset(env_ids, samples)` restores only the
selected environments without an internal physics step.

## Reproduction

Set `WUJI_DESCRIPTION_ROOT` to the official `v2026.9.11` checkout at commit
`c2cd7f8d1ef8b6dc8cb907c17daa5a88b4442d95` and `ISAACLAB_PYTHON` to the
Isaac Sim 6.0.1 / Isaac Lab 3.0 Python. Then run:

```bash
PYTHONPATH=source "$ISAACLAB_PYTHON" scripts/environments/generate_pregrasp_reset_candidates.py --seeds 5000 --max_accepted 32 --output /tmp/pregrasp_candidates.json
./scripts/run_isaaclab.sh scripts/environments/validate_pregrasp_reset_physx.py --candidates /tmp/pregrasp_candidates.json --bank docs/assets/phase0_pregrasp_reset_bank.json --report docs/assets/phase0_pregrasp_reset_physx_report.json --max 32 --headless --device cuda:0
./scripts/run_isaaclab.sh scripts/environments/validate_bimanual_reset.py --pregrasp_bank docs/assets/phase0_pregrasp_reset_bank.json --control_mode pose_command_wrist --num_envs 16 --iterations 100 --headless --device cuda:0
./scripts/run_isaaclab.sh scripts/environments/preview_grasp_initialization.py --pregrasp_bank docs/assets/phase0_pregrasp_reset_bank.json --sample_index 3 --interactive_static --viz kit --device cuda:0
```

Use `--headless --steps 2` instead of `--interactive_static` for a headless
preview smoke test. For smaller Reset fixtures, set `--num_envs 1` or `2`.

## Measured result on this workstation

On an RTX 5090 with Isaac Sim 6.0.1 and Isaac Lab 3.0, the deterministic
search accepted 32 of its first 2262 MJCF candidates (the remaining 2738
seeds of the requested 5000 were not needed). All 32 passed the native USD
one-step probe. The deliberate overlap was reported at -4.51 mm; the minimum
crossed-finger self overlap was reported at -18.66 mm. The minimum
reported separation among accepted candidates was +0.060 mm (some had no
contact points within the backend's reporting distance). No candidate
produced a contact impulse. The measured proxy gaps in the bank were
0.51–11.51 mm for left body digits and 0.50–8.89 mm for the right tripod's
collision bodies at the cap sidewall. The three right fingertip sites were
1.02–11.05 mm radially outside the cap sidewall. The full
[PhysX report](phase0_pregrasp_reset_physx_report.json)
and per-sample bank evidence record the exact numbers. The compact
[validation summary](phase0_pregrasp_reset_validation.json) records the 1, 2
and 16 environment Reset results.

The final 32-sample bank passed 100 selective Resets in each of 1, 2 and 16
environments. Untouched environments were bitwise unchanged at the immediate
Reset boundary. Maximum root restoration error was 0, 5.96e-8 and 2.38e-7
respectively; joint state/target error was zero. Detached thread state was
restored to engaged, and contact history was invalidated. The headless
preview ran two physics steps with no reported contact. None of these tests
proves that the hand can close and hold the bottle for one second.

The full CPU test suite reported 121 passed, 1 skipped and 2 failures in
unrelated, locally modified Allegro Fly and MANO helper files. The focused
pregrasp and existing grasp/Reset/contact contracts reported 27 passed. See the
working tree before treating the unrelated failures as a repository baseline.
