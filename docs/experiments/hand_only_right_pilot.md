# Right-hand fingertip tracking pilot (2026-11-08)

## Scope

This is a TeleDexter-inspired consecutive-subgoal task for the **right WujiHand2
only**. GRAB object trajectories are unused. Both robot arms and the left hand
are held at their home joint targets. This pilot validates data conversion,
simulation, and the PPO workflow; it does not establish a trained controller.

## Sequence library

`scripts/data/build_grab_right_hand_library.py` reads GRAB right MANO poses and
subject templates, reconstructs 21 wrist-relative landmarks, and calls the
local Wuji C SDK right-hand retargeter. It converts the SDK's 20 joint values
to the pinned right-hand MJCF joint order, then evaluates its five tip sites by
FK in the `r_wrist` frame. Source 120 Hz sequences are sampled at 30 Hz. Each
`.npz` stores `q_ref[T,20]`, `tip_ref[T,5,3]`, source frame indices, and FPS.
The source recordings are unchanged.

Local output: `data/processed/hand_only_right/manifest.json` and 797 motion
files (33,681,420 bytes total). The generated `.npz` files are ignored by Git;
rebuild them from the local source data when setting up another checkout.

| Split | Subjects | Motions | Frames |
| --- | --- | ---: | ---: |
| Train | s1, s2, s3, s4, s5 | 635 | 200,010 |
| Validation | s8 | 162 | 52,669 |
| Total | | 797 | 252,679 |

All 797 inputs were accepted; none were rejected, nonfinite, or outside model
joint limits. The SDK to model conversion clipped zero joint values. Across
motions, the median/p95/maximum of each motion's largest adjacent joint change
were 0.166/0.259/0.343 rad; for fingertip displacement they were
11.6/18.1/26.8 mm. Targets are **kinematically reachable by construction**;
these statistics alone do not guarantee dynamic reachability at 30 Hz. Saved
tip positions reconstruct from saved joints, and the simulation's initial
right-wrist FK differs by at most about 0.005 mm in the smoke runs.

## Task definition

- Isaac Lab 3.0 `DirectRLEnv`, Newton MJWarp, full 54 DoF packaged asset with
  hand collisions disabled; 20 right-hand actions, 105 policy observations.
- Policy observes right joint positions and velocities, five current and five
  goal fingertips in the right-wrist frame, their 15D error, and previous 20D
  action. No object features, left-hand joints, arm joints, or reference joint
  angles enter the policy observation.
- Each 30 Hz action adds at most 0.08 rad to each right-hand joint target, then
  clips to robot limits. Other joints hold home targets. At reset, right joints
  are set to a sampled motion frame and the goal is four library frames ahead.
- A subgoal is reached after all five tip errors remain below 25 mm for two
  consecutive policy steps. Reward is a +10 reached bonus plus mean
  `exp(-50 * fingertip_error_m)`, minus 0.02 times mean squared action change
  and a 0.01 step cost. Reach advances by four frames, clamping to the last
  frame. Unsafe joint states, more than 45 steps without a reach, completion,
  or the 8 s episode limit end an episode.

The original 15 mm, 30-step pilot setting yielded very few reached goals under
direct joint replay. With a 25 mm tolerance and 45-step limit, an eight-env,
120-step train-split replay reached 41 subgoals with 10 resets in one short
run; the same setting on held-out `s8` reached 33 subgoals with 8 resets.
The held-out run had 20.69 mm mean fingertip error and 0.1168 rad mean joint
target error. These are replay diagnostics, not policy scores.

## PPO workflow check

Bounded command: `isaaclab train --rl_library rsl_rl --task
DexGym-Wuji-Right-Hand-Tracking-Direct-v0 --num_envs 16 --max_iterations 10
--seed 42 physics=newton_mjwarp` (with `PYTHONPATH=source`). It finished with
exit code 0, 5,120 interaction steps, finite losses and rewards, and a final
checkpoint at
`logs/rsl_rl/dexgym_wuji_right_hand_tracking/2026-11-08_00-00-23/model_9.pt`.
The checkpoint's actor and critic tensors are finite. Reported mean reward
rose from 5.75 at iteration 0 to 14.21 at iteration 9, while mean tip error
at iteration 9 was 33.2 mm and goal reach fraction 0.0059. Ten iterations
are enough to verify the training loop, not to claim convergence or reliable
sequence tracking.

The host's existing `.venv` contains several damaged mapped native files
(MuJoCo, cuSPARSELt, Warp, USD, Pillow). The successful checks used clean
copies of those pinned packages in `/tmp/handonly-deps`; no project package or
SDK files were changed. A clean environment reinstall is advisable before a
larger run.

Verification on the original feature tree: the focused Ruff check passed and
default pytest passed 22 tests with 5 optional external-source skips. After
integrating into the newer `main` tree, the complete default suite passed 242
tests with 3 skips. Ruff passed on all new and modified hand-only Python files;
the wider `main` tree has 110 unrelated existing lint findings. An eight-env
held-out replay on the integrated tree reached 55 subgoals in 120 steps, and a
separate one-iteration, eight-env PPO startup check exited successfully. The
earlier eight-env held-out replay passed finite observation and reward checks,
initial FK parity, and subgoal progression.

## Next discussion

Before a larger run, decide whether to keep the 25 mm bootstrap tolerance or
use a staged tolerance/stride curriculum, and choose success metrics on held-out
subjects (subgoals per episode, completion rate, five-tip error, and control
smoothness). The present pilot does not evaluate a converged policy or physical
hand transfer.
