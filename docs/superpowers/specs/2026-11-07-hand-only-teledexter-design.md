# Right-hand GRAB tracking task design

## Goal and scope

Build a hand-only analogue of TeleDexter's consecutive-subgoal tracking task. The
policy controls the 20 joints of the right WujiHand2 Beta 2 hand to follow a
library of reachable fingertip sequences derived from local GRAB recordings.
There is no manipulated object, object observation, or object reward. The left
hand and both arms are held at a fixed reference pose and are excluded from the
policy interface. This phase stops after a small PPO integration run; large-scale
training and reward tuning are separate decisions.

## Data flow

1. Read each GRAB `rhand.params.fullpose` and its subject-specific right-hand
   template. Reconstruct 21 MANO landmarks in MediaPipe order, expressed
   relative to the MANO wrist. The source files remain unmodified.
2. Run the pinned local Wuji C SDK 2026.10.26 right-hand retarget session on the
   landmarks in time order at 30 Hz. Reset the session for each source sequence.
3. Map the SDK's 20 firmware-order output joints to the names and limits in the
   pinned Marvin/WujiHand2 model. Clamp to legal limits and report clipping.
   Evaluate the five fingertip sites with that model's forward kinematics in
   the robot right-wrist frame. These FK positions, not raw human landmarks,
   are the reachable tracking targets.
4. Save one compressed motion file per source recording with `q_ref[T,20]`,
   `tip_ref[T,5,3]`, source rate, output rate, source path and mapping metadata.
   Save a manifest with processing counts, rejection reasons and quality
   statistics. Reserve subject `s8` for validation and use the other available
   subjects for training.

## RL task

Use Isaac Lab 3.0 `DirectRLEnv` and the existing Newton MJWarp packaged 54-DoF
asset. The environment holds the arms and left hand at home while the action is
a 20D right-hand joint target increment. The policy observes right joint
position and velocity, current fingertip positions in the right-wrist frame,
the active five-fingertip goal and its error, and the previous action. It sees
no reference joint angles.

At reset, sample a training sequence and a valid starting frame, set right-hand
joints to its `q_ref`, and choose a later fingertip target. A goal is reached
when every fingertip stays within a configured position tolerance for a short
dwell. Reaching advances the frame index by a curriculum-controlled stride;
otherwise the goal remains active. Reward combines a sparse reached-goal bonus,
dense fingertip tracking, a small action-rate penalty and a time cost. Episodes
end on timeout, unsafe joint state, or prolonged failure to reach a goal.

The initial PPO run verifies data loading, vectorized stepping, finite values,
goal progression, reset behavior and checkpoint export. It is a workflow check,
not evidence of policy convergence or real-world transfer. We compare FK goal
tracking with a direct joint-target replay baseline before increasing scale.

## Acceptance checks

- The data conversion records frame counts, per-joint clipping and FK
  reconstruction error; no retained frame has nonfinite or out-of-limit joints.
- A generated target re-evaluated from its saved joint pose matches within
  numerical tolerance in the declared right-wrist frame.
- A right-hand-only task reports 20 actions and observations/rewards independent
  of object and left-hand state, and reaches/advances synthetic subgoals.
- A short headless PPO run finishes with finite losses and saves a checkpoint.
