# Marvin M6 Hand-centric Bimanual Whole-Body S0

The fixed-base task controls 14 Marvin arm joints and 40 Wuji Hand2.2 joints
with one goal-conditioned Actor. It uses synchronized EgoPro hand and upper
body poses and the real-aligned dual-arm MJCF. MuJoCo Warp collects 2 ms
physical steps. The policy supplies one 54-dimensional absolute position
target every 20 ms.

## Data and goal

The input manifest records verified EgoProStandard-body source episodes.
Each source episode belongs to one split before resampling and trimming.
The hand goal describes ten fingertip poses, finger shape in the palm
frame, 25 cross-hand fingertip vectors and relative palm orientation.
The mapped human target and the robot's achievable projection are
stored together. The projected target drives the policy and reward.

The offline solver prioritizes hand geometry, with wrist and elbow pose
as weaker objectives. It enforces joint bounds and checks collision
clearance. A subsequent refinement accepts smaller mapped errors if the
segment still completes a physical replay. Native MuJoCo audits every
integrated 2 ms state and can apply contact-Jacobian corrections to the
reference trajectory. The native PD controller is a data audit and an
evaluation comparator.

## Training and evaluation

The Actor, World Model, Termination Head and Critic start from random
weights. Physical transitions come from policy exploration. The Actor
updates by backpropagating the imagined tracking return through the
World Model. The task uses hand-centric reward, joint and command
regularization, and physical collision termination. The exact settings
for each run are stored in its checkpoint and provenance.

Example commands from the training worktree:

```bash
root="$HOME/dexgym_data/marvin54_hand_centric_s0_20261112"
model="$HOME/dexgym_data/marvin54_aris_single_clip_20261112/deploy/dual_arm_marvin.xml"
bash scripts/supermimic/host.sh -m dexgym.tasks.direct.marvin54_s0.train \
  --model "$model" --library "$root/audited_library/train.json" \
  --output "$root/s0_run" --num-envs 4096 --iterations 5000

bash scripts/supermimic/host.sh -m dexgym.tasks.direct.marvin54_s0.evaluate \
  --model "$model" --library "$root/audited_library/test.json" \
  --checkpoint "$root/s0_run/model_final.pt" --output "$root/s0_test" \
  --backend cpu --require-heldout --max-clips 20 --trace
```

The evaluator tracks source-episode overlap and reports clip and episode
statistics, completion, errors on observed prefixes, and contact depth.
Perturbed starts and wrist-preserving alternate arm configurations test
recovery. The arm-hold comparisons keep one or both arms' initial servo
targets fixed while the Actor continues controlling the other joints.
