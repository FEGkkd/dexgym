# Marvin 54 DoF Hand-centric Bimanual S0

Base commit `adfd64426c5650686ad9d4d755089613d28e3dc6`. Build the offline
motion-tracking phase of one policy for the fixed-base Marvin M6 dual arms
and dual Wuji Hand2.2. Extract EgoPro poses, retarget jointly with a fingertip
and bimanual goal, validate executable reference segments, then train the
existing differentiable World Model and a randomly initialized Actor using
batched MuJoCo physics. Export one 54-target policy and verify it on held-out
source episodes and perturbed initial poses in native MuJoCo.

Inputs are the verified EgoProStandard-body manifest and the existing
real-aligned 54-joint MJCF. Outputs are cleaned segments, a train/validation/test
library, checkpoints, per-clip quantitative evaluations and replay videos.
This phase uses robot geometry and self-contact in simulation.
