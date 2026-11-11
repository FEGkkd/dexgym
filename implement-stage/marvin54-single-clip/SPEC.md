# Marvin 54 DoF single-clip deployment

Base commit `b46ef239fa8518b317b29f4a12661dce5b696eac`.

Train one goal-conditioned 54-joint position policy on Adjust Shelf Labels with
the existing differentiable World Model and Actor/Critic optimization. Deliver
a closed-loop MuJoCo deployment on the paired model, full-clip Newton and MuJoCo
evaluations, baseline comparisons, and reproducible artifacts.
