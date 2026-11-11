# Assumption ledger
<!-- ASK mode: never -->

| ID | Choice | Class | Source |
| --- | --- | --- | --- |
| A-001 | Copy 14 arm position servos from the supplied XML after checking arm joints, local link transforms and inertials. Keep the existing Hand2.2 joints, mesh assets, contacts, attachment transforms, and USD. Override the Newton drive parameters from the composed XML. | interface | user and local |
| A-002 | Preserve the verified 50 Hz single clip and its source asset identity. Accept an aligned XML only when the sole physical change is arm actuation. Compare complete 1,691-step runs against same-backend reference PD with the existing first-event safety thresholds. Report accuracy and completion independently. | semantic | user and local |
| A-003 | Keep the deployed policy's 54 absolute position targets. Initialize it by imitating a tracking teacher fit by inverse dynamics and full-clip native MuJoCo rollouts, then run differentiable World Model Actor/Critic optimization in Newton. For the 2,000-iteration run use 8,000 initialization updates, 300 World Model warmup iterations, regularizer weight 1.5, and Actor learning rate 2e-6. Teacher labels are training inputs only. Separate the initialization checkpoint from RL checkpoints in physical evaluation. | semantic | local |
| A-004 | Compare checkpoints by full-clip safety and geometric tracking in both engines, starting with phase zero and exact reference state. Report the same perturbations in both engines separately. Draw independent zero-mean Gaussian errors per joint at the initial state with standard deviations 0.01 rad and 0.05 rad/s. Clamp initial positions to the joint limits. A successful deployment must complete in both engines, with tracking at least comparable to the matched reference PD. These are single-clip results. | semantic | local |
| A-005 | Use 2 ms physics, one held servo target per 20 ms, identical policy goal/history inputs and limits, and check physical failure at each substep. In MuJoCo detect contacts at the integrated state using forward position. | interface | local |

| ID | Choice | Class | Source |
| --- | --- | --- | --- |
| A-006 | The local-dynamics comparison starts from the same behavior-cloned Actor weights. Its World Model and Termination Head receive q, qdot, and commanded q minus measured q. Use 16-step imagination and train only the Actor output layer during model-based optimization. Apply the teacher penalty to reference-neighborhood observation windows instead of imagined states. For the perturbed-PD collection group sample normalized action noise scale uniformly from 0.015, 0.05, 0.12 and 0.24 at each control step. | semantic | local |
| A-007 | Rank post-warmup RL checkpoints by complete nominal rollouts on both backends and the mean of wrist and fingertip mean errors normalized by each backend's matched PD errors. Assess the selected checkpoint with three separately seeded matched reset perturbations. | semantic | local |
