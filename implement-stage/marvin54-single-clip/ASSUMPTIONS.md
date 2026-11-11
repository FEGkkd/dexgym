# Assumption ledger
<!-- ASK mode: never -->

| ID | Choice | Class | Source |
| --- | --- | --- | --- |
| A-001 | Copy 14 arm position servos from the supplied XML after checking arm joints, local link transforms and inertials. Keep the existing Hand2.2 joints, mesh assets, contacts, attachment transforms, and USD. Override the Newton drive parameters from the composed XML. | interface | user and local |
| A-002 | Preserve the verified 50 Hz single clip and its source asset identity. Accept an aligned XML only when the sole physical change is arm actuation. Compare complete 1,691-step runs against same-backend reference PD with the existing first-event safety thresholds. Report accuracy and completion independently. | semantic | user and local |
| A-003 | Keep the deployed policy's 54 absolute position targets. Initialize it by imitating a simulation-derived tracking teacher, then run differentiable World Model Actor/Critic optimization with a behavior regularizer and reduced Actor step size. Teacher labels are training inputs only. Separate the initialization checkpoint from RL checkpoints in physical evaluation. | semantic | local |
| A-004 | Compare checkpoints by full-clip safety and geometric tracking in both engines, starting with phase zero and exact reference state. Report deterministic perturbed starts separately. A successful deployment must complete in both engines, with tracking at least comparable to the matched reference PD. These are single-clip results. | semantic | local |
| A-005 | Use 2 ms physics, one held servo target per 20 ms, identical policy goal/history inputs and limits, and check physical failure at each substep. In MuJoCo detect contacts at the integrated state using forward position. | interface | local |
