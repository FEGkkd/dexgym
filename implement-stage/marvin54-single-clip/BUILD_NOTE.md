# Build record

| Rung | Acceptance |
| --- | --- |
| F0 | Compose updated arm and existing Hand2.2 with 54 valid position servos and run a full native MuJoCo baseline. |
| F1 | Run Actor initialization and evaluate the closed-loop checkpoint with no online teacher in both engines. |
| F2 | Train with Newton, World Model, Termination Head, Actor and Critic, then compare full-clip checkpoints. |
| F3 | Deliver a selected RL checkpoint, sim-to-sim replay, measured trace and reproducible invocation. |
