# Config-driven models and normalization

[`NetworkFactory`](../../source/dexgym/algorithms/supermimic/core/models.py) receives `RobotSpec`, `TaskSpec` and `NetworkSpec`. It builds:

| Module | Input | Output |
| --- | --- | --- |
| State delta model | robot state + action + dynamics context | robot physical delta |
| Goal encoder + Actor | observation history + task goal window | robot raw action |
| Critic | task Critic features | scalar value |
| Learned termination head | robot state + task termination features | arrival termination probability |

All widths and hidden layers come from specs. Marvin Tracking's 42/14/28/10×90/146 numbers are plugin values; Reach uses 42/14/28/1×6/48; Toy uses 2/1/2. The learned head is separate from dynamics, so a Marvin world model can be reused between two compatible tasks while Actor, Critic and termination weights remain task-specific.

FP32 RunningNorm modules are part of the corresponding model state dict. Only fresh real training-environment samples update them; sampled Replay windows, imagination, validation environments and evaluation do not. The observation normalizer sees each newest real frame once, not its repeated history copies. `CoreCheckpoint` restores these buffers strictly. Historical M4.1/M4.2 Actors retain their old normalization mode and are loaded by explicit compatibility functions; a new Actor factory is never used to reinterpret their weights.

The default Marvin task termination mode is analytic. A task-shaped learned head is available with masked BCE and real hard-termination labels at the correct arrival reference. This M4.3 short smoke did not collect enough positive failures to calibrate a learned head; contact-rich learned termination remains unvalidated.
