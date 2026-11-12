# Hand-centric Bimanual S0 build

| Rung | Acceptance |
| --- | --- |
| F0 | Verify the manifest and decode a genuine MCAP into aligned 50 Hz hand/body segments. |
| F1 | Generate hand-centric 54-joint segments and independently test their native MuJoCo geometry and physical replay. |
| F2 | Load the validated Motion Library, perform real batched MuJoCo collection and complete model-based Actor and World Model updates from random weights. |
| F3 | Train on multiple source episodes, evaluate on held-out episodes from perturbed starts, and export one native MuJoCo policy and measured videos. |

## Run record
