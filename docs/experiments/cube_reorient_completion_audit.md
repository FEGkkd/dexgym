# Completion audit: native cube task and from-scratch training

Scope: port the referenced Wuji Hand 2 Cube Reorientation task into dexgym,
retain the SuperMimic training method, and train a working native policy.
No numerical success threshold was supplied by the user. Evidence is therefore
reported in full rather than replacing the task with an easier success rule.
The reward, target-switch, hold and termination rules were not simplified.

| Requirement | Current evidence | Assessment |
| --- | --- | --- |
| Native task inside dexgym task definitions | `source/dexgym/tasks/direct/hand2_cube_supermimic/`; registered `DexGym-Wuji-Hand2-Cube-Reorientation-SuperMimic-Direct-v0`; training and evaluation instantiate it through `gym.make` | Implemented and executed |
| Single Wuji hand and source cube task | Bundled reference Hand 2.2/cube assets; ground and resolved solver options in `native_scene_parity.json` and `original_scene_probe.json` | Source task preserved with documented Newton adaptations |
| Existing PPO/no-object tasks remain separate | Changes under `source/dexgym/tasks/direct/` since the parent of initial port commit `70e4a6a` are confined to the new cube directory | No shared task-rule edits |
| SuperMimic collection, WM, differentiable actor and critic method | Generic core in `source/dexgym/algorithms/supermimic/dex_reference/`; `algorithm_parity_failure_active.json` compares WM8/Actor32 losses and parameter updates with pinned upstream code, including terminal data and an active failure head; all reported differences zero | Reference update parity verified for the tested case |
| No added actor trust gate | Core follows the pinned reference; task adapter supplies geometry, controller/command memory and reward; no trust-gating path | Requirement preserved |
| Correct native state/reset/DR boundary | `native_with_ground.json`: FK maximum 6.59e-7 m; gain/mass/size errors zero; pre-reset capture and partial-history reset checks pass; pushes exercised | Tested native boundary passes |
| 4096-environment training | `full_config_4096_scene_v2.json`, run provenance, and `final_training_audit.json` | Full configured batches and completed training verified |
| Train an actual policy from scratch | Original command has no checkpoint argument; `train.py` constructs fresh modules unless that argument is supplied; 8000 iterations and optimizer counters verified; two real-physics evaluations complete 2387 goals, including 292 multi-goal episodes with >=3 successes | Learned continuous reorientation demonstrated, not just a smoke test |
| Reviewable model, logs and reproduction commands | Final model path/hash, native/reference per-episode reports, task guide, live TensorBoard, and 60-second recorded behavior | Delivered |

The training process exited normally after saving both final checkpoints. Module
states in the evaluated `model_8000.pt` exactly match `model_final.pt`. The task
and optimizer core are unchanged from training commit `2e238d5`; subsequent
source changes are confined to evaluation, world-model diagnostics, rendering
and TensorBoard utilities.

The completed scope is the native task port and a verified trained policy. The
published actor remains stronger in mean completed goals: 6.47266 versus 4.66211
across the same two evaluation seeds. This is not a claim of exact released-model
performance, bitwise MuJoCo 3.11/3.12 trajectory identity, hardware transfer,
training-seed robustness, or guaranteed convergence on arbitrary future tasks.
The source-version and environment-count differences are separately audited;
none is asserted to be the cause of the performance gap.
