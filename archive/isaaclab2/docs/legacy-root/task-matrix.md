# Task Migration Matrix

| Legacy task | Isaac Lab form | Priority | Initial status | Main risk |
| --- | --- | ---: | --- | --- |
| ShadowHandReOrientation | DirectRLEnv | P0 | planned | quaternion and reset parity |
| ShadowHandLiftUnderarm | DirectRLEnv | P1 | planned | arm/hand asset composition |
| AllegroHandCatchUnderarm | DirectRLEnv | P1 | planned | two-arm action layout |
| ShadowHandCatchUnderarm | DirectRLEnv | P1 | planned | contact and success semantics |
| ShadowHandOver | DirectMARLEnv | P1 | reference available | multi-agent wrapper |
| ShadowHandTwoCatchUnderarm | DirectMARLEnv | P2 | planned | multi-agent reset |
| ShadowHandGraspAndPlace | DirectRLEnv | P2 | planned | object and goal randomization |
| ShadowHandPushBlock | DirectRLEnv | P2 | planned | contact reward stability |
| ShadowHandBlockStack | DirectRLEnv | P2 | planned | sequential object state |
| ShadowHandDoorOpen/Close | DirectRLEnv | P3 | planned | articulated object asset |
| ShadowHandBottleCap | DirectRLEnv | P3 | planned | jointed object and torque |
| ShadowHandKettle | DirectRLEnv | P3 | planned | object pose and liquid proxy |
| ShadowHandPen | DirectRLEnv | P3 | planned | thin object contact |
| ShadowHandScissors | DirectRLEnv | P3 | planned | coupled articulated object |
| ShadowHandSwitch | DirectRLEnv | P3 | planned | small contact geometry |
| Wuji Hand Reorientation | DirectRLEnv | P0 | official native USD runtime, 207/413 PPO smoke, checkpoint/play/JIT/ONNX export, and five-finger contact/friction validation passed; task success not yet demonstrated | initial contact retention, reward/cage tuning, native USD geometry/contact parity, disturbance/DR parity |
| Wuji Arm + Hand Reorientation | ManagerBasedRLEnv | P0 | MDP ownership scaffold; not registered | arm asset and wrist mounting contract |

The Wuji core alignment includes the 20-joint action order, 0.05 s control
period, filtered position targets, 207-dimensional history observation,
tag-frame goal representation, reset, success, cage, torque, and timeout
semantics. Isaac contact reporting and per-environment filtering are validated
by a direct force probe. The native USD exposes collision-bearing `right_finger{1..5}_tip_link` bodies, which are mapped by official name and calibrated with their mesh-center offsets. The current short PPO smoke is numerically stable and the 207/413
asymmetric actor-critic data path is validated, but task success is still zero
and learned behavior has not been demonstrated. Initial contact retention,
reward balance, disturbance/DR migration, and longer training remain open
task-level work. The arm-hand scaffold is deliberately not registered until a real
composite asset is selected.
