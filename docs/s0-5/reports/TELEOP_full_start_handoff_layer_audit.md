# Full START handoff replay and wrist error decomposition

`TELEOP-SHARED-RIGID-START-HANDOFF-FULL-01` replayed all 32 signed HMB roots through the existing MuJoCo Teleop chain using the measured START handoff. The original cohort contains 24 Train roots, 8 Validation roots, 308 partitions, 116,613 planned cutoffs and 114,532 HMB joint-valid cutoffs. All 32 roots reached START and completed one historical replay on their first attempt. The `START_NOT_REACHED` list is empty.

## Simulation and data retention

Every root used a fresh localhost simulation in ROS domains 206–213, the original MuJoCo driver and the verified shared mapping overlay. The first signed four-stream HMB anchor selects the original MANUS and Tracker samples. The simulated robot returned to measured palms-down Home before the controller captured one fixed shared map. The largest measured joint difference from the validated 03 Home reference was 0.00497 rad. Each root held its fixed target until measured wrist and hand criteria were met for a continuous feedback dwell. Elapsed time after Enable ranged from 1.733 to 3.305 s, with a median of 1.772 s.

All successful replays preserved the original input order under one clock shift after the causal anchor. They published 1,115,529 source messages and wrote matching input seals. The prefix before each eligible anchor remains in the signed planned denominator. The run verified clean shutdown of the owned simulation processes and empty isolated graphs. An unsuccessful strict START would retain its raw trace, original targets and plan rows while blocking the historical portion of that attempt.

The complete data export has one NPZ and one schema receipt per root. Each record retains the signed partition and cutoff, HMB masks, the original M1 wrist pose, a shared target computed from the causal HMB trackers, the controller's native unfiltered and intermediate targets, and separate actual issued joint angles and measured joint angles. The physical wrist comparisons use the same live `tcp_L` and `tcp_R` world frames. The unchanged signed bank remains the source of the full independent M1 geometry. Wrist scoring uses bilateral arm feedback and the complete 54-axis arrays retain their own group validity masks.

| Availability | Planned cutoffs | HMB joint-valid cutoffs |
| --- | ---: | ---: |
| Denominator | 116,613 | 114,532 |
| Two M1 wrist targets | 114,332 | 114,332 |
| Two shared HMB wrist targets | 114,532 | 114,532 |
| Two wrist FK poses from issued commands | 100,915 | 99,960 |
| Two wrist FK poses from simulation feedback | 100,807 | 99,854 |
| All 54 issued axes | 100,728 | 99,778 |
| All 54 measured axes | 95,912 | 95,018 |

## Same-cutoff error decomposition

On 99,562 HMB-valid cutoffs all four layers were available and the raw native shared target passed the independent reconstruction check. The check agreed within 0.1 mm and 0.001 rad on all 201,566 available side records. Each per-cutoff statistic uses the worse wrist. The adjacent vector differences telescope in world position. Their norms and percentiles, and the SO(3) geodesic angles, are separate statistics.

| Comparison on the same 99,562 cutoffs | Position P50 | Position P95 | Orientation P95 |
| --- | ---: | ---: | ---: |
| Frozen M1 → native shared target | 380.36 mm | 693.28 mm | 87.33° |
| Native shared target → issued wrist FK | 2.10 mm | 79.11 mm | 44.14° |
| Issued wrist FK → measured MuJoCo wrist FK | 3.79 mm | 18.66 mm | 4.03° |
| Frozen M1 → measured MuJoCo wrist FK | 379.70 mm | 699.49 mm | 88.35° |

Among the 4,979 cutoffs at or above the M1-to-feedback position P95, the M1-to-shared link had the largest position displacement on the same worse wrist in 4,972 cases. The shared-to-issued link was largest in 7 cases. The issued-to-feedback link was largest in none. The independent HMB shared target corroborates the result on 99,657 common cutoffs with a 693.24 mm M1-to-shared position P95.

The target difference contains both common translation and changes to the relative wrist vector. On the 99,562 common cutoffs, the M1-to-shared midpoint displacement P50/P95 was 123.68/361.00 mm. The difference between the two methods' right-minus-left wrist vectors was 576.59/930.82 mm. The midpoint explained 26.25% of the sum of squared position discrepancies on all common cutoffs and 65.57% within the worst position tail. This decomposition uses the two signed wrist displacement vectors. It distinguishes a single shared displacement from disagreement in the two-wrist geometry. [Mapping geometry receipt](../campaign/evidence/teleop_full_start_handoff_mapping_geometry.json)

## Remaining control discrepancies

Pair-specific comparisons locate a smaller but material discrepancy between the native filtered target and the solver step target. Their position and orientation P95 were 76.74 mm and 44.18°. This link contains workspace shaping and rate limiting. The step target to the actual IK request measured 0.94 mm and 0.23°, and IK request to issued FK measured 1.31 mm and 0.44°. These pairs have their own support counts in the index. The issued-to-feedback link is a same-cutoff discrepancy with independent command and feedback stamps. It includes response delay and servo dynamics.

On 94,907 HMB-valid cutoffs with complete issued and measured vectors, the 54-axis maximum absolute angle difference had P95 0.2645 rad. Arm-only P95 was 0.0626 rad on 99,839 cutoffs. The corresponding left and right hand P95 were 0.2147 and 0.1876 rad on their respective usable subsets. These joint results describe observed execution and do not provide a fingertip-level independent intent score.

The large absolute M1-relative wrist offset is present upstream of command generation. The largest observed secondary discrepancy is between the filtered and solver step targets. Comparing the two target definitions establishes their divergence. Choosing which target matches the intended physical calibration requires a separate reference validation.

## Artifacts and checks

The server archive is `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e/runs/TELEOP-SHARED-RIGID-START-HANDOFF-FULL-01/`. Use `layer_index.json` and `layers-verified/<capture-id>/wrist_layers.npz`. The index binds the 32 NPZ and schema receipts, outcomes, signed denominators and all result masks. Its SHA-256 is `2bc6a63922e9f7790b4366690401c1d3934bd405b1a135cab1266c068fb1acb4`.

The native cohort and verified layer scorer both exited successfully. The first score invocation lacked three archived helper modules. The retry used the same untouched native traces and preserved the first logs. Local Teleop tests passed 20 cases. Independent checks covered causal duplicate feedback handling, immutable denominators, same-session Home and START, matching trace and anchor hashes, and same-support error arithmetic. The result has diagnostic candidate status. [Full review receipt](../campaign/evidence/teleop_full_start_handoff_layer_receipt.json)
