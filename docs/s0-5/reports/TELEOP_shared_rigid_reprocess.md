# Shared rigid START replay of the signed HMB cohort

`TELEOP-SHARED-RIGID-REPROCESS-01` replayed all 32 signed HMB roots through the existing Ubuntu MuJoCo Teleop chain. The derived dataset retains the 24 Train and 8 Validation roots, 308 partitions, 116,613 planned cutoffs and 114,532 HMB joint-valid cutoffs. Its review status is `DIAGNOSTIC_CANDIDATE`.

## Mapping and execution

The controller applies one fixed world rotation to both wrists. The translation is adjusted once at START to align the midpoint of the two human wrist targets with the corresponding measured robot midpoint. The established tracker-to-Wuji, Wuji-to-native-TCP and side-specific base frame conversions remain fixed. A single transform preserves the target pair's relative pose after the fixed tracker-to-Wuji conversions. The executed pose still depends on native IK, limits, control decisions and Servo dynamics.

The source snapshot came from `teleop_ws` at `30cf13bdc496c43b3016462cfa0a206334d1ff7f`. The isolated overlay used the controller from the 03 comparison, with the localhost domain guard expanded from 205 to 206–213 for parallel cold replays. It used the existing MuJoCo driver, exact loaded model and original simulation bringup. The world rotation and mount bridge came from the same historical calibration as the selected 03 replay. Same-day sensor mounts were unavailable for independent verification for all historical roots.

Each root ran an independent simulated Home and one reconstructed START. The four signed human streams enter the original controller in recorder-monotonic order. The observer measures newly produced command messages and MuJoCo feedback. The historical source captures do not establish the original live START event or original arrival at the controller. No historical robot commands or feedback were injected.

## Derived data

The Ubuntu artifact directory is `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e/runs/TELEOP-SHARED-RIGID-REPROCESS-01/`. Use the `candidate-session-bound/manifest.json` index. Each of its 32 root directories contains `commands_feedback.npz` and `manifest.json`. The root manifests bind the source, trace, paired rows, 54 named axes, array schema and NPZ hash. Partition slices keep the original HMB identities and order.

- `q_issued_rad` holds the 14-arm, 20-left-hand and 20-right-hand command groups observed by the cutoff. Each group has a separate mask, timestamp and sequence. The groups can have different timestamps. These observations do not assert actuator acknowledgment.
- `q_measured_rad` holds the coherent same-stamp MuJoCo joint state available by that cutoff. Command and feedback arrays are separate.
- The command join uses both header time and observer receipt time, a 40 ms freshness limit and the common observed ACTIVE/TASK session. It verifies native arm session and expiry, each hand's side, and the named joint mapping.
- `qd_issued_backward_rad_s` uses a one-sided difference within a signed partition and active epoch. Missing angles and unavailable derivatives use `NaN`. The index retains planned samples with incomplete or inactive data.

The exact M1 target and HMB validity still come from the original signed data. This export leaves them and their denominators intact. The archive contains original traces, the original export and the subsequent stricter session-bound export.

| Observation | Planned cutoff count | HMB joint-valid cutoff count |
| --- | ---: | ---: |
| Denominator | 116,613 | 114,532 |
| All three issued command groups observed | 100,940 | 99,982 |
| Coherent 54-axis MuJoCo feedback | 100,933 | 99,976 |
| Accepted or limited flags matched to an issued command for both arms | 60,028 | 59,452 |

| Controller status on all planned cutoffs | Left arm | Right arm |
| --- | ---: | ---: |
| Accepted diagnostic matched to command, rate limit flag false | 41,936 | 48,890 |
| Accepted diagnostic matched to command, rate limit flag true | 18,987 | 19,968 |
| Explicit HOLD report | 17,319 | 3,629 |
| Explicit fallback report matched by the classifier | 0 | 0 |
| Rejected report | 21 | 10 |
| Unknown or ambiguous command diagnosis | 22,682 | 28,448 |
| Inactive session or outside replay timeline | 15,668 | 15,668 |

Native recovery and braking fields remain in the raw trace. The explicit fallback category reflects the exporter classifier's coverage. Exact M1 IK feasibility is a separate diagnostic.

## Full-cohort comparison

The benchmark pairs 91,062 HMB-valid cutoffs with coherent MuJoCo feedback in both the historical original mapping A and shared-START mapping B. The unchanged M1 wrist metrics have 90,893 usable cutoffs after the existing wrist bridge and target checks. The 169 additional paired cutoffs remain in the geometry denominator. P50 and P95 are calculated over frames for each replay on the common scoring set.

| M1 metric, larger error over the two wrists | Original START A | Shared START B |
| --- | ---: | ---: |
| Position P50 | 360.90 mm | 379.84 mm |
| Position P95 | 715.58 mm | 701.92 mm |
| Orientation P50 | 63.52° | 55.50° |
| Orientation P95 | 103.65° | 89.43° |

Historically A had 92,005 coherent feedback cutoffs of 116,613. The B pair receipt has 100,933. They share 91,951 planned cutoffs with coherent feedback. There are 8,982 B-only, 54 A-only and 15,626 without either. The control observations remain sensitive to lifecycle availability and replay timing. Full-cohort A is historical context. The selected 03 experiment was the same-build attribution test.

At 7.94 seconds on the prior 03 render, the selected B simulation reduced the palm-plane angle from 69.68° to 5.84°, the absolute palm height difference from 100.44 to 2.12 mm, and the absolute fore/aft difference from 179.48 to 5.07 mm. Robot closest fingertip separation was 216.56 mm in A and 17.22 mm in B, with the calibrated human estimate of 34.22 mm. This individual example and the aggregate wrist measurements use different geometric targets. Full-cohort palm and nearest-tip statistics describe executed robot geometry. The live and signed hand landmark models differ, which prevents an independent aggregate fingertip tracking inference from those statistics.

## Qualification and review

The controller diagnostic counts mark reported arm acceptance, limiting, HOLD, rejection and unknown observations. A reported accepted arm can still use a constrained solution. Coverage of the command columns and observed simulated feedback is an availability result. V1-A keeps its existing 85% active-tracking requirement, which this replay has not qualified. The newly observed control states also do not revise the fixed M1 error domain or certify residual feasibility inside ±8°/±5°. No training authorization follows from the candidate.

The independent review checked source-vs-receipt causality, lifecycle consistency, named command groups, immutable HMB denominators and the limits of the status flags. Its findings required an exact shared session, an epoch header barrier and hand side validation before the final export. The native trace audit separately verified 32 single-session replays with four human inputs and STOP acknowledgment. All connected, position-valid hand feedback reported all axes online and none stale. The full software simulation remained isolated from robot hardware.

The replay, pair and export, final session-bound seal and geometry summary all exited successfully. The local nominal replay suite passed 20 tests, including future message, expired command, side mismatch, mixed-session, HOLD and rejection controls. The candidate index SHA-256 is `0b3af0676138cd54658b8d870eed5a9349ec35a422b254059ec6ced3ae61fd6b`. The [review receipt](../campaign/evidence/teleop_shared_rigid_reprocess_receipt.json) binds the index, geometry summary, trace audit, runtime assets and actual exit codes.
