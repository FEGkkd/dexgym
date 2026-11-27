# Existing MuJoCo Teleop nominal replay

Run `TELEOP-NOMINAL-20261127-30cf13b-01` replays the four human-input streams from the signed HMB cohort through the installed simulator and Teleop coordinator in `/home/ubuntu/teleop_ws`. The source workspace was on `4sensor` at `30cf13bdc496c43b3016462cfa0a206334d1ff7f` with existing user changes. The run records the source status and diff hash, executable hashes, loaded model hash and effective ROS parameters. It did not rebuild or edit that workspace. The reference code provenance and the actual loaded runtime are recorded separately.

This diagnostic uses native Teleop START-relative mapping, Marvin IK, Wuji SDK, existing safety and the existing MuJoCo Servo. The effective arm pipeline was `manimux_analytic_ik_position`, with full-rate arm diagnostics. The simulator was configured for 500 Hz physics and 120 Hz publication. Observed feedback counts are reported separately. Each root uses an isolated localhost ROS domain and a measured Home followed by a simulator START. Historical START and lifecycle are unknown in the signed native input assets. A repeated initial human pose is used for the diagnostic START outside the scored replay window. The receipt binds the MCAP bytes, embedded schemas, adjacent SampleObservation writer and monotonic clocks, and the original HMB interval. The only injected topics are the two wrist and two MANUS streams. The command and feedback observers read live output topics. This diagnostic has no D1, Actor, training or hardware qualification authority.

## First complete recording

The first root is `capture:8c85b5bd93595270f134bf72` (Train, 7 partitions, 2,549 HMB planned ticks). The input interval is approximately 50.965 s. The video has 50.933 s of nominal motion at 1×, rendered from time-matched 54-joint simulated state. Accepted and limited totals come from the actual arm diagnostics during ACTIVE/TASK. Control-loop diagnostics and signed 50 Hz HMB ticks have different denominators.

| Measurement | Left | Right |
| --- | ---: | ---: |
| Native arm diagnostic samples | 12,710 | 12,710 |
| Controller arm-command accepted flag | 12,710 | 12,704 |
| Arm rate-limited flag | 97 | 799 |
| Issued/feedback max arm axis error P95, rad | 0.01395 | 0.02610 |
| Issued/feedback max hand axis error P95, rad | 0.05059 | 0.06313 |

There were 5,085 coherent feedback stamps in the ACTIVE/TASK summary. The issued-command comparison has 5,084 causal 54-joint matches and one stamp without a prior complete command. The render receipt contains 5,094 complete 54-joint feedback frames over its input-time envelope. Aggregated replay publishing lag was 1.778 ms P95 and 2.180 ms P99. Its time origin is reconstructed from recorder monotonic time. Timing evidence concerns this replay rather than historical controller arrival.

The 105-probe named kinematics check found identical wrist positions and rotations at the tested joint vectors. The two XML files differ. World landmarks differed by up to 6.271 mm by coordinate and local hand landmarks differed by 6.354 mm. Fingertip-level comparisons between the two geometries require that difference to be resolved or explicitly modeled. The joint comparisons and wrist probe have no plant or safety-equivalence implication.

## Full signed cohort

All 32 unique HMB roots have observable four-stream playback, ACTIVE/TASK, coherent 54-joint feedback and STOP evidence. One initial attempt failed ROS graph discovery before control activation. Its independent cold-start retry completed, and both attempt logs are retained. The native summary had to use the recorder monotonic schedule for eligibility. Raw log times occasionally go backwards in that schedule, and crossing those log times with the control timeline had initially rejected the otherwise valid traces.

| Measurement | Cohort result |
| --- | ---: |
| Human inputs published | 1,116,094 |
| Coherent ACTIVE/TASK 54-joint stamps | 183,899 |
| Observed 14-axis arm command messages | 458,590 |
| Observed hand command messages, left / right | 220,804 / 220,803 |
| Controller arm-command accepted flag, left | 440,819 / 457,364 (96.38%) |
| Controller arm-command accepted flag, right | 445,531 / 457,364 (97.41%) |
| Controller rate-limited flag, left / right | 86,879 / 105,932 of 457,364 each |
| Signed planned HMB ticks with causal coherent native feedback | 92,005 / 116,613 |
| Signed HMB-valid ticks with causal coherent native feedback | 91,113 / 114,532 |
| Planned ticks outside observed ACTIVE/TASK | 24,528 |
| Planned ticks outside replay, missing feedback, or stale feedback | 70 / 7 / 3 |

Online and offline share the underlying human capture. Online uses the live simulator, measured initial Home, one native START per root, online target mapping and native guard. DCR-008 uses partition/generation resets, M1 target poses, virtual previous-command feedback and additional swept admission. Its sealed traces report 42,358 LIMITED_ACTIVE, 71,974 HOLD and 2,281 FAILED on the same 116,613 planned ticks. The observed native acceptance rates and DCR 50 Hz classes use different units and eligibility.

On 90,944 HMB-valid paired ticks the actual/signed wrist FK agreed at the measured configuration. For those rows, the larger of the two wrist position offsets from M0 and M1 has P95 810 and 716 mm, respectively. Those offsets compare the native START-relative execution to the frozen absolute targets. The same-tick max absolute 54-joint difference between measured online and offline nominal has P95 2.045 rad. These results establish substantial discrepancies between operating points and target definitions. A target-only comparison at identical state, START and previous command is needed to attribute the offset to M1 design rather than the registration, start or state seam. The geometry difference prevents a matched fingertip accuracy conclusion.

## Signed whole-body path shadow

The fixed first-root plan selected 400 samples at 120 ms spacing from the replay origin. It was hashed before the guard evaluation. One sample preceded observable ACTIVE/TASK and 399 could be evaluated. The exact DCR-008 endpoint test and bounded straight joint-space proof were run on retrospectively assembled issued command snapshots.

| Starting state | Endpoint failed | Sweep unproven | Sweep proven |
| --- | ---: | ---: | ---: |
| Last observed issued snapshot | 154 | 3 | 242 |
| Last coherent measured snapshot | 154 | 12 | 233 |

All 154 endpoint failures violated signed collision clearance and 124 showed penetration at the proposed endpoint. Joint limits and J6/J7 coupling passed those 154 endpoints. The most frequent rejecting signed pairs were among the right finger tip colliders. Repeating the endpoint test on the exact live XML gave 236 endpoints that passed both models, 154 that failed both, and nine that passed the signed model and failed the live model. This compares the same 399 named configurations using `RobotModel.legality(q_observed_issued_proposal_rad)` on each XML. These results establish that imposing the extra static clearance and path proof can remove many commands that the native controller publishes. They cannot establish the swept path taken by the Servo, change the safety authority, or explain every DCR HOLD across the cohort.

## Evidence identity

The Ubuntu archive is under `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e/runs/TELEOP-NOMINAL-20261127-30cf13b-01/`. The local video is copied separately. Raw input and telemetry remain on the data host.

| Artifact | SHA-256 |
| --- | --- |
| Byte-verified `cohort.json` (32 roots, 308 partitions, 116,613 planned ticks, 128 human MCAPs) | `7d6c4702229aefd0d564f76dfae5a2a7b62a0340b9c4cae43fea512cc2d531c2` |
| `root01/root01-full.mp4` | `c1538efa8274c0c1923d20a2f4b049393f414fde58ea5b4f7f97cc7f466dd528` |
| `root01/root01-full.csv.gz` | `678076a8adab6d09f919ac730a202df2463196b8c720e54911547fd86211de73` |
| `root01/root01-full.png` | `94ea46ac9c4e5d724b9d6eec5e22bfdced22099427572aad631f1fe5b6a608cf` |
| `root01/native-nominal-summary.json` | `08f76ac1e3dacbb2505eebf3476a09f4191699f65b2a34bff1f3cd07de4abce5` |
| `kinematic-probe.json` | `8b5abd7a970452a05bf0a03f8a0c6b1d7e150bb5ac592db6738700357d0080db` |
| `native_cohort_summary.json` | `21ad4502dcb29fa4d81a05b96183f3dceea2be7396d2f1273a28fc23c1797358` |
| `paired_cohort_summary.json` | `4276c8d5e65975845ad301217b7ec276748c3051faf9820c5b8c16e551ebfe40` |
| `root01/shadow-plan.json` | `d37917fc0247a58a18e579905a2844ee237d4c32c385ab999edac0667e68d5f1` |
| `root01/shadow.json` | `927b187aa80f8b9d00ba76363d2e948f166642e9ec03c61eb8586366705cb719` |
| `root01/endpoint_collision_model_comparison.json` | `d8d5166e7cc7d8f397bfcb598292e1cf69ac92fed23fbdd585198c1c7c0a2bb7` |
| Actual Teleop executable | `51c8b624aa769a02d75116c48067c90247a2b6774a53eb6c87e46b0325ebdb85` |
| Actual MuJoCo driver executable | `03a3119b94359ff6d429cfb18c5e16d2eb612866f8d21a25f679a18186cf6d7f` |
| Loaded MuJoCo XML | `be69f387f5bc25c8011a992a88ac2c2449c2c0419e09318b740fcdcd1402871d` |
| Signed DCR-008 MuJoCo XML | `939b3651e2426e596eed482e70a68c2382720df1fd55b124aa5f0893b640e92d` |

The first video CSV denotes the issued target position as `requested_rad`. The current exporter uses `issued_rad` for the same field.
