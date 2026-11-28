# Measured START pre-position in the 03 MuJoCo replay

`TELEOP-SHARED-RIGID-START-HANDOFF-03-01` compared the same shared rigid START mapping with immediate playback and a measured pre-position before playback. Both conditions used the verified shared-map controller, installed MuJoCo driver and identical effective parameters. Each was a separate cold localhost simulation with one accepted native START and Enable and one accepted Stop. The original four human input streams share one source-event digest across the two completed replays.

## START construction and measurement

The signed causal HMB anchor is 80 ms after capture origin. It binds complete left and right MANUS plus Tracker samples to original physical message indices and CDR hashes. Twenty-eight messages precede the anchor. The runner keeps that prefix as source evidence and begins playback on the signed anchor timeline using a single timestamp shift. The Home anatomy check found both palm normals approximately 4.2° from world down in both conditions.

The measured Home wrists set the target midpoint. The fixed calibrated world rotation and the shared translation `p_home_mid - R p_human_mid` apply to both sides. A rigid transform preserves the separation, relative orientation and wrist-to-wrist geometry of the calibrated target pair. MANUS retains its original hand shape samples and the existing Wuji retarget session computes the hand commands. The generated transition is outside the scored historical timebase.

The Home wrist-origin separation was 523.04 mm. The calibrated human START mapped to a 484.26 mm separation with a midpoint coincident with Home to floating-point precision. The mapped anatomical palm plane angle was 63.99°. The measured settled angle was 64.04°. The robot therefore reached the angled START pose instead of remaining at its nearly parallel Home orientation.

After Enable, the pre-position condition held the frozen four-stream input while the native Marvin IK, Wuji SDK and existing controller limits moved the simulated robot. The pilot used a 10 s limit and a 0.2 s dwell after the 1.5 s native hand ramp interval. Candidate checks were 10 mm and 5° for both wrists, 0.10 rad for each hand's maximum measured-to-issued joint difference, and 0.16 rad/s for maximum measured joint speed. The native same-session ACTIVE/TASK and both arms' accepted target flags were required.

| Handoff observation | Immediate at Enable | Measured pre-position |
| --- | ---: | ---: |
| Time spent in explicit hold | 0 s | 3.185 s |
| Left wrist error to mapped START | 57.283 mm / 60.108° | 0.02254 mm / 0.00998° |
| Right wrist error to mapped START | 60.156 mm / 38.454° | 0.00049 mm / 0.00008° |
| Longest final continuous dwell satisfying the pilot thresholds | Not evaluated | 0.206 s |

At handoff, the measured hand-to-issued errors were below 0.00010 rad and max observed joint speed was 0.00727 rad/s. These are simulated feedback results. The wrist comparison uses the native unprojected endpoint reconstructed from the START latch and calibrated frames. The hand comparison uses the currently issued retargeted joint commands.

## Playback results

The 03 recording has 3,840 signed planned cutoffs and 3,831 HMB joint-valid cutoffs. Each condition produced coherent simulated feedback at 3,835 planned cutoffs. The common frozen M1 wrist scoring set contains 3,830 joint-valid cutoffs.

| Larger of the two M1 wrist errors | Immediate | Pre-position |
| --- | ---: | ---: |
| Position P95 | 416.765 mm | 416.546 mm |
| Orientation P95 | 113.768° | 114.107° |

At the first common historical tick, 100 ms after capture origin, the immediate condition still had both arms rate-limited. The pre-position condition began from the mapped pose with the recorded hand shape already retargeted. At the earlier video cut labeled 7.94 s, the same signed HMB cutoff is 56,720,000,069 ns after capture origin.

| Live robot geometry at the 7.94 s cut | Immediate | Pre-position |
| --- | ---: | ---: |
| Acute palm plane angle | 10.14° | 5.68° |
| Right-minus-left palm height | −5.06 mm | −1.92 mm |
| Right-minus-left palm fore/aft | 5.29 mm | 5.07 mm |
| Closest cross-hand fingertip separation | 24.28 mm | 16.96 mm |

The prior calibrated estimate at this cut put the human palm plane angle at 9.34° and closest tips 34.22 mm apart. Both measurements on the pre-position replay moved farther from those two human estimates. Closest tip distance can involve different finger pairs across conditions.

Pre-positioning resolves the initial condition in this example. The sustained absolute tracking error needs separate diagnosis of registration, target reachability and native execution. For a future root whose fixed START misses the acceptance window, record the actual failure and preserve the intent. A timeout alone leaves geometric reachability unknown. This single-recording experiment carries diagnostic status.

The Ubuntu archive is `/home/ubuntu/dexgym_data/s0-5/campaign/S05-V1-20261127-490770e/runs/TELEOP-SHARED-RIGID-START-HANDOFF-03-01/`. It contains the original traces, anchor, paired scores and `comparison-verified` plots. The [review receipt](../campaign/evidence/teleop_start_handoff_03_receipt.json) binds those artifacts to their hashes.
