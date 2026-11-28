# High-error roots at matched bimanual cutoffs

The three witnesses come from the sealed strict START rerun `TELEOP-SHARED-RIGID-START-HANDOFF-FULL-01`. Roots were ordered by the P95 of the larger M1-to-measured TCP position error across their paired HMB-valid cutoffs with complete measured 54-axis state. Eligibility also required the shared HMB target and a reconstructed native shared target. Each image uses the real cutoff closest to that root's P95. All three selected roots are in Train and illustrate high-error development examples.

| Root prefix and task | Signed cutoff | Paired root support | Root P95 |
| --- | ---: | ---: | ---: |
| `b70297470e5e`, fruit | 19.260000057 s | 2,102 | 1,042.30 mm |
| `6fb07249fb5b`, arm motion | 20.020000068 s | 2,277 | 823.83 mm |
| `2151546f1c0b`, arm and hand | 77.840000058 s | 4,165 | 813.82 mm |

## How to read the figures

The four panels use the signed causal Tracker and MANUS geometry, the frozen M1 target, the shared target built from the same raw cutoff and fixed START map, and actual MuJoCo feedback. The human reconstruction retains observed MANUS landmark distances and uses the historical tracker-to-palm calibration. That historical calibration has no per-recording physical mount verification. The human panel uses the same world placement as the shared map for display, so their coincidence is the expected property of the transform.

For shape comparison the upper panels place each wrist midpoint at the visible origin. All four share world orientation, an orthographic viewpoint and one scale. The lower views and per-panel coordinates preserve uncentered world midpoints. Wrist distance in the images uses the anatomical palm root. The rank uses the sealed common TCP-site score. The reported orientation is the directed angle between the two palm normals. The acute plane angle is also shown in each image.

| Root | Original wrist distance and normals | M1 | Shared | Measured MuJoCo |
| --- | ---: | ---: | ---: | ---: |
| `b70297470e5e` | 375.91 mm, 52.39° | 408.71 mm, 18.85° | 375.91 mm, 52.39° | 376.58 mm, 42.99° |
| `6fb07249fb5b` | 560.18 mm, 1.95° | 614.72 mm, 42.55° | 560.18 mm, 1.95° | 572.96 mm, 114.50° |
| `2151546f1c0b` | 830.34 mm, 1.78° | 893.87 mm, 42.30° | 830.34 mm, 1.78° | 806.44 mm, 8.61° |

## First point of divergence

The global registrations already differ before any per-hand mount or finger reconstruction. The historical vertical calibration maps the registered recording's `+Y` to MuJoCo world `+Z` in the shared mapping. The frozen M1 places the same axis along `(-0.847, -0.532, 0)` in the horizontal plane. The two global rotations differ by 95.73°. At this stage the paired world midpoints already differ by 836.37, 637.79 and 476.13 mm. Both unit-scale registrations preserve the raw Tracker-origin separation at this step. The three original separation values are 414.00, 617.28 and 887.77 mm, with differences below 0.003 mm after the global placements.

The anatomical geometry then diverges through the side-specific mounts, the TCP-to-palm offsets and the M1 robot hand reconstruction. The distances in the upper figure panels measure the resulting anatomical wrists. M1 increases them by 32.79, 54.54 and 63.53 mm relative to the primary human reconstruction. The relative SO(3) errors between the two palm frames are 54.25°, 41.87° and 45.35°. No hard-pinch adjustment fired in these three frames.

The shared map applies one rigid transform to the two calibrated human palms and all hand landmarks. Invariance in the first and third panels is verified numerically. With an alternate historical calibration as a sensitivity check, the shared target's relative palm errors were 3.34–3.59°, versus 39.95–52.41° for M1. Across the 25 matched cross-hand fingertip distances, the alternate-calibration mean absolute discrepancies were 3.10–3.94 mm for shared and 26.91–81.72 mm for M1. The reproduced values support the relative preservation claim for these witnesses under both historical calibration assumptions.

## Target and execution

The second case illustrates why target geometry and execution must be scored separately. The shared target retains nearly parallel same-direction normals at 1.95° on the selected cutoff. The measured normals are 114.50°. At that cutoff native shared-to-issued TCP orientation errors are 69.30° on the left and 59.61° on the right. The issued-to-measured errors are 0.38° and 0.24°. The native diagnostics record a left `backoff` and limiting in both arms. The large orientation discrepancy is already present in the issued references on this sample.

The original hand input records, sealed NPZ arrays, mapping configuration and simulated feedback are bound in the [witness receipt](../campaign/evidence/teleop_high_error_mapping_witness_receipt.json). The solver read-only replay archive is under `witnesses-high-error-01` in the full strict START run. It retains the extracted geometry and corrected millimetre figures. The figures use recorded data and the already completed simulation traces.
