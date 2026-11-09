# Hand 2.1 right-hand motion distribution (2026-11-09)

## Population and measurement

This analysis uses the exact `hand21_right_geometry_feasible` bank loaded by
the active RL environment: records with at least six 30 Hz frames. It covers
942 train segments (281,793 frames, 2 h 36 m 33 s) and 446 validation
segments (122,482 frames, 1 h 8 m 3 s). The manifest has 954 train and
450 validation records; the task filters 12 short train and four short
validation records. All speed and displacement differences are computed
within each segment, never across segment boundaries. `tip_ref` is robot FK
in the right-wrist frame; global wrist/arm motion is outside this analysis.

Statistics below are frame-weighted unless explicitly called clip-weighted.
The task samples segment IDs uniformly at reset, so its training exposure is
not exactly the same as this frame-weighted distribution. A speed cutoff of
50 mm/s is used only to describe movement, not as a task success rule.

## Speed and multi-finger activity

| Metric | Train | Validation |
| --- | ---: | ---: |
| Segment duration, clip-weighted median | 8.43 s | 7.83 s |
| Largest of five tip speeds, p50 / p90 / p99 | 63.5 / 187.1 / 348.5 mm/s | 58.1 / 166.5 / 340.4 mm/s |
| All five tips below 20 mm/s | 15.2% | 17.4% |
| At least one tip above 100 mm/s | 31.2% | 27.2% |
| No tip above 50 mm/s | 40.7% | 44.1% |
| Four or five tips above 50 mm/s | 30.0% | 24.7% |

The train set has a somewhat heavier fast-motion tail. It contains both slow
periods and coordinated movement: 40.7% of within-clip transitions have no
tip above 50 mm/s, while 30.0% have four or five above that descriptive
cutoff. A clip-weighted median of each clip's median maximum-tip speed is
66.1 mm/s for train and 63.9 mm/s for validation; the 90th percentiles are
102.1 and 91.0 mm/s.

| Finger | Train tip speed p90 | Validation tip speed p90 |
| --- | ---: | ---: |
| Thumb | 87.5 mm/s | 86.3 mm/s |
| Index | 125.7 mm/s | 108.9 mm/s |
| Middle | 138.8 mm/s | 122.3 mm/s |
| Ring | 142.0 mm/s | 123.6 mm/s |
| Little | 155.7 mm/s | 128.3 mm/s |

The thumb is the slowest tip in these wrist-relative references; the little
finger is the fastest. Joint-angle coverage is also uneven: the train set's
central 90% range is 0.34 rad for `r_RFJ3` and 2.19 rad for `r_LFJ0`.
This identifies less-exercised joints; it does not by itself establish a
retargeting defect or a physical joint-limit problem.

## Geometric target jump by reference span

For each span, the table measures the **largest of five fingertip distances**
between two reference frames separated by that many 30 Hz frames. These are
geometric goal jumps, not simulated tracking errors. The 15 mm tracking
tolerance is shown only to convey scale: a jump above 15 mm can still be
reached by the controller within the task's failure budget.

| Span | Train p50 / p90 | Train jumps <15 mm | Validation p50 / p90 | Validation jumps <15 mm |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.1 / 6.2 mm | 99.8% | 1.9 / 5.6 mm | 99.8% |
| 4 | 8.1 / 23.5 mm | 75.2% | 7.5 / 21.1 mm | 79.0% |
| 10 | 19.1 / 50.3 mm | 39.8% | 17.5 / 45.8 mm | 43.6% |
| 20 | 33.6 / 76.2 mm | 18.7% | 30.2 / 70.0 mm | 22.6% |
| 40 | 49.5 / 97.2 mm | 8.1% | 44.8 / 93.2 mm | 10.5% |

The curriculum changes the physical size of a typical target substantially.
At span 1, almost every reference jump is already below 15 mm; at span 40,
only 8.1% of train and 10.5% of validation jumps are. This makes performance
by exact span more informative than one overall reach metric. The validation
set is moderately slower and has smaller median target jumps than train, so
held-out performance should still be reported by span and subject.

The reproducible calculation is
`scripts/data/analyze_hand21_motion_distribution.py`. Local generated files,
excluded from Git, are `artifacts/hand21/motion_distribution_20261109/summary.json`
and `motion_distribution.png`.
