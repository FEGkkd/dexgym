# CMU BVH → Marvin motion library: M1 source gate

## Outcome

The requested Full-6D source gate **did not pass**. No Marvin retargeting candidate, robot reference NPZ, MuJoCo PD replay, Newton MJWarp PD replay, or train/validation/test split was produced. This follows the requested rule to stop when the Pilot cannot establish reliable complete bilateral hand orientation.

## Verified source and asset

- CMU BVH mirror: [una-dinosauria/cmu-mocap](https://github.com/una-dinosauria/cmu-mocap), commit `09a07f54f3bbb58797325f009282d0b2048a2871`.
- Git tree: 2,548 BVH files in 113 Subject directories. The shallow sparse checkout materialized 34 files in Subject 141. Five additional files from Subjects 018, 019, 079, 111, and 113 were fetched from the pinned commit and checked against their Git blob SHA1. Full byte inventory of all 2,548 files was not obtained because repeated large Git fetches terminated with TLS/HTTP errors; SHA256 and frame metadata exist for the 39 audited files only.
- Marvin MJCF: existing `dual_arm_marvin.xml`, SHA256 `6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce`, 54 joints/actuators. No robot asset was modified.
- `READMEFIRST.txt` states that the conversion added an artificial T-pose at frame 0, uses 120 fps, and that finger and thumb motion was not captured. It also states that the conversion does not remove source noise. The first frame was removed from processed clips, then time was reset to zero.

## M1 measured result

| Check | Result |
| --- | ---: |
| BVH motions audited | 39 across 6 Subjects |
| 20-clip Pilot parsed into diagnostic NPZ | 20 |
| Pilot diagnostic frames / duration | 4,747 / 94.5913 s |
| Bilateral `Hand` rotation channels declared | 39 / 39 |
| Bilateral world SO(3) reconstructable by FK | 39 / 39 |
| Bilateral local `Hand` Z/Y rotation exactly constant | 39 / 39 |
| Bilateral local `Hand` X rotation dynamic | 39 / 39 |
| Hand instances exceeding conservative angular-speed threshold | 11 |
| Hand instances with an unrepaired >45° sample step | 1 |
| Full-6D source-quality accepted clips | **0 / 39** |

In `141_16` (Wave Hello), both `Hand` nodes declare `Zrotation Yrotation Xrotation`. The left local ranges are `0°, 0°, 6.5364°`; the right are `0°, 0°, 104.2338°`. The right local wrist angular-speed P95 is about `1568°/s`. The rotation matrices and XYZW quaternions are mathematically valid; maximum quaternion norm error over 39 clips was `2.22e-16`, and FK parent/child orientation composition error was `0°` at the precision of the audit. These checks establish a **BVH-reconstructed** world frame, not independently observed full palm orientation. The source gives no evidence for the two constant local wrist axes in the audited clips. The high-speed oscillation visible in the Wave Hello preview adds a motion-quality concern.

M1 diagnostic positions use an explicit assumed T-pose stature of 1.70 m because the mirror does not certify a source length unit. For `141_16`, this gives `0.0703371162 m/source-unit`. This is a visualization and diagnostic scaling assumption, not a physical measurement. To preserve exact clip duration, the current diagnostic resampler uses a uniform grid with effective rates 49.83–50.06 Hz across the Pilot; no diagnostic NPZ is admitted as a formal 50 Hz training reference.

## Gate decisions

- **M1 parser/FK:** Implemented and verified on real BVH. Declared Euler orders, hierarchy, artificial first-frame removal, FK, quaternion continuity, and SO(3) resampling are tested.
- **M1 reliable bilateral full hand orientation:** Failed. The Pilot provides no reliable independent three-axis local hand orientation. Every diagnostic NPZ has `hand_orientation_valid=false` for both sides, while metadata separately records that FK reconstruction is mathematically valid.
- **M2:** Not run because M1 source gate failed. No TCP/IK/collision result is claimed.
- **M3:** Not run because M1 source gate failed. No robot reference or PD physics result is claimed.

## Reproduction

Use the verified source checkout and the repository's Python 3.12 environment:

```bash
uv sync --locked --extra motion --group dev
python scripts/motion/download_cmu.py --subjects 141
python scripts/motion/download_spot_checks.py
python scripts/motion/inspect_cmu.py
python scripts/motion/process_human_motion.py --motion-id 141_16
python scripts/motion/process_human_motion.py --pilot
python scripts/motion/audit_cmu_subset.py
python scripts/motion/validate_m1.py
python scripts/motion/preview_human_motion.py 141_16
PYTHONPATH=source python -m pytest -q tests/motion
```

The five cross-Subject spot-check BVHs are stored outside Git at `/home/ubuntu/dexgym_data/cmu/spot_checks/`, with each Git blob hash verified against the pinned commit. The audit script includes them when present.

## Output locations

- `/home/ubuntu/dexgym_data/reports/bvh_inventory.json`
- `/home/ubuntu/dexgym_data/reports/human_orientation_audit.json`
- `/home/ubuntu/dexgym_data/reports/pilot_human_orientation_audit.json`
- `/home/ubuntu/dexgym_data/reports/m1_npz_validation.json`
- `/home/ubuntu/dexgym_data/human_motion_v1/diagnostic_motions/` (20 actual CMU-derived NPZ)
- `/home/ubuntu/dexgym_data/human_motion_v1/metadata/` (source hashes, hierarchy, scale and per-hand audits)
- `/home/ubuntu/dexgym_data/human_motion_v1/previews/141_16_orientation.mp4`

A different source with independently calibrated bilateral hand markers or glove orientation is required to reach the requested hand-orientation confidence, followed by Marvin M2/M3 validation.
