# GRAB right-hand object-centric in-hand motion selection (2026-11-09)

## Result

For the object-centered selection, prioritize **camera, toothpaste tube,
game controller, and elephant toy**. Their non-lift actions (`browse`/
`takepicture`, `squeeze`, `play`, and `inspect`) have candidate windows in all
three requested categories. **Flashlight, mouse, and stapler** are useful
supplements for finger contact exchange and object actuation, with fewer
translation examples. The
[object-focused CSV](grab_inhand_object_focus_20261109.csv) has 87
source/category entries from 65 distinct non-lift recordings. It includes
raw and processed frame indices, so these windows can be extracted without
changing the source data.

| Object | Priority | Non-lift rotation / translation / contact-exchange windows | Subjects |
| --- | --- | ---: | ---: |
| Camera | Core | 18 / 6 / 5 | 10 |
| Toothpaste tube | Core | 6 / 3 / 2 | 6 |
| Game controller | Core | 4 / 3 / 2 | 6 |
| Elephant toy | Core | 7 / 1 / 2 | 8 |
| Flashlight | Supplement | 9 / 1 / 5 | 7 |
| Mouse | Supplement | 4 / 1 / 1 | 5 |
| Stapler | Supplement | 5 / 0 / 2 | 5 |

These are **object-centered candidate counts**, not confirmed robot in-hand
manipulation successes. Cups, mugs, glasses, bottles, and hammers have many
high-scoring windows, but their `drink`/`pour`/`use` actions often include
whole-arm transport or tool use; they are deferred from the first in-hand
object set. Spheres, cylinders, and toruses have pose-symmetry ambiguity for
rotation and are also deferred.

## Full scan and category shortlist

The 1,335 local GRAB recordings for subjects `s1`–`s10` were scanned at 30 Hz.
All sources mapped into the current Hand 2.1 `hand21_right_geometry_feasible`
MotionBank; the scan produced 24,144 overlapping 2–3 second windows with no
read errors. After contact, motion, and diversity screening, the priority
catalog contains **591 source/category entries** from **446 distinct GRAB
recordings**. There is one best window per source and category, so a source can
appear in more than one category.

| Priority category | Train (`s1`–`s7`) | Validation (`s8`–`s10`) | Median category motion |
| --- | ---: | ---: | --- |
| Object rotation relative to right wrist | 228 | 102 | 91.9° rotation; 55.9 mm translation |
| Object translation relative to right wrist | 87 | 57 | 52.2 mm translation; 18.1° rotation |
| Alternating finger contact / regrasp candidate | 82 | 35 | 3 gains and 3 losses; 57.4° rotation |

The [54-row priority shortlist](grab_inhand_shortlist_20261109.csv) contains 12
train and 6 validation windows for each category. It covers all ten subjects,
27 object types, 50 distinct source recordings, and 52 distinct windows (138 s
after deduplicating two windows with multiple labels). Each row includes the
original GRAB source and 30 Hz frame interval plus the processed Hand 2.1 file
and its local frame interval. The complete 591-entry catalog remains local at
`artifacts/hand21/grab_inhand_priority_bank.json`.

Representative windows from the shortlist:

| Category | GRAB source and time | Relative rotation | Relative translation | Contact evidence |
| --- | --- | ---: | ---: | --- |
| Rotation | `s3/lightbulb_screw_1.npz`, 4–6 s | 118.7° | 18.2 mm | right-hand contact 100% |
| Rotation | `s4/elephant_inspect_1.npz`, 4–6 s | 175.3° | 32.0 mm | right-hand contact 98% |
| Rotation, held out | `s8/bowl_drink_2.npz`, 2–4 s | 177.0° | 15.7 mm | right-hand contact 100% |
| Translation | `s6/phone_call_1.npz`, 2–4 s | 20.9° | 116.8 mm | right-hand contact 100% |
| Translation | `s7/binoculars_lift.npz`, 9–12 s | 23.8° | 100.2 mm | right-hand contact 90% |
| Translation, held out | `s9/stanfordbunny_inspect_1.npz`, 3–6 s | 24.8° | 117.2 mm | right-hand contact 100% |
| Regrasp | `s3/duck_inspect_1.npz`, 4–7 s | 139.8° | 88.9 mm | 7 finger-contact gains, 6 losses; no contact-free gap |
| Regrasp | `s6/stapler_staple_2.npz`, 2–5 s | 14.3° | 24.2 mm | 3 gains, 5 losses; no contact-free gap |
| Regrasp, held out | `s9/mouse_lift.npz`, 4–7 s | 24.8° | 52.6 mm | 4 gains, 4 losses; longest gap 2 frames |

## Selection method

The [official GRAB description](https://github.com/otaheri/GRAB) documents the
per-frame hand and object poses and body/object contact annotations. The scan
maps GRAB body-mesh right-hand contact vertices to each subject's MANO finger
vertices, using the subject templates and MANO skinning weights. A finger is
active when at least two mapped body vertices carry a nonzero contact label;
the binary signal is smoothed over seven 30 Hz frames. The MANO-to-body
template mapping matched approximately 767 of 778 hand vertices within 1 mm
for every subject.

Object motion is measured in the right-hand root frame:

\[
R_{h\to o}(t)=R_h(t)^{-1}R_o(t),\qquad
p_{h\to o}(t)=R_h(t)^{-1}(p_o(t)-p_h(t)).
\]

The window rotation is the angle between the first and last five-frame mean
relative rotations. Translation is the distance between the first and last
five-frame median relative positions. This removes bulk hand transport from
the metric. A scan window needs right-hand contact for at least 80% of frames,
two-finger contact for at least 40%, contact at both ends, and no contact-free
gap longer than nine frames. Rotation starts at 45° relative rotation;
translation starts at 30 mm and is capped at 45° rotation. A second pass
requires object-world motion and nontrivial Hand 2.1 `q_ref` and `tip_ref`
motion. Symmetric primitives are excluded from the rotation category; `fly`,
`pass`, and `offhand` actions are excluded from the priority catalog.

The final catalog tightens the rotation category to at most 100 mm relative
translation and at least 90% right-hand and 80% two-finger contact. Translation
requires at most 25° rotation, at most 130 mm relative translation, at least
85% right-hand and 70% two-finger contact. Regrasp requires at least 93%
right-hand and 75% two-finger contact, at least two stable gains and losses
across at least three fingers, and a contact-free gap of at most three frames.
The shortlist covers each subject first, then fills by score with at most two
windows per subject and object in each split/category.

Every catalog interval was checked against the processed manifest: source,
split, frame offset, and segment bounds all matched. A manual contact-timeline
check on representative `duck_inspect`, `piggybank_lift`, `lightbulb_screw`,
and `hammer_lift` windows prompted the stricter gap requirement: some otherwise
high-scoring clips briefly lost all right-hand contact.

## Use and limits

The shortlist is a **motion candidate index**, not a verified physical
in-hand manipulation label. GRAB contact annotations indicate body/object
contact, not contact force or grasp stability. Relative object-center motion
can also arise from changing grip geometry; a contact exchange can be a touch
shift rather than a full regrasp. Before using object-motion labels as task
ground truth, inspect the source hand/object meshes and object trajectory for
the desired clip. The Hand 2.1 MotionBank contains right-hand `q_ref` and
`tip_ref`, but the current hand-only RL task has no simulated object; these
clips can currently prioritize finger-motion training or guide a future
object-aware task, rather than measure robot in-hand object motion.

Raw GRAB recordings and generated libraries were not modified. To regenerate
the local catalog from the repository root with the GRAB/MANO inputs and the
existing Hand 2.1 MotionBank:

```bash
PYTHONPATH=source /home/ubuntu/miniconda3/envs/dextrack-grab/bin/python \
  scripts/data/select_grab_inhand_motions.py \
  --data-root data \
  --manifest data/processed/hand21_right_geometry_feasible/manifest.json \
  --output artifacts/hand21/grab_inhand_windows.json
PYTHONPATH=source /home/ubuntu/miniconda3/envs/dextrack-grab/bin/python \
  scripts/data/curate_grab_inhand_motions.py \
  --scan artifacts/hand21/grab_inhand_windows.json \
  --data-root data \
  --output artifacts/hand21/grab_inhand_enriched.json
python scripts/data/build_grab_inhand_priority_bank.py \
  --input artifacts/hand21/grab_inhand_enriched.json \
  --output artifacts/hand21/grab_inhand_priority_bank.json \
  --csv-output docs/experiments/grab_inhand_shortlist_20261109.csv \
  --object-csv-output docs/experiments/grab_inhand_object_focus_20261109.csv
```
