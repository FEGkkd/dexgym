"""Build a diverse right-hand in-hand motion shortlist from enriched GRAB windows.

The output is an index of existing raw and retargeted frames. It does not copy
or modify GRAB recordings or the Hand 2.1 MotionBank.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

CATEGORIES = ("rotation", "translation", "regrasp")
SPLITS = ("train", "validation")
EXCLUDED_ACTIONS = ("_fly", "_pass", "_offhand")
FOCUS_TIERS = {
    "camera": "core",
    "toothpaste": "core",
    "gamecontroller": "core",
    "elephant": "core",
    "flashlight": "supplementary",
    "mouse": "supplementary",
    "stapler": "supplementary",
}


def eligible(row: dict, category: str) -> bool:
    """Keep sustained, finger-active windows with one clear primary label."""
    if not row[f"high_confidence_{category}"]:
        return False
    if any(action in row["source"] for action in EXCLUDED_ACTIONS):
        return False
    if category == "rotation":
        return (row["relative_translation_mm"] <= 100
                and row["hand_contact_fraction"] >= 0.90
                and row["two_finger_contact_fraction"] >= 0.80)
    if category == "translation":
        return (row["relative_rotation_deg"] <= 25
                and row["relative_translation_mm"] <= 130
                and row["hand_contact_fraction"] >= 0.85
                and row["two_finger_contact_fraction"] >= 0.70)
    if category == "regrasp":
        return (row["hand_contact_fraction"] >= 0.93
                and row["two_finger_contact_fraction"] >= 0.75
                and row["max_no_contact_gap_frames"] <= 3
                and min(row["contact_gains"], row["contact_losses"]) >= 2
                and row["fingers_changed"] >= 3)
    raise ValueError(category)


def priority(row: dict, category: str) -> float:
    """Rank motion strength while rewarding sustained contact and robot motion."""
    tip_factor = min(row["robot_tip_motion_mean_mm"], 70) / 70
    contact = row["hand_contact_fraction"] * row["two_finger_contact_fraction"]
    if category == "rotation":
        return row["relative_rotation_deg"] * tip_factor * contact * (
            1 - row["relative_translation_mm"] / 200
        )
    if category == "translation":
        return row["relative_translation_mm"] * tip_factor * contact * (
            1 - row["relative_rotation_deg"] / 90
        )
    if category == "regrasp":
        exchanges = min(row["contact_gains"], row["contact_losses"], 6)
        return exchanges * row["fingers_changed"] * tip_factor * contact
    raise ValueError(category)


def best_per_source(rows: list[dict], category: str) -> list[dict]:
    """Collapse overlapping 2/3-second windows to one window per GRAB source."""
    best = {}
    for row in rows:
        if not eligible(row, category):
            continue
        source = row["source"]
        score = priority(row, category)
        if source not in best or score > best[source][0]:
            best[source] = score, row
    return [dict(row, priority_score=round(score, 3)) for score, row in sorted(
        best.values(), key=lambda pair: (-pair[0], pair[1]["source"])
    )]


def diverse_shortlist(rows: list[dict], count: int, subject_cap: int = 2,
                      object_cap: int = 2) -> list[dict]:
    """Cover every subject first, then fill by rank with subject/object caps."""
    chosen, used_sources = [], set()
    subjects, objects = Counter(), Counter()

    def take(row: dict) -> bool:
        subject = row["source"].split("/")[0]
        obj = row["object"]
        if (row["source"] in used_sources or subjects[subject] >= subject_cap
                or objects[obj] >= object_cap):
            return False
        chosen.append(row)
        used_sources.add(row["source"])
        subjects[subject] += 1
        objects[obj] += 1
        return True

    for subject in sorted({row["source"].split("/")[0] for row in rows}):
        for row in rows:
            if row["source"].split("/")[0] == subject and take(row):
                break
    for row in rows:
        if len(chosen) >= count:
            break
        take(row)
    return chosen[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("artifacts/hand21/grab_inhand_enriched.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/hand21/grab_inhand_priority_bank.json"))
    parser.add_argument("--csv-output", type=Path,
                        default=Path("artifacts/hand21/grab_inhand_shortlist.csv"))
    parser.add_argument("--object-csv-output", type=Path,
                        default=Path("artifacts/hand21/grab_inhand_object_focus.csv"))
    parser.add_argument("--train-shortlist", type=int, default=12)
    parser.add_argument("--validation-shortlist", type=int, default=6)
    args = parser.parse_args()
    enriched = json.loads(args.input.expanduser().read_text())
    result = {
        "input": str(args.input),
        "definition": "Candidate in-hand motion windows, not verified object manipulation labels",
        "selection": {
            "one_best_window_per_source_and_category": True,
            "excluded_actions": list(EXCLUDED_ACTIONS),
            "shortlist_subject_cap": 2,
            "shortlist_object_cap": 2,
        },
        "categories": {},
    }
    for category in CATEGORIES:
        category_result = {}
        for split in SPLITS:
            windows = [row for row in enriched["windows"] if row["split"] == split]
            catalog = best_per_source(windows, category)
            quota = args.train_shortlist if split == "train" else args.validation_shortlist
            shortlist = diverse_shortlist(catalog, quota)
            if len(shortlist) < quota:
                raise ValueError(f"Insufficient diverse {category}/{split} candidates: {len(shortlist)}/{quota}")
            category_result[split] = {
                "eligible_windows": sum(eligible(row, category) for row in windows),
                "catalog": catalog,
                "shortlist": shortlist,
            }
            print(f"{category}/{split}: {len(catalog)} source clips, "
                  f"{len(shortlist)} shortlisted", flush=True)
        result["categories"][category] = category_result
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"WROTE {output}")
    csv_output = args.csv_output.expanduser()
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "category", "split", "rank", "source", "object", "start_s", "stop_s",
        "start_frame", "stop_frame_exclusive", "processed_file", "processed_start_frame",
        "processed_stop_frame_exclusive", "relative_rotation_deg", "relative_translation_mm",
        "hand_contact_fraction", "two_finger_contact_fraction", "contact_gains",
        "contact_losses", "fingers_changed", "robot_tip_motion_mean_mm", "priority_score",
    )
    with csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for category in CATEGORIES:
            for split in SPLITS:
                for rank, row in enumerate(result["categories"][category][split]["shortlist"], 1):
                    writer.writerow({name: category if name == "category" else
                                     rank if name == "rank" else row[name] for name in fields})
    print(f"WROTE {csv_output}")
    object_csv_output = args.object_csv_output.expanduser()
    object_csv_output.parent.mkdir(parents=True, exist_ok=True)
    object_ranks = Counter()
    with object_csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("tier", *fields), lineterminator="\n")
        writer.writeheader()
        for category in CATEGORIES:
            for split in SPLITS:
                for row in result["categories"][category][split]["catalog"]:
                    obj = row["object"]
                    if obj not in FOCUS_TIERS:
                        continue
                    action = Path(row["source"]).stem.removeprefix(obj + "_")
                    if action.startswith("lift"):
                        continue
                    object_ranks[category, split, obj] += 1
                    rank = object_ranks[category, split, obj]
                    writer.writerow({name: FOCUS_TIERS[obj] if name == "tier" else
                                     category if name == "category" else
                                     rank if name == "rank" else row[name]
                                     for name in ("tier", *fields)})
    print(f"WROTE {object_csv_output}")


if __name__ == "__main__":
    main()
