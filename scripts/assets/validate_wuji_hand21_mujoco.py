"""Independent Hand 2.1 MJCF reference checks for saved FK and PD holds."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from dexgym.assets.wuji_hand21 import OFFICIAL_MJCF
from dexgym.data.wuji_retarget import RightHandFK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/hand21_right_geometry_feasible/manifest.json")
    )
    parser.add_argument("--samples-per-subject", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("artifacts/hand21/validate_mujoco.json"))
    args = parser.parse_args()
    path = args.manifest.expanduser().resolve()
    bank = json.loads(path.read_text())
    selected = []
    for subject in (f"s{i}" for i in range(1, 11)):
        rows = [
            x
            for x in bank["motions"]
            if x.get("origin_source", x["source"]).startswith(subject + "/") and x["frames"] >= 5
        ]
        if len(rows) < args.samples_per_subject:
            raise ValueError(f"Too few eligible {subject} trajectories")
        indices = np.linspace(0, len(rows) - 1, args.samples_per_subject, dtype=int)
        selected.extend(rows[index] for index in indices)
    fk = RightHandFK(OFFICIAL_MJCF)
    model = fk.model
    data = mujoco.MjData(model)
    counts = []
    for entry in selected:
        with np.load(path.parent / entry["file"], allow_pickle=False) as sample:
            q = np.asarray(sample["q_ref"], dtype=np.float32)
            tip = np.asarray(sample["tip_ref"], dtype=np.float32)
        start, goal = q[0], q[4]
        saved_fk_error_mm = float(np.linalg.norm(fk.tips(goal) - tip[4], axis=-1).max() * 1000)
        mujoco.mj_resetData(model, data)
        data.qpos[fk.q_indices] = start
        data.ctrl[:] = goal
        mujoco.mj_forward(model, data)
        for _ in range(round(1.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        actual_q = data.qpos[fk.q_indices].copy()
        actual_tips = fk.tips(actual_q)
        counts.append(
            {
                "source": entry["source"],
                "saved_fk_max_mm": saved_fk_error_mm,
                "joint_error_max_rad": float(np.abs(actual_q - goal).max()),
                "joint_speed_max_rad_s": float(np.abs(data.qvel).max()),
                "tip_error_mm": (np.linalg.norm(actual_tips - tip[4], axis=-1) * 1000).tolist(),
                "contact_count": int(data.ncon),
            }
        )
    tip_errors = np.array([row["tip_error_mm"] for row in counts])
    result = {
        "model": str(OFFICIAL_MJCF),
        "samples": len(counts),
        "saved_fk_max_mm": max(row["saved_fk_max_mm"] for row in counts),
        "hold_joint_error_max_rad": max(row["joint_error_max_rad"] for row in counts),
        "hold_joint_speed_max_rad_s": max(row["joint_speed_max_rad_s"] for row in counts),
        "hold_tip_per_finger_max_mm": tip_errors.max(axis=0).tolist(),
        "hold_all_fingers_within_10mm": int(np.all(tip_errors < 10, axis=1).sum()),
        "rows": counts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
