"""Audit native MuJoCo contacts and correct brief penetrations in references."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import mujoco
import numpy as np

from .egopro_handcentric import DT, IK_DEPTH_M, JOINTS, PHYSICAL_DEPTH_M, Robot, quaternion
from .egopro_handcentric_refine import load, measure, unpack


DEEP_M = 0.0005


def contact_correction(robot, contact, depth):
    """Prefer joint changes that separate the contact with small tip motion."""
    m, d = robot.model, robot.data
    point = np.zeros((2, 3, JOINTS))
    for i, geom in enumerate(contact.geom):
        mujoco.mj_jac(m, d, point[i], None, contact.pos, int(m.geom_bodyid[geom]))
    gradient = contact.frame[:3] @ (point[1] - point[0])
    tips = np.empty((len(robot.tips.flat), 3, JOINTS))
    for i, site in enumerate(robot.tips.flat):
        mujoco.mj_jacSite(m, d, tips[i], None, int(site))
    tip_j = tips.reshape(-1, JOINTS)
    weight = np.eye(JOINTS) + 220. * (tip_j.T @ tip_j)
    direction = np.linalg.solve(weight, gradient)
    return ((depth + 0.0007) / max(float(gradient @ direction), 1e-8)) * direction


def audit(robot, q, qd, tip_ref, *, directions=False):
    m, d = robot.model, robot.data
    mujoco.mj_resetData(m, d)
    d.qpos[:], d.qvel[:] = q[0], qd[0]
    mujoco.mj_forward(m, d)
    peak, contacts, deep = 0., 0, 0
    events, changes, tip_error = [], [], []
    completed = True
    for frame in range(1, len(q)):
        d.ctrl[:] = q[frame, robot.actuator_qpos]
        worst = DEEP_M
        for substep in range(1, 11):
            previous_time = d.time
            mujoco.mj_step(m, d)
            mujoco.mj_fwdPosition(m, d)
            distance = d.contact.dist
            depth = max(0.0, -float(np.min(distance, initial=0.0)))
            peak = max(peak, depth)
            contacts += int(depth > 1e-6)
            deep += int(depth > DEEP_M)
            if depth > worst:
                worst = depth
                c = d.contact[int(np.argmin(distance))]
                event = dict(
                    frame=frame, substep=substep, depth_mm=1000 * depth,
                    geoms=[m.geom(int(index)).name for index in c.geom],
                )
                change = contact_correction(robot, c, depth) if directions else None
            if (depth > PHYSICAL_DEPTH_M or d.time <= previous_time
                    or not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
                    or (d.qpos < robot.limits[:, 0] - 0.05).any()
                    or (d.qpos > robot.limits[:, 1] + 0.05).any()):
                completed = False
                break
        if worst > DEEP_M:
            events.append(event)
            if directions:
                changes.append((frame, change))
        if not completed:
            break
        tip_error.append(np.linalg.norm(d.site_xpos[robot.tips] - tip_ref[frame], axis=-1).mean())
    return dict(
        complete=completed, steps=len(tip_error), peak_mm=peak * 1000,
        penetrating_substeps=contacts, deep_substeps=deep,
        tip_p95_mm=float(np.percentile(tip_error, 95) * 1000) if tip_error else None,
        events=events,
    ), changes


def window_correction(q, changes, arm):
    delta, weights = np.zeros_like(q), np.zeros(len(q))
    limit = np.where(arm, 0.065, 0.105)
    for frame, change in changes:
        center = max(0, frame - 2)
        ids = np.arange(max(0, center - 12), min(len(q), center + 13))
        w = np.exp(-0.5 * ((ids - center) / 4.5)**2)
        delta[ids] += w[:, None] * np.clip(change, -limit, limit)
        weights[ids] += w
    return delta / np.maximum(weights[:, None], 1.0)


def repair(data, xml):
    robot = Robot(xml)
    goal = unpack(data)
    q = data["qpos"].astype(np.float64)
    qd = data["qvel"]
    initial, changes = audit(robot, q, qd, data["ref_tip_pos"], directions=True)
    before = after = initial
    if before["deep_substeps"] == 0 and before["complete"]:
        return data, dict(before=before, after=after, accepted_updates=0, frames=len(q))
    fk, original_error = measure(robot, q, goal)
    original_q = q.copy()
    accepted = 0
    speed = np.where(robot.arm, 0.063, 0.115)
    for _ in range(5):
        if not changes:
            break
        delta = window_correction(q, changes, robot.arm)
        for multiplier in (2., 1., 0.5, 0.25):
            candidate = np.clip(q + multiplier * delta, robot.lo, robot.hi)
            for frame in range(1, len(candidate)):
                np.clip(candidate[frame], candidate[frame - 1] - speed, candidate[frame - 1] + speed,
                        out=candidate[frame])
            candidate_fk, error = measure(robot, candidate, goal)
            if (error["geometric_depth_mm"] > IK_DEPTH_M * 1000
                    or error["tip_mm"] > original_error["tip_mm"] + 0.35
                    or error["cross_tip_mm"] > original_error["cross_tip_mm"] + 0.50
                    or error["shape_mm"] > original_error["shape_mm"] + 0.6
                    or error["tip_deg"] > original_error["tip_deg"] + 0.5):
                continue
            candidate_vel = np.gradient(candidate, DT, axis=0)
            result, _ = audit(robot, candidate, candidate_vel, candidate_fk["tip_pos"])
            if (result["complete"] and result["tip_p95_mm"] <= min(85., (before["tip_p95_mm"] or 85.) + 3.)
                    and result["peak_mm"] < after["peak_mm"]
                    and result["deep_substeps"] <= after["deep_substeps"]):
                q, qd, fk = candidate, candidate_vel, candidate_fk
                after, changes = audit(robot, q, qd, fk["tip_pos"], directions=True)
                accepted += 1
                break
        else:
            break
    record = dict(before=before, after=after, accepted_updates=accepted, frames=len(q),
                  before_mapped=original_error, after_mapped=measure(robot, q, goal)[1],
                  max_joint_change_rad=float(abs(q - original_q).max()))
    if not accepted:
        return data, record
    output = {**data, "qpos": q.astype(np.float32), "qvel": qd.astype(np.float32)}
    for key, values in fk.items():
        if key.endswith("_rot"):
            key = key.replace("_rot", "_quat_wxyz")
            values = quaternion(values)
        output[f"ref_{key}"] = values.astype(np.float32)
    return output, record


def work(path, xml, destination):
    result, report = repair(load(path), xml)
    if report["after"]["complete"]:
        if report["accepted_updates"]:
            np.savez_compressed(destination, **result)
        else:
            destination = path
    else:
        destination = None
    return {**report, "source": str(path), "path": str(destination) if destination else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--witness", type=Path)
    source.add_argument("--library", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.witness:
        result, report = repair(load(args.witness), args.model)
        np.savez_compressed(args.output / "witness_full.npz", **result)
        (args.output / "witness.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    paths = {}
    for split in ("train", "val", "test"):
        paths[split] = [Path(p) for p in json.loads((args.library / f"{split}.json").read_text())]
        (args.output / split).mkdir(exist_ok=True)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output / "progress.jsonl").open("w") as log:
        pending = {pool.submit(work, p, args.model, args.output / split / p.name): split
                   for split, group in paths.items() for p in group}
        for job in as_completed(pending):
            entry = {**job.result(), "split": pending[job]}
            reports.append(entry)
            print(json.dumps(entry), file=log, flush=True)
            if len(reports) % 50 == 0:
                print(f"Audited {len(reports)}/{len(pending)} references", flush=True)
    reports.sort(key=lambda row: row["source"])
    (args.output / "collision_audit.json").write_text(json.dumps(reports, indent=2) + "\n")
    for split in paths:
        kept = [r["path"] for r in reports if r["split"] == split and r["path"]]
        (args.output / f"{split}.json").write_text(json.dumps(kept, indent=2) + "\n")
        if not kept:
            raise RuntimeError(f"No audited clips in {split}")


if __name__ == "__main__":
    main()
