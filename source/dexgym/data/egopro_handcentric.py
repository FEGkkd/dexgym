"""Project synchronized EgoPro hand goals onto Marvin and dual Wuji Hand2.2."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .egopro_poses import FPS, MIN_FRAMES


DT = 1 / FPS
JOINTS = 54
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
SIDES = ("l", "r")
TIPS = np.array([4, 8, 12, 16, 20])
CLEARANCE_M = 0.004
IK_DEPTH_M = 0.0001
PHYSICAL_DEPTH_M = 0.003
MAX_CLIP_FRAMES = 601


def unit(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-9)


def palm_frame(points):
    forward = unit(points[..., 9, :] - points[..., 0, :])
    across = points[..., 5, :] - points[..., 17, :]
    normal = unit(np.cross(forward, across))
    return np.stack((forward, np.cross(normal, forward), normal), axis=-1)


def distal_frame(direction, normal):
    axis = unit(direction)
    across = np.cross(normal, axis)
    fallback = np.cross(np.eye(3)[np.abs(axis).argmin(axis=-1)], axis)
    across = unit(np.where(np.linalg.norm(across, axis=-1, keepdims=True) < 1e-5, fallback, across))
    return np.stack((axis, across, np.cross(axis, across)), axis=-1)


def skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def quaternion(rotation):
    out = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_quat(scalar_first=True)
    return out.reshape(*rotation.shape[:-2], 4)


class Robot:
    def __init__(self, xml):
        self.model = m = mujoco.MjModel.from_xml_path(str(xml))
        self.data = d = mujoco.MjData(m)
        if (m.nq, m.nv, m.nu) != (JOINTS, JOINTS, JOINTS) or not np.isclose(m.opt.timestep, 0.002):
            raise ValueError("Expected the fixed-base 54-joint model with 2 ms physics")
        self.names = [m.joint(i).name for i in np.argsort(m.jnt_qposadr)]
        self.actuator_qpos = m.jnt_qposadr[m.actuator_trnid[:, 0]]
        if (len(np.unique(self.actuator_qpos)) != JOINTS or not m.jnt_limited.all()
                or not m.actuator_ctrllimited.all() or not np.allclose(m.actuator_gear[:, 0], 1)):
            raise ValueError("Expected a limited unit-gear position actuator for each joint")
        self.arm = np.array([name.startswith("Joint") for name in self.names])
        self.limits = m.jnt_range[np.argsort(m.jnt_qposadr)]
        ctrl = m.actuator_ctrlrange[np.argsort(self.actuator_qpos)]
        self.lo = np.maximum(self.limits[:, 0], ctrl[:, 0]) + 0.003
        self.hi = np.minimum(self.limits[:, 1], ctrl[:, 1]) - 0.003
        self.wrist = np.array([m.body(f"{side}_wrist").id for side in SIDES])
        self.elbow = np.array([m.body(f"Link4_{side.upper()}").id for side in SIDES])
        self.shoulder = np.array([m.body(f"Link2_{side.upper()}").id for side in SIDES])
        self.tips = np.array([[m.site(f"{side}_{finger}_tip").id for finger in FINGERS] for side in SIDES])
        self.fingers = np.array([[
            [m.body(f"{side}_{finger}_{link}").id for link in ("proximal_abd", "middle", "distal")]
            for finger in FINGERS] for side in SIDES])
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        self.home = d.qpos.copy()
        self.forward(self.home)
        self.home_frames = self.frames()
        self.home_shoulders = d.xpos[self.shoulder].copy()

    def forward(self, q):
        self.data.qpos[:] = q
        mujoco.mj_fwdPosition(self.model, self.data)

    def depth(self):
        return max(0.0, -float(np.min(self.data.contact.dist, initial=0.0)))

    def frames(self):
        d = self.data
        wrist_p = d.xpos[self.wrist]
        wrist_r = d.xmat[self.wrist].reshape(2, 3, 3)
        tip_p = d.site_xpos[self.tips]
        chain = np.concatenate((d.xpos[self.fingers], tip_p[:, :, None]), axis=2)
        local = np.einsum("sji,sfkj->sfki", wrist_r, chain - wrist_p[:, None, None])
        return dict(tip_pos=tip_p.copy(), tip_rot=d.site_xmat[self.tips].reshape(2, 5, 3, 3).copy(),
                    wrist_pos=wrist_p.copy(), wrist_rot=wrist_r.copy(),
                    elbow_pos=d.xpos[self.elbow].copy(), shape=local)


def hand_goals(robot, arms, left, right):
    """Preserve shared fingertip placement and scale each finger to robot bones."""
    home = robot.home_frames
    shoulders = robot.home_shoulders
    up = np.array([0., 0., 1.])
    left_axis = unit((arms[:, 0, :3] - arms[:, 1, :3]) * [1, 1, 0])
    robot_left = unit((shoulders[0] - shoulders[1]) * [1, 1, 0])
    torso = np.stack((np.cross(left_axis, up), left_axis, np.broadcast_to(up, left_axis.shape)), axis=-1)
    robot_torso = np.stack((np.cross(robot_left, up), robot_left, up), axis=-1)
    align = robot_torso @ torso.swapaxes(-1, -2)
    center = arms[:, :2, :3].mean(axis=1)
    elbow = shoulders + np.linalg.norm(home["elbow_pos"] - shoulders, axis=-1)[:, None] * np.einsum(
        "tij,tsj->tsi", align, unit(arms[:, 2:4, :3] - arms[:, :2, :3])
    )
    output = {key: [] for key in ("tip_pos", "tip_rot", "wrist_pos", "wrist_rot", "shape")}
    for side, hand in enumerate((left, right)):
        p = hand[..., :3]
        human_palm = palm_frame(p)
        wrist_r = home["wrist_rot"][side]
        robot_local = home["shape"][side]
        neutral = np.zeros((21, 3))
        neutral[TIPS - 3] = robot_local[:, 0]
        neutral_palm = palm_frame(neutral)
        world_r = align @ human_palm @ neutral_palm.T
        length = np.linalg.norm(np.diff(robot_local, axis=1), axis=-1)
        chain = p[:, 1:].reshape(-1, 5, 4, 3)
        directions = unit(np.diff(chain, axis=2))
        in_wrist = np.einsum("tij,tfkj->tfki", neutral_palm @ human_palm.swapaxes(-1, -2), directions)
        shape = np.concatenate((
            np.broadcast_to(robot_local[:, :1], (len(p), 5, 1, 3)),
            robot_local[None, :, :1] + np.cumsum(in_wrist * length[None, :, :, None], axis=2),
        ), axis=2)
        offset = np.einsum("tij,tfj->tfi", world_r, shape[:, :, -1])
        center_goal = shoulders.mean(axis=0) + 1.12 * np.einsum("tij,tj->ti", align, p[:, TIPS].mean(1) - center)
        world_wrist = center_goal - offset.mean(axis=1)
        source_rot = Rotation.from_quat(hand[:, TIPS, 3:].reshape(-1, 4), scalar_first=True).as_matrix()
        source_rot = source_rot.reshape(-1, 5, 3, 3)
        human_distal = distal_frame(directions[0, :, -1], human_palm[0, :, 2])
        robot_distal = distal_frame(
            (robot_local[:, -1] - robot_local[:, -2]) @ wrist_r.T, wrist_r @ neutral_palm[:, 2]
        )
        site_frame = robot_distal.swapaxes(-1, -2) @ home["tip_rot"][side]
        calibration = source_rot[0].swapaxes(-1, -2) @ human_distal @ site_frame
        output["tip_pos"].append(world_wrist[:, None] + offset)
        output["tip_rot"].append(align[:, None] @ source_rot @ calibration)
        output["shape"].append(shape)
        output["wrist_rot"].append(world_r)
        output["wrist_pos"].append(shoulders.mean(axis=0) + 1.12 * np.einsum(
            "tij,tj->ti", align, arms[:, 4 + side, :3] - center
        ))
    return {**{key: np.stack(values, axis=1) for key, values in output.items()}, "elbow_pos": elbow}


def rotation_error(desired, measured):
    r = desired @ measured.swapaxes(-1, -2)
    return Rotation.from_matrix(r.reshape(-1, 3, 3)).as_rotvec().reshape(*r.shape[:-2], 3)


class Solver:
    def __init__(self, robot):
        self.robot = robot
        # Expanding proximity detection supplies a clearance gradient before contact.
        robot.model.geom_margin[:] += CLEARANCE_M + 0.002
        robot.model.pair_margin[:] += CLEARANCE_M + 0.002
        self.eye = np.eye(JOINTS)

    def residual(self, q, goal, prior, jacobian, wrists_first):
        r = self.robot
        m, d = r.model, r.data
        r.forward(q)
        actual = r.frames()
        e1, e2, j1, j2 = [], [], [], []

        def put(error, weight, jac=None, weak=False):
            (e2 if weak else e1).append(np.ravel(weight * error))
            if jacobian:
                (j2 if weak else j1).append((weight * jac).reshape(-1, JOINTS))

        def jac(kind, index):
            p = np.zeros((3, JOINTS))
            rot = np.zeros_like(p)
            kind(m, d, p, rot, int(index))
            return p, rot

        wrist_j = [jac(mujoco.mj_jacBody, index) for index in r.wrist] if jacobian else [(None, None)] * 2
        tip_j = [[jac(mujoco.mj_jacSite, index) for index in row] for row in r.tips] if jacobian else None
        if jacobian:
            tip_pj = np.array([[pair[0] for pair in row] for row in tip_j])
            tip_rj = np.array([[pair[1] for pair in row] for row in tip_j])
        put(goal["tip_pos"] - actual["tip_pos"], 1 / 0.021, tip_pj if jacobian else None, weak=wrists_first)
        put(rotation_error(goal["tip_rot"], actual["tip_rot"]), 0.65, tip_rj if jacobian else None,
            weak=wrists_first)
        rel = goal["tip_pos"][0, :, None] - goal["tip_pos"][1, None]
        measured = actual["tip_pos"][0, :, None] - actual["tip_pos"][1, None]
        weights = (0.20 + 0.46 * np.exp(-(np.linalg.norm(rel, axis=-1) / 0.10) ** 2)) / 0.035
        for i, j in np.ndindex(5, 5):
            put(rel[i, j] - measured[i, j], weights[i, j],
                tip_pj[0, i] - tip_pj[1, j] if jacobian else None, weak=wrists_first)
        relative_r = goal["wrist_rot"][0].T @ goal["wrist_rot"][1]
        put(rotation_error(actual["wrist_rot"][0] @ relative_r, actual["wrist_rot"][1]), 0.5,
            wrist_j[1][1] - wrist_j[0][1] if jacobian else None, weak=wrists_first)
        for side in range(2):
            jw, jwr = wrist_j[side]
            wr = actual["wrist_rot"][side]
            put(goal["wrist_pos"][side] - actual["wrist_pos"][side], 1 / (0.024 if wrists_first else 0.16),
                jw, weak=not wrists_first)
            put(rotation_error(goal["wrist_rot"][side], wr), 2.3 if wrists_first else 0.22,
                jwr, weak=not wrists_first)
            put(goal["elbow_pos"][side] - actual["elbow_pos"][side], 1 / 0.19,
                jac(mujoco.mj_jacBody, r.elbow[side])[0] if jacobian else None, weak=True)
            for finger in range(5):
                for link in range(1, 4):
                    offset = wr @ goal["shape"][side, finger, link]
                    error = wr @ (goal["shape"][side, finger, link] - actual["shape"][side, finger, link])
                    if jacobian:
                        jp = (tip_pj[side, finger] if link == 3 else
                              jac(mujoco.mj_jacBody, r.fingers[side, finger, link])[0])
                    put(error, 0.40 / 0.020, jp - jw + skew(offset) @ jwr if jacobian else None)
        put(prior - q, 0.52, self.eye, weak=True)
        near = {}
        for c in d.contact:
            pair = tuple(c.geom)
            if c.dist < CLEARANCE_M and (pair not in near or c.dist < near[pair].dist):
                near[pair] = c
        for c in near.values():
            grad = None
            if jacobian:
                pos_j = np.zeros((2, 3, JOINTS))
                for i, geom in enumerate(c.geom):
                    mujoco.mj_jac(m, d, pos_j[i], None, c.pos, int(m.geom_bodyid[geom]))
                grad = c.frame[:3] @ (pos_j[1] - pos_j[0])
            put(CLEARANCE_M - c.dist, 580., grad)
        return (np.concatenate(e1), np.concatenate(e2),
                np.concatenate(j1) if jacobian else None, np.concatenate(j2) if jacobian else None)

    def solve(self, q, goal, prior, lo, hi, iterations, wrists_first=False):
        q = q.copy()
        for _ in range(iterations):
            e, posture, j, j_posture = self.residual(q, goal, prior, True, wrists_first)
            rhs = j.T @ e
            free = ~(((q <= lo + 1e-8) & (rhs < 0)) | ((q >= hi - 1e-8) & (rhs > 0)))
            if not free.any():
                break
            jf = j[:, free]
            eye = self.eye[:free.sum(), :free.sum()]
            hessian = jf.T @ jf + 0.7 * eye
            rhs_and_null = np.column_stack((rhs[free], 0.7 * eye))
            result = np.linalg.solve(hessian, rhs_and_null)
            first, null = result[:, 0], result[:, 1:]
            js = j_posture[:, free]
            jn = js @ null
            second = np.linalg.solve(jn.T @ jn + 0.7 * eye, jn.T @ (posture - js @ first))
            update = np.zeros(JOINTS)
            update[free] = np.clip(first + null @ second, -0.17, 0.17)
            if np.abs(update).max() < 1e-5:
                break
            score, posture_score = e @ e, posture @ posture
            for amount in (1., 0.5, 0.25, 0.125):
                candidate = np.clip(q + amount * update, lo, hi)
                ep, es, _, _ = self.residual(candidate, goal, prior, False, wrists_first)
                if self.robot.depth() <= IK_DEPTH_M and (
                    ep @ ep < score - 1e-7 or (ep @ ep <= score + 1e-6 and es @ es < posture_score)
                ):
                    q = candidate
                    break
            else:
                break
        self.robot.forward(q)
        return q


def native_replay(robot, q, qd, goal):
    m, d = robot.model, robot.data
    mujoco.mj_resetData(m, d)
    d.qpos[:], d.qvel[:] = q[0], qd[0]
    peak, errors = 0.0, []
    for frame in range(1, len(q)):
        d.ctrl[:] = q[frame, robot.actuator_qpos]
        for _ in range(10):
            mujoco.mj_step(m, d)
            peak = max(peak, robot.depth())
            if (peak > PHYSICAL_DEPTH_M or not np.isfinite(d.qpos).all()
                    or np.any(d.qpos < robot.limits[:, 0] - 0.05) or np.any(d.qpos > robot.limits[:, 1] + 0.05)):
                return dict(complete=False, steps=frame, depth_mm=peak * 1000)
        mujoco.mj_kinematics(m, d)
        tip = d.site_xpos[robot.tips]
        errors.append(np.linalg.norm(tip - goal[frame], axis=-1).mean())
    return dict(complete=True, steps=len(q) - 1, depth_mm=peak * 1000,
                tip_mean_mm=float(1000 * np.mean(errors)), tip_p95_mm=float(1000 * np.percentile(errors, 95)))


def contiguous(mask):
    bounds = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(bounds[::2], bounds[1::2], strict=True)


def fit_segment(arms, left, right, solver, wrists_first):
    r = solver.robot
    goals = hand_goals(r, arms, left, right)
    q = [np.clip(r.home, r.lo, r.hi)]
    speed = np.where(r.arm, 0.056, 0.10)
    achieved, safe = [], []
    for i in range(len(arms)):
        previous = q[-1]
        target = {k: v[i] for k, v in goals.items()}
        prior = previous if i < 2 else previous + 0.55 * (previous - q[-2])
        lo, hi = (r.lo, r.hi) if i == 0 else (np.maximum(r.lo, previous - speed), np.minimum(r.hi, previous + speed))
        candidate = solver.solve(previous, target, prior, lo, hi, 48 if i == 0 else 4, wrists_first)
        # Check the interpolation midpoint when two safe endpoints approach closely.
        if i and r.depth() <= IK_DEPTH_M:
            r.forward(0.5 * (candidate + previous))
            if r.depth() > IK_DEPTH_M:
                candidate = previous
            r.forward(candidate)
        q.append(candidate)
        achieved.append(r.frames())
        safe.append(r.depth() <= IK_DEPTH_M)
    fk = {key: np.stack([a[key] for a in achieved]) for key in achieved[0]}
    tips = np.linalg.norm(fk["tip_pos"] - goals["tip_pos"], axis=-1)
    delta = fk["tip_pos"] - goals["tip_pos"]
    relative = np.linalg.norm(delta[:, 0, :, None] - delta[:, 1, None], axis=-1).mean(axis=(1, 2))
    return np.stack(q[1:]), fk, goals, tips.mean(axis=(1, 2)), relative, np.asarray(safe)


def retarget(path, xml, output, episode, *, wrists_first=False):
    r = Robot(xml)
    native = Robot(xml)
    solver = Solver(r)
    model_sha = hashlib.sha256(xml.read_bytes()).hexdigest()
    rows, total_frames, good_frames, kept = [], 0, 0, 0
    tip_sum, rel_sum = 0., 0.
    with np.load(path, allow_pickle=False) as source:
        offsets, clock = source["offsets"], source["header_start_ns"]
        arrays = [source[key] for key in ("arms", "left", "right")]
        if int(source["fps"]) != FPS or len(clock) != len(offsets) - 1:
            raise ValueError(f"Invalid synchronized episode {path}")
        for segment, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            q, fk, mapped, tip, relative, safe = fit_segment(
                *(data[start:end] for data in arrays), solver, wrists_first
            )
            total_frames += len(q)
            tip_sum += tip.sum()
            rel_sum += relative.sum()
            valid = safe & (tip < 0.055) & (relative < 0.065)
            good_frames += int(valid.sum())
            for begin, stop in contiguous(valid):
                for a in range(begin, stop, MAX_CLIP_FRAMES - 1):
                    b = min(stop, a + MAX_CLIP_FRAMES)
                    if b - a < MIN_FRAMES:
                        continue
                    sl = slice(a, b)
                    qd = np.gradient(q[sl], DT, axis=0)
                    physical = native_replay(native, q[sl], qd, fk["tip_pos"][sl])
                    row = dict(segment=segment, first_frame=int(a), frames=int(b - a), replay=physical,
                               mapped_tip_mean_mm=float(1000 * tip[sl].mean()),
                               mapped_cross_tip_mean_mm=float(1000 * relative[sl].mean()))
                    rows.append(row)
                    if not physical["complete"] or physical["tip_p95_mm"] > 85.0:
                        continue
                    output.mkdir(parents=True, exist_ok=True)
                    dest = output / f"{episode}_{segment:03d}_{a:06d}.npz"
                    fields = {}
                    for prefix, state in (("ref", fk), ("mapped", mapped)):
                        for key, value in state.items():
                            if key.endswith("_rot"):
                                value = quaternion(value[sl])
                                key = key.replace("_rot", "_quat_wxyz")
                            else:
                                value = value[sl]
                            fields[f"{prefix}_{key}"] = value.astype(np.float32)
                    np.savez_compressed(
                        dest, time_s=np.arange(b - a) * DT, qpos=q[sl].astype(np.float32),
                        qvel=qd.astype(np.float32), joint_names=r.names, model_sha256=model_sha,
                        source_episode=episode, source_header_start_ns=int(clock[segment]) + int(a) * 20_000_000,
                        **fields,
                    )
                    row["path"] = str(dest)
                    kept += int(b - a - 1)
    return dict(episode=episode, attempted_frames=total_frames, quality_frames=good_frames, kept_steps=kept,
                mapped_tip_mean_mm=float(1000 * tip_sum / max(1, total_frames)),
                mapped_cross_tip_mean_mm=float(1000 * rel_sum / max(1, total_frames)), clips=rows)


def work(row, xml, output):
    return {**retarget(Path(row["path"]), xml, output / row["split"], row["episode"]),
            **{k: row[k] for k in ("episode", "split", "task")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--witness", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--wrists-first", action="store_true", help="Matched retargeting ablation for the witness")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.witness:
        report = retarget(args.witness, args.model, args.output, "witness", wrists_first=args.wrists_first)
        (args.output / "witness.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "clips"}), flush=True)
        if not report["kept_steps"]:
            raise RuntimeError("The witness has no physically replayable segment")
        return
    rows = [r for r in json.loads(args.manifest.read_text()) if r["status"] == "ok"]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (args.output / "progress.jsonl").open("w") as log:
        jobs = [pool.submit(work, row, args.model, args.output) for row in rows]
        for job in as_completed(jobs):
            result = job.result()
            results.append(result)
            print(json.dumps(result), file=log, flush=True)
            print(f"Retargeted {len(results)}/{len(rows)} {result['task']}: {result['kept_steps']} usable steps", flush=True)
    results.sort(key=lambda r: r["episode"])
    (args.output / "retarget.json").write_text(json.dumps(results, indent=2) + "\n")
    for split in ("train", "val", "test"):
        paths = [clip["path"] for r in results if r["split"] == split for clip in r["clips"] if "path" in clip]
        (args.output / f"{split}.json").write_text(json.dumps(paths, indent=2) + "\n")
        if not paths:
            raise RuntimeError(f"No physically replayable segments in {split}")


if __name__ == "__main__":
    main()
