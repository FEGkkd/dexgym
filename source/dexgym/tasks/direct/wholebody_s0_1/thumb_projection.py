"""Project wrist-local thumb landmarks with the native Wuji model."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import mujoco
import numpy as np

from ..wholebody_s0.bag import digest
from ..wholebody_s0.prepare import write_npz
from . import DT, SCHEMA
from .posture import HYPEREXTENSION_RAD
from .reference_fit import ReferenceRobot


VERSION = "wuji_thumb_ip_v1"
CONTACT_TOLERANCE_M = 1e-7
LANDMARK_WEIGHTS = np.array([1., 1.45, 1.15])
THUMB_JOINTS = ("thumb_cmc_flex", "thumb_cmc_abd", "thumb_mcp", "thumb_ip")
PRESERVED_FIELDS = ("wrist", "rotation", "time_ns", "relative_position", "relative_rotation")


class ThumbProjector:
    """Search for a close thumb shape while respecting native contacts and the control envelope."""

    def __init__(self, metadata):
        self.robot = robot = ReferenceRobot(metadata["teleop_model"], metadata["home"], metadata)
        m = robot.model
        if robot.names != metadata["joint_names"]:
            raise ValueError("Joint order differs from the bank")
        self.joints = np.array([[m.joint(f"{side}_{joint}").id for joint in THUMB_JOINTS] for side in "lr"])
        self.indices = m.jnt_qposadr[self.joints]
        self.lower = np.maximum(robot.lo, metadata["position_lower_rad"])
        self.upper = np.minimum(robot.hi, metadata["position_upper_rad"])
        self.lower[self.indices[:, 2:]] = np.maximum(self.lower[self.indices[:, 2:]], -HYPEREXTENSION_RAD + 1e-7)
        self.hands = np.flatnonzero(~robot.arm)
        self.tip = robot.tips[:, 0]
        self.influence = np.arange(4)[None, :] < np.array([2, 3, 4])[:, None]
        self.initial = np.clip(robot.home, self.lower, self.upper).astype(np.float32)
        if not self.safe(self.initial):
            raise ValueError("The native Home shape with natural thumbs is in collision")

    def safe(self, q):
        # Re-evaluate the float32 representation that the library will write.
        self.robot.forward(q.astype(np.float32), collision=True)
        return self.robot.depth() <= CONTACT_TOLERANCE_M

    def thumb_points(self):
        robot = self.robot
        d = robot.data
        return np.concatenate((d.xanchor[self.joints[:, 2:]], d.site_xpos[self.tip, None]), axis=1)

    def local_points(self):
        wrist, rotation = self.robot.body_pose(self.robot.wrists)
        return (self.thumb_points() - wrist[:, None]) @ rotation

    def fit(self, seed, target, lower, upper, iterations):
        robot, ids = self.robot, self.indices
        q = seed.copy()
        robot.forward(q)
        wrist, rotation = robot.body_pose(robot.wrists)
        world = wrist[:, None] + target @ rotation.swapaxes(-1, -2)
        for _ in range(iterations):
            points = self.thumb_points()
            axes = robot.data.xaxis[self.joints]
            anchors = robot.data.xanchor[self.joints]
            jac = np.cross(axes[:, None], points[:, :, None] - anchors[:, None])
            jac *= (self.influence * LANDMARK_WEIGHTS[:, None])[None, :, :, None]
            jac = jac.transpose(0, 1, 3, 2).reshape(2, 9, 4)
            error = ((world - points) * LANDMARK_WEIGHTS[None, :, None]).reshape(2, 9, 1)
            jt = jac.swapaxes(-1, -2)
            rhs = (jt @ error)[..., 0]
            free = ~(((q[ids] <= lower[ids] + 1e-7) & (rhs < 0.))
                     | ((q[ids] >= upper[ids] - 1e-7) & (rhs > 0.)))
            system = (jt @ jac) * (free[:, :, None] & free[:, None, :]) + np.eye(4) * 7e-6
            delta = np.linalg.solve(system, (rhs * free)[..., None])[..., 0]
            proposal = np.clip(q[ids] + delta.clip(-.23, .23), lower[ids], upper[ids])
            if np.abs(proposal - q[ids]).max() < 1e-4:
                break
            q[ids] = proposal
            robot.forward(q)
        return q

    def clear_contacts(self, fitted, lower, upper, *, allow_arm_motion=False):
        """Move the colliding fingers if the archived non-thumb context itself overlaps."""
        robot = self.robot
        m, d = robot.model, robot.data
        q = fitted.copy()
        movable = np.arange(m.nv) if allow_arm_motion else self.hands
        mobility = np.where(robot.arm[movable], .32, 1.)
        jac = np.zeros((2, 3, m.nv))
        for _ in range(10):
            robot.forward(q.astype(np.float32), collision=True, jacobian=True)
            if robot.depth() <= CONTACT_TOLERANCE_M:
                return q
            active = {}
            for contact in d.contact:
                key = tuple(contact.geom)
                if contact.dist < 0. and (key not in active or contact.dist < active[key].dist):
                    active[key] = contact
            rows, desired = [], []
            for contact in active.values():
                for i, geom in enumerate(contact.geom):
                    body = int(m.geom_bodyid[geom])
                    mujoco.mj_jac(m, d, jac[i], None, contact.pos, body)
                row = (contact.frame[:3] @ (jac[1] - jac[0]))[movable]
                if abs(row).max() > 1e-8:
                    rows.append(row)
                    desired.append(.00015 - contact.dist)
            if not rows:
                break
            j = np.stack(rows)
            updated = q[movable]
            lower_h, upper_h = lower[movable], upper[movable]
            rhs = j.T @ desired
            free = ~(((updated <= lower_h + 1e-7) & (rhs < 0.))
                     | ((updated >= upper_h - 1e-7) & (rhs > 0.)))
            j *= free * mobility
            delta = mobility * (j.T @ np.linalg.solve(j @ j.T + np.eye(len(j)) * 1e-6, desired))
            proposal = np.clip(updated + delta.clip(-.12, .12), lower_h, upper_h)
            if np.abs(proposal - updated).max() < 1e-5:
                break
            q[movable] = proposal
        return q if self.safe(q) else None

    def frame(self, archived, target, previous):
        ids = self.indices
        lower, upper = self.lower.copy(), self.upper.copy()
        if previous is not None:
            slew = self.robot.slew[ids] - 2e-7
            lower[ids] = np.maximum(lower[ids], previous[ids] - slew)
            upper[ids] = np.minimum(upper[ids], previous[ids] + slew)
        seed = archived.astype(float)
        seed[ids] = np.clip((archived if previous is None else previous)[ids], lower[ids], upper[ids])
        fitted = self.fit(seed, target, lower, upper, 10 if previous is None else 4)
        if self.safe(fitted):
            return fitted, 0

        # Backtracking retains the other joints of this frame where possible.
        for ratio in (.75, .50, .25, 0.):
            q = seed.copy()
            q[ids] += ratio * (fitted[ids] - seed[ids])
            if self.safe(q):
                return q, 1
        repaired = self.clear_contacts(fitted, lower, upper)
        if repaired is not None:
            return repaired, 2

        # Inter-hand or arm contacts sometimes need a small wrist-context correction.
        arm = self.robot.arm
        lower[arm] = np.maximum(lower[arm], archived[arm] - .035)
        upper[arm] = np.minimum(upper[arm], archived[arm] + .035)
        repaired = self.clear_contacts(fitted, lower, upper, allow_arm_motion=True)
        if repaired is not None:
            return repaired, 4

        if previous is not None:
            for ratio in (.75, .50, .25, .125, .0625):
                candidate = previous + ratio * (fitted - previous)
                if self.safe(candidate):
                    return candidate, 5
        return (self.initial if previous is None else previous), 3

    def clip(self, q, qd, target, safe):
        if not len(q) or not np.isfinite(q).all() or not np.isfinite(target).all():
            raise ValueError("Projection requires a nonempty, finite reference clip")
        result = np.empty_like(q)
        points = np.empty_like(target)
        route = np.empty(len(q), dtype=np.uint8)
        depth = np.empty(len(q), dtype=np.float32)
        previous = None
        for i in range(len(q)):
            candidate, route[i] = self.frame(q[i], target[i], previous)
            result[i] = candidate
            if not self.safe(result[i]):
                raise RuntimeError(f"Thumb projection could not certify frame {i}")
            points[i] = self.local_points()
            depth[i] = self.robot.depth()
            previous = result[i]
        if len(q) > 1 and (np.abs(np.diff(result[:, self.indices], axis=0)) >
                           self.robot.slew[self.indices] + 2e-7).any():
            raise RuntimeError("Projected thumb exceeded the 50-Hz position rate")
        # Preserve archived reset dynamics on joints the projection did not move.
        changed = (np.abs(result - q) > 1e-7).any(axis=0)
        qd = qd.copy()
        if len(q) > 1:
            velocity = np.gradient(result[:, changed], DT, axis=0)
            rate = self.robot.slew[changed] / DT
            qd[:, changed] = np.clip(velocity, -rate, rate)
        else:
            qd[:, changed] = 0.
        return result, qd, points, depth, safe & (route != 3), route


def read_archive(path, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError(f"Bank checksum differs: {path}")
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def init_worker(metadata, arrays):
    global projector, archive
    projector, archive = ThumbProjector(metadata), arrays


def project_clip(clip):
    span = slice(clip["start"], clip["stop"])
    return span, projector.clip(archive["rsi_q"][span], archive["rsi_qd"][span],
                                archive["hand"][span, :, 0], archive["reset_safe"][span])


def project(args):
    metadata = json.loads(args.bank.read_text(encoding="utf-8"))
    if (metadata["schema"] != SCHEMA or metadata["rate_hz"] != round(1 / DT)
            or "wrist_calibration" not in metadata or "thumb_projection" in metadata):
        raise ValueError("Expected the unprojected physical-wrist bank")
    if digest(Path(metadata["teleop_model"])) != metadata["teleop_model_sha256"]:
        raise ValueError("Projection requires the MJCF bound to the bank")
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    prior_hash = digest(args.bank)
    metadata["thumb_projection"] = dict(version=VERSION, input_bank_sha256=prior_hash,
                                        contact_tolerance_m=CONTACT_TOLERANCE_M,
                                        mcp_ip_min_rad=-HYPEREXTENSION_RAD)
    if args.reference_bank is not None:
        training = json.loads(args.reference_bank.read_text(encoding="utf-8"))
        reference_input = metadata.get("reference_bank_sha256")
        linked_inputs = (training["thumb_projection"]["input_bank_sha256"],
                         training.get("wrist_refit", {}).get("input_bank_sha256"))
        if not reference_input or reference_input not in linked_inputs:
            raise ValueError("The source-clock bank must link to the matching projected training bank")
        metadata["reference_bank_sha256"] = digest(args.reference_bank)
    args.output.mkdir(parents=True, exist_ok=False)
    for split, info in metadata["splits"].items():
        arrays = read_archive(args.bank.parent / info["file"], info["sha256"])
        # fork keeps the input pack read-only in the workers while the parent collects results.
        raw_thumb = arrays["hand"][:, :, 0].copy()
        routes = np.empty(info["frames"], dtype=np.uint8)
        clips = info["clips"]
        starts, stops = [c["start"] for c in clips], [c["stop"] for c in clips]
        if (not clips or [0, *stops[:-1]] != starts or stops[-1] != info["frames"]
                or len(arrays["rsi_q"]) != info["frames"]):
            raise ValueError(f"Bank clip offsets differ from the {split} archive")
        print(f"Projecting {split}: {info['frames']} frames, {len(clips)} clips", flush=True)
        with mp.get_context("fork").Pool(args.workers, initializer=init_worker, initargs=(metadata, arrays)) as pool:
            for index, (span, values) in enumerate(pool.imap(project_clip, clips, chunksize=1), 1):
                q, qd, points, depth, safe, route = values
                arrays["rsi_q"][span], arrays["rsi_qd"][span] = q, qd
                arrays["hand"][span, :, 0] = points
                arrays["reset_depth_m"][span], arrays["reset_safe"][span], routes[span] = depth, safe, route
                if index % 500 == 0:
                    print(f"{split}: {index}/{len(clips)} clips", flush=True)
        arrays["thumb_projection_route"] = routes
        file = args.output / info["file"]
        write_npz(file, arrays)
        checksum = digest(file)
        change = np.linalg.norm(arrays["hand"][:, :, 0] - raw_thumb, axis=-1) * 1000
        stats = {source: summarize_projection(arrays, clips, source, change) for source in ("teleop", "egopro")}
        info.update(sha256=checksum, thumb_projection=stats)
        print(split, json.dumps(stats), flush=True)
    # A manifest becomes usable only after the complete dataset is ready.
    manifest = args.output / "manifest.json"
    partial = manifest.with_suffix(".tmp")
    partial.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    partial.replace(manifest)


def summarize_projection(arrays, clips, source, change):
    spans = [slice(c["start"], c["stop"]) for c in clips if c.get("source", "teleop") == source]
    if not spans:
        return {}
    depth = np.concatenate([arrays["reset_depth_m"][s] for s in spans])
    route = np.concatenate([arrays["thumb_projection_route"][s] for s in spans])
    diff = np.concatenate([change[s] for s in spans])
    return dict(frames=len(depth), maximum_penetration_m=float(depth.max()),
                native_fit=int((route == 0).sum()), backtracked=int((route == 1).sum()),
                adjusted_fingers=int((route == 2).sum()), fallback=int((route == 3).sum()),
                adjusted_arms=int((route == 4).sum()), blended_context=int((route == 5).sum()),
                landmark_shift_p95_mm=np.quantile(diff, .95, axis=(0, 1)).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-bank", type=Path, help="Matching projected training bank for source-clock replay")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    project(args)


if __name__ == "__main__":
    main()
