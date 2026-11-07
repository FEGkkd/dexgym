"""Seeded FK-generated dual TCP target bank; no inverse kinematics or teacher actions."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dexgym.motion.motion_manager import quaternion_xyzw_to_6d

DEFAULT_MJCF = (Path(__file__).resolve().parents[1]
                / "assets/marvin_wuji/source_model/dual_arm_marvin.xml")
FAMILIES = ("independent_strokes", "opposed_strokes", "oscillatory", "move_hold",
            "asymmetric_arms")
SPLITS = ("train", "validation", "test")


def reference_episode_control_steps(lengths: torch.Tensor, episode_length_s: float) -> int:
    """Require the Newton timeout to cover the longest 50 Hz target clip exactly."""
    if lengths.ndim != 1 or lengths.numel() == 0 or bool((lengths < 33).any()):
        raise ValueError("Reference lengths must be nonempty and allow BPTT32")
    steps = int(lengths.max()) - 1
    if abs(episode_length_s * 50 - steps) > 1e-4:
        raise ValueError("Newton episode length must match reference duration")
    return steps


def _smoothstep(x):
    return x * x * x * (10 + x * (-15 + 6 * x))


def _joint_path(rng, home_arm, frames, family, difficulty):
    """Smooth, diverse joint witnesses used only to create reachable TCP poses."""
    extent = (0.065, 0.11, 0.16)[difficulty]
    base = home_arm + rng.uniform(-0.07, 0.07, 14)
    anchors = np.empty((5, 14), np.float64)
    anchors[0] = base
    for index in range(1, 5):
        left = rng.uniform(-extent, extent, 7)
        right = rng.uniform(-extent, extent, 7)
        if family == "opposed_strokes":
            right = -left + rng.uniform(-extent * 0.2, extent * 0.2, 7)
        elif family == "asymmetric_arms":
            (left if index % 2 else right)[:] *= 0.25
        elif family == "move_hold" and index in (2, 4):
            anchors[index] = anchors[index - 1]
            continue
        elif family == "oscillatory" and index >= 3:
            anchors[index] = anchors[index - 2]
            continue
        anchors[index] = base + np.concatenate((left, right))
    # Four independently varying 1 s segments, with a zero-speed transition.
    sample = np.linspace(0, 4, frames)
    segment = np.minimum(sample.astype(int), 3)
    phase = _smoothstep(sample - segment)
    path = anchors[segment] * (1 - phase[:, None]) + anchors[segment + 1] * phase[:, None]
    return path


def _broad_joint_path(rng, home_arm, lower, upper, frames, family, difficulty):
    """Sample safe-region starts throughout joint bounds and smooth local strokes.

    The joint path is a target-generation witness, never an Actor input or a
    joint-space training target. Longer episodes allow larger TCP excursions
    without asking the position servo to follow an impossible frame jump.
    """
    fraction = (0.25, 0.55, 1.0)[difficulty]
    extent = (0.08, 0.16, 0.24)[difficulty]
    base = rng.uniform(home_arm + fraction * (lower - home_arm),
                       home_arm + fraction * (upper - home_arm))
    anchors = np.empty((5, 14), np.float64)
    anchors[0] = base
    for index in range(1, 5):
        left = rng.uniform(-extent, extent, 7)
        right = rng.uniform(-extent, extent, 7)
        if family == "opposed_strokes":
            right = -left + rng.uniform(-extent * 0.2, extent * 0.2, 7)
        elif family == "asymmetric_arms":
            (left if index % 2 else right)[:] *= 0.25
        elif family == "move_hold" and index in (2, 4):
            anchors[index] = anchors[index - 1]
            continue
        elif family == "oscillatory" and index >= 3:
            anchors[index] = anchors[index - 2]
            continue
        anchors[index] = np.clip(base + np.concatenate((left, right)), lower, upper)
    sample = np.linspace(0, 4, frames)
    segment = np.minimum(sample.astype(int), 3)
    phase = _smoothstep(sample - segment)
    return anchors[segment] * (1 - phase[:, None]) + anchors[segment + 1] * phase[:, None]


def generate_teleop_workspace_bank(path: str | Path, *, seed: int,
                                   split_counts=(256, 32, 32), frames=201,
                                   max_attempts_per_clip=120,
                                   profile="local_v1",
                                   mjcf: str | Path = DEFAULT_MJCF) -> dict:
    """Generate disjoint train/validation/test target streams at exactly 50 Hz.

    Each accepted frame is collision-free in source MuJoCo kinematics. This is
    a screen, not a claim that a policy has physically executed each target.
    """
    if len(split_counts) != 3 or min(split_counts) < 1:
        raise ValueError("Bank requires three nonempty splits")
    if profile == "local_v1":
        if frames != 201:
            raise ValueError("local_v1 requires a 201-frame, 4-second bank")
    elif profile == "broad_v2":
        if frames < 401 or (frames - 1) % 200:
            raise ValueError("broad_v2 requires at least 401 frames and whole 4-second blocks")
    else:
        raise ValueError("Unknown workspace target profile")
    path = Path(path)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    joint_names = tuple(f"Joint{j}_{side}" for side in ("L", "R") for j in range(1, 8))
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                 for name in joint_names]
    joint_addresses = np.array([model.jnt_qposadr[i] for i in joint_ids])
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{side}")
                for side in ("L", "R")]
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    home = model.key_qpos[key].copy()
    margin = 0.08 if profile == "broad_v2" else 0.03
    lower = model.jnt_range[joint_ids, 0] + margin
    upper = model.jnt_range[joint_ids, 1] - margin
    rng = np.random.default_rng(seed)
    poses, witnesses, families, difficulties, splits, names = [], [], [], [], [], []
    total = sum(split_counts)
    for index in range(total):
        split = next(i for i, cutoff in enumerate(np.cumsum(split_counts)) if index < cutoff)
        family_index = (index + split) % len(FAMILIES)
        difficulty = ((index % 3) if profile == "broad_v2"
                      else (index // len(FAMILIES)) % 3)
        family = FAMILIES[family_index]
        for _ in range(max_attempts_per_clip):
            witness = (_broad_joint_path(rng, home[joint_addresses], lower, upper,
                                         frames, family, difficulty)
                       if profile == "broad_v2" else
                       _joint_path(rng, home[joint_addresses], frames, family, difficulty))
            if (np.any(witness <= lower) or np.any(witness >= upper)
                    or np.max(np.abs(np.diff(witness, axis=0))) > 0.009):
                continue
            frame_pose = np.empty((frames, 2, 7), np.float32)
            valid = True
            for frame, arm_q in enumerate(witness):
                data.qpos[:] = home
                data.qpos[joint_addresses] = arm_q
                mujoco.mj_forward(model, data)
                if data.ncon:
                    valid = False
                    break
                for side, site_id in enumerate(site_ids):
                    frame_pose[frame, side, :3] = data.site_xpos[site_id]
                    quaternion = Rotation.from_matrix(
                        data.site_xmat[site_id].reshape(3, 3)).as_quat().astype(np.float32)
                    if frame and np.dot(quaternion, frame_pose[frame - 1, side, 3:]) < 0:
                        quaternion = -quaternion
                    frame_pose[frame, side, 3:] = quaternion
            if valid:
                break
        else:
            raise RuntimeError(f"Could not produce safe target clip {index} after bounded attempts")
        witnesses.append(witness.astype(np.float32))
        poses.append(frame_pose)
        families.append(family)
        difficulties.append(difficulty)
        splits.append(split)
        names.append(f"teleop_{SPLITS[split]}_{index:04d}_{family}_l{difficulty}")
    path.parent.mkdir(parents=True, exist_ok=True)
    version = ("marvin_teleop_workspace_bank_v2" if profile == "broad_v2"
               else "marvin_teleop_workspace_bank_v1")
    np.savez_compressed(path, version=version, seed=seed,
                        fps=50, joint_names_arm=np.array(joint_names),
                        names=np.array(names), split=np.array(splits, np.int8),
                        family=np.array(families), difficulty=np.array(difficulties, np.int8),
                        q_reset_witness=np.stack(witnesses),
                        tcp_pos=np.stack(poses)[..., :3],
                        tcp_quat_xyzw=np.stack(poses)[..., 3:])
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clips": total, "frames": frames, "split_counts": list(split_counts),
            "families": list(FAMILIES), "seed": seed, "profile": profile,
            "duration_s": (frames - 1) / 50}


def generate_teleop_reach_probe_bank(path: str | Path, *, seed: int,
                                     split_counts=(1, 1, 32), frames=601,
                                     max_attempts_per_clip=500,
                                     mjcf: str | Path = DEFAULT_MJCF) -> dict:
    """FK-only, held target probes from independently sampled safe starts.

    The goal joint witness is discarded after FK. The saved reset witness is
    deliberately different from the target, so this probes long-range reaching
    that smooth tracking clips starting on their target cannot measure.
    """
    if len(split_counts) != 3 or min(split_counts) < 1 or frames < 401:
        raise ValueError("Far-goal probe needs three splits and at least eight seconds")
    path = Path(path)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    names = tuple(f"Joint{joint}_{side}" for side in ("L", "R") for joint in range(1, 8))
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    addresses = model.jnt_qposadr[ids]
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"tcp_{side}")
             for side in ("L", "R")]
    home = model.key_qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")].copy()
    lower, upper = model.jnt_range[ids, 0] + 0.08, model.jnt_range[ids, 1] - 0.08
    rng = np.random.default_rng(seed)
    starts, target_poses, difficulties = [], [], []

    def pose_at(q):
        data.qpos[:] = home
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        if data.ncon:
            return None
        return np.stack([np.r_[data.site_xpos[site], Rotation.from_matrix(
            data.site_xmat[site].reshape(3, 3)).as_quat()] for site in sites])

    for index in range(sum(split_counts)):
        difficulty = index % 3
        radius = (0.65, 1.1, 1.8)[difficulty]
        for _ in range(max_attempts_per_clip):
            start = rng.uniform(lower, upper)
            start_pose = pose_at(start)
            if start_pose is None:
                continue
            goal_q = np.clip(start + rng.uniform(-radius, radius, 14), lower, upper)
            goal_pose = pose_at(goal_q)
            if goal_pose is None or np.linalg.norm(
                    goal_pose[:, :3] - start_pose[:, :3], axis=-1).min() <= 0.05:
                continue
            # Existence of a contact-free joint interpolation is a useful
            # static feasibility screen; Newton still decides dynamic safety.
            if any(pose_at(start * (1 - alpha) + goal_q * alpha) is None
                   for alpha in np.linspace(0, 1, 21)):
                continue
            starts.append(start.astype(np.float32))
            target_poses.append(goal_pose.astype(np.float32))
            difficulties.append(difficulty)
            break
        else:
            raise RuntimeError(f"Could not sample safe far-goal probe {index}")

    split = np.repeat(np.arange(3, dtype=np.int8), split_counts)
    targets = np.repeat(np.array(target_poses, np.float32)[:, None], frames, axis=1)
    reset = np.repeat(np.array(starts, np.float32)[:, None], frames, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, version="marvin_teleop_workspace_bank_v3", seed=seed,
                        fps=50, joint_names_arm=np.array(names),
                        names=np.array([f"far_reach_{SPLITS[int(s)]}_{i:04d}"
                                        for i, s in enumerate(split)]),
                        split=split, family=np.array(["static_far_reach"] * len(split)),
                        difficulty=np.array(difficulties, np.int8),
                        q_reset_witness=reset,
                        tcp_pos=targets[..., :3], tcp_quat_xyzw=targets[..., 3:])
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clips": len(split), "frames": frames, "profile": "static_far_reach_probe_v3"}


class TeleopWorkspaceReferenceBank:
    """GPU batched TCP lookup; joint witness is exposed only to physical reset."""

    def __init__(self, path: str | Path, device: str | torch.device):
        path = Path(path)
        self.device = torch.device(device)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            self.version = str(data["version"])
            if self.version not in ("marvin_teleop_workspace_bank_v1",
                                    "marvin_teleop_workspace_bank_v2",
                                    "marvin_teleop_workspace_bank_v3") or int(data["fps"]) != 50:
                raise ValueError("Target bank version or 50 Hz rate differs")
            self.names = tuple(str(x) for x in data["names"])
            self.joint_names_arm = tuple(str(x) for x in data["joint_names_arm"])
            pos = torch.as_tensor(data["tcp_pos"].copy(), device=self.device)
            quat = torch.as_tensor(data["tcp_quat_xyzw"].copy(), device=self.device)
            self._reset_q = torch.as_tensor(data["q_reset_witness"].copy(), device=self.device)
            self.split = torch.as_tensor(data["split"].copy(), device=self.device, dtype=torch.long)
            self.difficulty = torch.as_tensor(data["difficulty"].copy(), device=self.device, dtype=torch.long)
            self.family = tuple(str(x) for x in data["family"])
        if (pos.ndim != 4 or pos.shape[-2:] != (2, 3)
                or quat.shape != (*pos.shape[:-1], 4)
                or self._reset_q.shape != (*pos.shape[:2], 14)
                or len(self.joint_names_arm) != 14
                or not bool(torch.isfinite(pos).all())
                or not bool(torch.isfinite(quat).all())):
            raise ValueError("Target bank tensors are malformed or nonfinite")
        self._poses = torch.cat((pos, quaternion_xyzw_to_6d(quat)), dim=-1).flatten(-2)
        self.lengths = torch.full((len(self.names),), pos.shape[1],
                                  device=self.device, dtype=torch.long)
        # The key is part of existing v1 checkpoint manifests, not a display name.
        hash_key = ("teleop_workspace_bank_v1" if self.version.endswith("_v1")
                    else self.version)
        self.hashes = {hash_key: digest}

    def _check(self, motion_id, frame):
        if (motion_id.ndim != 1 or motion_id.shape != frame.shape
                or motion_id.dtype != torch.long or frame.dtype != torch.long
                or bool(((motion_id < 0) | (motion_id >= len(self.names))).any())
                or bool(((frame < 0) | (frame >= self.lengths[motion_id])).any())):
            raise ValueError("Target bank motion ID or frame is invalid")

    def pose_at(self, motion_id, frame):
        self._check(motion_id, frame)
        return self._poses[motion_id, frame]

    def reset_q_at(self, motion_id, frame):
        self._check(motion_id, frame)
        return self._reset_q[motion_id, frame]

    def sample_ids(self, count, *, split: str, max_difficulty=2):
        if split not in SPLITS or not 0 <= max_difficulty <= 2:
            raise ValueError("Unknown bank split or difficulty")
        eligible = torch.nonzero((self.split == SPLITS.index(split))
                                 & (self.difficulty <= max_difficulty)).flatten()
        if len(eligible) == 0:
            raise ValueError("No target clips in requested split/difficulty")
        return eligible[torch.randint(len(eligible), (count,), device=self.device)]
