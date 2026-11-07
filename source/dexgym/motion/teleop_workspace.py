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


def generate_teleop_workspace_bank(path: str | Path, *, seed: int,
                                   split_counts=(256, 32, 32), frames=201,
                                   max_attempts_per_clip=120,
                                   mjcf: str | Path = DEFAULT_MJCF) -> dict:
    """Generate disjoint train/validation/test target streams at exactly 50 Hz.

    Each accepted frame is collision-free in source MuJoCo kinematics. This is
    a screen, not a claim that a policy has physically executed each target.
    """
    if len(split_counts) != 3 or min(split_counts) < 1 or frames != 201:
        raise ValueError("Bank requires three nonempty 201-frame splits at 50 Hz")
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
    lower = model.jnt_range[joint_ids, 0] + 0.03
    upper = model.jnt_range[joint_ids, 1] - 0.03
    rng = np.random.default_rng(seed)
    poses, witnesses, families, difficulties, splits, names = [], [], [], [], [], []
    total = sum(split_counts)
    for index in range(total):
        split = next(i for i, cutoff in enumerate(np.cumsum(split_counts)) if index < cutoff)
        family_index = (index + split) % len(FAMILIES)
        difficulty = (index // len(FAMILIES)) % 3
        family = FAMILIES[family_index]
        for _ in range(max_attempts_per_clip):
            witness = _joint_path(rng, home[joint_addresses], frames, family, difficulty)
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
    np.savez_compressed(path, version="marvin_teleop_workspace_bank_v1", seed=seed,
                        fps=50, joint_names_arm=np.array(joint_names),
                        names=np.array(names), split=np.array(splits, np.int8),
                        family=np.array(families), difficulty=np.array(difficulties, np.int8),
                        q_reset_witness=np.stack(witnesses),
                        tcp_pos=np.stack(poses)[..., :3],
                        tcp_quat_xyzw=np.stack(poses)[..., 3:])
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "clips": total, "frames": frames, "split_counts": list(split_counts),
            "families": list(FAMILIES), "seed": seed}


class TeleopWorkspaceReferenceBank:
    """GPU batched TCP lookup; joint witness is exposed only to physical reset."""

    def __init__(self, path: str | Path, device: str | torch.device):
        path = Path(path)
        self.device = torch.device(device)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            if str(data["version"]) != "marvin_teleop_workspace_bank_v1" or int(data["fps"]) != 50:
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
        self.hashes = {"teleop_workspace_bank_v1": digest}

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
