"""Seeded FK goal sampling; witness joints never leave this sampler."""

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from dexgym.algorithms.supermimic.rewards import so3_geodesic
from .se3_reach_hold import pack_dual_tcp_pose


@dataclass(frozen=True)
class SE3CurriculumLevel:
    start_span_rad: float
    target_span_rad: float
    min_arm_joint_motion_rad: float
    min_position_m: float
    max_position_m: float
    min_rotation_rad: float
    max_rotation_rad: float


CURRICULUM = {
    "near": SE3CurriculumLevel(0.08, 0.045, 0.030, 0.005, 0.08, 0.005, 0.25),
    "medium": SE3CurriculumLevel(0.15, 0.11, 0.075, 0.012, 0.18, 0.012, 0.55),
    "wider_feasible": SE3CurriculumLevel(0.22, 0.19, 0.13, 0.020, 0.30, 0.020, 1.0),
    # Train-only near-target resets expose the actual low-speed hold region.
    # Both TCPs start outside the strict 5 mm position-success boundary.
    "near_hold_bootstrap": SE3CurriculumLevel(0.08, 0.018, 0.012,
                                                 0.006, 0.012, 0.015, 0.07),
}


@dataclass
class SE3GoalSamples:
    start_q: torch.Tensor
    goal_pose: torch.Tensor
    position_difficulty_m: torch.Tensor
    rotation_difficulty_rad: torch.Tensor

    def index(self, indices):
        return SE3GoalSamples(self.start_q[indices], self.goal_pose[indices],
                              self.position_difficulty_m[indices],
                              self.rotation_difficulty_rad[indices])


class SE3GoalSampler:
    """Static MuJoCo collision/path screening followed by batched Marvin FK."""

    def __init__(self, robot, *, train_seed: int, evaluation_seed: int,
                 evaluation_set_size: int = 64):
        if evaluation_set_size < 1 or train_seed == evaluation_seed:
            raise ValueError("Training and evaluation goal generators need distinct seeds")
        self.robot = robot
        self.train_rng = np.random.default_rng(train_seed)
        self.eval_rng = np.random.default_rng(evaluation_seed)
        self.evaluation_set_size = evaluation_set_size
        self._evaluation_sets = {}
        self._evaluation_cursors = {}
        self.model = mujoco.MjModel.from_xml_path(str(robot.native.mjcf))
        self.data = mujoco.MjData(self.model)
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        self.home_full = self.model.key_qpos[key].copy()
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                     for name in robot.spec.joint_order]
        self.qpos_ids = self.model.jnt_qposadr[joint_ids]
        self.home_arm = self.home_full[self.qpos_ids].copy()
        self.lower = (robot.action_map.lower + 0.02).detach().cpu().numpy()
        self.upper = (robot.action_map.upper - 0.02).detach().cpu().numpy()

    def _collision_free_path(self, start: np.ndarray, target: np.ndarray) -> bool:
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            self.data.qpos[:] = self.home_full
            self.data.qpos[self.qpos_ids] = start + fraction * (target - start)
            mujoco.mj_forward(self.model, self.data)
            if self.data.ncon:
                return False
        return True

    def _draw(self, count: int, level: str, rng: np.random.Generator) -> SE3GoalSamples:
        if count < 1 or level not in CURRICULUM:
            raise ValueError("Unknown SE(3) curriculum level or batch size")
        config = CURRICULUM[level]
        accepted_start, accepted_pose, accepted_position, accepted_rotation = [], [], [], []
        attempts = 0
        while len(accepted_start) < count:
            remaining = count - len(accepted_start)
            starts, witnesses = [], []
            for _ in range(max(remaining * 4, 16)):
                attempts += 1
                if attempts > count * 500:
                    raise RuntimeError("Could not sample safe dual-arm SE(3) goals")
                start = self.home_arm + rng.uniform(-config.start_span_rad,
                                                    config.start_span_rad, 14)
                witness = start + rng.uniform(-config.target_span_rad,
                                               config.target_span_rad, 14)
                if (np.any(start < self.lower) or np.any(start > self.upper)
                        or np.any(witness < self.lower) or np.any(witness > self.upper)):
                    continue
                joint_motion = np.linalg.norm((witness - start).reshape(2, 7), axis=-1)
                if np.any(joint_motion < config.min_arm_joint_motion_rad):
                    continue
                if not self._collision_free_path(start, witness):
                    continue
                starts.append(start)
                witnesses.append(witness)
            if not starts:
                continue
            device = self.robot.native.device
            start_q = torch.tensor(np.stack(starts), device=device, dtype=torch.float32)
            witness_q = torch.tensor(np.stack(witnesses), device=device, dtype=torch.float32)
            start_position, start_rotation = self.robot.fk(start_q)
            target_position, target_rotation = self.robot.fk(witness_q)
            start_position, start_rotation = start_position[:, :, 2], start_rotation[:, :, 2]
            target_position, target_rotation = target_position[:, :, 2], target_rotation[:, :, 2]
            position = torch.linalg.vector_norm(target_position - start_position, dim=-1)
            rotation = so3_geodesic(start_rotation, target_rotation)
            valid = ((position >= config.min_position_m).all(-1)
                     & (position <= config.max_position_m).all(-1)
                     & (rotation >= config.min_rotation_rad).all(-1)
                     & (rotation <= config.max_rotation_rad).all(-1))
            pose = pack_dual_tcp_pose(target_position, target_rotation)
            for index in torch.nonzero(valid, as_tuple=False).flatten().tolist():
                accepted_start.append(start_q[index])
                accepted_pose.append(pose[index])
                accepted_position.append(position[index])
                accepted_rotation.append(rotation[index])
                if len(accepted_start) == count:
                    break
        return SE3GoalSamples(torch.stack(accepted_start), torch.stack(accepted_pose),
                              torch.stack(accepted_position), torch.stack(accepted_rotation))

    def sample(self, count: int, level: str) -> SE3GoalSamples:
        if level == "near_hold_mix":
            if count < 1:
                raise ValueError("Goal batch size must be positive")
            hold_count = int(self.train_rng.binomial(count, 0.5))
            parts = []
            if hold_count:
                parts.append(self._draw(hold_count, "near_hold_bootstrap", self.train_rng))
            if count > hold_count:
                parts.append(self._draw(count - hold_count, "near", self.train_rng))
            combined = SE3GoalSamples(*(
                torch.cat([getattr(part, field) for part in parts], dim=0)
                for field in ("start_q", "goal_pose", "position_difficulty_m",
                              "rotation_difficulty_rad")))
            order = torch.as_tensor(self.train_rng.permutation(count),
                                    device=combined.start_q.device, dtype=torch.long)
            return combined.index(order)
        return self._draw(count, level, self.train_rng)

    def sample_evaluation(self, count: int, level: str) -> SE3GoalSamples:
        if level not in ("near", "medium", "wider_feasible"):
            raise ValueError("Evaluation goal sets use the original Near/Medium/Wider levels")
        if level not in self._evaluation_sets:
            self._evaluation_sets[level] = self._draw(self.evaluation_set_size,
                                                      level, self.eval_rng)
            self._evaluation_cursors[level] = 0
        saved = self._evaluation_sets[level]
        index = (torch.arange(count, device=saved.start_q.device)
                 + self._evaluation_cursors[level]) % self.evaluation_set_size
        self._evaluation_cursors[level] = (
            self._evaluation_cursors[level] + count) % self.evaluation_set_size
        return saved.index(index)
