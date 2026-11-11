"""Reference library and differentiable task for the DexGym SuperMimic engine."""

import hashlib

import mujoco
import numpy as np
import torch

from ..hand2_cube_supermimic.kinematics import HandKinematics
from ..hand2_cube_supermimic.rotations import mm, quat_to_mat

TASK_ID = "DexGym-MarvinM6-WujiHand22-MotionTracking-SuperMimic-v0"
JOINTS = 54
DT = 0.02
IMAGINATION_STEPS = 32
FINGERS = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")


class MotionLibrary:
    """Concatenate clips with explicit ends. Indices can never cross a clip."""

    def __init__(self, paths, model, xml, device):
        self.names = [model.joint(i).name for i in np.argsort(model.jnt_qposadr)]
        if len(self.names) != JOINTS or model.nq != JOINTS or model.nv != JOINTS:
            raise ValueError("Expected 54 scalar joints")
        self.model_sha256 = hashlib.sha256(xml.read_bytes()).hexdigest()
        fields = {
            "qpos": (54,), "qvel": (54,), "ref_wrist_pos": (2, 3),
            "ref_wrist_quat_wxyz": (2, 4), "ref_elbow_pos": (2, 3),
            "ref_tip_pos": (2, 5, 3), "ref_tip_quat_wxyz": (2, 5, 4),
        }
        parts = {k: [] for k in fields}
        ends, starts, reset_limits = [], [], []
        self.sha256 = []
        offset = 0
        for path in paths:
            with path.open("rb") as file:
                self.sha256.append(hashlib.file_digest(file, "sha256").hexdigest())
            with np.load(path, allow_pickle=False) as clip:
                if list(clip["joint_names"]) != self.names or clip["model_sha256"].item() != self.model_sha256:
                    raise ValueError(f"Asset or joint ordering mismatch: {path}")
                time = clip["time_s"]
                n = len(time)
                if n <= IMAGINATION_STEPS or not np.allclose(time, np.arange(n) * DT, rtol=0, atol=1e-7):
                    raise ValueError(f"Expected at least 33 consecutive 50 Hz frames: {path}")
                for name, shape in fields.items():
                    value = clip[name]
                    if value.shape != (n, *shape) or not np.isfinite(value).all():
                        raise ValueError(f"Invalid {name} in {path}")
                    if name.endswith("_quat_wxyz") and not np.allclose(
                        np.linalg.norm(value, axis=-1), 1.0, rtol=0, atol=1e-4
                    ):
                        raise ValueError(f"Invalid quaternion in {name}: {path}")
                    parts[name].append(value)
                starts.append(offset)
                offset += n
                ends.append(torch.full((n,), offset - 1, dtype=torch.long, device=device))
                reserve = IMAGINATION_STEPS if n <= 2 * IMAGINATION_STEPS else 1
                reset_limits.append(torch.full((n,), offset - 1 - reserve, dtype=torch.long, device=device))
        self.frames = offset
        self.start = torch.tensor(starts, dtype=torch.long, device=device)
        self.end = torch.cat(ends)
        self.reset_limit = torch.cat(reset_limits)
        for k, values in parts.items():
            tensor = torch.as_tensor(np.concatenate(values), dtype=torch.float32, device=device)
            if k.endswith("_quat_wxyz"):
                k = k.replace("_quat_wxyz", "_rot")
                tensor = quat_to_mat(tensor)
            setattr(self, k, tensor)

    def at(self, frame, delta=1):
        return torch.minimum(frame + delta, self.end[frame])

    def sample(self, n):
        # Short clips reserve a complete imagination window at every reset.
        frame = torch.randint(self.frames, (n,), device=self.end.device)
        return torch.minimum(frame, self.reset_limit[frame])


class TrackingGeometry:
    def __init__(self, model, library, device):
        tips = [model.site(f"{side}_{finger}_tip").id for side in ("l", "r") for finger in FINGERS]
        self.wrists = [model.body(f"{side}_wrist").id for side in ("l", "r")]
        self.elbows = [model.body(f"Link4_{side}").id for side in ("L", "R")]
        ids = [model.joint(name).id for name in library.names]
        if np.any(model.qpos0):
            raise ValueError("FK expects the current MJCF's zero hinge references")
        self.tree = HandKinematics(model, range(model.nbody), ids, tips, device)
        self.tip_rotation = quat_to_mat(torch.as_tensor(model.site_quat[tips], device=device, dtype=torch.float32))

    def pose(self, q):
        kin = self.tree.forward(q, torch.zeros_like(q))
        p, r, _, _ = kin
        tip_pos, _ = self.tree.tips(kin)
        tip_rot = mm(r[:, self.tree.tip_bodies], self.tip_rotation)
        return p[:, self.wrists], r[:, self.wrists], p[:, self.elbows], tip_pos, tip_rot


class JointCodec:
    @staticmethod
    def difference(a, b):
        return b - a

    @staticmethod
    def integrate(x, delta):
        return x + delta


class TrackingTask:
    task_id = TASK_ID
    state_dim = dynamics_dim = 108
    action_dim = 54
    contact_dim = 2
    policy_dim = context_dim = 324
    error_floor = [0.002] * 54 + [0.05] * 54
    codec = JointCodec

    def __init__(self, env, model, library, nominal):
        self.env = self.reader = env
        self.device = env.device
        self.library = library
        self.geometry = TrackingGeometry(model, library, self.device)
        self.simulator_context = torch.zeros(env.num_envs, 1, device=self.device)
        self.encoder_bias = torch.zeros(env.num_envs, 54, device=self.device)
        joint_ids = [model.joint(name).id for name in library.names]
        actuator_ids = [int(np.flatnonzero(model.actuator_trnid[:, 0] == j).item()) for j in joint_ids]
        self.kp = self.tensor(model.actuator_gainprm[actuator_ids, 0])
        self.kd = self.tensor(-model.actuator_biasprm[actuator_ids, 2])
        self.effort = self.tensor(model.actuator_forcerange[actuator_ids, 1])
        self.joint_lo, self.joint_hi = self.tensor(model.jnt_range[joint_ids]).T
        ctrl_lo, ctrl_hi = self.tensor(model.actuator_ctrlrange[actuator_ids]).T
        self.lo = torch.maximum(self.joint_lo, ctrl_lo)
        self.hi = torch.minimum(self.joint_hi, ctrl_hi)
        if ((library.qpos < self.lo - 1e-6) | (library.qpos > self.hi + 1e-6)).any():
            raise ValueError("Motion Library poses exceed model limits")
        arm_scale, hand_scale = (0.008, 0.025) if nominal == "feedforward" else (0.18, 0.25)
        self.scale = self.tensor([arm_scale if name.startswith("Joint") else hand_scale for name in library.names])
        self.nominal = library.qpos
        if nominal == "feedforward":
            self.nominal = self.feedforward_targets(model)

    def feedforward_targets(self, model):
        data = mujoco.MjData(model)
        q = self.library.qpos.cpu().numpy()
        qd = self.library.qvel.cpu().numpy()
        bias = np.empty_like(q)
        for i in range(len(q)):
            data.qpos[:] = q[i]
            data.qvel[:] = qd[i]
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            bias[i] = data.qfrc_bias - data.qfrc_passive
        velocity = self.library.qvel
        return (self.library.qpos + (self.tensor(bias) + self.kd * velocity) / self.kp).clamp(self.lo, self.hi)

    def tensor(self, x):
        return torch.as_tensor(x, device=self.device, dtype=torch.float32)

    def read(self):
        return self.reader.state(), self.env.memory, self.simulator_context

    def current_goal(self, m):
        return self.library.at(m["frame"])

    def sample_goal(self, n):
        # The next target is deterministic. Only resets sample reference phase.
        return None

    def advance_memory(self, x1, m, a, target, cage, next_goal, soft=True):
        return self.replay_failure_memory(m, a)

    def replay_failure_memory(self, m, a):
        return {"frame": self.current_goal(m), "action": a}

    def imagination_start_mask(self, m, horizon):
        frame = m["frame"]
        return frame + horizon <= self.library.end[frame]

    @staticmethod
    def encode_memory(m):
        return torch.cat((m["frame"][:, None].double(), m["action"]), dim=-1)

    @staticmethod
    def decode_memory(encoded):
        return {"frame": encoded[:, 0].long(), "action": encoded[:, 1:].float()}

    def policy_observation(self, x, m, aux):
        lib = self.library
        goal = self.current_goal(m)
        preview = lib.at(m["frame"], 5)
        return torch.cat((x, lib.qpos[goal], lib.qvel[goal], lib.qpos[preview], m["action"]), dim=-1)

    def context(self, x, m, aux, sim):
        return self.policy_observation(x, m, aux)

    @staticmethod
    def critic(context):
        return context

    @staticmethod
    def history_from_obs(obs):
        return obs[:, None]

    @staticmethod
    def flatten_history(history):
        return history[:, -1]

    @staticmethod
    def dynamics(x, sim, aux):
        return x

    @staticmethod
    def replay_push(x1, push, aux):
        return x1, aux

    @staticmethod
    def imagine_push(x1, m, aux):
        return x1, aux

    def control_target(self, action, m, bias):
        u = (self.nominal[self.current_goal(m)] + self.scale * action).clamp(self.lo, self.hi)
        return u, u

    def torque(self, x, u, sim):
        q, qd = x.split(JOINTS, dim=-1)
        return (self.kp * (u - q) - self.kd * qd).clamp(-self.effort, self.effort)

    def position_violation(self, x):
        q = x[:, :JOINTS]
        return torch.maximum(q - self.joint_hi, self.joint_lo - q).amax(-1)

    def termination(self, x1, m):
        overflow = self.position_violation(x1)
        limit = torch.sigmoid((overflow - 0.05) / 0.007)
        end = self.current_goal(m) == self.library.end[m["frame"]]
        return torch.where(end, 1.0, limit)

    def geometric_errors(self, x, m):
        q = x[:, :JOINTS]
        lib = self.library
        goal = self.current_goal(m)
        wp, wr, ep, fp, fr = self.geometry.pose(q)
        wp_delta = wp - lib.ref_wrist_pos[goal]
        return {
            "wrist": wp_delta,
            "wrist_rot": wr - lib.ref_wrist_rot[goal],
            "elbow": ep - lib.ref_elbow_pos[goal],
            "tip": fp - lib.ref_tip_pos[goal].flatten(1, 2),
            "tip_rot": fr - lib.ref_tip_rot[goal].flatten(1, 2),
            "relative_wrist": wp_delta[:, 0] - wp_delta[:, 1],
            "q": q - lib.qpos[goal],
            "qd": x[:, JOINTS:] - lib.qvel[goal],
        }

    def reward(self, x1, m, a, target, contact, torque, soft=False):
        error = self.geometric_errors(x1, m)
        def mse(v):
            return v.flatten(1).square().mean(-1)

        weights = {
            "wrist": (2.0, 0.045), "wrist_rot": (0.7, 0.20), "elbow": (0.5, 0.07),
            "tip": (2.0, 0.045), "tip_rot": (0.7, 0.25), "relative_wrist": (0.4, 0.045),
            "q": (0.8, 0.12), "qd": (0.15, 2.0),
        }
        terms = {k: w * torch.exp(-mse(error[k] / scale)) for k, (w, scale) in weights.items()}
        terms["smooth"] = -0.08 * mse(a - m["action"])
        terms["residual"] = -0.02 * mse(a)
        terms["contact"] = -0.25 * contact[:, 1]
        return sum(terms.values()), contact[:, 1], terms

    @staticmethod
    def angle(error):
        # ||R1 - R2||_F^2 = 4 (1 - cos(theta)).
        cosine = 1 - error.square().sum(dim=(-2, -1)) / 4
        return torch.acos(cosine.clamp(-1, 1))

    def rotation_error(self, x, m):
        return self.angle(self.geometric_errors(x, m)["wrist_rot"]).mean(-1)
