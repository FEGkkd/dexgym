"""Common closed-loop Isaac Lab / Newton transition for both learner tasks."""

from types import SimpleNamespace

import torch

from ..marvin54_s0.physics import NONFINITE, OVERFLOW, SUBSTEPS
from .domain import Domain, IMPULSE_PROBABILITY, apply_newton_domain
from .residual import N

BAD_ACTION = 32


class ResidualEnv:
    def __init__(self, backend, bank, task_class, *, robust, seed, transfer_model=None, reward_profile="v1"):
        self.backend = backend
        self.device, self.num_envs = backend.device, backend.num_envs
        self.common_step_counter = 0
        self.disturbance = SimpleNamespace(adaptive_scale=float(robust), success_scale=0.)
        arm = torch.as_tensor([name.startswith("Joint") for name in bank.names], device=self.device)
        self.domain = Domain(self.num_envs, self.device, seed + 37, robust, arm, transfer=transfer_model is not None)
        self.task = task_class(self, backend.model, bank, domain=self.domain, transfer_model=transfer_model,
                               reward_profile=reward_profile, hand_model=transfer_model)
        self.domain_buffers = apply_newton_domain(backend, self.task)
        self._state = backend.state()
        self.memory = {
            "frame": torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
            "cmd": self._state[:, :N],
            "action": torch.zeros(self.num_envs, N, device=self.device),
            "delay": self.domain.delay,
        }
        self.transition = None

    def state(self):
        return self._state

    def observation(self):
        task = self.task
        x, m = self._state, self.memory
        return {
            "policy": task.policy_observation(x, m, self.domain.observation_bias),
            "critic": task.context(x, m, self.domain.observation_bias, self.domain.physical),
        }

    def reset(self, env_ids=None, frame=None, *, exact=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        bank = self.task.bank
        if frame is None:
            frame = bank.sample(len(env_ids), self.domain.generator)
        q, v = bank.q[frame], bank.qd[frame]
        if not exact:
            q, v = self.domain.reset_noise(q, v)
        q = q.clamp(self.task.lo, self.task.hi)
        self.backend.reset(env_ids, q, v)
        self._state = self._state.index_copy(0, env_ids, torch.cat((q, v), -1))
        # Replay retains the pre-step memory until the collector writes it.
        self.memory = {key: value.clone() for key, value in self.memory.items()}
        self.memory["frame"][env_ids] = frame
        self.memory["cmd"][env_ids] = q
        self.memory["action"][env_ids] = 0.
        return self.observation(), {}

    def apply_impulse(self, x, done):
        if not self.domain.robust or self.common_step_counter % 25 != 24:
            return x
        ids = ((self.domain.uniform((self.num_envs,)) < 25 * IMPULSE_PROBABILITY) & ~done).nonzero().flatten()
        if not len(ids):
            return x
        dq = x[ids, N:] + self.domain.velocity_impulse((len(ids), N))
        backend = self.backend
        with backend.simulation_stream():
            backend.robot.write_joint_velocity_to_sim_index(velocity=dq, env_ids=ids)
            backend.scene.write_data_to_sim()
            backend.sim.forward()
            backend.scene.update(0.0)
        return torch.cat((x[:, :N], x[:, N:].index_copy(0, ids, dq)), -1)

    def step(self, action):
        task = self.task
        before = self.memory
        bad = ~torch.isfinite(action).all(-1)
        action = action.nan_to_num(nan=0., posinf=0., neginf=0.).clamp(-1., 1.)
        target, applied = task.control_target(action, before)
        measured, torque, depth, substep, flags, deep_steps = self.backend.interval(applied, task)
        flags = flags | (bad.int() * BAD_ACTION)
        invalid = ((flags & (NONFINITE | OVERFLOW | BAD_ACTION)) != 0)
        full = ((substep == 0) | (substep == SUBSTEPS)) & ~invalid
        x = measured.nan_to_num(nan=0., posinf=0., neginf=0.)
        contact = task.contact_labels(depth)
        reward, _, errors = task.reward(x, before, action, target, contact, torque)
        reward = torch.where(invalid, -1., reward)
        frame = task.current_goal(before)
        completed = (frame == task.bank.end[frame]) & (flags == 0)
        terminated = completed | (flags != 0)
        self._state = self.apply_impulse(x, terminated)
        self.memory = task.advance_memory(self._state, before, action, target, None, None, soft=False)
        proposed = task.bank.q[frame] + task.scale * action
        stats = {
            "wrist_mm": errors["wrist"].sqrt().mean(-1) * 1000,
            "local_tip_mm": errors["local"][:, :, :, -1].sqrt().mean((1, 2)) * 1000,
            "world_tip_mm": errors["tip"].sqrt().mean((1, 2)) * 1000,
            "relative_wrist_mm": errors["relative"].sqrt() * 1000,
            "wrist_chord": errors["rotation"].sqrt().mean(-1),
            "projection_fraction": ((proposed - target).abs() > 1e-6).float().mean(-1),
            "contact_substep_fraction": deep_steps.float() / SUBSTEPS,
            "mean_compensation_rad": ((target - task.bank.q[frame]).abs()).mean(-1),
        }
        if task.human_intent:
            stats.update({
                "intent_wrist_mm": errors["intent_wrist"].sqrt().mean(-1) * 1000,
                "intent_local_tip_mm": errors["intent_local_tip"].sqrt().mean((1, 2)) * 1000,
                "intent_wrist_chord": errors["intent_rotation"].sqrt().mean(-1),
                "intent_relative_mm": errors["intent_relative"].sqrt() * 1000,
                "intent_score": errors["intent_score"],
                "execution_score": errors["execution_score"],
                "action_rate": errors["action_rate"],
                "action_magnitude": errors["action_magnitude"],
            })
        if task.retarget_pose:
            def degrees(key):
                return (2. * (.5 * errors[key].sqrt()).clamp(0., 1.).asin()).rad2deg()

            stats.update({
                "wrist_deg": degrees("rotation").mean(-1),
                "wrist_local_tip_mm": errors["wrist_local_tip"].sqrt().mean((1, 2)) * 1000.,
                "wrist_local_tip_deg": degrees("wrist_local_rotation").mean((1, 2)),
                "relative_wrist_deg": degrees("relative_rotation"),
                "pose_score": errors["pose_score"],
                "action_rate": errors["action_rate"],
            })
        self.transition = {
            "x1": x, "c": target, "u": applied, "contact": contact, "torque": torque,
            "depth": depth, "nan": invalid, "drop": flags != 0, "full_step": full,
            "failure_bits": flags, "safety_substep": substep, "deep_substeps": deep_steps,
            "safety_failure": flags != 0, "rollout_stats": stats,
            "rollout_event_flags": {"reference_complete": completed, "hard_contact": contact[:, 1] > 0,
                                    "solver_invalid": invalid},
        }
        self.common_step_counter += 1
        return self.observation(), reward, terminated, torch.zeros_like(terminated), {}

    @torch.no_grad()
    def contract(self):
        bank = self.task.bank
        ids = torch.arange(self.num_envs, device=self.device)
        frames = ids % bank.frames
        self.reset(frame=frames, exact=True)
        q, v = self.backend.state().split(N, -1)
        expected = self.task.goal_geometry[0][frames]
        sites = [self.backend.robot.body_names.index(f"TCP_Link_{side}") for side in "LR"]
        measured = self.backend.robot.data.body_pose_w.torch[:, sites, :3] - self.backend.scene.env_origins[:, None]
        errors = {
            "q_rad": float((q - bank.q[frames]).abs().max()),
            "qd_rad_s": float((v - bank.qd[frames]).abs().max()),
            "tcp_m": float((measured - expected).abs().max()),
            "source_tcp_m": float((bank.stored_wrists[..., :3] - self.task.goal_geometry[0]).abs().max()),
            "source_tip_m": float((bank.stored_tips[..., :3] - self.task.goal_geometry[-1]).abs().max()),
        }
        if any(not 0 <= value < .0003 for value in errors.values()):
            raise RuntimeError(f"Reference/Newton FK and state contract failed: {errors}")
        out = {"errors": errors, "envs": self.num_envs, "solver": type(self.backend.solver).__name__,
               "physical_dt_s": .002, "control_dt_s": .020, "substeps": SUBSTEPS}
        if self.domain.robust:
            out["domain_fields"] = {
                key: float((buffer - expected).abs().max()) for key, (buffer, expected) in self.domain_buffers.items()
            }
            out["domain_world_max_span"] = {
                key: float((expected - expected[:1]).abs().max()) for key, (_, expected) in self.domain_buffers.items()
            }
            out["delay_worlds"] = int(self.domain.delay.sum())
            if self.task.transfer_delta is not None:
                import warp as wp

                mix = self.domain.physical[:, -1]
                out["transfer_worlds"] = {
                    "packaged": int((mix == 0).sum()), "teleop": int((mix == 1).sum()),
                    "intermediate": int(((mix > 0) & (mix < 1)).sum()),
                }
                self.step(torch.zeros(self.num_envs, N, device=self.device))
                self.common_step_counter = 0
                out["domain_after_interval_max_error"] = {
                    key: float((buffer - expected).abs().max()) for key, (buffer, expected) in self.domain_buffers.items()
                }
                if any(value > 1e-5 for value in out["domain_after_interval_max_error"].values()):
                    raise RuntimeError(f"Newton replaced per-world domain parameters: {out['domain_after_interval_max_error']}")
                with self.backend.simulation_stream():
                    gravity_torque = wp.to_torch(self.backend.mjw_data.qfrc_gravcomp).abs().amax(-1)
                    out["gravity_compensation_peak_nm"] = {
                        key: float(gravity_torque[mask].max())
                        for key, mask in (("packaged", mix == 0), ("teleop", mix == 1)) if mask.any()
                    }
                peaks = out["gravity_compensation_peak_nm"]
                if peaks.get("packaged", 1.) < 1e-4 or peaks.get("teleop", 0.) > 1e-4:
                    raise RuntimeError(f"Gravity compensation was not applied by the Newton solver: {peaks}")
        return out
