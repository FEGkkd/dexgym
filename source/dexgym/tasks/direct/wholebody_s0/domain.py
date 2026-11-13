"""Shared physical and observation disturbances for training and evaluation."""

import torch

from .residual import DR_DIM, N

PHYSICAL_FIELDS = ("arm_kp", "hand_kp", "kd", "effort", "contact_friction", "joint_friction")
SCALE_RANGES = ((.78, 1.22), (.72, 1.25), (.75, 1.27), (.83, 1.12), (.68, 1.32), (.65, 1.35))
DELAY_PROBABILITY = .35
IMPULSE_PROBABILITY = .007


class Domain:
    def __init__(self, num_envs, device, seed, robust, arm, *, transfer=False):
        self.device, self.robust, self.arm = device, robust, arm
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.physical = torch.ones(num_envs, DR_DIM, device=device)
        self.delay = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.observation_bias = torch.zeros(num_envs, 2 * N, device=device)
        if robust:
            limits = torch.tensor(SCALE_RANGES, device=device)
            self.physical = limits[:, 0] + self.uniform((num_envs, DR_DIM)) * (limits[:, 1] - limits[:, 0])
            self.delay = self.uniform((num_envs,)) < DELAY_PROBABILITY
            q = self.normal((num_envs, N)) * torch.where(arm, .002, .004)
            dq = self.normal((num_envs, N)) * torch.where(arm, .010, .023)
            self.observation_bias = torch.cat((q, dq), -1)
        if transfer:
            profile = self.uniform((num_envs, 1))
            # Include both real configurations and intermediate calibration values.
            mix = torch.where(profile < .20, 0., torch.where(profile < .65, 1., self.uniform((num_envs, 1))))
            self.physical = torch.cat((self.physical, mix), -1)

    def uniform(self, shape):
        return torch.rand(shape, device=self.device, generator=self.generator)

    def normal(self, shape):
        return torch.randn(shape, device=self.device, generator=self.generator)

    def reset_noise(self, q, v):
        if not self.robust:
            return q, v
        return (q + self.normal(q.shape) * torch.where(self.arm, .005, .012),
                v + self.normal(v.shape) * torch.where(self.arm, .035, .080))

    def velocity_impulse(self, shape):
        return self.normal(shape) * torch.where(self.arm, .12, .32)

    def imagine_push(self, x):
        if not self.robust:
            return 0.
        dv = self.velocity_impulse((len(x), N)) * (self.uniform((len(x), 1)) < IMPULSE_PROBABILITY)
        return torch.cat((torch.zeros_like(dv), dv), -1)


def apply_newton_domain(backend, task):
    """Write per-world MJWarp parameters and preserve them through drive refreshes."""
    if not task.domain.robust:
        return {}
    import warp as wp

    values = task.simulator_context
    source = backend.solver.mj_model
    joint_by_dof = {dof: j for j, dof in enumerate(source.jnt_dofadr)}
    position_ids, joint_ids = [], []
    for dof in backend.dof_ids.tolist():
        joint = joint_by_dof[dof]
        ids = [i for i, j in enumerate(source.actuator_trnid[:, 0]) if j == joint and source.actuator_biasprm[i, 1] < 0]
        if len(ids) != 1:
            raise ValueError("Expected one MJWarp position channel per joint")
        position_ids.append(ids[0])
        joint_ids.append(joint)
    kp, kd, effort = task.servo_scales(values)
    servo_fields = ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "jnt_actfrcrange")
    if task.transfer_delta is not None:
        servo_fields += ("body_gravcomp",)
    desired = {}
    with backend.simulation_stream():
        for name in servo_fields + ("dof_frictionloss", "geom_friction"):
            buffer = wp.to_torch(getattr(backend.mjw_model, name))
            if buffer.shape[0] != len(values):
                raise ValueError(f"MJWarp field {name} does not expose a per-world axis")
            field = buffer[:1].expand_as(buffer).clone()
            if name == "actuator_gainprm":
                field[:, position_ids, 0] *= kp
            elif name == "actuator_biasprm":
                field[:, position_ids, 1] *= kp
                field[:, position_ids, 2] *= kd
            elif name == "actuator_forcerange":
                field[:, position_ids] *= effort.unsqueeze(-1)
            elif name == "jnt_actfrcrange":
                field[:, joint_ids] *= effort.unsqueeze(-1)
            elif name == "body_gravcomp":
                if not field.any():
                    raise ValueError("The packaged Newton model has no gravity compensation to vary")
                field *= 1. - values[:, DR_DIM:DR_DIM + 1]
            elif name == "dof_frictionloss":
                field *= values[:, 5:6]
            else:
                field *= values[:, 4:5, None]
            buffer.copy_(field)
            desired[name] = (buffer, field)
        # Newton refreshes joint friction as well as drive gains before integration.
        # Its post-actuator callback restores these per-world values before solving.
        backend.servo_parameters = [desired[key] for key in servo_fields + ("dof_frictionloss",)]
    return desired
