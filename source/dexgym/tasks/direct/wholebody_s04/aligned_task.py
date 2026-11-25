"""Keep the calibrated drive state explicit in SuperMimic's imagined rollouts."""

import numpy as np
import torch

from dexgym.tasks.direct.marvin54_s0.physics import SUBSTEPS
from dexgym.tasks.direct.wholebody_s04.task import Env, Task


class ServoTask(Task):
    """The actor still observes and outputs the standard S04 contract."""

    def __init__(self, env, backend, profile):
        super().__init__(env)
        self.arm_columns = backend.arm
        step = backend.model.opt.timestep
        powers = np.exp(-1000 * step * np.arange(1, SUBSTEPS + 1) / profile["target_filter_ms"])
        self.decay = float(powers[-1])
        self.mean_decay = float(powers.mean())

    def memory(self):
        row = super().memory()
        row["servo"] = self.env.direct.backend.filtered_target.clone()
        return row

    def encode_memory(self, memory):
        return torch.cat((super().encode_memory(memory), memory["servo"]), -1)

    def decode_memory(self, encoded):
        row = super().decode_memory(encoded[..., :-len(self.arm_columns)])
        row["servo"] = encoded[..., -len(self.arm_columns):]
        return row

    def applied_target(self, previous, command):
        applied = command.clone()
        applied[:, self.arm_columns] = torch.lerp(command[:, self.arm_columns], previous, self.mean_decay)
        return applied

    def control_target(self, action, memory, bias=None):
        command, _ = super().control_target(action, memory, bias)
        return command, self.applied_target(memory["servo"], command)

    def advance_memory(self, x, memory, action, command, geometry, next_goal, soft=True):
        row = super().advance_memory(x, memory, action, command, geometry, next_goal, soft=soft)
        row["servo"] = torch.lerp(command[:, self.arm_columns], memory["servo"], self.decay)
        return row


class ServoEnv(Env):
    def __init__(self, backend, bank, model, profile, seed, *, stationary_reset_fraction):
        super().__init__(backend, bank, model, seed, evaluation=True, stationary_reset_fraction=stationary_reset_fraction)
        self.task = ServoTask(self, backend, profile)

    def step(self, action):
        previous = self.direct.backend.filtered_target.clone()
        result = super().step(action)
        transition = self.transition
        transition["u"] = self.task.applied_target(previous, transition["c"])
        return result
