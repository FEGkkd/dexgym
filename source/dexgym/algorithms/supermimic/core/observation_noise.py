"""Task-schema observation noise for Actor sensing only, never dynamics targets."""

from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class ObservationNoiseSpec:
    distribution: str
    observation_schema: str
    scale: tuple[float, ...]
    units: tuple[str, ...] = ()
    version: str = "actor_observation_noise_v1"

    def __post_init__(self):
        if self.version != "actor_observation_noise_v1":
            raise ValueError("Unknown observation noise contract version")
        if self.distribution not in ("disabled", "uniform", "gaussian"):
            raise ValueError("Unknown observation noise distribution")
        if not self.observation_schema or not self.scale:
            raise ValueError("Observation schema and feature scales are required")
        if any(not math.isfinite(value) or value < 0 for value in self.scale):
            raise ValueError("Observation noise scales must be finite and nonnegative")
        if self.distribution != "disabled" and len(self.units) != len(self.scale):
            raise ValueError("Noisy observations require one unit per feature")

    @classmethod
    def disabled(cls, observation_schema: str, dimension: int):
        return cls("disabled", observation_schema, (0.0,) * dimension)

    def manifest(self):
        return asdict(self)


class ObservationNoiseModel:
    def __init__(self, spec: ObservationNoiseSpec, *, seed: int,
                 noisy_evaluation: bool = False):
        self.spec, self.seed, self.noisy_evaluation = spec, int(seed), bool(noisy_evaluation)
        self._generators = {}

    def validate(self, task_spec):
        schema = task_spec.observation_schema or task_spec.goal_schema
        if self.spec.observation_schema != schema or len(self.spec.scale) != task_spec.observation_dim:
            raise ValueError("Observation noise schema or dimension differs from TaskSpec")

    def _generator(self, device):
        key = str(device)
        if key not in self._generators:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            self._generators[key] = generator
        return self._generators[key]

    def __call__(self, clean: torch.Tensor, *, mode: str = "train") -> torch.Tensor:
        if mode not in ("train", "eval"):
            raise ValueError("Observation noise mode must be train or eval")
        if clean.shape[-1] != len(self.spec.scale) or not bool(torch.isfinite(clean).all()):
            raise ValueError("Clean observation schema or finite values differ")
        if self.spec.distribution == "disabled" or (mode == "eval" and not self.noisy_evaluation):
            return clean
        shape = clean.shape
        generator = self._generator(clean.device)
        if self.spec.distribution == "uniform":
            sample = 2 * torch.rand(shape, device=clean.device, dtype=clean.dtype,
                                    generator=generator) - 1
        else:
            sample = torch.randn(shape, device=clean.device, dtype=clean.dtype,
                                 generator=generator)
        scale = clean.new_tensor(self.spec.scale)
        return clean + sample * scale

    def state_dict(self):
        return {"spec": self.spec.manifest(), "seed": self.seed,
                "noisy_evaluation": self.noisy_evaluation,
                "generators": {key: value.get_state() for key, value in self._generators.items()}}

    def load_state_dict(self, record):
        if record["spec"] != self.spec.manifest() or int(record["seed"]) != self.seed:
            raise ValueError("Observation noise checkpoint contract differs")
        self.noisy_evaluation = bool(record["noisy_evaluation"])
        self._generators.clear()
        for key, state in record["generators"].items():
            generator = torch.Generator(device=key)
            generator.set_state(state)
            self._generators[key] = generator
