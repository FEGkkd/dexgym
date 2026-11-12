"""Shared SuperMimic learner configuration for DexGym tasks."""

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass
class SuperMimicConfig:
    seed: int = 42
    max_iterations: int = 8000
    buffer_steps: int = 1000
    warmup_steps: int = 64
    utd: float = 0.4
    wm_only_iters: int = 500
    wm_window: int = 8
    wm_failure_frac: float = 0.25
    wm_failure_horizon: int = 16
    final_state_weight: float = 1.0
    contact_weight: float = 0.1
    contact_partial_intervals: bool = False
    contact_depth_weight: float = 0.0
    kinematic_loss_weight: float = 0.0
    wm_batches: int = 10
    wm_batch: int = 2048
    wm_hidden: tuple = (2048, 2048, 2048)
    isolate_wm_init: bool = False
    control_error_features: bool = False
    pi_batch: int = 1024
    horizon: int = 32
    imagination_start_reserve: int | None = None
    actor_bootstrap_weight: float = 1.0
    max_step_reward: float | None = None
    critic_epochs: int = 5
    critic_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 2e-4
    optimizer: str = "adam"
    actor_lr: float | None = None
    lr_decay_iters: int = 250000
    lr_final_ratio: float = 0.03
    action_noise: float = 0.1
    failure_warmup_iters: int = 300
    failure_head: bool = True
    failure_clip: float = 0.1
    amp: bool = True
    compile: bool = True
    compile_policy: bool = True
    profile_training: bool = False
    save_interval: int = 1000
    log_interval: int = 50
    holdout_fraction: float = 0.1

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        if self.optimizer not in ("adam", "fused-adam", "muon"):
            raise ValueError("Unknown SuperMimic optimizer")
        if self.imagination_start_reserve is not None and self.imagination_start_reserve < self.horizon:
            raise ValueError("Imagination start reserve must cover the rollout horizon")
        if any(not isfinite(weight) or weight < 0 for weight in (
            self.contact_depth_weight, self.kinematic_loss_weight
        )):
            raise ValueError("World-model auxiliary loss weights must be finite and non-negative")
        if self.max_step_reward is not None and (
            not isfinite(self.max_step_reward) or self.max_step_reward < 0 or not 0 <= self.gamma < 1
        ):
            raise ValueError("A finite value bound requires a non-negative reward maximum and 0 <= gamma < 1")
        if not 0 <= self.actor_bootstrap_weight <= 1:
            raise ValueError("Actor bootstrap weight must be in [0,1]")
        if not 0 < self.utd <= 1:
            raise ValueError("UTD must be in (0,1]")
        if self.warmup_steps < max(self.horizon, self.wm_window):
            raise ValueError("Insufficient replay warmup")
        if not 0 <= self.holdout_fraction < 1:
            raise ValueError("Invalid holdout fraction")
        if not 0 <= self.wm_failure_frac < 1:
            raise ValueError("Invalid failure mixture")
