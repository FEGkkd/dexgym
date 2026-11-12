"""Pinned Dex-Supermimic fixed-variant training recipe (2becb738)."""

from dataclasses import asdict, dataclass


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
    wm_batches: int = 10
    wm_batch: int = 2048
    wm_hidden: tuple = (2048, 2048, 2048)
    isolate_wm_init: bool = False
    control_error_features: bool = False
    pi_batch: int = 1024
    horizon: int = 32
    actor_bootstrap_weight: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 2e-4
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
    save_interval: int = 1000
    log_interval: int = 50
    holdout_fraction: float = 0.1

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
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
