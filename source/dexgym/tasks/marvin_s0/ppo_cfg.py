"""RSL-RL PPO baseline for goal-conditioned Marvin S0."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.utils import check_rsl_rl_version, handle_deprecated_rsl_rl_cfg


@configclass
class MarvinS0PPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30
    save_interval = 10
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    experiment_name = "dexgym_marvin_bimanual_s0_v0"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.4),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


def compatible_agent_cfg(cfg: MarvinS0PPOCfg) -> MarvinS0PPOCfg:
    """Apply the official play entrypoint's RSL-RL version migration."""
    return handle_deprecated_rsl_rl_cfg(cfg, check_rsl_rl_version())
