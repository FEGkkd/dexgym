from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class WujiHand21TeleDexterPPOCfg(RslRlOnPolicyRunnerCfg):
    """PPO learner configuration for the standalone Hand21 task."""

    num_steps_per_env = 32
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=3,
        num_mini_batches=2,
        learning_rate=0.0003,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    experiment_name = "dexgym_wuji_hand21_frame_tracking_15mm_tanh_std1_entropy005"
    max_iterations = 50000
    save_interval = 50
    actor.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0)
    actor.distribution_cfg.class_name = "dexgym.algorithms.action_distributions:TanhGaussianDistribution"
    algorithm.entropy_coef = 0.005
