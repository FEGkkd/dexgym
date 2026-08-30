from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class ShadowHandReorientationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "dexteroushands_shadow_hand_reorientation_direct"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.998,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
