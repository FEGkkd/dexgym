"""PPO configuration for the Wuji Hand DirectTask."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class WujiHandReorientationActorCfg(RslRlMLPModelCfg):
    hidden_dims = [512, 256, 128]
    activation = "elu"
    obs_normalization = True
    distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5)


@configclass
class WujiHandReorientationCriticCfg(RslRlMLPModelCfg):
    hidden_dims = [512, 512, 256, 128]
    activation = "elu"
    obs_normalization = True


@configclass
class WujiHandReorientationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.002
    num_learning_epochs = 5
    num_mini_batches = 32
    learning_rate = 1.0e-4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class WujiHandReorientationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Use the asymmetric actor-critic contract for the official native USD task.
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    num_steps_per_env = 40
    max_iterations = 5000
    save_interval = 50
    experiment_name = "dexteroushands_wuji_hand_reorientation_direct_isaac"
    actor = WujiHandReorientationActorCfg()
    critic = WujiHandReorientationCriticCfg()
    algorithm = WujiHandReorientationAlgorithmCfg()
