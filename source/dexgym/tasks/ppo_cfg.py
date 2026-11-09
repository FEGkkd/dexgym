"""Shared official Isaac Lab RSL-RL PPO configuration."""

from copy import deepcopy

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class MatchedBetaDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    """Beta head with initial action std matched to the Gaussian pilots."""

    class_name: str = "dexgym.algorithms.action_distributions:MatchedBetaDistribution"
    init_std: float = 0.3


@configclass
class JointTrackingPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 10
    save_interval = 5
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    experiment_name = "dexgym_joint_tracking"
    actor = RslRlMLPModelCfg(
        hidden_dims=[32, 32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[32, 32], activation="elu", obs_normalization=False)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=3,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RightHandTrackingPPOCfg(RslRlOnPolicyRunnerCfg):
    """Bounded PPO workflow check; training scale is decided after task review."""

    num_steps_per_env = 32
    max_iterations = 10
    save_interval = 5
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    experiment_name = "dexgym_wuji_right_hand_tracking"
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
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class RightHandCleanTrackingPPOCfg(RightHandTrackingPPOCfg):
    """Separate logs for the official Hand 2.1 tracking experiment."""

    experiment_name = "dexgym_wuji_hand21_right_hand_deadzone_tracking"


@configclass
class RightHandCleanObs125PPOCfg(RightHandCleanTrackingPPOCfg):
    """Separate logs for the observation-only ablation."""

    experiment_name = "dexgym_wuji_hand21_right_hand_deadzone_obs125"


@configclass
class RightHandCleanHybrid125PPOCfg(RightHandCleanTrackingPPOCfg):
    """Separate logs for the observation-plus-reward ablation."""

    experiment_name = "dexgym_wuji_hand21_right_hand_deadzone_hybrid125"


@configclass
class RightHandCleanHybrid125Entropy0PPOCfg(RightHandCleanHybrid125PPOCfg):
    """Paired continuation that changes only the PPO entropy coefficient."""

    experiment_name = "dexgym_wuji_hand21_right_hand_deadzone_hybrid125_entropy0"
    algorithm = deepcopy(RightHandCleanHybrid125PPOCfg().algorithm)
    algorithm.entropy_coef = 0.0


@configclass
class RightHandCleanHybrid125TanhGaussianPPOCfg(RightHandCleanHybrid125Entropy0PPOCfg):
    """Entropy-free pilot with a tanh-squashed Gaussian action head."""

    experiment_name = "dexgym_wuji_hand21_right_hand_deadzone_hybrid125_tanh_gaussian"
    actor = deepcopy(RightHandCleanHybrid125Entropy0PPOCfg().actor)
    actor.distribution_cfg.class_name = "dexgym.algorithms.action_distributions:TanhGaussianDistribution"


@configclass
class RightHandCleanHybrid125BetaPPOCfg(RightHandCleanHybrid125Entropy0PPOCfg):
    """Entropy-free pilot with a spread-matched RSL-RL bounded Beta action head."""

    experiment_name = "dexgym_wuji_hand21_right_hand_deadzone_hybrid125_beta"
    actor = deepcopy(RightHandCleanHybrid125Entropy0PPOCfg().actor)
    actor.distribution_cfg = MatchedBetaDistributionCfg()


@configclass
class WujiHand21TeleDexterPPOCfg(RightHandTrackingPPOCfg):
    """Current Hand 2.1 PPO: paper-style Gaussian std and entropy bonus."""

    experiment_name = "dexgym_wuji_hand21_teledexter_20mm"
    max_iterations = 1000
    save_interval = 50
    actor = deepcopy(RightHandTrackingPPOCfg().actor)
    actor.distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0)
    algorithm = deepcopy(RightHandTrackingPPOCfg().algorithm)
    algorithm.entropy_coef = 0.005
