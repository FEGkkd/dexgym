"""Conservative Phase 1 RSL-RL PPO baseline; training metrics need GPU validation."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BimanualActorCfg(RslRlMLPModelCfg):
    hidden_dims = [512, 256, 128]
    activation = "elu"
    obs_normalization = True
    distribution_cfg = RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.25)


@configclass
class BimanualCriticCfg(RslRlMLPModelCfg):
    hidden_dims = [512, 512, 256]
    activation = "elu"
    obs_normalization = True


@configclass
class BimanualAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "dexteroushands_lab.tasks.direct.bimanual_bottle.agents.ppo_diagnostics:BimanualDiagnosticPPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.001
    num_learning_epochs = 5
    num_mini_batches = 8
    learning_rate = 1.0e-4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class BimanualBottlePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 25
    experiment_name = "dexteroushands_bimanual_bottle_phase1"
    actor = BimanualActorCfg()
    critic = BimanualCriticCfg()
    algorithm = BimanualAlgorithmCfg()
