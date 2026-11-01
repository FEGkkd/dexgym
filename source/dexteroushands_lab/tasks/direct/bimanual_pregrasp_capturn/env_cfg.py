"""Phase 1B single-task PGS Mimic PPO environment configuration."""

from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from dexteroushands_lab.tasks.direct.bimanual_bottle.bimanual_bottle_env_cfg import BimanualBottleDirectEnvCfg


@configclass
class PregraspCapTurn30DirectEnvCfg(BimanualBottleDirectEnvCfg):
    task_kind = "pregrasp_cap_turn_30"
    unified_reward_profile = "baseline"
    no_left_bottle_contact_counterfactual = False
    thread_backend = "mimic"
    stable_reset_probability = 0.0
    pregrasp_bank_path = "docs/assets/phase1b_pregrasp_bank.json"
    pregrasp_reset_levels = ("A",)
    pregrasp_split_key = "right_natural_source_seed"
    pregrasp_split_manifest_path = "docs/assets/phase1b_pregrasp_split.json"
    right_cap_contact_gate = "manifold_sidewall"
    right_cap_proximity_mode = "sidewall_tripod"
    right_contact_balance_weight = 0.2
    wrist_translation_scale = 0.0005
    wrist_rotation_scale = 0.02
    finger_increment_scale = 0.01
    sim = SimulationCfg(
        dt=0.002, render_interval=10,
        physics=PhysxCfg(solver_type=0, min_position_iteration_count=8,
                         max_position_iteration_count=8,
                         enable_enhanced_determinism=True),
    )
