"""Readable Newton configuration bound to M4.3.1 checkpoints and evaluation."""

import hashlib
from pathlib import Path


def marvin_physics_config(cfg, robot, *, task_id: str) -> dict:
    reset_path = (cfg.reset_state_path if task_id == "marvin_dual_tcp_reach"
                  else cfg.reference_path)
    return {
        "backend": "newton_mjwarp_v1",
        "num_envs": cfg.scene.num_envs,
        "physics_dt": cfg.sim.dt,
        "policy_dt": cfg.sim.dt * cfg.decimation,
        "decimation": cfg.decimation,
        "episode_length_s": cfg.episode_length_s,
        "controller_kp_source": robot.native.controller_kp_source,
        "controller_kd_source": robot.native.controller_kd_source,
        "controller_damping_scale_source": robot.native.controller_damping_scale_source,
        "hands_fixed_home": True,
        "hand_contact_configuration": "validated_disabled",
        "audit_contacts_each_physics_step": cfg.audit_contacts_each_physics_step,
        "reset_asset_sha256": hashlib.sha256(Path(reset_path).read_bytes()).hexdigest(),
        "reset_random_reference_start": (cfg.random_reference_start
                                         if task_id == "marvin_motion_tracking" else False),
        "reset_joint_perturb_rad": (cfg.reset_joint_perturb_rad
                                    if task_id == "marvin_motion_tracking" else 0.0),
        "active_reset_fraction": (cfg.active_reset_fraction
                                   if task_id == "marvin_motion_tracking" else 0.0),
        "early_reset_fraction": (cfg.early_reset_fraction
                                  if task_id == "marvin_motion_tracking" else 0.0),
        "observation_position_noise_rad": (cfg.observation_position_noise_rad
                                            if task_id == "marvin_motion_tracking" else 0.0),
        "observation_velocity_noise_rad_s": (cfg.observation_velocity_noise_rad_s
                                               if task_id == "marvin_motion_tracking" else 0.0),
        "initialize_hands_from_pd_replay": (cfg.initialize_hands_from_pd_replay
                                            if task_id == "marvin_motion_tracking" else False),
    }
