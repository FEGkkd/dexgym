"""Compare actor-in-model closed loop and action-conditioned model against Newton."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import launch_simulation

from dexgym.algorithms.supermimic.closed_loop import model_actor_rollout
from dexgym.algorithms.supermimic.networks import DeltaWorldModel, GoalActor
from dexgym.algorithms.supermimic.rewards import so3_geodesic
from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
from dexgym.tasks.marvin_supermimic.robot_adapter import DEFAULT_MJCF


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metrics(predicted, physical, fk, action_difference=None):
    pred_pos, pred_rot = fk(predicted[:, :14])
    true_pos, true_rot = fk(physical[:, :14])
    tcp = (pred_pos[:, :, 2] - true_pos[:, :, 2]).norm(dim=-1)
    rot = so3_geodesic(pred_rot[:, :, 2], true_rot[:, :, 2])
    result = {
        "joint_position_rmse_rad": float((predicted[:, :14] - physical[:, :14]).square().mean().sqrt()),
        "joint_velocity_rmse_rad_s": float((predicted[:, 14:28] - physical[:, 14:28]).square().mean().sqrt()),
        "tcp_position_mean_mm": float(tcp.mean() * 1000),
        "tcp_orientation_mean_deg": float(rot.mean() * 180 / math.pi),
    }
    if action_difference is not None:
        result["action_abs_difference_mean"] = float(action_difference.abs().mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=140116)
    parser.add_argument("--horizon", type=int, default=32)
    args = parser.parse_args()
    record = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if record.get("schema_version") != "m42_v1":
        raise ValueError("Dynamics validation requires an M4.2 checkpoint")
    metadata, configuration = record["metadata"], record["config"]
    if metadata["reference_hashes"].get(str(args.reference)) != _hash(args.reference):
        raise ValueError("Checkpoint reference hash differs")
    if metadata["robot_asset_hashes"]["mjcf"] != _hash(DEFAULT_MJCF):
        raise ValueError("Checkpoint robot asset hash differs")
    actor = GoalActor(goal_latent_tanh=True,
                      normalization_mode=configuration["normalization_mode"]).to("cuda").eval()
    legacy_head = any(name.startswith("termination_head.")
                      for name in record["models"]["world_model"])
    model = DeltaWorldModel(hidden=configuration["world_hidden"],
                            legacy_termination_head=legacy_head).to("cuda").eval()
    actor.load_state_dict(record["models"]["actor"], strict=True)
    model.load_state_dict(record["models"]["world_model"], strict=True)
    cfg = MarvinWaveDirectCfg()
    cfg.reference_path = str(args.reference)
    cfg.action_schema = metadata["action_schema"]
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.random_reference_start = False
    cfg.reset_joint_perturb_rad = 0.0
    cfg.initialize_hands_from_pd_replay = cfg.action_schema == "legacy_wave_v1"
    cfg.reward_profile = configuration["reward_profile"]
    cfg.episode_length_s = 28.0
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-SuperMimic-Wave-Direct-v0", cfg=cfg)
        try:
            observation, _ = env.reset()
            native = env.unwrapped
            initial_state = native.dynamics_adapter.physical_state(native).detach().clone()
            initial_history = observation["policy"][:, :560].reshape(1, 10, 56).detach().clone()
            initial_frame = native.motion.frame.detach().clone()
            real_states, real_actions, rewards = [], [], []
            real_terminated = False
            for _ in range(min(args.horizon, native.motion.length - 1)):
                with torch.no_grad():
                    action = actor(observation["policy"][:, :560].reshape(1, 10, 56),
                                   observation["policy"][:, 560:].reshape(1, 10, 90))
                    observation, reward, terminated, truncated, _ = env.step(action)
                    real_states.append(native.last_physical_arm_state.detach().clone())
                    real_actions.append(action.detach().clone())
                    rewards.append(float(reward.mean()))
                    real_terminated = bool(terminated.any() or truncated.any())
                    if real_terminated:
                        break
            physical = torch.stack(real_states)
            actions = torch.stack(real_actions)
            open_loop = model_actor_rollout(
                actor, model, native.action_map, native.motion, initial_state, initial_history,
                initial_frame, horizon=len(physical), physical_actions=actions)
            closed_loop = model_actor_rollout(
                actor, model, native.action_map, native.motion, initial_state, initial_history,
                initial_frame, horizon=len(physical))
            output = {
                "reference_sha256": _hash(args.reference), "checkpoint": str(args.checkpoint),
                "action_schema": metadata["action_schema"], "seed": args.seed,
                "requested_horizon": args.horizon, "real_steps": len(physical),
                "real_terminated": real_terminated, "real_reward_mean": sum(rewards) / len(rewards),
                "action_conditioned_open_loop": {}, "policy_closed_loop": {},
            }
            for horizon in (1, 8, 32):
                if horizon <= len(physical):
                    output["action_conditioned_open_loop"][str(horizon)] = _metrics(
                        open_loop["states"][horizon - 1], physical[horizon - 1], native.fk)
                    output["policy_closed_loop"][str(horizon)] = _metrics(
                        closed_loop["states"][horizon - 1], physical[horizon - 1], native.fk,
                        closed_loop["actions"][:horizon] - actions[:horizon])
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output / "report.json").write_text(json.dumps(output, indent=2))
            np.savez_compressed(args.output / "trace.npz",
                                physical_states=physical.cpu().numpy(),
                                physical_actions=actions.cpu().numpy(),
                                action_conditioned_states=open_loop["states"].cpu().numpy(),
                                policy_closed_loop_states=closed_loop["states"].cpu().numpy(),
                                policy_closed_loop_actions=closed_loop["actions"].cpu().numpy())
            print(json.dumps(output), flush=True)
        finally:
            env.close()


if __name__ == "__main__":
    main()
