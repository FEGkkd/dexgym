"""Held-out action-conditioned SE(3) model rollout diagnostics from real Replay."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.algorithms.supermimic.core.models import NetworkFactory
from dexgym.algorithms.supermimic.core.replay_buffer import SchemaReplay
from dexgym.algorithms.supermimic.core.specs import NetworkSpec
from dexgym.algorithms.supermimic.core.transition import TaskState
from dexgym.algorithms.supermimic.rewards import so3_geodesic
from dexgym.tasks.marvin_supermimic.core_plugins import MarvinCoreRobot
from dexgym.tasks.marvin_supermimic.se3_reach_hold import MarvinBimanualSE3ReachHold


@torch.no_grad()
def closed_loop_actor_model_vs_newton(model, actor, robot, task, transitions):
    """Compare independent Actor feedback from the same real Newton starts."""
    device = next(model.parameters()).device
    result = {"evaluation_kind": "policy_closed_loop_model_vs_newton",
              "start_indices": [0, 32, 64, 96, 128, 160]}
    for horizon in (1, 8, 32):
        position_errors, rotation_errors, action_errors = [], [], []
        for start_index in result["start_indices"]:
            if start_index + horizon > len(transitions):
                continue
            segment = transitions[start_index:start_index + horizon]
            validity = (~torch.stack([item.done for item in segment]).any(0)).to(device)
            if not bool(validity.any()):
                continue
            first = segment[0]
            state = first.state.to(device)
            history = first.history.to(device)
            context = first.dynamics_context.to(device)
            task_state = TaskState(*(getattr(first.task_state, key).to(device)
                                     for key in ("task_id", "motion_id", "frame", "data")))
            for step, real in enumerate(segment):
                action = actor(history, task.goal(state, task_state))
                real_action = real.action.to(device)
                if step == 0 and not torch.allclose(action[validity], real_action[validity],
                                                    atol=1e-5, rtol=1e-5):
                    raise ValueError("Actor checkpoint differs from independent Newton trace")
                action_errors.append((action[validity].tanh()
                                      - real_action[validity].tanh()).abs())
                next_state = model(state, action, context, robot)
                next_task_state = task.advance(task_state)
                observation = task.observation(next_state, action, next_task_state)
                history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
                state, task_state = next_state, next_task_state
            truth = segment[-1].next_state.to(device)
            predicted_position, predicted_rotation = robot.fk(state[validity, :14])
            true_position, true_rotation = robot.fk(truth[validity, :14])
            position_errors.append(torch.linalg.vector_norm(
                predicted_position[:, :, 2] - true_position[:, :, 2], dim=-1))
            rotation_errors.append(so3_geodesic(
                predicted_rotation[:, :, 2], true_rotation[:, :, 2]))
        if not position_errors:
            result[str(horizon)] = {"available": False}
            continue
        position = torch.cat(position_errors)
        rotation = torch.cat(rotation_errors)
        result[str(horizon)] = {
            "samples": len(position),
            "tcp_position_mean_m": position.mean(0).tolist(),
            "tcp_position_p95_m": torch.quantile(position, 0.95, dim=0).tolist(),
            "tcp_so3_mean_rad": rotation.mean(0).tolist(),
            "tcp_so3_p95_rad": torch.quantile(rotation, 0.95, dim=0).tolist(),
            "bounded_actor_action_error_mean": float(torch.cat(action_errors).mean()),
        }
    return result


@torch.no_grad()
def evaluate(checkpoint_path: Path, *, seed: int = 71, batch: int = 64,
             physics_trace_path: Path | None = None, closed_loop: bool = False):
    if closed_loop and physics_trace_path is None:
        raise ValueError("Policy closed-loop comparison requires an independent Newton trace")
    record = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if physics_trace_path is None and record.get("replay") is None:
        raise ValueError("Model diagnostic needs a checkpoint with real Replay")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    robot = MarvinCoreRobot(SimpleNamespace(joint_names_arm=ARM_JOINT_NAMES), device,
                            action_schema="arm_joint_delta_v1")
    task = MarvinBimanualSE3ReachHold(robot)
    network = NetworkSpec(**record["contract_manifest"]["networks"]["world_model"]["network_spec"])
    modules = NetworkFactory.build(robot.spec, task.spec, network, robot_adapter=robot)
    model = modules.world_model.to(device)
    model.load_state_dict(record["modules"]["world_model"], strict=True)
    model.eval()
    if closed_loop:
        actor = modules.actor.to(device)
        actor.load_state_dict(record["modules"]["actor"], strict=True)
        actor.eval()
    if physics_trace_path is None:
        saved_replay = record["replay"]
        replay = SchemaReplay(saved_replay["capacity"], saved_replay["num_envs"],
                              robot.spec, task.spec, device)
        replay.load_state_dict(saved_replay, fresh_physics=False)
        collection = record.get("physics_config", {}).get("collection_contract", {})
        if collection.get("version") == "dual_source_geometry_trust_v2":
            noisy_start = collection["train_env_count"]
            deterministic_start = noisy_start + collection["validation_exploration_env_count"]
            selections = (("heldout_exploratory_actor_replay", (noisy_start, deterministic_start), 3),
                          ("heldout_deterministic_actor_replay", (deterministic_start, replay.num_envs), 0))
        else:
            selections = (("heldout_actor_replay", (int(0.75 * replay.num_envs), replay.num_envs), 0),)
    else:
        trace = torch.load(physics_trace_path, map_location="cpu", weights_only=False)
        if trace.get("evaluation_kind") != "independent_physics_evaluation":
            raise ValueError("Model policy-distribution validation needs an independent Newton trace")
        transitions = trace["transitions"]
        if not transitions:
            raise ValueError("Independent Newton trace has no transitions")
        replay = SchemaReplay(len(transitions), len(transitions[0].state),
                              robot.spec, task.spec, device)
        for transition in transitions:
            transition.validate(robot.spec, task.spec, num_envs=replay.num_envs)
            state, action, command = (transition.state.to(device),
                                      transition.action.to(device),
                                      transition.executed_command.to(device))
            if not torch.allclose(robot.executed_command(state, action), command,
                                  atol=1e-6, rtol=1e-6):
                raise ValueError("Independent Newton trace action contract differs")
            replay.append(transition)
        selections = (("independent_deterministic_actor_newton", (0, replay.num_envs), 0),)
    torch.manual_seed(seed)
    result = {"checkpoint": str(checkpoint_path),
              "physics_trace": str(physics_trace_path) if physics_trace_path else None,
              "evaluation_kind": "action_conditioned_open_loop", "distributions": {}}
    for source, environment_range, source_id in selections:
        distribution = {"source_id": source_id, "environment_range": environment_range}
        result["distributions"][source] = distribution
        for horizon in (1, 8, 32):
            try:
                window = replay.sample_windows(batch, horizon, env_range=environment_range,
                                               source_id=source_id, require_full=True)
            except ValueError as error:
                distribution[str(horizon)] = {"available": False, "reason": str(error)}
                continue
            state = window.state[:, 0]
            for step in range(horizon):
                state = model(state, window.action[:, step],
                              window.dynamics_context[:, step], robot)
            truth = window.next_state[:, -1]
            pred_position, pred_rotation = robot.fk(state[:, :14])
            true_position, true_rotation = robot.fk(truth[:, :14])
            pos_error = torch.linalg.vector_norm(
                pred_position[:, :, 2] - true_position[:, :, 2], dim=-1)
            rot_error = so3_geodesic(pred_rotation[:, :, 2], true_rotation[:, :, 2])
            distribution[str(horizon)] = {
                "samples": batch,
                "joint_q_rmse_rad": float((state[:, :14] - truth[:, :14]).square().mean().sqrt()),
                "joint_qd_rmse_rad_s": float((state[:, 14:28] - truth[:, 14:28]).square().mean().sqrt()),
                "tcp_position_mean_m": pos_error.mean(0).tolist(),
                "tcp_position_p95_m": torch.quantile(pos_error, 0.95, dim=0).tolist(),
                "tcp_so3_mean_rad": rot_error.mean(0).tolist(),
                "tcp_so3_p95_rad": torch.quantile(rot_error, 0.95, dim=0).tolist(),
            }
    if closed_loop:
        result["closed_loop"] = closed_loop_actor_model_vs_newton(
            model, actor, robot, task, transitions)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--physics-trace", type=Path)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, physics_trace_path=args.physics_trace,
                      closed_loop=args.closed_loop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
