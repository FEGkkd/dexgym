"""Compare model and physics responses to identical commands near hand contact."""

import math

import torch

from dexgym.algorithms.supermimic.dex_reference.networks import FailureHead, WorldModel

from .task import HARD_CONTACT_DEPTH_M, JOINTS


def probe_frames(library, clip_index, count=16):
    start = int(library.start[clip_index])
    stop = int(library.end[start])
    points = library.ref_tip_pos[start:stop]
    distance = (points[:, 0, :, None] - points[:, 1, None, :]).norm(dim=-1).amin(dim=(1, 2))
    selected = [start]
    for i in distance.argsort(stable=True).tolist():
        frame = start + i
        if all(abs(frame - other) >= 5 for other in selected):
            selected.append(frame)
            if len(selected) == count + 1:
                break
    return [(frame, float(distance[frame - start])) for frame in selected]


@torch.no_grad()
def diagnose_model(env, actor, checkpoint, clip_index):
    task, lib = env.task, env.task.library
    cfg = checkpoint["config"]
    wm = WorldModel(task.dynamics_dim, task.contact_dim, task.codec, task.action_dim, task.error_floor,
                    tuple(cfg["wm_hidden"]), False).to(env.device)
    failure_head = FailureHead(task.termination_context_dim, task.state_dim).to(env.device)
    wm.load_state_dict(checkpoint["modules"]["wm"])
    failure_head.load_state_dict(checkpoint["modules"]["fail_head"])
    wm.eval()
    failure_head.eval()
    failure_active = cfg["failure_head"] and checkpoint["iteration"] > (
        cfg["wm_only_iters"] + cfg["failure_warmup_iters"]
    )
    policies = (task.baseline_name, "policy")
    records = []
    for frame, distance in probe_frames(lib, clip_index):
        frames = torch.full((2,), frame, dtype=torch.long, device=env.device)
        obs, _ = env.reset(frame=frames)
        x, memory, aux = task.read()
        action = actor(obs["policy"])
        action[0] = task.baseline_action(memory)[0]
        _, u = task.control_target(action, memory, task.encoder_bias)
        sim = task.simulator_context
        predicted, _ = wm(x, u, task.dynamics(x, sim, aux))
        if task.transition_termination:
            features = failure_head.features(x, u)
        else:
            next_memory = task.replay_failure_memory(memory, action)
            context = task.context(predicted, next_memory, aux, sim)
            features = failure_head.features(predicted, context)
        probability = failure_head(features).sigmoid()
        risk = probability.clamp(max=cfg["failure_clip"]) if failure_active else torch.zeros_like(probability)
        terminated_probability = 1 - (1 - task.termination(predicted, memory)) * (1 - risk)
        _, _, _, _, events = env.step(action)
        measured = env.transition["x1"]
        pred_pose = task.geometry.pose(predicted[:, :JOINTS])
        real_pose = task.geometry.pose(measured[:, :JOINTS])
        wrist_mm = (pred_pose[0] - real_pose[0]).norm(dim=-1).mean(-1) * 1000
        tip_mm = (pred_pose[3] - real_pose[3]).norm(dim=-1).mean(-1) * 1000
        depth = env.transition["depth"]
        row = {"frame": frame - int(lib.start[clip_index]), "reference_interhand_tip_gap_mm": 1000 * distance}
        for i, name in enumerate(policies):
            penetration = float(1000 * depth[i])
            at_control_boundary = bool(events["full_step"][i])
            row[name] = {
                "max_real_penetration_mm": penetration if math.isfinite(penetration) else None,
                "hard_contact": bool(depth[i] > HARD_CONTACT_DEPTH_M),
                "physical_terminated": bool(events["safety_failure"][i]),
                "failure_substep": int(events["safety_substep"][i]) or None,
                "termination_probability": float(probability[i]),
                "imagined_termination_probability": float(terminated_probability[i]),
                "wrist_prediction_error_mm": float(wrist_mm[i])
                if at_control_boundary and torch.isfinite(wrist_mm[i]) else None,
                "tip_prediction_error_mm": float(tip_mm[i])
                if at_control_boundary and torch.isfinite(tip_mm[i]) else None,
            }
            if not records:
                row[name]["executed_target_rad"] = u[i].tolist()
        records.append(row)
    result = {
        "threshold": 0.5,
        "hard_contact_threshold_mm": HARD_CONTACT_DEPTH_M * 1000,
        "counts": {name: summarize([row[name] for row in records]) for name in policies},
        "samples": records,
    }
    if task.first_event:
        result["continuation_probe"] = continuation_probe(env, actor, failure_head, clip_index, cfg["gamma"])
        result["reference_perturbation_probe"] = reference_perturbation_probe(env, failure_head, clip_index)
    return result


@torch.no_grad()
def continuation_probe(env, actor, failure_head, clip_index, gamma):
    """Compare safety-only and combined continuation on measured rewards."""
    task, library = env.task, env.task.library
    frame = library.start[clip_index]
    horizon = min(32, int(library.end[frame] - frame))
    obs, _ = env.reset(frame=frame.repeat(2))
    alive = torch.ones(2, dtype=torch.bool, device=env.device)
    survival = torch.ones(2, dtype=torch.float64, device=env.device)
    combined_survival = survival.clone()
    real_return = torch.zeros_like(survival)
    weighted_return = torch.zeros_like(survival)
    combined_return = torch.zeros_like(survival)
    executed = torch.zeros(2, dtype=torch.long, device=env.device)
    for step in range(horizon):
        x, memory, _ = task.read()
        action = actor(obs["policy"])
        action[0] = task.baseline_action(memory)[0]
        _, u = task.control_target(action, memory, task.encoder_bias)
        probability = failure_head(failure_head.features(x, u)).sigmoid().double()
        obs, reward, terminated, truncated, _ = env.step(action)
        discounted = gamma**step * torch.where(alive, reward, 0.0)
        real_return += discounted
        weighted_return += survival * discounted
        combined_return += combined_survival * discounted
        analytic = task.termination(env.state().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0), memory).double()
        survival *= torch.where(alive, 1 - probability, 1.0)
        combined_survival *= torch.where(alive, (1 - probability) * (1 - analytic), 1.0)
        executed += alive
        alive &= ~(terminated | truncated)
        if not alive.any():
            break
        inactive = (~alive).nonzero().flatten()
        if len(inactive):
            obs, _ = env.reset(inactive, frame=library.at(frame.repeat(len(inactive)), step + 1))
    return {
        name: {
            "executed_steps": int(executed[i]),
            "physical_discounted_reward": float(real_return[i]),
            "physical_reward_weighted_by_head": float(weighted_return[i]),
            "physical_reward_weighted_by_combined_continuation": float(combined_return[i]),
        }
        for i, name in enumerate((task.baseline_name, "policy"))
    }


@torch.no_grad()
def reference_perturbation_probe(env, failure_head, clip_index):
    task = env.task
    scales = (0.12, 0.24)
    sigma = torch.tensor(scales, device=env.device)[:, None]
    generator = torch.Generator(device=env.device).manual_seed(2026)
    records = [[], []]
    for frame, _ in probe_frames(task.library, clip_index):
        frames = torch.full((2,), frame, dtype=torch.long, device=env.device)
        for _ in range(4):
            env.reset(frame=frames)
            x, memory, _ = task.read()
            reference = task.baseline_action(memory)
            noise = torch.randn(reference.shape, generator=generator, device=env.device)
            action = (reference + sigma * noise).clamp(-1, 1)
            _, u = task.control_target(action, memory, task.encoder_bias)
            p = failure_head(failure_head.features(x, u)).sigmoid()
            _, _, _, _, events = env.step(action)
            for i, rows in enumerate(records):
                rows.append({
                    "hard_contact": bool(env.transition["depth"][i] > HARD_CONTACT_DEPTH_M),
                    "physical_terminated": bool(events["safety_failure"][i]),
                    "termination_probability": float(p[i]),
                })
    return {
        str(scale): summarize(rows)
        for scale, rows in zip(scales, records, strict=True)
    }


def summarize(rows):
    positives = sum(row["physical_terminated"] for row in rows)
    hard_contacts = sum(row["hard_contact"] for row in rows)
    false_negatives = sum(row["physical_terminated"] and row["termination_probability"] < 0.5 for row in rows)
    false_positives = sum(not row["physical_terminated"] and row["termination_probability"] >= 0.5 for row in rows)
    return {
        "samples": len(rows), "physical_terminations": positives, "hard_contacts": hard_contacts,
        "missed_hard_contacts": sum(row["hard_contact"] and row["termination_probability"] < 0.5 for row in rows),
        "false_negatives": false_negatives, "false_positives": false_positives,
        "false_negative_rate": false_negatives / positives if positives else None,
        "false_positive_rate": false_positives / (len(rows) - positives) if positives < len(rows) else None,
    }
