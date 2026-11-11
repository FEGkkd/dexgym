"""Compare model and physics responses to identical commands near hand contact."""

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
    failure_head = FailureHead(task.context_dim, task.state_dim).to(env.device)
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
        predicted, contact = wm(x, u, task.dynamics(x, sim, aux))
        next_memory = task.replay_failure_memory(memory, action)
        context = task.context(predicted, next_memory, aux, sim)
        generic = failure_head(failure_head.features(predicted, context)).sigmoid()
        hard_contact = task.imagined_contact_failure(contact.sigmoid())
        risk = generic.clamp(max=cfg["failure_clip"]) if failure_active else torch.zeros_like(generic)
        if hard_contact is not None:
            risk = torch.maximum(risk, hard_contact)
        terminated_probability = 1 - (1 - task.termination(predicted, memory)) * (1 - risk)
        _, _, done, _, _ = env.step(action)
        measured = env.transition["x1"]
        pred_pose = task.geometry.pose(predicted[:, :JOINTS])
        real_pose = task.geometry.pose(measured[:, :JOINTS])
        wrist_mm = (pred_pose[0] - real_pose[0]).norm(dim=-1).mean(-1) * 1000
        tip_mm = (pred_pose[3] - real_pose[3]).norm(dim=-1).mean(-1) * 1000
        depth = env.transition["depth"]
        row = {"frame": frame - int(lib.start[clip_index]), "reference_interhand_tip_gap_mm": 1000 * distance}
        for i, name in enumerate(policies):
            row[name] = {
                "max_real_penetration_mm": float(1000 * depth[i]),
                "hard_contact": bool(depth[i] > HARD_CONTACT_DEPTH_M),
                "physical_terminated": bool(done[i]),
                "generic_failure_probability": float(generic[i]),
                "hard_contact_probability": float(hard_contact[i]) if hard_contact is not None else None,
                "imagined_termination_probability": float(terminated_probability[i]),
                "wrist_prediction_error_mm": float(wrist_mm[i]),
                "tip_prediction_error_mm": float(tip_mm[i]),
            }
            if not records:
                row[name]["executed_target_rad"] = u[i].tolist()
        records.append(row)
    return {
        "threshold": 0.5,
        "label_depth_mm": HARD_CONTACT_DEPTH_M * 1000,
        "counts": {name: summarize([row[name] for row in records]) for name in policies},
        "samples": records,
    }


def summarize(rows):
    positives = sum(row["hard_contact"] for row in rows)
    # Old checkpoints use the generic termination head as the collision proxy.
    probabilities = [
        row["generic_failure_probability"] if row["hard_contact_probability"] is None
        else row["hard_contact_probability"] for row in rows
    ]
    false_negatives = sum(row["hard_contact"] and p < 0.5 for row, p in zip(rows, probabilities, strict=True))
    false_positives = sum(not row["hard_contact"] and p >= 0.5 for row, p in zip(rows, probabilities, strict=True))
    return {
        "samples": len(rows), "hard_contacts": positives,
        "false_negatives": false_negatives, "false_positives": false_positives,
        "false_negative_rate": false_negatives / positives if positives else None,
        "false_positive_rate": false_positives / (len(rows) - positives) if positives < len(rows) else None,
    }
