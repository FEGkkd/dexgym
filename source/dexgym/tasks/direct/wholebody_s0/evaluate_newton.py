"""Held-out nominal replay through the same Isaac Lab Newton task as training."""

import argparse
from collections import Counter
import json
from pathlib import Path

import mujoco
import torch

from ..marvin54_s0.physics import COLLISION, FAILURES, SUBSTEPS, open_physics
from .bag import digest
from .evaluate_residual import checkpoint_geometry, load_actor, stats
from .prepare import write_json
from .residual import HARD_DEPTH, HUMAN_INTENT_WEIGHTS, N, ReferenceBank, ResidualTask, human_intent_score
from .robot import NEWTON_USD
from .residual_env import BAD_ACTION, ResidualEnv


@torch.inference_mode()
def run_case(env, checkpoint, *, compare_earlier_library=False, intent_geometry=None, intent_goals=None):
    task, backend = env.task, env.backend
    bank = task.bank
    actor = load_actor(checkpoint, task, compare_earlier_library=compare_earlier_library)
    obs, _ = env.reset(frame=bank.start, exact=True)
    with backend.simulation_stream() as caller:
        peak = backend.contact_depth().clone()
        peak.record_stream(caller)
    planned = bank.end[bank.start] - bank.start
    bits = (peak > HARD_DEPTH).int() * COLLISION
    failure_frame = torch.zeros_like(planned)
    substeps, contacts = torch.zeros_like(planned), torch.zeros_like(planned)
    states, frames, masks = [], [], []
    for i in range(1, int(planned.max()) + 1):
        active = (bits == 0) & (i <= planned)
        frame = task.current_goal(env.memory)
        action = actor(obs["policy"]) if actor is not None else torch.zeros_like(env.memory["action"])
        obs, _, _, _, _ = env.step(action)
        info = env.transition
        step_bits = info["failure_bits"]
        new_failure = active & (step_bits != 0)
        bits = torch.where(new_failure, step_bits, bits)
        failure_frame = torch.where(new_failure, i, failure_frame)
        peak = torch.where(active, torch.maximum(peak, info["depth"]), peak)
        step_substeps = info["safety_substep"]
        substeps += active * torch.where(step_substeps > 0, step_substeps, SUBSTEPS)
        contacts += active * info["deep_substeps"]
        states.append(info["x1"][:, :N])
        frames.append(frame)
        masks.append(active & info["full_step"])
    states, frames, masks = (torch.stack(seq, 1) for seq in (states, frames, masks))
    complete = bits == 0
    keys = {**FAILURES, BAD_ACTION: "invalid_action"}
    failures = Counter()
    errors, passed, clips = [], [], []
    for j, entry in enumerate(bank.entries):
        reasons = [value for flag, value in keys.items() if int(bits[j]) & flag]
        failures.update(reasons)
        mask = masks[j]
        clips.append({"id": entry["id"], "episode": entry["source_episode"],
                      "planned_intervals": int(planned[j]), "measured_intervals": int(mask.sum()),
                      "complete": bool(complete[j]), "failure_frame": int(failure_frame[j]) if reasons else None,
                      "failures": reasons, "peak_depth_mm": 1000. * float(peak[j]),
                      "substeps": int(substeps[j]), "contact_substeps": int(contacts[j])})
        if not mask.any():
            continue
        q = states[j, mask]
        goal = frames[j, mask]
        e = task.error_terms(task.measured_pose(q), goal)
        def millimeters(key):
            return 1000. * e[key].sqrt().cpu().numpy()
        def degrees(key):
            return (2. * (e[key].sqrt() / 2.).clamp(0., 1.).asin()).rad2deg().cpu().numpy()
        errors.append({"wrist_mm": millimeters("wrist"), "wrist_deg": degrees("rotation"),
                       "local_tip_mm": millimeters("local")[..., -1], "tip_mm": millimeters("tip"),
                       "relative_mm": millimeters("relative"), "relative_deg": degrees("relative_rotation")})
        if task.retarget_pose:
            errors[-1].update(wrist_local_tip_mm=millimeters("wrist_local_tip"),
                              wrist_local_tip_deg=degrees("wrist_local_rotation"))
        if intent_goals is not None:
            score, intent = human_intent_score(intent_geometry.pose(q), (value[goal] for value in intent_goals))
            errors[-1].update({
                "intent_loss": (sum(HUMAN_INTENT_WEIGHTS.values()) - score).clamp_min(0.).cpu().numpy(),
                "intent_wrist_mm": 1000. * intent["intent_wrist"].sqrt().cpu().numpy(),
                "intent_local_tip_mm": 1000. * intent["intent_local_tip"].sqrt().cpu().numpy(),
                "intent_relative_mm": 1000. * intent["intent_relative"].sqrt().cpu().numpy(),
            })
        if complete[j]:
            passed.append(errors[-1])
    return {"complete": int(complete.sum()), "clips": len(clips), "failures": dict(failures),
            "coverage": float(masks.sum() / planned.sum()), "peak_depth_mm": 1000. * float(peak.max()),
            "contact_substep_rate": float(contacts.sum() / substeps.sum().clamp_min(1)),
            "all_measured": stats(errors), "completed_only": stats(passed), "clip_results": clips}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--checkpoint", type=Path, action="append", default=[])
    p.add_argument("--compare-earlier-library", action="store_true")
    args = p.parse_args()
    if not (args.baseline or args.checkpoint):
        p.error("Choose --baseline or --checkpoint")
    torch.set_num_threads(1)
    source = json.loads(args.manifest.read_text())["provenance"]["robot_model"]
    if digest(source["path"]) != source["sha256"]:
        raise ValueError("Reference asset checksum mismatch")
    model = mujoco.MjModel.from_xml_path(source["path"])
    candidates = ([None] if args.baseline else []) + [p.resolve(strict=True) for p in args.checkpoint]
    metadata = {ckpt: json.loads((ckpt.parent / "provenance.json").read_text()) if ckpt else {} for ckpt in candidates}
    intent_info = [m for m in metadata.values() if m.get("reward_profile") == "human_intent"]
    if intent_info:
        goal = intent_info[0]["intent"]
        if any(m["intent"] != goal for m in intent_info):
            raise ValueError("Newton cases must share the same human-intent geometry and goal version")
    intent_model = checkpoint_geometry(intent_info[0]) if intent_info else None
    bank = ReferenceBank(args.manifest, args.split, model, "cuda:0", intent_model=intent_model)
    with open_physics("newton", model, bank.names, len(bank.entries), robot_usd=NEWTON_USD) as backend:
        env = ResidualEnv(backend, bank, ResidualTask, robust=False, seed=0)
        contract = env.contract()
        results, tasks = [], {}
        for checkpoint in candidates:
            meta = metadata[checkpoint]
            profile = meta.get("reward_profile", "v1")
            if profile not in tasks:
                hand_model = intent_model if profile == "human_intent" else checkpoint_geometry(meta)
                tasks[profile] = ResidualTask(env, model, bank, domain=env.domain, reward_profile=profile,
                                              hand_model=hand_model)
        intent = tasks.get("human_intent")
        for checkpoint in candidates:
            env.task = tasks[metadata[checkpoint].get("reward_profile", "v1")]
            row = run_case(env, checkpoint, compare_earlier_library=args.compare_earlier_library,
                           intent_geometry=intent.reward_geometry if intent else None,
                           intent_goals=intent.intent_geometry if intent else None)
            row.update(checkpoint=str(checkpoint) if checkpoint else None,
                       checkpoint_sha256=digest(checkpoint) if checkpoint else None,
                       split=args.split, model_sha256=source["sha256"], reference_manifest_sha256=bank.manifest_sha256)
            results.append(row)
            print(f"{row['checkpoint'] or 'zero_residual'}: {row['complete']}/{row['clips']} complete", flush=True)
        write_json(args.output, {"newton": contract, "cases": results})


if __name__ == "__main__":
    main()
