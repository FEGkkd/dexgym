"""Replay selected GRAB joint references through the stable full-contact drive."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from dexgym.data.quality_filter import replay_contract_sha256, sha256_file
from dexgym.tasks import RIGHT_HAND_ID
from dexgym.tasks.direct.right_hand_tracking.drive_experiment_cfg import make_drive_experiment_cfg
from isaaclab.app import launch_simulation
from isaaclab.cli import load_external_tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-path", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--motion", action="append", required=True, help="Source path such as s8/flashlight_on_1.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.library_path.expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    selected = set(args.motion)
    records = [entry for entry in manifest["motions"] if entry["source"] in selected]
    if len(records) != len(selected) or any(entry["split"] != args.split for entry in records):
        parser.error("All selected motions must exist in the requested split")
    npz_hashes = {entry["source"]: sha256_file(manifest_path.parent / entry["file"]) for entry in records}
    controller_sha256 = replay_contract_sha256(Path(__file__).resolve().parents[1])
    cfg = make_drive_experiment_cfg(physics_dt=1 / 500, num_envs=len(records), hand_contacts=True, profile="hand_x2")
    cfg.library_path = str(manifest_path)
    cfg.split = args.split
    cfg.decimation = 1
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    load_external_tasks()
    with launch_simulation(cfg):
        raw_env = gym.make(RIGHT_HAND_ID, cfg=cfg)
        try:
            task = raw_env.unwrapped
            raw_env.reset(seed=20261108)
            robot = task.robot
            scene = task.scene
            sim = task.sim
            motion_ids = torch.tensor(
                [
                    next(i for i, entry in enumerate(task.bank.records) if entry["source"] == record["source"])
                    for record in records
                ],
                dtype=torch.long,
                device=task.device,
            )
            lengths = task.bank.lengths[motion_ids]
            frames = torch.zeros(len(records), dtype=torch.long, device=task.device)
            start_q = task.bank.joints(motion_ids, frames)
            full_start = task.home_q[None].repeat(len(records), 1)
            full_start[:, task.right_ids] = start_q
            robot.write_joint_position_to_sim_index(position=full_start)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(full_start))
            scene.write_data_to_sim()
            sim.forward()
            scene.update(0.0)

            def step_to(right_q: torch.Tensor) -> None:
                target = task.home_q[None].repeat(len(records), 1)
                target[:, task.right_ids] = right_q
                robot.actuators.target_command.set_position_index(value=target, full_data=True)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(cfg.sim.dt)

            with torch.inference_mode():
                for _ in range(round(1.5 / cfg.sim.dt)):
                    step_to(start_q)
                errors = [[] for _ in records]
                prev_frame = -1
                total_steps = int(np.ceil(int(lengths.max()) / manifest["fps"] / cfg.sim.dt))
                for step in range(total_steps):
                    current_frame = min(int(step * cfg.sim.dt * manifest["fps"]), int(lengths.max()) - 1)
                    clamped = torch.minimum(torch.full_like(frames, current_frame), lengths - 1)
                    step_to(task.bank.joints(motion_ids, clamped))
                    next_frame = int((step + 1) * cfg.sim.dt * manifest["fps"])
                    if next_frame != current_frame or step + 1 == total_steps:
                        actual = task._current_tips()
                        target_tips = task.bank.fingertips(motion_ids, clamped)
                        current_error = (actual - target_tips).norm(dim=-1).detach().cpu().numpy() * 1000
                        for index, length in enumerate(lengths.tolist()):
                            if current_frame < length and current_frame != prev_frame:
                                errors[index].append(current_error[index])
                        prev_frame = current_frame
            report = {
                "library_path": str(manifest_path),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "controller_sha256": controller_sha256,
                "physics_dt_s": cfg.sim.dt,
                "motions": [],
            }
            for record, samples in zip(records, errors):
                values = np.asarray(samples, dtype=np.float32)
                worst = values.max(axis=-1)
                report["motions"].append(
                    {
                        "source": record["source"],
                        "input_npz_sha256": npz_hashes[record["source"]],
                        "sampled_frames": len(values),
                        "mean_five_finger_max_error_mm": float(worst.mean()),
                        "p95_five_finger_max_error_mm": float(np.percentile(worst, 95)),
                        "max_five_finger_max_error_mm": float(worst.max()),
                        "all_five_within_10mm_frames": int(np.count_nonzero(worst < 10.0)),
                    }
                )
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
        finally:
            raw_env.close()


if __name__ == "__main__":
    main()
