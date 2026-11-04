"""Train the S0 Actor on synthetic FK pairs and write an RSL-RL bootstrap checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import torch
from dexgym.assets.marvin_wuji.controller import ARM_JOINT_NAMES
from dexgym.assets.marvin_wuji.robot_cfg import JOINT_ORDER, PACKAGED_KEYFRAMES, SOURCE_SHA256, resolve_source_mjcf
from dexgym.tasks.marvin_s0.kinematics import MarvinArmFK
from dexgym.tasks.marvin_s0.teacher import sample_fk_teacher_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True, help="RSL-RL model_0.pt for the same 73D/14D network")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--bimanual-fraction", type=float, default=0.0)
    parser.add_argument("--policy-std", type=float, default=0.2, help="Initial PPO exploration standard deviation")
    args = parser.parse_args()
    if args.updates <= 0 or args.batch_size <= 0:
        parser.error("updates and batch size must be positive")
    if not 0.0 <= args.bimanual_fraction <= 1.0:
        parser.error("bimanual fraction must be in [0, 1]")
    if not 0.0 < args.policy_std <= 1.0:
        parser.error("policy std must be in (0, 1]")
    source = resolve_source_mjcf()
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("The source MJCF differs from the pinned FK model")
    model = mujoco.MjModel.from_xml_path(str(source))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    fk = MarvinArmFK(model, device)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
    lower = torch.tensor(model.jnt_range[ids, 0].copy() + 0.03, dtype=torch.float32, device=device)
    upper = torch.tensor(model.jnt_range[ids, 1].copy() - 0.03, dtype=torch.float32, device=device)
    keyframes = json.loads(PACKAGED_KEYFRAMES.read_text())
    if tuple(keyframes["joint_order"]) != JOINT_ORDER:
        raise ValueError("Packaged Home order differs from the articulation")
    home = torch.tensor(keyframes["keyframes"]["home"], dtype=torch.float32, device=device)
    arm_ids = [JOINT_ORDER.index(name) for name in ARM_JOINT_NAMES]
    home_arm = home[arm_ids]
    network = torch.nn.Sequential(
        torch.nn.Linear(73, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 14),
    ).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    with torch.no_grad():
        val_obs, val_target = sample_fk_teacher_batch(
            fk, home_arm, lower, upper, 8192, bimanual_fraction=args.bimanual_fraction
        )
    with torch.no_grad():
        initial_mse = float(torch.nn.functional.mse_loss(network(val_obs), val_target))
    losses = []
    for update in range(args.updates):
        with torch.no_grad():
            obs, target = sample_fk_teacher_batch(
                fk, home_arm, lower, upper, args.batch_size, bimanual_fraction=args.bimanual_fraction
            )
        predicted = network(obs)
        loss = torch.nn.functional.mse_loss(predicted, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        if (update + 1) % 100 == 0 or update + 1 == args.updates:
            with torch.no_grad():
                val_mse = float(torch.nn.functional.mse_loss(network(val_obs), val_target))
            losses.append({"update": update + 1, "train_mse": float(loss.detach()), "validation_mse": val_mse})
            print(losses[-1], flush=True)
    checkpoint = torch.load(args.template, map_location="cpu", weights_only=True)
    if tuple(checkpoint["actor_state_dict"]["mlp.0.weight"].shape) != (256, 73):
        raise ValueError("Template checkpoint does not match the 73D S0 actor")
    for name, tensor in network.state_dict().items():
        key = f"mlp.{name}"
        if key not in checkpoint["actor_state_dict"] or checkpoint["actor_state_dict"][key].shape != tensor.shape:
            raise ValueError(f"Template Actor layer mismatch: {key}")
        checkpoint["actor_state_dict"][key] = tensor.detach().cpu()
    checkpoint["actor_state_dict"]["distribution.std_param"][:] = args.policy_std
    checkpoint["iter"] = 0
    checkpoint["infos"] = {"bootstrap": "offline_fk_teacher", "source_sha256": SOURCE_SHA256}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    summary = {
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "synthetic_training_pairs": args.updates * args.batch_size,
        "initial_validation_mse": initial_mse,
        "final_validation_mse": losses[-1]["validation_mse"],
        "losses": losses,
        "checkpoint": str(args.output.resolve()),
        "source_mjcf_sha256": SOURCE_SHA256,
        "actor_observation_size": 73,
        "actor_action_size": 14,
        "deployment_uses_ik": False,
        "bimanual_fraction": args.bimanual_fraction,
        "policy_std": args.policy_std,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "losses"}, indent=2))


if __name__ == "__main__":
    main()
