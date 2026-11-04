"""Compare full and collision-free Marvin/Wuji inertial data inside Newton MJWarp."""

import argparse
import json
from pathlib import Path

import torch
from dexgym.assets.marvin_wuji.robot_cfg import NO_HAND_CONTACT_USD, PACKAGED_USD, make_robot_cfg
from isaaclab.app import launch_simulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg


def validate(full_usd: Path, no_hand_usd: Path) -> dict:
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        full = make_robot_cfg(full_usd, prim_path="{ENV_REGEX_NS}/FullRobot")
        no_hand = make_robot_cfg(no_hand_usd, prim_path="{ENV_REGEX_NS}/NoHandRobot")

    cfg = SimulationCfg(
        dt=0.002, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=1000, nconmax=500), use_cuda_graph=False)
    )
    with launch_simulation(cfg):
        sim = SimulationContext(cfg)
        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
        sim.reset()
        full = scene["full"]
        no_hand = scene["no_hand"]
        if full.joint_names != no_hand.joint_names or full.body_names != no_hand.body_names:
            raise ValueError("Joint or rigid-body names/order changed in Newton")
        if (full.num_joints, full.num_bodies) != (54, 74):
            raise ValueError("The full robot structure changed in Newton")
        differences = {}
        for name in (
            "body_mass",
            "body_inertia",
            "joint_stiffness",
            "joint_damping",
            "joint_armature",
            "joint_friction_coeff",
            "joint_effort_limits",
            "joint_pos_limits_lower",
            "joint_pos_limits_upper",
        ):
            original = getattr(full.data, name).torch
            changed = getattr(no_hand.data, name).torch
            if (
                original.shape != changed.shape
                or not torch.isfinite(original).all()
                or not torch.isfinite(changed).all()
            ):
                raise ValueError(f"Invalid Newton {name} arrays")
            differences[name] = float(torch.max(torch.abs(original - changed)))
            if differences[name] != 0.0:
                raise ValueError(f"Newton {name} changed: {differences[name]}")
    return {"joints": 54, "bodies": 74, "max_absolute_differences": differences, "pass": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-usd", type=Path, default=PACKAGED_USD)
    parser.add_argument("--no-hand-usd", type=Path, default=NO_HAND_CONTACT_USD)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.full_usd, args.no_hand_usd)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
