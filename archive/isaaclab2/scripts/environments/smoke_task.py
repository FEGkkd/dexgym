"""Run a short reset/step smoke test for a registered DexterousHands task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Smoke test an Isaac Lab task.")
parser.add_argument("--task", default="DexterousHands-Shadow-Hand-Reorientation-Direct-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--diagnose", action="store_true")
parser.add_argument("--force-contact-probe", action="store_true")
parser.add_argument("--ground-contact-probe", action="store_true")
parser.add_argument("--large-contact-probe", action="store_true")
parser.add_argument("--probe-steps", type=int, default=1)
parser.add_argument("--trace", action="store_true", help="Print per-step object/contact state.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
import dexteroushands_lab.tasks as dexteroushands_tasks  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

dexteroushands_tasks.register_tasks()


def _check_observations(obs, cfg, num_envs):
    """Check the selected task's declared spaces, without assuming Wuji v1."""
    assert "policy" in obs
    assert obs["policy"].shape == (num_envs, cfg.observation_space)
    assert torch.isfinite(obs["policy"]).all().item()
    if cfg.state_space:
        assert "critic" in obs
        assert obs["critic"].shape == (num_envs, cfg.state_space)
        assert torch.isfinite(obs["critic"]).all().item()


def main() -> None:
    print("SMOKE_STAGE before_parse_cfg", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.large_contact_probe:
        env_cfg.object.spawn.size = (0.30, 0.30, 0.30)
    print("SMOKE_STAGE before_make", flush=True)
    print("SMOKE_STAGE before_make_call", flush=True)
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        print("SMOKE_STAGE after_make", flush=True)
    except BaseException as exc:
        print(f"SMOKE_MAKE_ERROR type={type(exc).__name__} message={exc!r}", flush=True)
        raise
    try:
        print("SMOKE_STAGE before_reset", flush=True)
        obs, _ = env.reset(seed=0)
        print("SMOKE_STAGE after_reset", flush=True)
        _check_observations(obs, env_cfg, args_cli.num_envs)

        if args_cli.force_contact_probe or args_cli.ground_contact_probe or args_cli.large_contact_probe:
            robot = env.unwrapped.robot
            object_asset = env.unwrapped.object
            probe_body_id = env.unwrapped._tip_body_ids[0]
            probe_tip_pos_w = robot.data.body_link_pose_w.torch[:, probe_body_id, :3].clone()
            probe_pose = object_asset.data.default_root_pose.torch.clone()
            probe_pose[:, :3] = env.unwrapped.scene.env_origins + torch.tensor(
                (10.0, 10.0, 10.0), device=env.unwrapped.device, dtype=probe_pose.dtype
            )
            if args_cli.ground_contact_probe:
                probe_pose[:, :3] = env.unwrapped.scene.env_origins
                probe_pose[:, 2] = 0.027
            elif args_cli.large_contact_probe:
                probe_pose[0, :3] = robot.data.root_pose_w.torch[0, :3]
            else:
                probe_pose[0, :3] = probe_tip_pos_w[0]
            probe_pose[:, 3:7] = torch.tensor(
                (0.0, 0.0, 0.0, 1.0), device=env.unwrapped.device, dtype=probe_pose.dtype
            )
            probe_velocity = object_asset.data.default_root_vel.torch.clone()
            probe_velocity.zero_()
            probe_env_ids = torch.arange(
                args_cli.num_envs, device=env.unwrapped.device, dtype=torch.long
            )
            object_asset.write_root_pose_to_sim_index(
                root_pose=probe_pose, env_ids=probe_env_ids
            )
            object_asset.write_root_velocity_to_sim_index(
                root_velocity=probe_velocity, env_ids=probe_env_ids
            )

        if args_cli.force_contact_probe or args_cli.ground_contact_probe or args_cli.large_contact_probe:
            env.unwrapped.scene.write_data_to_sim()
            for _ in range(args_cli.probe_steps):
                env.unwrapped.sim.step(render=False)
                env.unwrapped.scene.update(dt=env.unwrapped.physics_dt)
            direct_contact_forces = env.unwrapped._contact_forces_with_object()
            direct_net_forces = env.unwrapped.contact_sensor.data.net_forces_w.torch
            direct_max_force = torch.linalg.norm(direct_contact_forces, dim=-1).amax(dim=1)
            print(
                "DIAG direct_probe_force="
                f"filtered={direct_max_force.detach().cpu().tolist()} "
                f"force_by_body_env0={torch.linalg.norm(direct_contact_forces[0], dim=-1).detach().cpu().tolist()} "
                f"net={direct_net_forces.detach().cpu().tolist()} "
                f"object_pos={env.unwrapped.object.data.root_pos_w.torch.detach().cpu().tolist()}",
                flush=True,
            )
            if args_cli.force_contact_probe or args_cli.large_contact_probe:
                assert direct_max_force[0] > env.unwrapped.cfg.contact_force_threshold
                if args_cli.num_envs > 1:
                    assert torch.all(
                        direct_max_force[1:] <= env.unwrapped.cfg.contact_force_threshold
                    ).item()

        if args_cli.diagnose and hasattr(env.unwrapped, "robot"):


            robot = env.unwrapped.robot
            object_asset = env.unwrapped.object
            body_pose = robot.data.body_link_pose_w.torch
            root_pose = robot.data.root_pose_w.torch
            object_pos = object_asset.data.root_pos_w.torch
            body_rel = body_pose[:, :, :3] - root_pose[:, None, :3]
            print(f"DIAG body_names={robot.body_names}")
            print(f"DIAG root_pose={root_pose[0].detach().cpu().tolist()}")
            print(f"DIAG object_pos={object_pos[0].detach().cpu().tolist()}")
            if hasattr(env.unwrapped, "_compute_cage_escape"):
                print(f"DIAG cage_escape={env.unwrapped._compute_cage_escape()[0].item():.6f}")
            if hasattr(env.unwrapped, "_wuji_contact_body_paths"):
                print(
                    "DIAG contact_body_paths="
                    f"{env.unwrapped._wuji_contact_body_paths}"
                )
            if hasattr(env.unwrapped, "contact_sensors"):
                for sensor_index, sensor in enumerate(env.unwrapped.contact_sensors):
                    print(
                        "DIAG contact_sensor="
                        f"{sensor_index} cfg={sensor.cfg.prim_path} "
                        f"bodies={sensor.body_names} "
                        f"view={sensor.body_physx_view.prim_paths} "
                        f"filters={sensor.contact_view.filter_count}"
                    )
            print(
                "DIAG body_rel_min_max="
                f"{body_rel[0].amin(dim=0).detach().cpu().tolist()} "
                f"{body_rel[0].amax(dim=0).detach().cpu().tolist()}"
            )
            tip_rel = body_pose[0, env.unwrapped._tip_body_ids, :3] - root_pose[0, None, :3]
            cube_rel = object_pos[0] - root_pose[0, :3]
            print(
                "DIAG tip_rel=" + str(tip_rel.detach().cpu().tolist()) +
                " cube_rel=" + str(cube_rel.detach().cpu().tolist()) +
                " tip_dist=" + str(torch.linalg.norm(tip_rel - cube_rel, dim=-1).detach().cpu().tolist())
            )

        if args_cli.diagnose:
            stage = env.unwrapped.sim.stage
            tip_path = "/World/envs/env_0/WujiHand/" + robot.body_names[env.unwrapped._tip_body_ids[0]]
            tip_root = stage.GetPrimAtPath(tip_path)
            print(
                f"DIAG tip_descendants={tip_path} instance={tip_root.IsInstanceProxy()} "
                f"all_children={len(tip_root.GetAllChildren())}",
                flush=True,
            )
            collision_prim = stage.GetPrimAtPath(tip_path + "/collisions")
            collision_proto = collision_prim.GetPrototype()
            print(
                f"DIAG tip_collision_proto valid={collision_proto.IsValid()} "
                f"path={collision_proto.GetPath()} children={len(collision_proto.GetAllChildren())}",
                flush=True,
            )
            if collision_proto.IsValid():
                for collision_child in Usd.PrimRange(
                    collision_proto, Usd.TraverseInstanceProxies()
                ):
                    if collision_child.IsA(UsdGeom.Gprim):
                        print(
                            f"DIAG collision_geom={collision_child.GetPath()} "
                            f"type={collision_child.GetTypeName()} "
                            f"apis={collision_child.GetAppliedSchemas()}",
                            flush=True,
                        )
            for prim in Usd.PrimRange(tip_root, Usd.TraverseInstanceProxies()):
                if prim.GetPath() != tip_root.GetPath():
                    print(
                        f"DIAG tip_prim={prim.GetPath()} type={prim.GetTypeName()} "
                        f"apis={prim.GetAppliedSchemas()}",
                        flush=True,
                    )

        for _ in range(args_cli.steps):
            actions = torch.zeros(
                (args_cli.num_envs, env.action_space.shape[-1]),
                device=env.unwrapped.device,
            )
            obs, reward, terminated, truncated, _ = env.step(actions)
            _check_observations(obs, env_cfg, args_cli.num_envs)
            assert torch.isfinite(reward).all().item()
            assert terminated.shape[0] == args_cli.num_envs
            assert truncated.shape[0] == args_cli.num_envs
            if args_cli.trace:
                object_position = env.unwrapped.object.data.root_pos_w.torch - env.unwrapped.scene.env_origins
                palm_position = env.unwrapped.robot.data.root_pose_w.torch[:, :3] - env.unwrapped.scene.env_origins
                contact_magnitude = torch.linalg.norm(env.unwrapped._contact_forces_with_object(), dim=-1)
                print(
                    "TRACE step=" + str(_ + 1) +
                    " object_rel=" + str((object_position - palm_position).detach().cpu().tolist()) +
                    " cage=" + str(env.unwrapped._compute_cage_escape().detach().cpu().tolist()) +
                    " tip_contacts=" + str((contact_magnitude[:, env.unwrapped._tip_body_ids] > env.unwrapped.cfg.contact_force_threshold).sum(dim=-1).detach().cpu().tolist()) +
                    " terminated=" + str(terminated.detach().cpu().tolist()) +
                    " truncated=" + str(truncated.detach().cpu().tolist()),
                    flush=True,
                )
            if args_cli.force_contact_probe or args_cli.ground_contact_probe or args_cli.large_contact_probe:
                object_position = env.unwrapped.object.data.root_pos_w.torch
                tip_position = env.unwrapped.robot.data.body_link_pose_w.torch[:, env.unwrapped._tip_body_ids[0], :3]
                contact_forces = env.unwrapped._contact_forces_with_object()
                net_forces = env.unwrapped.contact_sensor.data.net_forces_w.torch
                raw_net_forces = env.unwrapped.contact_sensor.contact_view.get_net_contact_forces(
                    dt=env.unwrapped.cfg.sim.dt
                )
                raw_matrix = env.unwrapped.contact_sensor.contact_view.get_contact_force_matrix(
                    dt=env.unwrapped.cfg.sim.dt
                )
                print(
                    "DIAG contact_probe_positions="
                    f"object={object_position.detach().cpu().tolist()} "
                    f"tip={tip_position.detach().cpu().tolist()}",
                    flush=True,
                )
                print(
                    "DIAG contact_probe_raw="
                    f"shape={getattr(raw_matrix, 'shape', None)} "
                    f"type={type(raw_matrix).__name__} "
                    f"max={wp.to_torch(raw_matrix).abs().max().item()} "
                    f"net={net_forces.detach().cpu().tolist()} "
                    f"raw_net_max={wp.to_torch(raw_net_forces).abs().max().item()} "
                    f"object_vel={env.unwrapped.object.data.root_lin_vel_w.torch.detach().cpu().tolist()}",
                    flush=True,
                )
                contact_magnitude = torch.linalg.norm(contact_forces, dim=-1)
                max_force_per_env = contact_magnitude.amax(dim=1)
                print(
                    "DIAG contact_probe_max_force="
                    f"{max_force_per_env.detach().cpu().tolist()}",
                    flush=True,
                )
                if _ == args_cli.steps - 1:
                    if args_cli.ground_contact_probe:
                        assert torch.linalg.norm(net_forces[0]).item() > 1.0e-6
                    else:
                        assert max_force_per_env[0] > env.unwrapped.cfg.contact_force_threshold
                        if args_cli.num_envs > 1:
                            assert torch.all(
                                max_force_per_env[1:] <= env.unwrapped.cfg.contact_force_threshold
                            ).item()
        print(
            f"SMOKE_OK task={args_cli.task} envs={args_cli.num_envs} "
            f"obs={tuple(obs['policy'].shape)} critic={tuple(obs['critic'].shape) if 'critic' in obs else None} action={env.action_space.shape}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
