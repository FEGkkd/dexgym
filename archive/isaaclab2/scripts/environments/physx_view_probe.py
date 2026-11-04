"""Probe native PhysX body/contact view path semantics without ContactSensor."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab_physx.physics import PhysxManager  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

from dexteroushands_lab.assets.wuji_hand import make_wuji_hand_cfg  # noqa: E402


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    robot = make_wuji_hand_cfg().replace(prim_path="{ENV_REGEX_NS}/Hand")
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.054, 0.054, 0.054),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.60)),
    )


def try_view(sim_view, pattern: str, filters: list[str]) -> None:
    print(f"TRY pattern={pattern} filters={filters}", flush=True)
    try:
        view = sim_view.create_rigid_contact_view(
            pattern,
            filter_patterns=filters,
            max_contact_data_count=128,
        )
        print(f"VIEW attrs={[name for name in dir(view) if not name.startswith('_')]}", flush=True)
        print(f"VIEW num_bodies={view.num_bodies} filter_count={view.filter_count}", flush=True)
    except Exception as exc:
        print(f"VIEW_ERROR {type(exc).__name__}: {exc}", flush=True)


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    scene = InteractiveScene(ProbeSceneCfg(num_envs=args.num_envs, env_spacing=1.0))
    sim.reset()
    scene.reset()
    sim_view = PhysxManager.get_physics_sim_view()
    print(f"BODY_NAMES={scene['robot'].body_names}", flush=True)
    stage = sim_utils.get_current_stage()
    for name in ('r_wrist', 'right_finger2_tip_link', 'right_finger2_tip_link'):
        prim = stage.GetPrimAtPath(f"/World/envs/env_0/Hand/{name}")
        print(f"API path={prim.GetPath()} schemas={prim.GetAppliedSchemas()}", flush=True)
    for pattern in (
        "/World/envs/env_0/Hand/right_finger2_tip_link/collisions/*",
        "/World/envs/env_0/Hand/right_finger2_tip_link/visuals/*",
        "/World/envs/env_0/Hand/right_finger2_tip_link/collisions",
        "/World/envs/env_0/Hand/right_finger2_tip_link",
        "/World/envs/env_*/Hand/right_finger2_tip_link",
        "/World/env_. */Hand/right_finger2_tip_link",
        "/World/envs/env_*/Hand/*",
    ):
        pattern = pattern.replace("env_. *", "env_.*")
        body_view = sim_view.create_rigid_body_view(pattern)
        print(f"BODY_VIEW pattern={pattern} count={body_view.count} paths={body_view.prim_paths}", flush=True)
    patterns = [
        "/World/envs/env_.*/Hand/right_finger2_tip_link",
        "/World/envs/env_*/Hand/right_finger2_tip_link",
        "/World/envs/env_0/Hand/right_finger2_tip_link",
        "/World/envs/env_.*/Hand/*",
    ]
    for pattern in patterns:
        for filter_pattern in (
            "/World/envs/env_0/Object",
            "/World/envs/env_*/Object",
            "/World/envs/env_. */Object",
        ):
            try_view(sim_view, pattern, [filter_pattern.replace("env_. *", "env_.*")])
    sim.clear()


try:
    main()
finally:
    app.close()
