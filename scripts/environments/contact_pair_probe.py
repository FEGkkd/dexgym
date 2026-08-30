"""Probe one native PhysX contact-view sensor/filter pair."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--sensor", required=True)
parser.add_argument("--filter", dest="filter_path", action="append", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab_physx.physics import PhysxManager  # noqa: E402

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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.60)),
    )


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args.device)
    )
    scene = InteractiveScene(ProbeSceneCfg(num_envs=args.num_envs, env_spacing=1.0))
    sim.reset()
    scene.reset()
    sim_view = PhysxManager.get_physics_sim_view()
    print(f"PAIR sensor={args.sensor} filter={args.filter_path}", flush=True)
    try:
        view = sim_view.create_rigid_contact_view(
            args.sensor,
            filter_patterns=args.filter_path,
            max_contact_data_count=32,
        )
        print(f"PAIR_VIEW attrs={[name for name in dir(view) if not name.startswith('_')]}", flush=True)
        print(f"PAIR_SENSOR_PATHS={view.sensor_paths}", flush=True)
        print(f"PAIR_FILTER_PATHS={view.filter_paths}", flush=True)
        print(f"PAIR_COUNTS sensor={view.sensor_count} filters={view.filter_count}", flush=True)
        forces = view.get_contact_force_matrix(dt=1.0 / 120.0)
        print(f"PAIR_FORCE type={type(forces).__name__} shape={getattr(forces, 'shape', None)}", flush=True)
    except Exception as exc:
        print(f"PAIR_ERROR {type(exc).__name__}: {exc}", flush=True)
    sim.clear()


try:
    main()
finally:
    app.close()
