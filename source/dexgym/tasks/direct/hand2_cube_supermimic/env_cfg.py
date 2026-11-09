"""Native Isaac Lab 3 / Newton scene for the pinned Dex-Supermimic cube task."""

import json

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.utils import PresetCfg

from dexgym.assets.wuji_hand2_cube import ROOT
from dexgym.assets.wuji_hand2_cube.rig import (
    HAND2_REORIENT_ACTUATOR_FORCE_LIMITS,
    HAND2_REORIENT_ACTUATOR_GAINS,
    REORIENT_HAND2_CUBE_INIT_POS,
    REORIENT_HAND2_JOINT_POS,
    REORIENT_HAND2_ROOT_POS,
    REORIENT_HAND2_ROOT_ROT,
)

_METADATA = json.loads((ROOT / "generated/model_metadata.json").read_text())
JOINT_NAMES = _METADATA["joint_names"]


@configclass
class Hand2CubeSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Hand",
        spawn=UsdFileCfg(
            usd_path=str(ROOT / "generated/usd/hand2_cube_rig/hand2_cube_rig.usda"), variants={"Physics": "mujoco"}
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=REORIENT_HAND2_ROOT_POS,
            rot=REORIENT_HAND2_ROOT_ROT,
            joint_pos=REORIENT_HAND2_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        joint_ordering=JOINT_NAMES,
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["right_finger.*"],
                stiffness={k: v[0] for k, v in HAND2_REORIENT_ACTUATOR_GAINS.items()},
                damping={k: v[1] for k, v in HAND2_REORIENT_ACTUATOR_GAINS.items()},
                joint_effort_limit=HAND2_REORIENT_ACTUATOR_FORCE_LIMITS,
            )
        },
    )
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=UsdFileCfg(usd_path=str(ROOT / "generated/cube_usd/cube/cube.usda"), variants={"Physics": "mujoco"}),
        init_state=RigidObjectCfg.InitialStateCfg(pos=REORIENT_HAND2_CUBE_INIT_POS["right"]),
    )


@configclass
class Hand2CubePhysicsCfg(PresetCfg):
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1500,
            nconmax=180,
            integrator="implicitfast",
            enable_multiccd=True,
            ccd_iterations=50,
            iterations=10,
            ls_iterations=20,
            tolerance=1e-8,
            use_mujoco_contacts=True,
        ),
        num_substeps=1,
        use_cuda_graph=False,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class Hand2CubeSuperMimicCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 50.0
    action_space = 20
    observation_space = 207
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=2, physics=Hand2CubePhysicsCfg())
    scene: Hand2CubeSceneCfg = Hand2CubeSceneCfg(num_envs=16, env_spacing=2.0)
    auto_reset: bool = False
    observation_noise: bool = True
    randomize_reset: bool = True
    startup_randomization: bool = True
    velocity_disturbance: bool = True
