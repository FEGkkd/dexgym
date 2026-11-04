"""Shared minimal Cartpole asset and Newton physics configuration.

Source: Isaac Lab v3.0.0 Cartpole, https://github.com/isaac-sim/IsaacLab/tree/v3.0.0/source/isaaclab_tasks/isaaclab_tasks/core/cartpole
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.utils import PresetCfg


@configclass
class JointTrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = replace(
        CARTPOLE_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "cart": ImplicitActuatorCfg(
                joint_names_expr=["slider_to_cart"], joint_effort_limit=400.0, stiffness=200.0, damping=20.0
            ),
            "pole": ImplicitActuatorCfg(
                joint_names_expr=["cart_to_pole"], joint_effort_limit=400.0, stiffness=0.0, damping=0.0
            ),
        },
    )


@configclass
class JointTrackingPhysicsCfg(PresetCfg):
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=5, nconmax=3), num_substeps=1, use_cuda_graph=True
    )
    default: NewtonCfg = newton_mjwarp
