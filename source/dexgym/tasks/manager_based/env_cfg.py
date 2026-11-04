"""ManagerBasedRLEnv joint tracking with six active MDP managers."""

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from dexgym.assets.cartpole import JointTrackingPhysicsCfg, JointTrackingSceneCfg
from dexgym.tasks.manager_based import mdp


@configclass
class ActionsCfg:
    cart = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=1.5,
        use_default_offset=False,
        clip={"slider_to_cart": (-1.5, 1.5)},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        state_goal = ObsTerm(func=mdp.cart_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    cart_goal: mdp.CartGoalCommandCfg = mdp.CartGoalCommandCfg()


@configclass
class EventsCfg:
    reset_cart = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-0.5, 0.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_pole = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=base_mdp.is_alive, weight=1.0)
    tracking_error = RewTerm(func=mdp.cart_tracking_error, weight=-1.0)
    cart_velocity = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=base_mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-2.8, 2.8)},
    )


@configclass
class JointTrackingManagerCfg(ManagerBasedRLEnvCfg):
    scene: JointTrackingSceneCfg = JointTrackingSceneCfg(num_envs=16, env_spacing=4.0, clone_in_fabric=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = JointTrackingPhysicsCfg()
