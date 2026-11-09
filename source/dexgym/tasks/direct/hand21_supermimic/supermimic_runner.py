"""Construct the standalone hand's shared training/evaluation components."""

import subprocess
from pathlib import Path

from dexgym.algorithms.supermimic.core.dynamics_contract import world_model_dynamics_signature
from dexgym.algorithms.supermimic.core.models import NetworkFactory, NetworkModules, TaskTerminationHead
from dexgym.algorithms.supermimic.core.specs import NetworkSpec

from .supermimic_dex_model import HandDexTerminationHead, HandDexWorldModel
from .supermimic_model import HandWorldModel
from .supermimic_networks import HandCritic, HandDexActor, HandDexCritic, HandDirectActor
from .supermimic_robot import Hand21DynamicsAdapter, HandDexDynamicsAdapter
from .supermimic_task import Hand21Task


def components(env, experiment):
    robot_type = HandDexDynamicsAdapter if experiment["version"] == "hand21_s0_v4" else Hand21DynamicsAdapter
    robot = robot_type(device=env.device)
    task = Hand21Task(robot, env.bank, env.cfg)
    network_fields = experiment["network"].copy()
    for key in ("world_hidden", "goal_hidden", "actor_hidden", "critic_hidden", "termination_hidden"):
        network_fields[key] = tuple(network_fields[key])
    network = NetworkSpec(**network_fields)
    if experiment["version"] == "hand21_s0_v4":
        world = HandDexWorldModel(robot.spec, network, task)
        world.dynamics_signature = world_model_dynamics_signature(robot, network)
        modules = NetworkModules(
            world,
            HandDexActor(robot.spec, task.spec, network),
            HandDexCritic(task.spec, network),
            HandDexTerminationHead(robot.spec, task.spec, network),
        )
    elif experiment["version"] == "hand21_s0_v3":
        world = HandWorldModel(robot.spec, network, task)
        world.dynamics_signature = world_model_dynamics_signature(robot, network)
        modules = NetworkModules(
            world,
            HandDirectActor(robot.spec, task.spec, network),
            HandCritic(task.spec, network),
            TaskTerminationHead(robot.spec, task.spec, network),
        )
    else:
        modules = NetworkFactory.build(robot.spec, task.spec, network, robot_adapter=robot)
        dynamics_signature = modules.world_model.dynamics_signature
        modules.world_model = HandWorldModel(robot.spec, network, task)
        modules.world_model.dynamics_signature = dynamics_signature
    for module in (modules.world_model, modules.actor, modules.critic, modules.termination):
        module.to(env.device)
    return robot, task, modules


def physics_contract(cfg, robot):
    from dexgym.assets.wuji_hand21.robot_cfg import OFFICIAL_DAMPING, OFFICIAL_EFFORT_LIMIT, OFFICIAL_STIFFNESS

    return {
        "num_envs": cfg.scene.num_envs,
        "seed": cfg.seed,
        "control_dt": 1 / 30,
        "physics_dt": cfg.sim.dt,
        "integrator": "implicitfast",
        "gravity": tuple(cfg.sim.gravity),
        "stiffness": OFFICIAL_STIFFNESS,
        "damping": OFFICIAL_DAMPING,
        "effort_limits": OFFICIAL_EFFORT_LIMIT,
        "action_scale": 0.1,
        "deadzone": 0.1,
        "robot": robot.checkpoint_contract(),
        "full_self_collision": True,
    }


def source_revision():
    """Record the source checkout, independent of the caller's working directory."""
    try:
        directory = str(Path(__file__).resolve().parent)
        root = Path(
            subprocess.check_output(
                ["git", "-C", directory, "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        expected = root / "source/dexgym/tasks/direct/hand21_supermimic/supermimic_runner.py"
        if Path(__file__).resolve() != expected.resolve():
            return "unversioned-package"
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned-package"
