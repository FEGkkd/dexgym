"""Goal command and MDP terms for cart joint tracking."""

from collections.abc import Sequence

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class CartGoalCommand(CommandTerm):
    """Hold a sampled legal slider target for each episode."""

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        self._goal = torch.zeros((self.num_envs, 1), device=self.device)
        self._cart_ids, _ = env.scene["robot"].find_joints("slider_to_cart")

    @property
    def command(self) -> torch.Tensor:
        return self._goal

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        self._goal[env_ids] = torch.empty_like(self._goal[env_ids]).uniform_(-1.0, 1.0)

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        robot = self._env.scene["robot"]
        self.metrics["position_error"] = (self._goal - robot.data.joint_pos.torch[:, self._cart_ids]).abs().squeeze(-1)


@configclass
class CartGoalCommandCfg(CommandTermCfg):
    class_type: type[CartGoalCommand] = CartGoalCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)


def cart_observation(env) -> torch.Tensor:
    robot = env.scene["robot"]
    cart_ids, _ = robot.find_joints("slider_to_cart")
    pos = robot.data.joint_pos.torch[:, cart_ids]
    vel = robot.data.joint_vel.torch[:, cart_ids]
    goal = env.command_manager.get_command("cart_goal")
    return torch.cat((pos, vel, goal, goal - pos), dim=-1)


def cart_tracking_error(env) -> torch.Tensor:
    robot = env.scene["robot"]
    cart_ids, _ = robot.find_joints("slider_to_cart")
    pos = robot.data.joint_pos.torch[:, cart_ids]
    goal = env.command_manager.get_command("cart_goal")
    return (goal - pos).square().sum(dim=-1)
