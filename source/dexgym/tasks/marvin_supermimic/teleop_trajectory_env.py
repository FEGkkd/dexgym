"""Newton task for causal, workspace-distributed dual TCP SE(3) target streams."""

from pathlib import Path

import torch
from isaaclab.utils import configclass

from dexgym.motion.teleop_workspace import TeleopWorkspaceReferenceBank

from .se3_trajectory_env import (MarvinContinuousSE3FollowCfg,
                                 MarvinContinuousSE3FollowEnv)
from .teleop_trajectory import MarvinTeleopWorkspaceFollow


@configclass
class MarvinTeleopWorkspaceCfg(MarvinContinuousSE3FollowCfg):
    observation_space = 10 * 68 + 18
    reference_path = ""  # The pinned procedural trajectory is not an input to this task.
    reference_bank_path = ""
    evaluation_split = "test"
    evaluation_max_difficulty = 2
    curriculum_level = 2
    # First 75% of Newton environments train; the rest are held out for WM trust.
    train_env_fraction = 0.75


class MarvinTeleopWorkspaceEnv(MarvinContinuousSE3FollowEnv):
    """Reuse only the validated arm servo/physics loop; target MDP is independent."""

    cfg: MarvinTeleopWorkspaceCfg

    def _make_reference(self, cfg):
        if not cfg.reference_bank_path or not Path(cfg.reference_bank_path).is_file():
            raise ValueError("A generated 50 Hz workspace target bank is required")
        return TeleopWorkspaceReferenceBank(cfg.reference_bank_path, self.device)

    @staticmethod
    def _make_task(reference, robot):
        return MarvinTeleopWorkspaceFollow(reference, robot)

    def _sample_reference(self, env_ids):
        count = len(env_ids)
        if self.cfg.evaluation_mode:
            maximum = self.cfg.evaluation_max_difficulty
            if not 0 <= maximum <= int(self.reference.difficulty.max()):
                raise ValueError("Evaluation difficulty is outside the target bank")
            if self.cfg.evaluation_motion_id is not None:
                motion_id = torch.full((count,), self.cfg.evaluation_motion_id,
                                       device=self.device, dtype=torch.long)
                required_split = ("train", "validation", "test").index(self.cfg.evaluation_split)
                if bool((self.reference.split[motion_id] != required_split).any()):
                    raise ValueError("Selected evaluation clip does not belong to its declared split")
                if bool((self.reference.difficulty[motion_id] > maximum).any()):
                    raise ValueError("Selected evaluation clip exceeds the declared difficulty")
            else:
                eligible = torch.nonzero(
                    (self.reference.split == ("train", "validation", "test").index(
                        self.cfg.evaluation_split))
                    & (self.reference.difficulty <= maximum)).flatten()
                if len(eligible) == 0:
                    raise ValueError("No target clips match evaluation split and difficulty")
                motion_id = eligible[torch.arange(count, device=self.device) % len(eligible)]
            frame = torch.zeros(count, device=self.device, dtype=torch.long)
        else:
            split = torch.where(env_ids < int(self.num_envs * self.cfg.train_env_fraction),
                                torch.zeros(count, device=self.device, dtype=torch.long),
                                torch.ones(count, device=self.device, dtype=torch.long))
            motion_id = torch.empty(count, device=self.device, dtype=torch.long)
            for value, name in ((0, "train"), (1, "validation")):
                selected = split == value
                if bool(selected.any()):
                    motion_id[selected] = self.reference.sample_ids(
                        int(selected.sum()), split=name,
                        max_difficulty=self.cfg.curriculum_level)
            frame = (torch.randint(self.cfg.max_reference_start_frame + 1,
                                   (count,), device=self.device)
                     if self.cfg.random_reference_start else
                     torch.zeros(count, device=self.device, dtype=torch.long))
        return motion_id, frame
