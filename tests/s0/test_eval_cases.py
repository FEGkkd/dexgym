"""Fixed external cases must set TCP goals without giving goal joint angles to the actor."""

import os

import dexgym.tasks  # noqa: F401
import gymnasium as gym
import pytest
import torch
from dexgym.tasks.marvin_s0.env_cfg import MarvinBimanualS0Cfg
from isaaclab.app import launch_simulation


@pytest.mark.skipif(not os.environ.get("DEXGYM_MARVIN_MJCF"), reason="External MJCF unavailable")
def test_fixed_cases_override_reset_and_keep_actor_contract():
    cfg = MarvinBimanualS0Cfg()
    cfg.scene.num_envs = 2
    cfg.bimanual_fraction = 1.0
    cfg.sim.physics = cfg.sim.physics.newton_mjwarp
    with launch_simulation(cfg):
        env = gym.make("DexGym-Marvin-Bimanual-SE3-Direct-v0", cfg=cfg)
        try:
            env.reset(seed=991)
            raw = env.unwrapped
            assert torch.all(raw.active_mask == 1.0)
            assert torch.all(raw.mode_one_hot[:, 2] == 1.0)
            starts = raw.home_arm.repeat(2, 1)
            goals = raw.fk(starts)
            masks = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=raw.device)
            raw.set_eval_cases(starts, goals, masks)
            obs = raw._get_observations()["policy"]
            assert obs.shape == (2, 73)
            assert torch.isfinite(obs).all()
            assert torch.all(raw.active_mask == masks)
            assert torch.linalg.vector_norm(obs[:, 28:34], dim=-1).max() < 1e-4
        finally:
            env.close()
