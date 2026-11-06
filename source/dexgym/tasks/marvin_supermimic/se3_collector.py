"""Time-correlated, teacher-free physical exploration for the SE(3) task."""

import torch

from .core_collector import CoreNewtonCollector


class TemporalJointExploration:
    """Seeded AR(1) perturbation of raw Actor actions, before the delta map."""

    def __init__(self, num_envs, action_dim, *, device, seed, rho=0.95, sigma=0.8):
        if not 0 <= rho < 1 or sigma < 0:
            raise ValueError("Exploration correlation or scale is invalid")
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.state = torch.zeros((num_envs, action_dim), device=self.device)
        self.rho, self.sigma = rho, sigma

    def __call__(self, clean_action, *, active_mask=None):
        if clean_action.shape != self.state.shape or not bool(torch.isfinite(clean_action).all()):
            raise ValueError("Actor action differs from temporal exploration contract")
        if active_mask is None:
            active_mask = torch.ones(len(clean_action), dtype=torch.bool, device=self.device)
        if (active_mask.shape != (len(clean_action),)
                or active_mask.device != self.device or active_mask.dtype != torch.bool):
            raise ValueError("Exploration lane mask differs from environment contract")
        innovation = torch.randn(self.state.shape, device=self.device,
                                 generator=self.generator)
        self.state = (self.rho * self.state
                      + self.sigma * (1 - self.rho**2)**0.5 * innovation)
        self.state = torch.where(active_mask[:, None], self.state,
                                 torch.zeros_like(self.state))
        return clean_action + self.state

    def reset(self, env_ids):
        self.state[env_ids] = 0


class SE3CorrelatedExplorationCollector(CoreNewtonCollector):
    """Store deterministic Actor lanes as source 0 and noisy Actor lanes as 3."""

    def __init__(self, env, robot, task, *, seed, rho=0.95, sigma=0.8,
                 exploration_enabled=True, exploration_env_count=None,
                 train_env_count=None, validation_exploration_env_count=0):
        super().__init__(env, robot, task, action_noise_std=0)
        self.exploration_enabled = exploration_enabled
        self.exploration_env_count = (self.num_envs if exploration_env_count is None
                                      else int(exploration_env_count))
        if not 0 <= self.exploration_env_count <= self.num_envs:
            raise ValueError("Exploratory environment count is outside the collector")
        env_index = torch.arange(self.num_envs, device=self.native.device)
        self.exploration_mask = env_index < self.exploration_env_count
        if validation_exploration_env_count:
            if (train_env_count is None or not 0 < self.exploration_env_count < train_env_count
                    or not 0 < validation_exploration_env_count < self.num_envs - train_env_count):
                raise ValueError("Dual-distribution validation lanes are invalid")
            self.exploration_mask |= ((env_index >= train_env_count)
                                      & (env_index < train_env_count
                                         + validation_exploration_env_count))
        self.exploration = TemporalJointExploration(
            self.num_envs, robot.spec.action_dim, device=self.native.device,
            seed=seed, rho=rho, sigma=sigma)

    @torch.no_grad()
    def step(self, actor, *, policy="actor"):
        if policy == "actor" and self.exploration_enabled:
            original_actor = actor

            def actor(history, goal):
                return self.exploration(original_actor(history, goal),
                                        active_mask=self.exploration_mask)

        transition = super().step(actor, policy=policy)
        if policy == "actor" and self.exploration_enabled:
            # 0 means the current deterministic Actor; 3 is explicitly noisy
            # Actor exploration. Trust and Actor starts select source 0.
            transition.source_id[self.exploration_mask] = 3
        done = torch.nonzero(transition.done, as_tuple=False).flatten()
        if len(done):
            self.exploration.reset(done)
        return transition
