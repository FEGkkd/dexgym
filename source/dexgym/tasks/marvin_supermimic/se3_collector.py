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

    def __call__(self, clean_action):
        if clean_action.shape != self.state.shape or not bool(torch.isfinite(clean_action).all()):
            raise ValueError("Actor action differs from temporal exploration contract")
        innovation = torch.randn(self.state.shape, device=self.device,
                                 generator=self.generator)
        self.state = (self.rho * self.state
                      + self.sigma * (1 - self.rho**2)**0.5 * innovation)
        return clean_action + self.state

    def reset(self, env_ids):
        self.state[env_ids] = 0


class SE3CorrelatedExplorationCollector(CoreNewtonCollector):
    """Core transitions remain Actor-sourced; Replay stores the executed raw action."""

    def __init__(self, env, robot, task, *, seed, rho=0.95, sigma=0.8,
                 exploration_enabled=True):
        super().__init__(env, robot, task, action_noise_std=0)
        self.exploration_enabled = exploration_enabled
        self.exploration = TemporalJointExploration(
            self.num_envs, robot.spec.action_dim, device=self.native.device,
            seed=seed, rho=rho, sigma=sigma)

    @torch.no_grad()
    def step(self, actor, *, policy="actor"):
        if policy == "actor" and self.exploration_enabled:
            original_actor = actor

            def actor(history, goal):
                return self.exploration(original_actor(history, goal))

        transition = super().step(actor, policy=policy)
        done = torch.nonzero(transition.done, as_tuple=False).flatten()
        if len(done):
            self.exploration.reset(done)
        return transition
