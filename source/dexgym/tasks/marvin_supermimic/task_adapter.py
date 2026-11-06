"""Marvin six-link tracking MDP plugged into the common SuperMimic trainer."""

import torch

from dexgym.algorithms.supermimic.imagination import critic_td_lambda_batch, imagined_rollout
from dexgym.algorithms.supermimic.rewards import REWARD_PROFILES
from dexgym.algorithms.supermimic.termination import TerminationThresholds
from dexgym.algorithms.supermimic.transition import arrival_reference_features

from .contact_audit import unsafe_arm_states


class MarvinTrackingTask:
    adapter_version = "marvin_tracking_v2"
    observation_schema = "arm_proprio_history_10x56_v1"
    goal_schema = "six_link_future_10x90_v1"

    def __init__(self, reference, robot, *, reward_profile: str = "paper_multilink",
                 termination_mode: str = "analytic",
                 thresholds: TerminationThresholds = TerminationThresholds(),
                 termination_module=None):
        if reward_profile not in REWARD_PROFILES:
            raise ValueError("Unknown tracking reward profile")
        if termination_mode not in ("analytic", "learned"):
            raise ValueError("Unknown termination mode")
        self.reference = reference
        self.robot = robot
        self.reward_profile = reward_profile
        self.termination_mode = termination_mode
        self.thresholds = thresholds
        self.termination_module = termination_module

    def arrival_goal(self, frame: torch.Tensor) -> torch.Tensor:
        return arrival_reference_features(self.reference, frame)

    def future_goal(self, frame: torch.Tensor) -> torch.Tensor:
        return self.reference.future_goal_at(frame)

    def hard_termination(self, q: torch.Tensor, qd: torch.Tensor, tcp_error: torch.Tensor,
                         contacts: torch.Tensor) -> torch.Tensor:
        mapping = self.robot.action_map
        return unsafe_arm_states(q, qd, mapping.lower, mapping.upper, tcp_error, contacts,
                                 thresholds=self.thresholds)

    def imagine(self, actor, world_model, critic, start: dict, *, horizon: int) -> dict:
        return imagined_rollout(actor, world_model, critic, self.robot.fk,
                                self.robot.action_map, self.reference, start,
                                horizon=horizon, weights=REWARD_PROFILES[self.reward_profile],
                                termination_mode=self.termination_mode,
                                termination_thresholds=self.thresholds,
                                bootstrap_weight=0.0 if horizon == 1 else 1.0,
                                termination_module=self.termination_module)

    def critic_batch(self, critic, rollout: dict):
        return critic_td_lambda_batch(critic, self.reference, rollout)
