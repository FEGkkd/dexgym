"""Causal hand-only TELEDEXTER task adapter; reference joints never enter Actor."""

from dataclasses import asdict

import torch

from dexgym.algorithms.supermimic.core.specs import TaskSpec
from dexgym.algorithms.supermimic.core.transition import TaskState, TaskStep

from .logic import build_right_hand_observation, teledexter_hand_dwell
from .subgoals import HandGoals, advance_hand_goals, paper_hand_reward, unsafe_hand_state


class Hand21Task:
    termination_mode = "learned"

    def __init__(self, robot, bank, cfg):
        self.robot, self.bank, self.cfg = robot, bank, cfg
        self.spec = TaskSpec(
            "hand21_teledexter_s0_v1",
            125,
            10,
            15,
            1,
            11,
            162,
            "current_5_tip_xyz_wrist_v1",
            "ppo125_bounded_previous_action_v1",
            "state_landmarks_goal_landmarks_counters_clock_v1",
            termination_feature_dim=26,
        )
        self.reward_head = None

    @staticmethod
    def physical_task_state(env):
        clock = env.episode_length_buf.detach().clone().long()
        zero = torch.zeros_like(clock)
        return TaskState(zero, zero.clone(), clock, HandGoals.from_env(env).pack())

    def goal(self, state, task_state):
        goals = HandGoals.unpack(task_state.data)
        return self.bank.fingertips(goals.motion_ids, goals.goal_frame).flatten(1)[:, None]

    def observation(self, state, previous_action, task_state):
        return build_right_hand_observation(
            state[:, :20],
            state[:, 20:40],
            self.robot.fk(state[:, :20]),
            self.goal(state, task_state).reshape(-1, 5, 3),
            self.robot.bounded_action(previous_action),
            command_q=state[:, 40:60],
        )

    def goal_landmarks(self, task_state):
        goals = HandGoals.unpack(task_state.data)
        return self.robot.fk.landmarks(self.bank.joints(goals.motion_ids, goals.goal_frame))

    def critic_features(self, state, task_state):
        return torch.cat(
            (
                state,
                self.robot.fk.landmarks(state[:, :20]).flatten(1),
                self.goal_landmarks(task_state).flatten(1),
                task_state.data,
                task_state.frame[:, None].float(),
            ),
            -1,
        )

    def exact_reward(self, next_state, old_task_state, *, fingertip_errors=None):
        old = HandGoals.unpack(old_task_state.data)
        landmarks = self.robot.fk.landmarks(next_state[:, :20])
        errors, _, reached, _ = teledexter_hand_dwell(
            landmarks[:, 0],
            self.goal(next_state, old_task_state).reshape(-1, 5, 3),
            old.dwell,
            old.dwell_required,
            tolerance_m=self.cfg.goal_tolerance_m,
        )
        if fingertip_errors is not None:
            # True physics measurements are authoritative at a discontinuous
            # threshold: sub-micron FK rounding can change the sparse indicator.
            errors = fingertip_errors
            reached = (errors < self.cfg.goal_tolerance_m).all(-1) & (old.dwell + 1 >= old.dwell_required)
        return paper_hand_reward(
            errors,
            reached,
            old.goal_span,
            landmarks,
            self.goal_landmarks(old_task_state),
            dense_scale=self.cfg.reward_dense_scale,
            time_cost=self.cfg.reward_time_cost,
        )

    def reach_features(self, next_state, old_task_state):
        goals = HandGoals.unpack(old_task_state.data)
        errors = torch.linalg.vector_norm(
            self.robot.fk(next_state[:, :20]) - self.goal(next_state, old_task_state).reshape(-1, 5, 3), dim=-1
        )
        return torch.cat(
            (errors / self.cfg.goal_tolerance_m, goals.dwell[:, None] / 15.0, goals.dwell_required[:, None] / 15.0), -1
        )

    def transition(self, state, next_state, action, task_state, previous_action):
        old = HandGoals.unpack(task_state.data)
        landmarks = self.robot.fk.landmarks(next_state[:, :20])
        result = advance_hand_goals(landmarks[:, 0], old, self.bank, self.cfg)
        # This is a supervised reward model of the exact event, never a change
        # to physical success labels. Dense geometry/score remain analytic.
        if self.reward_head is None:
            raise RuntimeError("Attach the trained world model reach head before imagination")
        probability = self.reward_head(self.reach_features(next_state, task_state)).squeeze(-1).sigmoid()
        reward = paper_hand_reward(
            result.errors,
            probability,
            old.goal_span,
            landmarks,
            self.goal_landmarks(task_state),
            dense_scale=self.cfg.reward_dense_scale,
            time_cost=self.cfg.reward_time_cost,
        )
        arrival = TaskState(task_state.task_id, task_state.motion_id, task_state.frame + 1, result.goals.pack())
        return TaskStep(arrival, reward)

    def hard_termination(self, state, task_state):
        goals = HandGoals.unpack(task_state.data)
        return (
            unsafe_hand_state(state[:, :20], state[:, 20:40], self.robot.fk.limits)
            | (goals.fail_frames > goals.fail_budget)
            | goals.completed
        )

    def termination_features(self, state, task_state):
        scales = state.new_tensor([1000.0, 1000.0, 40.0, 15.0, 15.0, 300.0, 300.0, 25600.0, 900.0, 100.0, 1.0])
        return torch.cat((self.goal(state, task_state)[:, 0], task_state.data / scales), -1)

    def valid_imagination_start(self, task_state, horizon):
        return (task_state.frame + horizon < round(self.cfg.episode_length_s * 30)) & ~HandGoals.unpack(
            task_state.data
        ).completed

    @staticmethod
    def reward_upper_bound():
        return 375.0 + 0.1 * (4.2 + 0.6 + 0.4) - 0.1

    def checkpoint_contract(self):
        names = (
            "goal_tolerance_m",
            "curriculum_stride_min",
            "curriculum_stride_max",
            "curriculum_anneal_steps",
            "curriculum_sigma_min",
            "reference_start_fraction",
            "dwell_min_steps",
            "dwell_max_steps",
            "cross_trajectory_probability",
            "failure_tolerance_scale",
            "cross_trajectory_failure_budget",
            "episode_length_s",
        )
        return {
            "version": "hand_s0_task_v1",
            "spec": asdict(self.spec),
            "rules": {name: getattr(self.cfg, name) for name in names},
            "reward": "paper_hand_mean_knuckle_error_v1",
            "reward_model": "real_reach_BCE_7_64_64_1_unweighted_v1",
            "reference_in_actor": "current_fingertips_only",
            "nominal_fixed_wrist": True,
        }
