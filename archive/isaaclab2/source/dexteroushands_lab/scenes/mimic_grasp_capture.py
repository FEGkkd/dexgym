"""Capture one live Mimic screw grasp as an unvalidated Reset candidate.

The caller must independently replay the returned state under the same
physics settings. Capturing contact never marks a sample validated.
"""

from __future__ import annotations

from dataclasses import replace

from dexteroushands_lab.scenes.grasp_init import BodyInit, HandInit


def capture_mimic_grasp_sample(env, template, *, source: str,
                              metadata_extra: dict | None = None):
    if env.num_envs != 1 or not env._mimic_backend:
        raise ValueError("Live Mimic grasp capture requires exactly one environment")
    state = env.thread.view.read()

    def body(pose, velocity):
        return BodyInit(tuple(float(x) for x in pose[0].detach().cpu()),
                        tuple(float(x) for x in velocity[0].detach().cpu()))

    def hand(asset, spec, wrist, target):
        root = body(asset.data.root_pose_w.torch, asset.data.root_vel_w.torch)
        order = [asset.joint_names.index(name) for name in spec.joint_names]
        wrist_target = (*wrist.target_pos[0].detach().cpu().tolist(),
                        *wrist.target_quat[0].detach().cpu().tolist())
        return HandInit(
            root.root_pose, root.root_velocity,
            tuple(float(asset.data.joint_pos.torch[0, i]) for i in order),
            tuple(float(asset.data.joint_vel.torch[0, i]) for i in order),
            tuple(float(x) for x in wrist_target),
            tuple(float(target[0, i]) for i in order),
        )

    metadata = {
        **template.metadata,
        "wrist_control_mode": "pose_command_wrist",
        "mimic_spin_velocity_rad_s": float(state.spin_velocity_rad_s[0]),
        "mimic_drive_state": {
            "stick_reference_rad": float(env.thread.drive.stick_reference[0]),
            "sliding": bool(env.thread.drive.sliding[0]),
            "breakaway_seen": bool(env.thread.drive.breakaway_seen[0]),
        },
        "mimic_candidate_unverified": True,
        **(metadata_extra or {}),
    }
    return replace(
        template, mode="stable_pregrasp", validated=False, validation=None,
        source=source,
        left=hand(env.left, env.specs[0], env.wrists[0], env._finger_targets[0]),
        right=hand(env.right, env.specs[1], env.wrists[1], env._finger_targets[1]),
        bottle=body(state.root_pose_w, state.root_velocity_w),
        cap=body(state.cap_pose_w, env.cap.data.root_vel_w.torch),
        thread_initial_angle_rad=float(state.theta_rad[0]),
        thread_initial_engagement_m=(env.bottle_cfg.initial_engagement
                                     - float(state.axial_m[0])),
        metadata=metadata,
    )
