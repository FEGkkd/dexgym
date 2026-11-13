"""Export a standalone CPU controller with the same observation and command projection."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import torch
from torch import nn

from .bag import digest
from .domain import Domain
from .evaluate_residual import checkpoint_geometry, load_actor, task_for_checkpoint
from .intent import GOAL_VERSION
from .prepare import write_json
from .residual import DT, N, ReferenceBank, encode_intent_offsets, encode_policy_observation, project_command


class Controller(nn.Module):
    def __init__(self, task, actor):
        super().__init__()
        self.actor = actor
        for name in ("home", "lo", "hi", "scale", "slew"):
            self.register_buffer(name, getattr(task, name))

    def forward(self, q, qd, reference, previous_reference, previous_command, previous_action):
        observation = encode_policy_observation(q, qd, reference, previous_reference,
                                                previous_command, previous_action, self.home)
        action = self.actor(observation)
        command = project_command(reference, action, previous_command, self.lo, self.hi, self.scale, self.slew)
        return command, action


class IntentController(Controller):
    def forward(self, q, qd, reference, previous_reference, previous_command, previous_action,
                human_wrist, human_rotation, human_local_tip, reference_wrist, reference_rotation, reference_local_tip):
        observation = encode_policy_observation(q, qd, reference, previous_reference,
                                                previous_command, previous_action, self.home)
        intent = encode_intent_offsets(human_wrist, human_rotation, human_local_tip,
                                      reference_wrist, reference_rotation, reference_local_tip)
        action = self.actor(torch.cat((observation, intent), dim=-1))
        command = project_command(reference, action, previous_command, self.lo, self.hi, self.scale, self.slew)
        return command, action


@torch.no_grad()
def export(args):
    torch.set_num_threads(1)
    model_info = json.loads(args.manifest.read_text())["provenance"]["robot_model"]
    if digest(model_info["path"]) != model_info["sha256"]:
        raise ValueError("Reference asset checksum mismatch")
    model = mujoco.MjModel.from_xml_path(model_info["path"])
    meta = json.loads((args.checkpoint.parent / "provenance.json").read_text())
    intent_model = checkpoint_geometry(meta) if meta.get("reward_profile") == "human_intent" else None
    bank = ReferenceBank(args.manifest, "validation", model, "cpu", intent_model=intent_model)
    arm = torch.as_tensor([name.startswith("Joint") for name in bank.names])
    task = task_for_checkpoint(args.checkpoint, SimpleNamespace(device="cpu"), model, bank, Domain(1, "cpu", 0, False, arm))
    actor = load_actor(args.checkpoint, task)
    controller = (IntentController if task.human_intent else Controller)(task, actor).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    frames = torch.linspace(0, bank.frames - 1, 512).long()
    previous_ref = bank.q[frames]
    generator = torch.Generator().manual_seed(901)
    q_noise, v_noise, cmd_noise, previous_action = torch.randn(4, len(frames), N, generator=generator)
    q = (previous_ref + .03 * q_noise).clamp(task.lo, task.hi)
    qd = bank.qd[frames] + .10 * v_noise
    previous_cmd = (previous_ref + .04 * cmd_noise).clamp(task.lo, task.hi)
    previous_action = previous_action.tanh()
    x = torch.cat((q, qd), -1)
    memory = {"frame": frames, "cmd": previous_cmd, "action": previous_action,
              "delay": torch.zeros(len(q), dtype=torch.bool)}
    inputs = (q, qd, bank.q[bank.next(frames)], previous_ref, previous_cmd, previous_action)
    input_order = ["q_measured_rad", "qd_measured_rad_s", "q_reference_rad", "previous_reference_rad",
                   "previous_projected_command_rad", "previous_normalized_action"]
    if task.human_intent:
        goal = bank.next(frames)
        human = bank.intent_targets
        inputs += (human.wrist[goal], human.rotation[goal], human.local[goal, ..., -1, :],
                   *(value[goal] for value in task.intent_reference))
        input_order += ["human_wrist_world_m", "human_wrist_world_rotation", "human_tip_palm_m",
                        "reference_wrist_world_m", "reference_wrist_world_rotation", "reference_tip_palm_m"]
    expected_action = actor(task.policy_observation(x, memory, torch.zeros_like(x)))
    expected_command, _ = task.control_target(expected_action, memory)
    traced = torch.jit.freeze(torch.jit.trace(controller, tuple(item[:1] for item in inputs)))
    path = args.output / "controller.pt"
    traced.save(str(path))
    loaded = torch.jit.load(str(path), map_location="cpu")
    actual_command, actual_action = loaded(*inputs)
    for actual, expected in ((actual_action, expected_action), (actual_command, expected_command)):
        if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Exported controller deviates from the training task")
    if ((actual_command < task.lo - 1e-6) | (actual_command > task.hi + 1e-6)
            | ((actual_command - previous_cmd).abs() > task.slew + 1e-6)).any():
        raise RuntimeError("Exported controller violated the command limits")
    write_json(args.output / "contract.json", {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest(args.checkpoint),
        "controller_sha256": digest(path), "reference_manifest_sha256": bank.manifest_sha256,
        "joint_names": bank.names, "dtype": "float32", "input_shape": ["batch", N], "control_dt_s": DT,
        "reward_profile": task.reward_profile, "actor_observation_dim": task.policy_dim,
        "input_order": input_order,
        "output_order": ["projected_command_rad", "normalized_action"],
        "position_lower_rad": task.lo.tolist(), "position_upper_rad": task.hi.tolist(),
        "residual_scale_rad": task.scale.tolist(), "command_delta_limit_rad": task.slew.tolist(),
        "parity_frames": len(frames), "action_max_error": float((actual_action - expected_action).abs().max()),
        "command_max_error_rad": float((actual_command - expected_command).abs().max()),
        **({"goal_version": GOAL_VERSION,
            "input_shapes": {name: ["batch", *value.shape[1:]] for name, value in zip(input_order, inputs, strict=True)}}
           if task.human_intent else {}),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
