"""Exercise disk RMB decoding through real MuJoCo and SuperMimic inference."""

import argparse
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from ......algorithms.supermimic.dex_reference.config import SuperMimicConfig
from ...p02.model import RobotModel
from ..rmb.build import MODEL_SHA, output_fields
from ..rmb.mapping import packed_geometry
from ..rmb.reader import RobotReferenceBank
from ..rmb.schema import SCHEMA, TICK_NS, add_arrays, empty_geometry, file_sha, write_json, write_partition
from ..task import HISTORY, JOINTS, SUBSTEPS
from ..task.physics import ServoProfile
from ..task.reference import RmbReference
from ..task.smoke import make_task
from .adapter import ImaginationAdapter, ResidualEnvAdapter
from .catalog import FrameCatalog
from .train import PreActorLearner


FRAMES = 32
COLLECTION_STEPS = 20
ROOTS = ("synthetic_fit", "synthetic_holdout")


def write_fixture(directory, robot):
    geometry = robot.geometry(robot.home)
    # Explicit fictional deviation distinguishes independent targets from nominal FK.
    displacement = np.array([.002, -.001, .003])
    mapped = packed_geometry(geometry.wrists + displacement, geometry.rotations, geometry.local,
                             geometry.landmarks + displacement)
    arrays = output_fields(FRAMES)
    arrays.update(tick_ns=np.arange(FRAMES, dtype="<i8") * TICK_NS,
                  inputs_stream_age_ns=np.zeros((FRAMES, 4), dtype="<i8"),
                  inputs_stream_valid=np.ones((FRAMES, 4), dtype=bool),
                  audit_clock_eligible=np.ones((FRAMES, 4), dtype=bool),
                  inputs_safety_permission=np.ones(FRAMES, dtype=bool))
    arrays["nominal_q_nominal"][:] = robot.home
    for name in ("nominal_joint_valid", "nominal_velocity_valid", "domain_mapped_valid",
                 "domain_nominal_legal", "inputs_source_changed", "mapped_geometry_velocity_valid"):
        arrays[name][:] = True
    arrays["audit_generation"][:] = 0
    arrays["audit_segment"][:] = 0
    fields = empty_geometry(FRAMES)
    for name, value in mapped.items():
        fields[name][:] = value
        fields[f"{name}_mask"][:] = True
    add_arrays(arrays, "mapped", fields)
    parts = []
    for name in ROOTS:
        relative = f"partitions/Train/{name}"
        parts.append({"key": name, "root": name, "split": "Train", "relative_path": relative,
                      "arrays": write_partition(directory / relative, arrays)})
    manifest = directory / "rmb_manifest.json"
    write_json(manifest, {"schema": SCHEMA, "qualification": "SMOKE_ONLY", "partitions": parts})
    return file_sha(manifest)


def smoke_catalog(directory, robot, device):
    identity = write_fixture(directory, robot)
    try:
        RobotReferenceBank(directory, expected_sha=identity)
    except ValueError as error:
        if "not eligible" not in str(error):
            raise
    else:
        raise AssertionError("Synthetic manifest gained formal RMB admission")
    bank = RobotReferenceBank(directory, expected_sha=identity, require_qualified=False)
    try:
        RmbReference(bank, device=device)
    except ValueError as error:
        if "Unqualified RMB" not in str(error):
            raise
    else:
        raise AssertionError("Synthetic manifest gained production reference access")
    partitions = {item.key: item for item in bank.partitions("Train")}
    generations = {name: index for index, name in enumerate(ROOTS)}
    # Explicit diagnostic provider only. The production constructor above stays closed.
    provider = SimpleNamespace(device=device, partitions=partitions,
                               current=partial(RmbReference.pack_current, partitions, generations, device))
    return FrameCatalog.from_rmb(provider, fit_roots=ROOTS[:1], holdout_roots=ROOTS[1:]), identity


@torch.no_grad()
def run(model, device):
    if file_sha(model) != MODEL_SHA:
        raise ValueError("Synthetic preflight requires the signed S0-5 model")
    torch.manual_seed(49)
    robot = RobotModel(model)
    task, _, _ = make_task(robot, 2, ServoProfile(), "B54", device)
    with TemporaryDirectory(prefix="s05-rmb-synthetic-") as temporary:
        directory = Path(temporary)
        catalog, identity = smoke_catalog(directory, robot, device)
        env = ResidualEnvAdapter(task, catalog, max_episode=12)
        adapter = ImaginationAdapter(env, smoke=True)
        cfg = SuperMimicConfig(seed=49, max_iterations=1, buffer_steps=128, warmup_steps=COLLECTION_STEPS, horizon=8,
                               wm_hidden=(128, 128, 128), amp=False, optimizer="adam",
                               compile=False, compile_policy=False, profile_training=False, action_noise=0.)
        engine = PreActorLearner(adapter, cfg, directory / "engine")
        initial = engine.obs.clone()
        if initial.shape != (2, HISTORY * task.builder.frame_dim) or initial[:, :-task.builder.frame_dim].count_nonzero():
            raise AssertionError("Actor initial history does not match the real task ABI")
        decoded = env.task.pending.target
        live = env.task.pending.feedback.geometry
        if (decoded.geometry.tips - live.tips).abs().amax() < .001:
            raise AssertionError("Fixture does not distinguish target from physical FK")
        engine.collect(COLLECTION_STEPS)
        rows = engine.buffer.chronological()
        full_steps = int(engine.buffer.data["full_step"][rows].sum())
        if engine.sim_steps != 2 * COLLECTION_STEPS or full_steps != engine.sim_steps:
            raise AssertionError("Synthetic collection lost a physical interval")
        if not torch.allclose(env.last.u, env.last.applied.sum(1) / SUBSTEPS):
            raise AssertionError("Replay model input differs from applied physical control")
        for holdout in (False, True):
            window = engine.buffer.sample(2, window=8, holdout=holdout)
            if (not (window["env"] == int(holdout)).all() or not window["full_step"].all()
                    or not (window["clock_tick"].diff(dim=1) == TICK_NS).all()):
                raise AssertionError("The source role or contiguous physical window leaked")
            x = window["x"][:, 0]
            aux = window["aux"][:, 0]
            sim = adapter.simulator_context[window["env"]]
            for u, command in zip(window["u"].unbind(1), window["c"].unbind(1), strict=True):
                x, contact = engine.step_wm(x, u, adapter.dynamics(x, sim, aux))
                if (x.shape != (2, 2 * JOINTS) or contact.shape != (2, adapter.contact_dim)
                        or not torch.isfinite(x).all() or not torch.isfinite(contact).all()):
                    raise AssertionError("World Model inference outputs do not match the actual replay ABI")
                # The next interval uses the real safe issued command, not the raw action.
                aux = adapter.replay_auxiliary(aux, command)
        if engine.update_counts or any(optimizer.state for optimizer in engine.optimizers.values()):
            raise AssertionError("Synthetic inference performed an optimizer update")
        return {"scope": "DIAGNOSTIC_ONLY_SYNTHETIC_RMB_PHYSICAL_INFERENCE",
                "model_sha256": MODEL_SHA, "synthetic_manifest_sha256": identity,
                "device": device, "torch_version": torch.__version__, "physics_backend": type(task.physics).__name__,
                "actor_observation_shape": list(initial.shape), "action_width": JOINTS, "causal_history_frames": HISTORY,
                "physics_substeps_per_tick": SUBSTEPS, "physical_transitions": engine.sim_steps,
                "physical_full_steps": full_steps,
                "dynamics_state_width": adapter.state_dim, "world_model_contact_width": adapter.contact_dim,
                "world_model_replay_inference_horizon": 8, "isolated_fit_and_holdout": True,
                "production_bank_admission_denied": True, "optimizer_updates": 0,
                "training_artifact_written": False, "formal_training_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    args = parser.parse_args()
    result = run(args.model, args.device)
    write_json(args.out, result)
    print(f"{result['scope']} physical={result['physical_transitions']} sha256={file_sha(args.out)}")


if __name__ == "__main__":
    main()
