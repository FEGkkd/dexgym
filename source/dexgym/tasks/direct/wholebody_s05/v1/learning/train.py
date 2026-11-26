"""Run real S0-5 C0 collection and SuperMimic world-model updates.

Smoke is a physical integration check and can never authorize or produce an
Actor candidate. Production fitting requires the frozen A and B inputs and
explicitly reserved resources. No ROS or hardware drivers are imported.
"""
import argparse
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from ......algorithms.supermimic.dex_reference.config import SuperMimicConfig
from ......algorithms.supermimic.dex_reference.engine import SuperMimicEngine
from ...p02.model import RobotModel
from ..rmb.reader import RobotReferenceBank
from ..task.reference import RmbReference
from ..task.smoke import make_task
from ..task.physics import ServoProfile, signature
from .adapter import ImaginationAdapter, ResidualEnvAdapter
from .catalog import FrameCatalog

DESIGN_FREEZE = "cd5bff29a5b5f683fae328898aa9060b40fc4be00e193a22d9bf18cb2d10b60b"


def digest(file):
    with Path(file).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def json_file(file, data):
    file = Path(file)
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(file)


def as_numbers(metrics):
    result = {}
    for key, value in metrics.items():
        their = value.detach().cpu().item() if torch.is_tensor(value) else value
        if isinstance(their, (float, int)) and not np.isfinite(their):
            raise FloatingPointError(f"Non-finite metric {key}")
        result[key] = their
    return result


class PreActorLearner(SuperMimicEngine):
    """Fail closed regardless of learner iteration or inherited configuration."""

    def __init__(self, task, cfg, output):
        self.update_counts = Counter()
        super().__init__(task, cfg, output)

    def policy_update(self):
        raise RuntimeError("C0 cannot update Actor or Critic")

    def optimize(self, name, loss):
        if name not in {"wm", "fail_head"}:
            raise RuntimeError("C0 optimizer cannot update " + name)
        super().optimize(name, loss)
        self.update_counts[name] += 1


def configuration(args):
    smoke = args.smoke
    return SuperMimicConfig(
        seed=49, max_iterations=args.iterations, buffer_steps=args.buffer_steps,
        warmup_steps=args.warmup, utd=.5, wm_only_iters=350, wm_window=8,
        wm_batches=1 if smoke else 3, wm_batch=8 if smoke else 768,
        wm_hidden=(128, 128, 128) if smoke else (1280, 1280, 1280),
        pi_batch=384, horizon=8, actor_bootstrap_weight=.65, max_step_reward=1.,
        critic_epochs=3, gamma=.99, lam=.95, lr=1.1e-4, actor_lr=4e-5,
        optimizer="adam" if args.device == "cpu" else "fused-adam",
        action_noise=0., failure_head=False, amp=False,
        compile=False, compile_policy=False, profile_training=not smoke,
        contact_weight=.2, contact_partial_intervals=True, wm_failure_frac=0.,
        save_interval=1000, log_interval=10 if not smoke else 1,
    )


def read_receipt(path, expected_sha, component):
    if not path or not expected_sha or digest(path) != expected_sha:
        raise ValueError(f"missing or corrupted {component} gate receipt")
    data = json.loads(Path(path).read_bytes())
    if component == "V1-A":
        stage_matches = data.get("gate") == "G-A" and data.get("stage", component) == component
        reviewer = data.get("independent_reviewer")
        authors = {data.get("implementation_agent"), *data.get("core_authors", [])}
    else:
        stage_matches = data.get("stage") == component
        reviewer = data.get("reviewer")
        authors = set(data.get("core_authors", []))
    if (not stage_matches or data.get("verdict") != "TECHNICAL_PASS"
            or data.get("frozen_design_sha256") != DESIGN_FREEZE or not reviewer or reviewer in authors):
        raise ValueError(f"wrong scope or missing independent reviewer for {component}")
    return data


def production_catalog(args):
    ga = read_receipt(args.g_a, args.g_a_sha, "V1-A")
    gb = read_receipt(args.g_b, args.g_b_sha, "V1-B")
    if not args.bank or not args.bank_sha or ga.get("manifest_sha256") != args.bank_sha:
        raise ValueError("A did not qualify this RMB identity")
    if gb.get("rmb_manifest_sha256") != args.bank_sha or gb.get("model_sha256") != digest(args.model):
        raise ValueError("B did not qualify these physical assets and input bank")
    if args.device != "cuda:0":
        raise ValueError("production C0 requires the independently qualified MuJoCo Warp backend")
    if args.environments not in {64, 256, 1024, 2048, 4096} or args.environments > gb.get("qualified_environment_count", 0):
        raise ValueError("requested physical environment count lacks bounded qualification")
    if not args.root_plan or not args.root_plan_sha or digest(args.root_plan) != args.root_plan_sha:
        raise ValueError("a separately preregistered root allocation is required")
    roles = json.loads(args.root_plan.read_bytes())
    fit, tune, qualification = (set(roles[k]) for k in ("fit_roots", "tune_roots", "qualification_roots"))
    if len(fit) != 18 or len(tune) != 3 or len(qualification) != 3 or len(fit | tune | qualification) != 24:
        raise ValueError("expected 18 disjoint fit roots, 3 tune and 3 independent qualification")
    bank = RobotReferenceBank(args.bank, expected_sha=args.bank_sha,
                              review_receipt=args.g_a, expected_review_sha=args.g_a_sha)
    provider = RmbReference(bank, device=args.device)
    if {row.root for row in provider.partitions.values()} != fit | tune | qualification:
        raise ValueError("root allocation does not cover this RMB Train split")
    # Independent qualification roots are deliberately never stored in the
    # trainee's catalog, even as read-only holdout lanes.
    provider.partitions = {key: row for key, row in provider.partitions.items() if row.root not in qualification}
    return FrameCatalog.from_rmb(provider, fit_roots=fit, holdout_roots=tune)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--environments", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=66)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--buffer-steps", type=int, default=128, choices=[128, 256, 512, 1024, 1800])
    parser.add_argument("--reserved-physical-transitions", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--g-a", type=Path)
    parser.add_argument("--g-a-sha")
    parser.add_argument("--g-b", type=Path)
    parser.add_argument("--g-b-sha")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--bank-sha")
    parser.add_argument("--root-plan", type=Path)
    parser.add_argument("--root-plan-sha")
    args = parser.parse_args()
    if args.iterations < 1 or args.environments < 2:
        parser.error("need positive iterations and disjoint source lanes")
    upper = args.environments * (args.warmup + 2 * args.iterations)
    if args.reserved_physical_transitions < upper:
        raise ValueError("insufficient non-reusable physical transition reservation")
    if args.smoke and (args.environments > 64 or args.iterations > 4):
        raise ValueError("smoke may not be used as a training entry point")
    np.random.seed(49)
    random.seed(49)
    torch.manual_seed(49)
    cfg = configuration(args)
    catalog = None if args.smoke else production_catalog(args)
    args.output.mkdir(parents=True, exist_ok=False)
    robot = RobotModel(args.model)
    task, _, _ = make_task(robot, args.environments, ServoProfile(), "B54", args.device)
    if catalog is None:
        catalog = FrameCatalog.smoke(task)
    env = ResidualEnvAdapter(task, catalog, holdout_envs=max(1, args.environments // 8))
    adapter = ImaginationAdapter(env, smoke=args.smoke)
    engine = PreActorLearner(adapter, cfg, args.output)
    receipt = {
        "scope": "PHYSICAL_SYNTHETIC_PREACTOR_SMOKE" if args.smoke else "C0_WORLD_MODEL_FITTING_ONLY",
        "design_freeze_sha256": DESIGN_FREEZE, "model_sha256": digest(args.model),
        "physical_profile": signature(robot, task.physics.profile),
        "rmb_manifest_sha256": args.bank_sha, "root_plan_sha256": args.root_plan_sha,
        "g_a_receipt_sha256": args.g_a_sha, "g_b_receipt_sha256": args.g_b_sha,
        "physical_backend": type(task.physics).__name__, "environments": args.environments,
        "reserved_physical_transitions": args.reserved_physical_transitions,
        "expected_hard_upper_bound": upper, "actor_locked": True, "critic_locked": True,
    }
    json_file(args.output / "identity.json", receipt)
    start = perf_counter()
    with (args.output / "metrics.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
        engine.collect(args.warmup)
        for i in range(args.iterations):
            engine.iteration = i
            physics = engine.collect(2)
            training = {}
            for _ in range(cfg.wm_batches):
                training = engine.fit_world_model()
            held = engine.fit_world_model(holdout=True)
            record = {
                "learner_iteration": i + 1, "iteration": i + 1, "physical_transitions": engine.sim_steps,
                "imagined_transitions": 0, "environments": args.environments,
                "elapsed_seconds": perf_counter() - start, "wm_update_count": engine.update_counts["wm"],
                "actor_update_count": 0, "critic_update_count": 0,
                **as_numbers(physics), **as_numbers(training),
                **{"tune/" + key: value for key, value in as_numbers(held).items()},
            }
            line = json.dumps(record, allow_nan=False)
            stream.write(line + "\n")
            print(line, flush=True)
            if (i + 1) % cfg.save_interval == 0:
                engine.save(args.output / f"preactor_{i + 1:06d}.pt", next_iteration=i + 1)
    checkpoint = args.output / "preactor_last.pt"
    engine.save(checkpoint, next_iteration=args.iterations)
    receipt.update(
        state="SMOKE_ONLY" if args.smoke else "FIT_COMPLETE_REQUIRES_INDEPENDENT_QUALIFICATION",
        physical_transitions=engine.sim_steps, imagined_transitions=0, learner_iterations=args.iterations,
        wm_updates=engine.update_counts["wm"], failure_head_updates=engine.update_counts["fail_head"],
        actor_updates=0, critic_updates=0, checkpoint=checkpoint.name, checkpoint_sha256=digest(checkpoint),
        replay_allocated_bytes=engine.buffer.allocated_bytes, elapsed_seconds=perf_counter() - start,
        resume_class="RESUME_REFILL",
        wm_gate="NOT_QUALIFIED",
        **env.report_exploration(),
    )
    json_file(args.output / "fit_report.json", receipt)


if __name__ == "__main__":
    main()
