"""Validate the user-authorized, resource-only overlay against the frozen v2.0.1 design."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

BASELINE = "490770e1df45832d0693380563152bd19eefa3f4"
FROZEN_SHA = "cd5bff29a5b5f683fae328898aa9060b40fc4be00e193a22d9bf18cb2d10b60b"
ATTACHMENT_SHA = "77b5118f392450eb7c3123e1206f9e89aa4e5ba630e0eaee781cc9e7df2a7867"
REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "docs/s0-5/campaign/execution_config.json"
DCR = REPO / "docs/s0-5/dcr/DCR-007_campaign_training_scale_monitoring.md"
LIMITS = {
    "train_total": 280_000_000, "train_primary_total": 250_000_000,
    "train_static_bucket": 50_000_000, "train_precursor": 10_000_000,
    "train_dynamic": 100_000_000, "train_mixed": 100_000_000,
    "train_d2": 15_000_000, "train_e2": 15_000_000,
    "eval_total": 10_000_000, "eval_c0": 1_500_000, "eval_training_watch": 5_500_000,
    "eval_d0": 600_000, "eval_d1": 400_000, "eval_d2_e": 1_400_000, "eval_d3": 600_000,
    "gpu_count": 1, "gpu_seconds": 172_800, "elapsed_seconds": 259_200,
    "resource_wait_seconds": 86_400, "persistent_bytes": 250 * (1024 ** 3),
}
PROHIBITIONS = {"REAL_ROBOT_ACTUATION", "REAL_DRIVERS", "FORMAL_PHYSICAL_SCORING", "INDEPENDENT_FINAL_TEST"}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def expect(condition, message):
    if not condition:
        raise ValueError(message)


def validate(config, frozen_blob):
    expect(sha(frozen_blob) == FROZEN_SHA, "historical freeze mismatch")
    frozen = json.loads(frozen_blob)
    auth = config["authorization"]
    expect(config["design_commit"] == BASELINE and config["design_freeze_sha256"] == FROZEN_SHA,
           "unapproved design")
    expect(auth["attachment_sha256"] == ATTACHMENT_SHA, "unrecognized authorization")
    expect(set(auth["scope"]) == set(frozen["policy"]["stages"]), "authorization stages differ")
    expect(set(auth["forbidden"]) == PROHIBITIONS and auth["m1_continuous_simulation"] is True and
           auth["ego_scope"] == "CONDITIONAL_E0_TO_E3", "authorization boundary differs")
    expect(config["budget"] == LIMITS, "authorized overlay is exact and indivisible")
    train = config["training"]
    expect(train["target_physical_envs"] == 4096 and train["qualification_ladder"] == [64, 256, 1024, 2048, 4096],
           "physical scale differs")
    expect(train["target_cumulative_learner_iterations"] == 30_000 and train["utd"] == 0.5 and
           train["collection_ticks_per_iteration"] == 2 and
           train["scaled_rollout_target"] == 4096 * 2 * 30_000, "iteration/transition accounting differs")
    expect(train["reviewed_resource_fallbacks"] == [2048, 1024], "invalid resource fallback")
    expect(train["algorithm"] == "SuperMimic" and train["action_dim"] == 54 and train["history"] == 5,
           "frozen task identity differs")
    cadence = config["cadence"]
    expect(cadence == {"log_iter_range": [10, 50], "checkpoint_and_physical_eval_iters": 1000,
                       "review_iter_range": [1000, 2000], "wm_watchdog_physical_transitions": 250000},
           "training/evaluation cadence differs")
    expect(config["execution"]["dashboard_bind"] == "127.0.0.1", "dashboard must remain local")
    expect(config["replay"]["capacity_candidates"] == [128, 256, 512, 1024, 1800], "replay range differs")


def read():
    raw = CONFIG.read_bytes()
    frozen = subprocess.check_output([
        "git", "-C", str(REPO), "show", f"{BASELINE}:docs/s0-5/design_freeze_candidate.json",
    ])
    config = json.loads(raw)
    validate(config, frozen)
    return config, sha(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    config, checksum = read()
    print(json.dumps({
        "campaign_id": config["campaign_id"], "execution_config_sha256": checksum,
        "design_sha256": FROZEN_SHA, "dcr_sha256": sha(DCR.read_bytes()), "exit": 0,
    }))


if __name__ == "__main__":
    main()
