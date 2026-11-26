"""Targeted controls for the pre-Actor wiring and fail-closed source admission."""
import json

import numpy as np
import pytest
import torch

from dexgym.tasks.direct.wholebody_s05.v1.learning.adapter import Actor, PhysicalCodec
from dexgym.tasks.direct.wholebody_s05.v1.learning.catalog import FrameCatalog
from dexgym.tasks.direct.wholebody_s05.v1.learning.train import DESIGN_FREEZE, PreActorLearner, digest, read_receipt


def test_causal_partition_and_generation_limits():
    allowed = np.array([True, True, False, True, True, True, True, True])
    generation = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    ticks = np.array([0, 1, 2, 3, 4, 5, 7, 8]) * 20_000_000
    assert FrameCatalog._lengths(allowed, generation, ticks).tolist() == [2, 1, 0, 2, 1, 1, 2, 1]


def test_actor_controls_stay_inside_54_axis_abi():
    torch.manual_seed(49)
    actor = Actor(25)
    value = actor(torch.rand((3, 25)))
    assert value.shape == (3, 54)
    assert [head.out_features for head in actor.heads] == [7, 7, 20, 20]
    assert torch.all(value.abs() < 1)
    assert torch.equal(actor(torch.zeros(1, 25)), actor(torch.zeros(1, 25)))


def test_physical_joint_codec_uses_bounded_joint_coordinates():
    start = torch.zeros(3, 108)
    target = (torch.arange(108) / 108).expand(3, -1) - 0.5
    assert torch.equal(PhysicalCodec.integrate(start, PhysicalCodec.difference(target, start)), target)


def test_pre_actor_learner_has_no_actor_or_critic_path():
    premature = PreActorLearner.__new__(PreActorLearner)
    with pytest.raises(RuntimeError, match="cannot update Actor or Critic"):
        premature.policy_update()
    with pytest.raises(RuntimeError, match="cannot update critic"):
        premature.optimize("critic", torch.zeros(()))


def test_receipt_aliases_cannot_forge_rmb_admission(tmp_path):
    path = tmp_path / "review.json"
    proof = {"gate": "G-A", "verdict": "TECHNICAL_PASS", "frozen_design_sha256": DESIGN_FREEZE,
             "manifest_sha256": "f" * 64, "implementation_agent": "implementer",
             "independent_reviewer": "reviewer"}
    path.write_text(json.dumps(proof))
    good = digest(path)
    assert read_receipt(path, good, "V1-A")["manifest_sha256"] == "f" * 64
    with pytest.raises(ValueError, match="wrong scope"):
        read_receipt(path, good, "V1-B")
    path.write_text(json.dumps({**proof, "independent_reviewer": "implementer", "reviewer": "reviewer"}))
    with pytest.raises(ValueError, match="corrupted"):
        read_receipt(path, good, "V1-A")
    with pytest.raises(ValueError, match="independent reviewer"):
        read_receipt(path, digest(path), "V1-A")
