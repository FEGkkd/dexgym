from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.environments.prepare_beta2_reset_candidates import rebind_candidates


def test_beta2_rebinding_discards_beta1_physics_certificate():
    source = {"seed": 7, "validated": False, "metadata": {
        "asset_variant": "beta1", "geometry_qualified": True,
        "hold_validated": True, "phase1b_pgs_reach_evidence": {"pgs_compatible": True},
        "geometry_evidence": {"native_physx": {"qualified": True}},
        "beta1_tip_marker_mass_override_kg": 0.001}}
    original = deepcopy(source)
    hand = lambda side: SimpleNamespace(source_commit="official", usd_sha256=side + "-usd",
                                        asset_tree_sha256=side + "-tree")
    result = rebind_candidates([source], "a" * 64, hand("left"), hand("right"))[0]
    assert source == original
    assert result["validated"] is False
    meta = result["metadata"]
    assert meta["asset_variant"] == "beta2"
    assert meta["left_usd_sha256"] == "left-usd"
    assert meta["right_asset_tree_sha256"] == "right-tree"
    assert meta["geometry_qualified"] is False
    assert meta["hold_validated"] is False
    assert "phase1b_pgs_reach_evidence" not in meta
    assert "geometry_evidence" not in meta
    assert "beta1_tip_marker_mass_override_kg" not in meta
    assert meta["beta2_candidate_provenance"]["beta1_physics_evidence_discarded"]


def test_beta2_rebinding_rejects_a_validated_grasp():
    with pytest.raises(ValueError, match="unvalidated"):
        rebind_candidates([{"validated": True, "metadata": {"asset_variant": "beta1"}}],
                          "a" * 64, None, None)
