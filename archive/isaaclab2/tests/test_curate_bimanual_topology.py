"""A held thumb-index pair must not be mislabeled as the target tripod."""

import importlib.util
from pathlib import Path


def test_right_loaded_contact_topology(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "source"))
    path = (root /
            "scripts/environments/curate_bimanual_grasp_bank.py")
    spec = importlib.util.spec_from_file_location("curate_bimanual_grasp_bank", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    hold = {"duration_s": 1.0,
            "right_finger_contact_time_s": {"thumb": 1.0, "index": 1.0}}
    assert module._right_contact_topology([hold, hold, hold]) == (
        "thumb_index_pinch", ("thumb", "index"))
    hold["right_finger_contact_time_s"]["middle"] = 1.0
    assert module._right_contact_topology([hold]*3)[0] == "thumb_index_middle_tripod"
