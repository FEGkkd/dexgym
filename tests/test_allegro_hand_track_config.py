from __future__ import annotations

import importlib

import pytest

pytest.importorskip("isaaclab.sim")


def test_allegro_config_declares_contract_and_resolves_reference(monkeypatch, tmp_path):
    asset_root = tmp_path / "asset"
    asset_root.mkdir()
    (asset_root / "allegro_hand_description_right_fly.usd").write_text(
        "#usda 1.0\n", encoding="utf-8"
    )
    data_root = tmp_path / "grab"
    (data_root / "data").mkdir(parents=True)
    sample = data_root / "data" / "sample.npy"
    sample.write_bytes(b"placeholder")
    monkeypatch.setenv("ALLEGRO_DESCRIPTION_ROOT", str(asset_root))
    monkeypatch.setenv("DEXTRACK_DATA_ROOT", str(data_root))

    module = importlib.import_module(
        "dexteroushands_lab.tasks.direct.allegro_hand_track.allegro_hand_track_env_cfg"
    )
    contract = importlib.import_module(
        "dexteroushands_lab.tasks.direct.allegro_hand_track.task_contract"
    )
    cfg = module.AllegroHandTrackDirectEnvCfg()

    assert cfg.action_space == 22
    assert cfg.observation_space == contract.ALLEGRO_TRACK_CONTRACT.observation_dim
    assert cfg.resolved_reference_path.endswith("data/sample.npy")
