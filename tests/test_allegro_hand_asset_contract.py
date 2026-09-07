from __future__ import annotations

from pathlib import Path

import pytest

from dexteroushands_lab.assets.allegro_hand import (
    allegro_hand_asset_path,
    ALLEGRO_FLY_FINGER_JOINT_NAMES,
    ALLEGRO_FLY_ROOT_JOINT_NAMES,
    ALLEGRO_FLY_TIP_BODY_NAMES,
    allegro_description_root,
    allegro_hand_usd_path,
)


def test_allegro_joint_and_tip_contract_is_unique():
    assert len(ALLEGRO_FLY_ROOT_JOINT_NAMES) == 6
    assert len(ALLEGRO_FLY_FINGER_JOINT_NAMES) == 16
    assert len(set(ALLEGRO_FLY_ROOT_JOINT_NAMES + ALLEGRO_FLY_FINGER_JOINT_NAMES)) == 22
    assert len(ALLEGRO_FLY_TIP_BODY_NAMES) == 4
    assert len(set(ALLEGRO_FLY_TIP_BODY_NAMES)) == 4


def test_allegro_description_root_requires_an_existing_environment_root(monkeypatch, tmp_path):
    monkeypatch.delenv("ALLEGRO_DESCRIPTION_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="ALLEGRO_DESCRIPTION_ROOT"):
        allegro_description_root()

    monkeypatch.setenv("ALLEGRO_DESCRIPTION_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        allegro_description_root()


def test_allegro_asset_path_requires_an_asset_file(monkeypatch, tmp_path):
    monkeypatch.setenv("ALLEGRO_DESCRIPTION_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="asset"):
        allegro_hand_asset_path()

    usd = tmp_path / "allegro_hand_description_right_fly.usd"
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setenv("ALLEGRO_ASSET_PATH", usd.name)
    assert Path(allegro_hand_usd_path()) == usd
