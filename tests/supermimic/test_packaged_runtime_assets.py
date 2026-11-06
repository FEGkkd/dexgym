"""Pinned runtime assets must work from a checkout outside the original host path."""

import hashlib
from pathlib import Path

import mujoco


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_marvin_mjcf_is_packaged_and_loads_all_meshes():
    from dexgym.tasks.marvin_supermimic.robot_adapter import DEFAULT_MJCF

    packaged = Path(__file__).resolve().parents[2] / "source/dexgym/assets/marvin_wuji/source_model"
    assert DEFAULT_MJCF == packaged / "dual_arm_marvin.xml"
    assert _sha256(DEFAULT_MJCF) == "6e0d569ec28c8ac26a84bfc9a48652b7a11a2f113193c2d3ea4191621280b3ce"
    assert len(list((packaged / "assets").rglob("*.STL"))) == 70
    assert mujoco.MjModel.from_xml_path(str(DEFAULT_MJCF)).njnt == 54


def test_default_task_references_are_packaged_and_unchanged():
    from dexgym.assets.supermimic import PD_WAVE_STATES, PROCEDURAL_REFERENCE, WAVE_REFERENCE
    from dexgym.tasks.marvin_supermimic.env_cfg import MarvinWaveDirectCfg
    from dexgym.tasks.marvin_supermimic.reach_env import MarvinReachDirectCfg

    assert WAVE_REFERENCE.is_file() and PROCEDURAL_REFERENCE.is_file() and PD_WAVE_STATES.is_file()
    assert _sha256(WAVE_REFERENCE) == "b7af95b289e8aff898df7f5dd06b169f74dba2a506aa8fc5bf589ee3fee8106f"
    assert _sha256(PROCEDURAL_REFERENCE) == "ca4741f01734389dd62f9e64c66cdc9c237c90da6370cf76b13bcda89e143af2"
    assert _sha256(PD_WAVE_STATES) == "8f8c6c466c75141db01d55ca0bb0ea4f668f36c7541111ffe52783eecd5f91de"
    assert MarvinWaveDirectCfg().reference_path == str(WAVE_REFERENCE)
    assert MarvinReachDirectCfg().reset_state_path == str(PROCEDURAL_REFERENCE)
