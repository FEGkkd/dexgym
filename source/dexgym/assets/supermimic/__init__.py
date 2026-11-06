"""Byte-identical reference files used by the Marvin SuperMimic examples."""

from pathlib import Path

_REFERENCES = Path(__file__).parent / "references"
WAVE_REFERENCE = _REFERENCES / "cmu_141_16_wave_bimanual.npz"
PROCEDURAL_REFERENCE = _REFERENCES / "procedural_bimanual_reach_v1.npz"
PD_WAVE_STATES = _REFERENCES / "newton_pd_states.npz"
WAVE_VALIDATION = _REFERENCES / "cmu_141_16_wave_bimanual_validation.json"

__all__ = ["WAVE_REFERENCE", "PROCEDURAL_REFERENCE", "PD_WAVE_STATES", "WAVE_VALIDATION"]
